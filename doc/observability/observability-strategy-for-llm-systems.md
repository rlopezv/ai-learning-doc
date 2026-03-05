## Chapter 1 — Observability Strategy for LLM Systems

### 1.1 The Three Pillars Applied to AI Systems

The three observability pillars — metrics, logs, and traces — each address a different question about system behaviour:

**Metrics** answer "what is happening?" — aggregated numerical signals over time. For LLM systems, metrics capture throughput, latency percentiles, token usage, cost, error rates, and quality scores. Metrics are cheap to store and query at scale; they are the first layer of awareness.

**Logs** answer "why did it happen?" — discrete, structured event records. For LLM systems, logs capture the full request context: query text, retrieved documents, generated answer, token counts, and quality verdicts. Logs are expensive but essential for debugging specific incidents and building quality datasets from production traffic.

**Traces** answer "where did time go?" — the causal chain of operations across service boundaries. For RAG systems, a trace captures: receive query → retrieve documents → call LLM → parse response → return answer, with latency attributed to each step. Traces reveal which component is the bottleneck and where errors originate.

A mature LLM observability strategy uses metrics for alerting, logs for debugging and quality sampling, and traces for performance attribution and request-level investigation.

---

### 1.2 What Makes LLM Observability Different

Traditional application observability treats all requests as equivalent — a 200 response is success. LLM systems require two additional observability dimensions:

**Quality observability.** A request can return HTTP 200 with a hallucinated or irrelevant answer. Quality signals (faithfulness, relevancy, user ratings) must be captured and treated as first-class metrics, not incidental feedback.

**Cost observability.** Every LLM request has a direct monetary cost proportional to token consumption. Observability must track cost per team, per service, per model, and per query type — not as a finance exercise, but as an engineering signal for optimisation.

**Content observability.** The content of queries and responses is itself a signal. Observability infrastructure must capture query topics, answer characteristics, refusal rates, and citation patterns — not just performance numbers.

**Prompt version correlation.** When quality changes, the change must be correlatable with a prompt version, model version, or corpus update. Observability data must carry version metadata to enable this correlation.

---

### 1.3 Observability Data Model

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class RAGRequestTrace:
    """
    Complete observability record for a single RAG request.
    Captured once per request and written to the observability store.
    """
    # Identity
    request_id: str
    correlation_id: str
    trace_id: str             # OpenTelemetry trace ID
    session_id: Optional[str] = None
    team_id: str = "unknown"
    service_id: str = "unknown"

    # Request
    query_text: str = ""
    query_length_chars: int = 0
    query_category: Optional[str] = None  # Classified post-hoc

    # Retrieval
    retrieval_top_k: int = 5
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0

    # Generation
    prompt_template_id: str = ""
    prompt_version: str = ""
    model_alias: str = "default"
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_latency_ms: float = 0.0
    is_refusal: bool = False

    # Response
    answer_text: str = ""
    answer_length_chars: int = 0
    cited_doc_ids: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    http_status: int = 200

    # Cost
    estimated_cost_usd: float = 0.0

    # Quality (filled asynchronously post-request)
    user_rating: Optional[float] = None     # 1–5 or thumbs 0/1
    faithfulness_score: Optional[float] = None
    relevancy_score: Optional[float] = None
    online_judge_score: Optional[float] = None

    # Versions (for regression correlation)
    corpus_version: str = ""
    index_version: str = ""
    app_version: str = ""

    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_log_dict(self, include_content: bool = True) -> dict:
        """Serialise to structured log record."""
        d = self.__dict__.copy()
        if not include_content:
            d.pop("query_text", None)
            d.pop("answer_text", None)
        return d
```

---

### 1.4 Instrumentation Architecture

```python
import time
import functools
from typing import Callable, Any

class ObservabilityContext:
    """
    Thread-local context propagated through the request lifecycle.
    Allows any function in the call stack to add observability data
    without passing the trace object through every function signature.
    """
    import threading
    _local = __import__("threading").local()

    @classmethod
    def current(cls) -> "RAGRequestTrace":
        return getattr(cls._local, "trace", None)

    @classmethod
    def set(cls, trace: "RAGRequestTrace"):
        cls._local.trace = trace

    @classmethod
    def clear(cls):
        cls._local.trace = None

def observe_request(emit_fn: Callable):
    """
    Decorator that creates a RAGRequestTrace, populates it during
    the request lifecycle, and emits it on completion.
    """
    def decorator(handler: Callable) -> Callable:
        @functools.wraps(handler)
        async def wrapper(request, *args, **kwargs):
            import uuid
            trace = RAGRequestTrace(
                request_id=uuid.uuid4().hex,
                correlation_id=getattr(request, "correlation_id",
                                       uuid.uuid4().hex),
                trace_id=uuid.uuid4().hex,
                team_id=getattr(request, "team_id", "unknown"),
                service_id=getattr(request, "service_id", "unknown"),
                query_text=getattr(request, "question", ""),
                query_length_chars=len(getattr(request, "question", "")),
                timestamp=__import__("datetime").datetime.utcnow().isoformat()
            )
            ObservabilityContext.set(trace)
            t0 = time.perf_counter()
            try:
                result = await handler(request, *args, **kwargs)
                trace.http_status = 200
                if hasattr(result, "answer"):
                    trace.answer_text = result.answer or ""
                    trace.answer_length_chars = len(trace.answer_text)
                    trace.is_refusal = getattr(result, "is_refusal", False)
                return result
            except Exception as exc:
                trace.http_status = 500
                raise
            finally:
                trace.total_latency_ms = (time.perf_counter() - t0) * 1000
                ObservabilityContext.clear()
                emit_fn(trace)
        return wrapper
    return decorator
```

---

> ### 📋 Chapter Summary
>
> - Metrics answer "what is happening?", logs answer "why?", and traces answer "where did time go?" — all three are required for complete LLM observability.
> - LLM systems require two additional dimensions beyond traditional observability: **quality** (faithfulness, relevancy) and **cost** (tokens, USD per request).
> - `RAGRequestTrace` is the core observability record — one per request, carrying identity, retrieval, generation, quality, and version fields.
> - `ObservabilityContext` propagates the trace through the call stack without polluting function signatures.

---

> ### ❓ Comprehension Questions
>
> 1. A production incident reveals that answer quality degraded after a corpus update. None of the HTTP error rate metrics triggered an alert. What observability gap does this reveal, and which fields in `RAGRequestTrace` would have caught it earlier?
> 2. `ObservabilityContext` uses thread-local storage. A FastAPI async handler uses `async/await`. What concurrency problem can arise with thread-local storage in async contexts, and how would you solve it?
> 3. `RAGRequestTrace.answer_text` stores the full response. For a high-traffic service at 1000 QPS, this generates significant log volume. Design a sampling strategy that retains full text for quality analysis while managing storage costs.
> 4. Prompt version is stored in `RAGRequestTrace`. After a prompt update, quality metrics improve on average but degrade for the `adversarial` query category. What query and visualisation would you run in your log analytics tool to surface this?
> 5. `estimated_cost_usd` is computed per request. Over a month, the sum of per-request costs is 12% higher than the cloud provider's invoice. What are the likely causes of this discrepancy?

---

## References

### Documentation
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — Unified observability framework.
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [LangSmith Tracing](https://docs.smith.langchain.com/) — LLM-native tracing and evaluation.

### Books
- *Observability Engineering* — Majors, Fong-Jones & Miranda (O'Reilly). The three pillars and their application to distributed systems.
- *Site Reliability Engineering* — Beyer et al. (Google). Monitoring chapter on USE and RED methods.

---

---
[« Back to observability Index](index.md) | [🏠 Home](../index.md)