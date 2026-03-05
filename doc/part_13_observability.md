# Part XIII — Observability

---

> **Navigation**
> [← Part XII — Infrastructure](part_12_infrastructure.md) | [→ Part XIV — Security](part_14_security.md)

---

## Contents

- [Chapter 1 — Observability Strategy for LLM Systems](#chapter-1--observability-strategy-for-llm-systems)
  - [1.1 The Three Pillars Applied to AI Systems](#11-the-three-pillars-applied-to-ai-systems)
  - [1.2 What Makes LLM Observability Different](#12-what-makes-llm-observability-different)
  - [1.3 Observability Data Model](#13-observability-data-model)
  - [1.4 Instrumentation Architecture](#14-instrumentation-architecture)
- [Chapter 2 — Metrics and Dashboards](#chapter-2--metrics-and-dashboards)
  - [2.1 Core Metrics Taxonomy](#21-core-metrics-taxonomy)
  - [2.2 Prometheus Instrumentation](#22-prometheus-instrumentation)
  - [2.3 RED Method for LLM Services](#23-red-method-for-llm-services)
  - [2.4 Grafana Dashboard Design](#24-grafana-dashboard-design)
- [Chapter 3 — Structured Logging](#chapter-3--structured-logging)
  - [3.1 Log Schema for RAG Systems](#31-log-schema-for-rag-systems)
  - [3.2 Implementing Structured Logging](#32-implementing-structured-logging)
  - [3.3 Log Sampling and Cost Control](#33-log-sampling-and-cost-control)
  - [3.4 Log-Based Quality Signals](#34-log-based-quality-signals)
- [Chapter 4 — Distributed Tracing](#chapter-4--distributed-tracing)
  - [4.1 Tracing the RAG Request Lifecycle](#41-tracing-the-rag-request-lifecycle)
  - [4.2 OpenTelemetry for LLM Pipelines](#42-opentelemetry-for-llm-pipelines)
  - [4.3 Trace Sampling Strategies](#43-trace-sampling-strategies)
  - [4.4 Java Tracing with Spring Boot and OTel](#44-java-tracing-with-spring-boot-and-otel)
- [Chapter 5 — Alerting and Incident Response 🧪](#chapter-5--alerting-and-incident-response-)
  - [5.1 Alert Design Principles](#51-alert-design-principles)
  - [5.2 Prometheus Alerting Rules](#52-prometheus-alerting-rules)
  - [5.3 Quality Degradation Alerts](#53-quality-degradation-alerts)
  - [5.4 Runbook Integration](#54-runbook-integration)
  - [🧪 Hands-on Lab: Instrumented RAG Service](#-hands-on-lab-instrumented-rag-service)

---

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

## Chapter 2 — Metrics and Dashboards

### 2.1 Core Metrics Taxonomy

```python
from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# ── Request metrics ────────────────────────────────────────────────────
rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG requests",
    ["team_id", "service_id", "status", "is_refusal"]
)

rag_request_latency = Histogram(
    "rag_request_latency_ms",
    "End-to-end RAG request latency in milliseconds",
    ["team_id", "model_alias"],
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000]
)

# ── Component latency ─────────────────────────────────────────────────
rag_retrieval_latency = Histogram(
    "rag_retrieval_latency_ms",
    "Retrieval component latency",
    ["team_id"],
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)

rag_generation_latency = Histogram(
    "rag_generation_latency_ms",
    "LLM generation latency",
    ["team_id", "model_name"],
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000]
)

# ── Token and cost metrics ────────────────────────────────────────────
rag_tokens_total = Counter(
    "rag_tokens_total",
    "Total tokens consumed",
    ["team_id", "service_id", "model_name", "token_type"]  # token_type: prompt|completion
)

rag_cost_usd_total = Counter(
    "rag_cost_usd_total",
    "Estimated cost in USD",
    ["team_id", "service_id", "model_name"]
)

# ── Quality metrics ───────────────────────────────────────────────────
rag_refusal_rate = Gauge(
    "rag_refusal_rate",
    "Rolling refusal rate (fraction of requests that returned a refusal)",
    ["team_id"]
)

rag_user_rating = Histogram(
    "rag_user_rating",
    "User satisfaction ratings (1–5)",
    ["team_id", "service_id"],
    buckets=[1, 2, 3, 4, 5]
)

rag_quality_score = Gauge(
    "rag_quality_score",
    "Online quality score (LLM-judge, 0–1)",
    ["team_id", "metric"]   # metric: faithfulness | relevancy | overall
)

# ── Retrieval quality ─────────────────────────────────────────────────
rag_retrieval_hit_rate = Gauge(
    "rag_retrieval_hit_rate",
    "Hit rate@K from offline evaluation",
    ["index_id", "k"]
)

rag_index_vector_count = Gauge(
    "rag_index_vector_count",
    "Number of vectors in the active index",
    ["index_id", "collection"]
)
```

---

### 2.2 Prometheus Instrumentation

```python
import time
from contextlib import contextmanager

class RAGMetricsInstrumentor:
    """
    Wraps RAG pipeline components with automatic metrics emission.
    Used as a decorator or context manager at the service layer.
    """

    def __init__(self, team_id: str, service_id: str):
        self.team_id = team_id
        self.service_id = service_id

    @contextmanager
    def track_request(self, model_alias: str):
        """Context manager: tracks end-to-end request latency and status."""
        t0 = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            rag_request_latency.labels(
                team_id=self.team_id,
                model_alias=model_alias
            ).observe(elapsed)
            rag_requests_total.labels(
                team_id=self.team_id,
                service_id=self.service_id,
                status=status,
                is_refusal="false"
            ).inc()

    @contextmanager
    def track_retrieval(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            rag_retrieval_latency.labels(
                team_id=self.team_id
            ).observe((time.perf_counter() - t0) * 1000)

    @contextmanager
    def track_generation(self, model_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            rag_generation_latency.labels(
                team_id=self.team_id,
                model_name=model_name
            ).observe((time.perf_counter() - t0) * 1000)

    def record_tokens(self, model_name: str, prompt_tokens: int,
                      completion_tokens: int, cost_usd: float):
        rag_tokens_total.labels(
            team_id=self.team_id, service_id=self.service_id,
            model_name=model_name, token_type="prompt"
        ).inc(prompt_tokens)
        rag_tokens_total.labels(
            team_id=self.team_id, service_id=self.service_id,
            model_name=model_name, token_type="completion"
        ).inc(completion_tokens)
        rag_cost_usd_total.labels(
            team_id=self.team_id, service_id=self.service_id,
            model_name=model_name
        ).inc(cost_usd)

    def record_refusal(self, is_refusal: bool):
        # Update rolling gauge (in production: use windowed calculation)
        pass

    def record_user_rating(self, rating: float):
        rag_user_rating.labels(
            team_id=self.team_id,
            service_id=self.service_id
        ).observe(rating)


# Usage in the RAG pipeline
class InstrumentedRAGPipeline:
    def __init__(self, retriever, generator, team_id: str, service_id: str):
        self.retriever = retriever
        self.generator = generator
        self.metrics = RAGMetricsInstrumentor(team_id, service_id)

    def query(self, question: str, model_alias: str = "default") -> dict:
        with self.metrics.track_request(model_alias):
            # Retrieval
            with self.metrics.track_retrieval():
                retrieved = self.retriever(question, top_k=5)

            # Generation
            model_name = "gpt-4o-mini"
            with self.metrics.track_generation(model_name):
                context = [r["content"] for r in retrieved]
                result = self.generator(question, context)

            # Token and cost recording
            self.metrics.record_tokens(
                model_name=model_name,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                cost_usd=result.get("cost_usd", 0.0)
            )
            return result
```

---

### 2.3 RED Method for LLM Services

The RED method (Rate, Errors, Duration) from SRE practice maps naturally to LLM services, with extensions for AI-specific signals:

```python
# prometheus/rules/rag_red.yml — Prometheus recording rules for RED metrics

RED_RECORDING_RULES = """
groups:
  - name: rag_red_metrics
    interval: 30s
    rules:
      # Rate: requests per second (5-minute window)
      - record: rag:request_rate5m
        expr: |
          rate(rag_requests_total[5m])

      # Error rate: fraction of requests with status=error
      - record: rag:error_rate5m
        expr: |
          rate(rag_requests_total{status="error"}[5m])
          /
          rate(rag_requests_total[5m])

      # Duration: P50, P95, P99 latency
      - record: rag:latency_p50
        expr: histogram_quantile(0.50, rate(rag_request_latency_ms_bucket[5m]))

      - record: rag:latency_p95
        expr: histogram_quantile(0.95, rate(rag_request_latency_ms_bucket[5m]))

      - record: rag:latency_p99
        expr: histogram_quantile(0.99, rate(rag_request_latency_ms_bucket[5m]))

      # AI extensions: refusal rate and cost rate
      - record: rag:refusal_rate5m
        expr: |
          rate(rag_requests_total{is_refusal="true"}[5m])
          /
          rate(rag_requests_total[5m])

      - record: rag:cost_rate_per_hour
        expr: |
          increase(rag_cost_usd_total[1h])

      # Component breakdown: what fraction of latency is retrieval vs generation
      - record: rag:retrieval_fraction
        expr: |
          rate(rag_retrieval_latency_ms_sum[5m])
          /
          rate(rag_request_latency_ms_sum[5m])
"""
```

---

### 2.4 Grafana Dashboard Design

```python
GRAFANA_DASHBOARD_SPEC = {
    "title": "AI Platform — RAG Service Overview",
    "uid": "rag-overview-v2",
    "tags": ["ai", "rag", "production"],
    "refresh": "30s",
    "panels": [
        # Row 1: Traffic and availability
        {
            "title": "Request Rate (RPS)",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 0, "w": 6, "h": 6},
            "targets": [{
                "expr": "sum(rag:request_rate5m) by (team_id)",
                "legendFormat": "{{team_id}}"
            }]
        },
        {
            "title": "Error Rate",
            "type": "timeseries",
            "gridPos": {"x": 6, "y": 0, "w": 6, "h": 6},
            "fieldConfig": {"defaults": {"unit": "percentunit", "max": 1}},
            "targets": [{
                "expr": "sum(rag:error_rate5m) by (team_id)",
                "legendFormat": "{{team_id}}"
            }]
        },
        {
            "title": "P99 Latency (ms)",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 0, "w": 6, "h": 6},
            "targets": [{
                "expr": "rag:latency_p99",
                "legendFormat": "P99"
            }, {
                "expr": "rag:latency_p95",
                "legendFormat": "P95"
            }],
            "alert": {
                "name": "High P99 Latency",
                "conditions": [{"evaluator": {"type": "gt", "params": [3000]}}]
            }
        },
        # Row 2: Quality signals
        {
            "title": "Refusal Rate",
            "type": "gauge",
            "gridPos": {"x": 0, "y": 6, "w": 4, "h": 4},
            "fieldConfig": {
                "defaults": {"unit": "percentunit", "min": 0, "max": 1,
                             "thresholds": {"steps": [
                                 {"value": 0, "color": "green"},
                                 {"value": 0.15, "color": "yellow"},
                                 {"value": 0.30, "color": "red"}
                             ]}}
            },
            "targets": [{"expr": "avg(rag:refusal_rate5m)"}]
        },
        {
            "title": "User Satisfaction (P50 rating)",
            "type": "stat",
            "gridPos": {"x": 4, "y": 6, "w": 4, "h": 4},
            "fieldConfig": {
                "defaults": {"unit": "none", "min": 1, "max": 5,
                             "thresholds": {"steps": [
                                 {"value": 1, "color": "red"},
                                 {"value": 3.5, "color": "yellow"},
                                 {"value": 4.0, "color": "green"}
                             ]}}
            },
            "targets": [{
                "expr": "histogram_quantile(0.50, sum(rate(rag_user_rating_bucket[1h])) by (le))"
            }]
        },
        {
            "title": "Online Quality Score (Faithfulness)",
            "type": "timeseries",
            "gridPos": {"x": 8, "y": 6, "w": 8, "h": 4},
            "fieldConfig": {"defaults": {"unit": "percentunit", "min": 0, "max": 1}},
            "targets": [{
                "expr": "rag_quality_score{metric='faithfulness'}",
                "legendFormat": "Faithfulness"
            }, {
                "expr": "rag_quality_score{metric='relevancy'}",
                "legendFormat": "Relevancy"
            }]
        },
        # Row 3: Cost
        {
            "title": "Hourly Cost by Team (USD)",
            "type": "bargauge",
            "gridPos": {"x": 0, "y": 10, "w": 12, "h": 6},
            "fieldConfig": {"defaults": {"unit": "currencyUSD"}},
            "targets": [{
                "expr": "sum(rag:cost_rate_per_hour) by (team_id)",
                "legendFormat": "{{team_id}}"
            }]
        },
        {
            "title": "Token Usage by Model",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 10, "w": 12, "h": 6},
            "targets": [{
                "expr": "sum(rate(rag_tokens_total[5m])) by (model_name, token_type)",
                "legendFormat": "{{model_name}} / {{token_type}}"
            }]
        },
    ]
}
```

---

> ### 📋 Chapter Summary
>
> - AI service metrics extend the RED method with **refusal rate**, **quality scores**, and **cost rate** — none of which have equivalents in traditional services.
> - `RAGMetricsInstrumentor` uses context managers to wrap retrieval, generation, and the full request — injecting metrics at each component boundary.
> - Prometheus recording rules pre-compute rate, error rate, and latency percentiles at 30-second intervals, making dashboard queries fast.
> - Grafana dashboard rows map to concerns: traffic/availability → quality → cost. Each row has its own thresholds and alert anchors.

---

> ### ❓ Comprehension Questions
>
> 1. The RED method tracks Rate, Errors, and Duration. A quality degradation (faithfulness drops from 0.92 to 0.74) occurs with no change in any RED metric. What does this demonstrate about the sufficiency of RED for LLM services?
> 2. `rag_refusal_rate` is a Gauge updated per request. At high traffic (500 RPS), individual Gauge updates create contention. How would you compute a windowed refusal rate using a Counter instead?
> 3. The Grafana dashboard refreshes every 30 seconds. A team reports that latency spikes are invisible because they last only 5 seconds. What combination of recording rule window and dashboard refresh rate would capture sub-30-second spikes?
> 4. `rag_cost_usd_total` is a Counter (monotonically increasing). To show "cost in the last hour", the query uses `increase(rag_cost_usd_total[1h])`. What does `increase` compute, and why is it preferable to `rate` for cost reporting?
> 5. The user satisfaction histogram uses `buckets=[1, 2, 3, 4, 5]`. A product manager wants the P25 satisfaction score. Write the PromQL expression, and explain why the bucket boundaries matter for percentile accuracy.

---

## References

### Documentation
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Prometheus Recording Rules](https://prometheus.io/docs/practices/rules/)
- [RED Method](https://www.weave.works/blog/the-red-method-key-metrics-for-microservices-architecture) — Rate, Error, Duration for microservices.

---

## Chapter 3 — Structured Logging

### 3.1 Log Schema for RAG Systems

Unstructured log messages ("Answering query for user") cannot be queried, aggregated, or analysed at scale. Structured logs encode every field as a typed key-value pair, enabling exact and range queries in log analytics systems.

```python
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
from enum import Enum
import json

class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"

class EventType(str, Enum):
    REQUEST_RECEIVED       = "request.received"
    RETRIEVAL_COMPLETED    = "retrieval.completed"
    GENERATION_COMPLETED   = "generation.completed"
    REQUEST_COMPLETED      = "request.completed"
    QUALITY_SCORED         = "quality.scored"
    REFUSAL_RETURNED       = "refusal.returned"
    ERROR_OCCURRED         = "error.occurred"
    CACHE_HIT              = "cache.hit"

@dataclass
class StructuredLogRecord:
    """
    Canonical log schema for all RAG service events.
    All fields are typed; no free-form message strings for queryable data.
    """
    # Mandatory fields — every log record
    timestamp: str
    level: str
    event_type: str
    service_id: str
    team_id: str
    request_id: str
    trace_id: str

    # Message (human-readable, not parsed programmatically)
    message: str = ""

    # Request context
    query_length_chars: Optional[int] = None
    session_id: Optional[str] = None

    # Retrieval context
    retrieved_count: Optional[int] = None
    retrieval_latency_ms: Optional[float] = None
    top_score: Optional[float] = None

    # Generation context
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    generation_latency_ms: Optional[float] = None
    is_refusal: Optional[bool] = None

    # Response context
    answer_length_chars: Optional[int] = None
    citation_count: Optional[int] = None
    total_latency_ms: Optional[float] = None
    http_status: Optional[int] = None

    # Cost
    estimated_cost_usd: Optional[float] = None

    # Quality
    faithfulness_score: Optional[float] = None
    relevancy_score: Optional[float] = None
    user_rating: Optional[float] = None

    # Error context
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Versions
    app_version: Optional[str] = None
    index_version: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {k: v for k, v in asdict(self).items() if v is not None},
            default=str
        )
```

---

### 3.2 Implementing Structured Logging

```python
import logging
import sys
from datetime import datetime

class StructuredLogger:
    """
    Wraps Python's logging with structured JSON output.
    Produces one JSON object per line — compatible with
    Elasticsearch, Loki, CloudWatch Logs Insights, and GCP Logs.
    """
    def __init__(self, service_id: str, app_version: str):
        self.service_id = service_id
        self.app_version = app_version
        self._logger = logging.getLogger(service_id)
        self._configure_json_handler()

    def _configure_json_handler(self):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

    def emit(self, record: StructuredLogRecord):
        record.service_id = record.service_id or self.service_id
        record.app_version = record.app_version or self.app_version
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }
        self._logger.log(
            level_map.get(record.level, logging.INFO),
            record.to_json()
        )

    def request_completed(self, trace: "RAGRequestTrace"):
        self.emit(StructuredLogRecord(
            timestamp=datetime.utcnow().isoformat(),
            level=LogLevel.INFO,
            event_type=EventType.REQUEST_COMPLETED,
            service_id=trace.service_id,
            team_id=trace.team_id,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            message="RAG request completed",
            query_length_chars=trace.query_length_chars,
            retrieved_count=len(trace.retrieved_doc_ids),
            retrieval_latency_ms=round(trace.retrieval_latency_ms, 2),
            model_name=trace.model_name,
            prompt_version=trace.prompt_version,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            generation_latency_ms=round(trace.generation_latency_ms, 2),
            is_refusal=trace.is_refusal,
            answer_length_chars=trace.answer_length_chars,
            total_latency_ms=round(trace.total_latency_ms, 2),
            http_status=trace.http_status,
            estimated_cost_usd=trace.estimated_cost_usd,
            index_version=trace.index_version,
        ))

    def error(self, request_id: str, trace_id: str,
              team_id: str, error_type: str, error_message: str):
        self.emit(StructuredLogRecord(
            timestamp=datetime.utcnow().isoformat(),
            level=LogLevel.ERROR,
            event_type=EventType.ERROR_OCCURRED,
            service_id=self.service_id,
            team_id=team_id,
            request_id=request_id,
            trace_id=trace_id,
            message=f"Error: {error_type}",
            error_type=error_type,
            error_message=error_message[:500]  # Truncate for log safety
        ))


class JsonFormatter(logging.Formatter):
    """Output log records as raw JSON (already formatted by StructuredLogRecord)."""
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


# Example log query patterns (Elasticsearch / Loki / CloudWatch)
LOG_QUERY_EXAMPLES = {
    "high_latency_requests": {
        "description": "Requests with P99 > 3000ms in the last hour",
        "loki": '{service_id="rag-api"} | json | total_latency_ms > 3000',
        "elasticsearch": {
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"total_latency_ms": {"gt": 3000}}},
                        {"range": {"@timestamp": {"gte": "now-1h"}}}
                    ]
                }
            }
        }
    },
    "refusal_rate_by_team": {
        "description": "Refusal rate per team in last 24h",
        "loki": 'sum by (team_id) (count_over_time({service_id="rag-api"} | json | is_refusal="true" [24h])) / sum by (team_id) (count_over_time({service_id="rag-api"} | json [24h]))',
    },
    "expensive_requests": {
        "description": "Top 10 most expensive requests today",
        "loki": '{service_id="rag-api"} | json | sort by (estimated_cost_usd) desc | limit 10',
    },
}
```

---

### 3.3 Log Sampling and Cost Control

At high traffic volumes, logging every request becomes prohibitively expensive. Sampling reduces log volume while preserving visibility into important events.

```python
import random
import hashlib
from enum import Enum

class SamplingDecision(str, Enum):
    ALWAYS   = "always"    # Always log regardless of sampling rate
    SAMPLED  = "sampled"   # Log at configured rate
    NEVER    = "never"     # Drop this log record

class AdaptiveLogSampler:
    """
    Adaptive log sampler:
    - Errors:           always logged (rate=1.0)
    - Refusals:         always logged (quality signal)
    - High latency:     always logged (performance signal)
    - High cost:        always logged (cost signal)
    - Normal requests:  sampled at configured base_rate
    """
    def __init__(
        self,
        base_rate: float = 0.10,         # Sample 10% of normal requests
        high_latency_threshold_ms: float = 2000,
        high_cost_threshold_usd: float = 0.01
    ):
        self.base_rate = base_rate
        self.high_latency_ms = high_latency_threshold_ms
        self.high_cost_usd = high_cost_threshold_usd

    def should_log(self, trace: "RAGRequestTrace") -> tuple[bool, str]:
        """Returns (should_log, reason)."""
        # Always log errors
        if trace.http_status >= 500:
            return True, "always:error"

        # Always log refusals (quality monitoring)
        if trace.is_refusal:
            return True, "always:refusal"

        # Always log high-latency requests (performance debugging)
        if trace.total_latency_ms > self.high_latency_ms:
            return True, "always:high_latency"

        # Always log high-cost requests (cost monitoring)
        if trace.estimated_cost_usd > self.high_cost_usd:
            return True, "always:high_cost"

        # Deterministic sampling based on request_id for reproducibility
        # (same request always makes same sampling decision across retries)
        sample_hash = int(hashlib.md5(trace.request_id.encode()).hexdigest(), 16)
        if (sample_hash % 10000) < int(self.base_rate * 10000):
            return True, f"sampled:{self.base_rate:.0%}"

        return False, "dropped"

    def get_sample_rate_for_metric(self, reason: str) -> float:
        """Return effective sample rate for metric adjustment."""
        if reason.startswith("always"):
            return 1.0
        return self.base_rate
```

---

### 3.4 Log-Based Quality Signals

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

class LogQualityAnalyser:
    """
    Computes quality signals from recent log records.
    Acts as a lightweight alternative to full evaluation pipeline
    for real-time quality monitoring.
    """
    def __init__(self, window_minutes: int = 60):
        self.window = timedelta(minutes=window_minutes)
        self._records: deque = deque()

    def ingest(self, log_record: dict):
        """Ingest a parsed log record."""
        self._records.append({
            **log_record,
            "_ingested_at": datetime.utcnow()
        })
        self._evict_old()

    def compute_signals(self) -> dict:
        """Compute quality signals over the current window."""
        records = list(self._records)
        if not records:
            return {}

        total = len(records)
        refusals = sum(1 for r in records if r.get("is_refusal"))
        errors = sum(1 for r in records if (r.get("http_status") or 0) >= 500)
        high_latency = sum(1 for r in records
                           if (r.get("total_latency_ms") or 0) > 2000)

        latencies = [r["total_latency_ms"] for r in records
                     if r.get("total_latency_ms")]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        by_team = defaultdict(lambda: {"total": 0, "refusals": 0, "cost": 0.0})
        for r in records:
            team = r.get("team_id", "unknown")
            by_team[team]["total"] += 1
            if r.get("is_refusal"):
                by_team[team]["refusals"] += 1
            by_team[team]["cost"] += r.get("estimated_cost_usd") or 0

        return {
            "window_minutes": self.window.seconds // 60,
            "total_requests": total,
            "error_rate": round(errors / total, 4) if total else 0,
            "refusal_rate": round(refusals / total, 4) if total else 0,
            "high_latency_rate": round(high_latency / total, 4) if total else 0,
            "latency_p50_ms": round(latencies_sorted[n // 2], 1) if n else None,
            "latency_p99_ms": round(latencies_sorted[int(n * 0.99)], 1) if n else None,
            "total_cost_usd": round(sum(r.get("estimated_cost_usd") or 0
                                        for r in records), 4),
            "by_team": {
                team: {
                    "requests": v["total"],
                    "refusal_rate": round(v["refusals"] / v["total"], 4)
                                    if v["total"] else 0,
                    "cost_usd": round(v["cost"], 4)
                }
                for team, v in by_team.items()
            }
        }

    def _evict_old(self):
        cutoff = datetime.utcnow() - self.window
        while self._records and self._records[0]["_ingested_at"] < cutoff:
            self._records.popleft()
```

---

> ### 📋 Chapter Summary
>
> - **Structured logs** encode every field as typed key-value pairs — enabling exact queries, aggregations, and anomaly detection at scale.
> - `StructuredLogRecord` with `EventType` enumeration provides a consistent schema across all RAG service events.
> - **Adaptive sampling** always logs errors, refusals, high-latency, and high-cost requests; samples normal requests at a configurable rate — typically 10%.
> - `LogQualityAnalyser` computes quality signals (refusal rate, error rate, latency percentiles, cost) from recent logs, enabling lightweight real-time quality monitoring.

---

> ### ❓ Comprehension Questions
>
> 1. A log record omits `query_text` by default for privacy but includes it when the request is sampled. An engineer needs to debug a specific refusal. How would the `should_log` method ensure the query text is available for all refusals?
> 2. `AdaptiveLogSampler` uses `hashlib.md5(trace.request_id)` for deterministic sampling. Why is determinism important here, and what failure mode does it prevent compared to `random.random()`?
> 3. Log storage costs $0.50/GB. At 1000 RPS with an average log record size of 1KB, calculate the daily log volume and cost at: (a) 100% sampling rate, (b) 10% base rate with always-log for errors (2%), refusals (5%), and high-latency (3%).
> 4. `LogQualityAnalyser.compute_signals()` holds all window records in memory. At 1000 RPS with a 60-minute window, how many records could this deque hold, and what is the approximate memory cost?
> 5. The log schema includes `index_version`. After correlating logs from the past week, you find that refusal rate is 8% on `index_version=v23` but 22% on `index_version=v24`. What investigation steps follow this finding?

---

## References

### Documentation
- [Grafana Loki](https://grafana.com/docs/loki/latest/) — Log aggregation system.
- [OpenSearch / Elasticsearch](https://opensearch.org/docs/latest/) — Log analytics.
- [Structlog](https://www.structlog.org/en/stable/) — Python structured logging library.
- [AWS CloudWatch Logs Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AnalyzingLogData.html)

---

## Chapter 4 — Distributed Tracing

### 4.1 Tracing the RAG Request Lifecycle

A single RAG request spans multiple service boundaries and internal components. Distributed tracing captures the causal chain with precise timing:

```
RAG Request Trace (example: 847ms total)

├── [0ms]     receive_request          2ms
├── [2ms]     validate_and_classify    3ms
├── [5ms]  ┬─ retrieve_documents      120ms
│          ├── embed_query             45ms   (embedding service call)
│          └── vector_search           75ms   (Qdrant query)
├── [125ms] ┬─ generate_answer        710ms
│           ├── render_prompt          2ms    (prompt service call)
│           ├── llm_gateway_call       700ms  (OpenAI API)
│           └── parse_response         8ms
└── [835ms]    format_and_return       12ms
```

Each box is a **span** — a named, timed operation with a parent-child relationship. The root span represents the entire request. Spans carry attributes (key-value pairs) and events (timestamped annotations).

---

### 4.2 OpenTelemetry for LLM Pipelines

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
import time

# ── Initialise OTel tracer ────────────────────────────────────────────
def configure_tracing(service_name: str, otlp_endpoint: str = "http://jaeger:4317"):
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "2.3.1",
        "deployment.environment": "production"
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

tracer = configure_tracing("rag-api")


# ── Instrumented RAG pipeline ─────────────────────────────────────────
class TracedRAGPipeline:
    """
    RAG pipeline with full OpenTelemetry instrumentation.
    Every logical operation is a child span with relevant attributes.
    """
    def __init__(self, retriever, generator, embedder, prompt_client):
        self.retriever = retriever
        self.generator = generator
        self.embedder = embedder
        self.prompt_client = prompt_client

    def query(self, question: str, team_id: str,
              model_alias: str = "default") -> dict:
        with tracer.start_as_current_span(
            "rag.request",
            attributes={
                "rag.team_id":        team_id,
                "rag.model_alias":    model_alias,
                "rag.query.length":   len(question),
            }
        ) as root_span:
            try:
                # ── Embed query ──────────────────────────────────────
                with tracer.start_as_current_span("rag.embed_query") as span:
                    t0 = time.perf_counter()
                    query_embedding = self.embedder.embed([question])[0]
                    span.set_attribute("rag.embed.latency_ms",
                                       (time.perf_counter() - t0) * 1000)
                    span.set_attribute("rag.embed.dimensions",
                                       len(query_embedding))

                # ── Vector search ─────────────────────────────────────
                with tracer.start_as_current_span("rag.vector_search") as span:
                    t0 = time.perf_counter()
                    retrieved = self.retriever.search(query_embedding, top_k=5)
                    elapsed = (time.perf_counter() - t0) * 1000
                    span.set_attribute("rag.retrieval.count",
                                       len(retrieved))
                    span.set_attribute("rag.retrieval.latency_ms", elapsed)
                    if retrieved:
                        span.set_attribute("rag.retrieval.top_score",
                                           retrieved[0].get("score", 0))

                root_span.set_attribute("rag.retrieval.doc_ids",
                                        str([r["id"] for r in retrieved]))

                # ── Render prompt ─────────────────────────────────────
                with tracer.start_as_current_span("rag.render_prompt") as span:
                    context_text = "\n".join(
                        [f"[{i+1}] {r['content']}"
                         for i, r in enumerate(retrieved)]
                    )
                    messages = self.prompt_client.get_messages(
                        "support_rag",
                        {"context": context_text, "question": question}
                    )
                    span.set_attribute("rag.prompt.template_id", "support_rag")
                    span.set_attribute("rag.prompt.message_count", len(messages))

                # ── LLM generation ─────────────────────────────────────
                with tracer.start_as_current_span("rag.llm_generation") as span:
                    t0 = time.perf_counter()
                    response = self.generator(messages, model_alias)
                    elapsed = (time.perf_counter() - t0) * 1000
                    span.set_attribute("rag.generation.latency_ms", elapsed)
                    span.set_attribute("rag.generation.model",
                                       response.get("model", ""))
                    span.set_attribute("rag.generation.prompt_tokens",
                                       response.get("prompt_tokens", 0))
                    span.set_attribute("rag.generation.completion_tokens",
                                       response.get("completion_tokens", 0))
                    span.set_attribute("rag.generation.cost_usd",
                                       response.get("cost_usd", 0.0))

                root_span.set_status(Status(StatusCode.OK))
                return response

            except Exception as exc:
                root_span.set_status(Status(StatusCode.ERROR, str(exc)))
                root_span.record_exception(exc)
                raise
```

---

### 4.3 Trace Sampling Strategies

```python
from opentelemetry.sdk.trace.sampling import (
    Sampler, SamplingResult, Decision, ParentBased, TraceIdRatioBased
)
from opentelemetry.trace import SpanKind
from opentelemetry.util.types import Attributes
from typing import Optional, Sequence

class AdaptiveTraceSampler(Sampler):
    """
    Custom OTel sampler with adaptive rates:
    - Errors:          always sample (Decision.RECORD_AND_SAMPLE)
    - High latency:    always sample (detected via baggage)
    - Health checks:   never sample (Decision.DROP)
    - Normal traffic:  sample at base_rate (1% in production)
    """
    def __init__(self, base_rate: float = 0.01):
        self.base_rate = base_rate
        self._ratio_sampler = TraceIdRatioBased(base_rate)

    def should_sample(
        self,
        parent_context,
        trace_id: int,
        name: str,
        kind: SpanKind = None,
        attributes: Attributes = None,
        links: Sequence = None,
        trace_state=None,
    ) -> SamplingResult:
        attrs = attributes or {}

        # Drop health check spans (very high frequency, zero signal)
        if name in ("/health", "/ready", "/metrics"):
            return SamplingResult(Decision.DROP)

        # Always sample error conditions
        if attrs.get("error") or attrs.get("http.status_code", 0) >= 500:
            return SamplingResult(Decision.RECORD_AND_SAMPLE)

        # Always sample refusals and high-cost requests
        if attrs.get("rag.is_refusal") or attrs.get("rag.generation.cost_usd", 0) > 0.05:
            return SamplingResult(Decision.RECORD_AND_SAMPLE)

        # Fall through to ratio-based sampling
        return self._ratio_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"AdaptiveTraceSampler(base_rate={self.base_rate})"
```

---

### 4.4 Java Tracing with Spring Boot and OTel

```java
// build.gradle — OpenTelemetry dependencies for Spring Boot
dependencies {
    implementation("io.opentelemetry:opentelemetry-api:1.42.0")
    implementation("io.opentelemetry:opentelemetry-sdk:1.42.0")
    implementation("io.opentelemetry.instrumentation:opentelemetry-spring-boot-starter:2.8.0")
}

// application.yml
// otel:
//   exporter:
//     otlp:
//       endpoint: http://jaeger:4317
//   traces:
//     sampler: parentbased_traceidratio
//     sampler.arg: "0.01"

// ── Traced RAG service ────────────────────────────────────────────────
@Service
public class TracedRAGService {

    private final Tracer tracer;
    private final DocumentRetriever retriever;
    private final ChatLanguageModel llm;

    public TracedRAGService(OpenTelemetry openTelemetry,
                            DocumentRetriever retriever,
                            ChatLanguageModel llm) {
        this.tracer = openTelemetry.getTracer("rag-service", "2.3.1");
        this.retriever = retriever;
        this.llm = llm;
    }

    public RAGResponse query(String question, String teamId) {
        Span rootSpan = tracer.spanBuilder("rag.request")
            .setAttribute("rag.team_id", teamId)
            .setAttribute("rag.query.length", question.length())
            .startSpan();

        try (Scope scope = rootSpan.makeCurrent()) {
            // Retrieval span
            var docs = tracedRetrieve(question);

            // Generation span
            var answer = tracedGenerate(question, docs);

            rootSpan.setStatus(StatusCode.OK);
            return new RAGResponse(answer, docs);

        } catch (Exception e) {
            rootSpan.setStatus(StatusCode.ERROR, e.getMessage());
            rootSpan.recordException(e);
            throw new RAGServiceException("Query failed", e);
        } finally {
            rootSpan.end();
        }
    }

    private List<Document> tracedRetrieve(String question) {
        Span span = tracer.spanBuilder("rag.retrieval").startSpan();
        try (Scope scope = span.makeCurrent()) {
            long t0 = System.currentTimeMillis();
            var docs = retriever.retrieve(question, 5);
            span.setAttribute("rag.retrieval.count", docs.size());
            span.setAttribute("rag.retrieval.latency_ms",
                              System.currentTimeMillis() - t0);
            return docs;
        } finally {
            span.end();
        }
    }

    private String tracedGenerate(String question, List<Document> docs) {
        Span span = tracer.spanBuilder("rag.generation").startSpan();
        try (Scope scope = span.makeCurrent()) {
            long t0 = System.currentTimeMillis();
            String answer = llm.generate(buildPrompt(question, docs));
            span.setAttribute("rag.generation.latency_ms",
                              System.currentTimeMillis() - t0);
            return answer;
        } finally {
            span.end();
        }
    }

    private String buildPrompt(String question, List<Document> docs) {
        String context = IntStream.range(0, docs.size())
            .mapToObj(i -> "[" + (i+1) + "] " + docs.get(i).content())
            .collect(Collectors.joining("\n"));
        return "Context:\n" + context + "\n\nQuestion: " + question;
    }
}
```

---

> ### 📋 Chapter Summary
>
> - A RAG request trace decomposes into spans: embed_query → vector_search → render_prompt → llm_generation, each with latency and attributes.
> - OpenTelemetry provides a vendor-neutral tracing API — the same code exports to Jaeger, Zipkin, Grafana Tempo, or any OTLP-compatible backend.
> - **Adaptive trace sampling** drops health checks, always captures errors and refusals, and samples normal traffic at 1% — balancing signal with storage cost.
> - Java Spring Boot OTel integration uses the same span structure and attributes as the Python implementation, enabling unified trace analysis.

---

> ### ❓ Comprehension Questions
>
> 1. A P99 latency alert fires. A developer opens Jaeger and finds that all slow requests have `rag.vector_search.latency_ms > 2000` while `rag.llm_generation.latency_ms` is normal. What does this trace pattern indicate and what would you investigate next?
> 2. `AdaptiveTraceSampler` drops spans named `/health` and `/ready`. These endpoints are called by Kubernetes every 5 seconds per pod. Justify this sampling decision with a storage cost calculation for 20 pods over 30 days.
> 3. A span attribute `rag.retrieval.doc_ids` stores the IDs of retrieved documents as a string. At 500 RPS, this attribute is stored for 1% of requests (5 RPS). A developer wants to increase sampling to 10% (50 RPS). Estimate the additional trace storage cost per month assuming 50 bytes per span attribute.
> 4. The `tracedGenerate` Java method creates a child span manually. Spring Boot OTel auto-instrumentation already instruments HTTP and database calls. What types of spans would NOT be captured by auto-instrumentation and therefore require manual span creation?
> 5. Two requests have identical total latency (950ms) but different span breakdowns: Request A (embed=40ms, search=810ms, generate=100ms) vs Request B (embed=40ms, search=100ms, generate=810ms). What different optimisations would each request trace suggest?

---

## References

### Documentation
- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)
- [OpenTelemetry Java](https://opentelemetry.io/docs/languages/java/)
- [Jaeger Tracing](https://www.jaegertracing.io/docs/)
- [Grafana Tempo](https://grafana.com/docs/tempo/latest/)
- [Spring Boot OTel Starter](https://opentelemetry.io/docs/zero-code/java/spring-boot-starter/)

### Papers
- [Dapper: Google's Distributed Tracing Infrastructure](https://research.google/pubs/pub36356/) — Sigelman et al., 2010. Foundational paper on distributed tracing design.

---

## Chapter 5 — Alerting and Incident Response 🧪

### 5.1 Alert Design Principles

Good alerts are **actionable** (the on-call engineer knows what to do), **precise** (few false positives), and **sensitive enough** (few false negatives). For LLM systems, four alert categories are required:

**Availability alerts** — the service is down or error rate is unacceptably high. These are the same as any web service: P99 latency breached, error rate above threshold, pod crash loops.

**Quality degradation alerts** — faithfulness or user satisfaction drops below baseline. These are unique to LLM systems and have no equivalent in traditional services.

**Cost alerts** — daily spend exceeds budget. Token costs can spike unexpectedly due to prompt changes, traffic increases, or routing errors.

**Data freshness alerts** — the knowledge base has not been updated within the expected window. Stale data causes gradual quality degradation without triggering availability alerts.

---

### 5.2 Prometheus Alerting Rules

```yaml
# infrastructure/prometheus/alerts/rag_alerts.yml
groups:
  - name: rag_availability
    rules:
      - alert: RAGHighErrorRate
        expr: |
          sum(rate(rag_requests_total{status="error"}[5m]))
          /
          sum(rate(rag_requests_total[5m]))
          > 0.02
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "RAG error rate {{ $value | humanizePercentage }} exceeds 2%"
          description: "Error rate has been above 2% for 2 minutes. Check gateway logs and LLM provider status."
          runbook: "https://wiki.company.com/runbooks/rag-high-error-rate"

      - alert: RAGP99LatencyHigh
        expr: |
          histogram_quantile(0.99,
            sum(rate(rag_request_latency_ms_bucket[5m])) by (le)
          ) > 3000
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "RAG P99 latency {{ $value | humanize }}ms exceeds 3000ms"
          description: "P99 latency has been above 3s for 5 minutes. Check vector DB and LLM gateway."
          runbook: "https://wiki.company.com/runbooks/rag-high-latency"

      - alert: RAGServiceDown
        expr: |
          absent(up{job="rag-api"} == 1)
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "RAG API service is not responding"
          runbook: "https://wiki.company.com/runbooks/rag-service-down"

  - name: rag_quality
    rules:
      - alert: RAGHighRefusalRate
        expr: |
          sum(rate(rag_requests_total{is_refusal="true"}[30m]))
          /
          sum(rate(rag_requests_total[30m]))
          > 0.25
        for: 10m
        labels:
          severity: warning
          team: ai-platform
        annotations:
          summary: "RAG refusal rate {{ $value | humanizePercentage }} exceeds 25%"
          description: "High refusal rate may indicate knowledge base staleness or query distribution shift."
          runbook: "https://wiki.company.com/runbooks/rag-high-refusal-rate"

      - alert: RAGQualityDegraded
        expr: |
          rag_quality_score{metric="faithfulness"} < 0.80
        for: 15m
        labels:
          severity: warning
          team: ai-platform
        annotations:
          summary: "RAG faithfulness score {{ $value | humanize }} below 0.80"
          description: "Online faithfulness score has been below threshold for 15 minutes."
          runbook: "https://wiki.company.com/runbooks/rag-quality-degraded"

      - alert: RAGUserSatisfactionLow
        expr: |
          histogram_quantile(0.50,
            sum(rate(rag_user_rating_bucket[1h])) by (le)
          ) < 3.0
        for: 30m
        labels:
          severity: warning
          team: ai-platform
        annotations:
          summary: "RAG user satisfaction P50 {{ $value | humanize }}/5 below 3.0"

  - name: rag_cost
    rules:
      - alert: RAGDailyCostBudgetWarning
        expr: |
          sum(increase(rag_cost_usd_total[24h])) by (team_id) > 100
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Team {{ $labels.team_id }} daily spend ${{ $value | humanize }} exceeds $100"

      - alert: RAGCostSpike
        expr: |
          rate(rag_cost_usd_total[10m]) > 2 * rate(rag_cost_usd_total[1h] offset 1h)
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "RAG cost rate 2x higher than 1h ago — possible runaway request"

  - name: rag_data_freshness
    rules:
      - alert: KnowledgeBaseStale
        expr: |
          (time() - rag_index_last_updated_timestamp) > 172800
        labels:
          severity: warning
          team: ai-platform
        annotations:
          summary: "Knowledge base has not been updated in >48 hours"
          description: "Last update was {{ $value | humanizeDuration }} ago."
          runbook: "https://wiki.company.com/runbooks/knowledge-base-stale"
```

---

### 5.3 Quality Degradation Alerts

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import statistics

@dataclass
class QualityBaseline:
    metric_name: str
    baseline_value: float
    warning_drop: float     # Alert if drops by this fraction (e.g. 0.05 = 5%)
    critical_drop: float    # Critical alert if drops by this fraction
    window_hours: int = 24

class QualityDegradationDetector:
    """
    Detects statistically significant quality degradation by comparing
    current rolling average against a historical baseline.
    Uses statistical significance testing to reduce false positives.
    """
    def __init__(self, baselines: list[QualityBaseline]):
        self.baselines = {b.metric_name: b for b in baselines}
        self._history: dict[str, list] = {}

    def record(self, metric_name: str, value: float):
        if metric_name not in self._history:
            self._history[metric_name] = []
        self._history[metric_name].append({
            "value": value,
            "timestamp": datetime.utcnow()
        })
        # Keep last 7 days only
        cutoff = datetime.utcnow() - timedelta(days=7)
        self._history[metric_name] = [
            r for r in self._history[metric_name]
            if r["timestamp"] > cutoff
        ]

    def check(self, metric_name: str) -> Optional[dict]:
        """Check if current window shows degradation vs baseline."""
        baseline = self.baselines.get(metric_name)
        if not baseline or metric_name not in self._history:
            return None

        cutoff = datetime.utcnow() - timedelta(hours=baseline.window_hours)
        recent = [r["value"] for r in self._history[metric_name]
                  if r["timestamp"] > cutoff]

        if len(recent) < 10:
            return None  # Insufficient data

        current_mean = statistics.mean(recent)
        drop = (baseline.baseline_value - current_mean) / baseline.baseline_value

        if drop <= 0:
            return None  # No degradation

        severity = None
        if drop >= baseline.critical_drop:
            severity = "critical"
        elif drop >= baseline.warning_drop:
            severity = "warning"

        if severity:
            return {
                "metric": metric_name,
                "severity": severity,
                "baseline": baseline.baseline_value,
                "current": round(current_mean, 4),
                "drop_pct": round(drop * 100, 1),
                "sample_count": len(recent)
            }
        return None

DEFAULT_QUALITY_BASELINES = [
    QualityBaseline("faithfulness",    baseline_value=0.90,
                    warning_drop=0.05, critical_drop=0.15),
    QualityBaseline("relevancy",       baseline_value=0.85,
                    warning_drop=0.05, critical_drop=0.15),
    QualityBaseline("user_rating",     baseline_value=4.1,
                    warning_drop=0.08, critical_drop=0.20),
    QualityBaseline("refusal_rate",    baseline_value=0.08,
                    warning_drop=-0.10, critical_drop=-0.20),  # Inverted: higher is worse
]
```

---

### 5.4 Runbook Integration

```python
RUNBOOK_REGISTRY = {
    "RAGHighErrorRate": {
        "title": "RAG High Error Rate",
        "severity": "critical",
        "expected_resolution_minutes": 30,
        "steps": [
            {
                "step": 1,
                "title": "Check LLM provider status",
                "action": "Visit https://status.openai.com — if there is an ongoing incident, the alert is external. Monitor and wait.",
                "commands": ["curl -s https://status.openai.com/api/v2/status.json | python3 -m json.tool"],
            },
            {
                "step": 2,
                "title": "Check gateway error logs",
                "action": "Query logs for error details in the last 15 minutes.",
                "commands": [
                    "kubectl logs -n ai-platform -l app=rag-api --since=15m | grep '\"level\":\"ERROR\"' | head -20",
                    "kubectl logs -n ai-platform -l app=llm-gateway --since=15m | grep error | tail -20",
                ],
            },
            {
                "step": 3,
                "title": "Check rate limit status",
                "action": "Verify rate limit errors are not the primary cause.",
                "commands": [
                    "kubectl exec -n ai-platform -it deploy/llm-gateway -- python3 -c \"from src.rate_limiter import RateLimiter; print(RateLimiter().get_usage_summary())\"",
                ],
            },
            {
                "step": 4,
                "title": "Activate fallback provider",
                "action": "If OpenAI is down, update gateway config to route to Anthropic or Ollama.",
                "commands": [
                    "kubectl set env -n ai-platform deploy/llm-gateway DEFAULT_PROVIDER=anthropic",
                    "kubectl rollout status deploy/llm-gateway -n ai-platform",
                ],
                "requires_approval": True,
            },
            {
                "step": 5,
                "title": "Escalate",
                "action": "If error rate remains above 5% after 20 minutes, page the AI Platform on-call lead.",
                "escalation_contact": "ai-platform-oncall@company.com",
            },
        ]
    },
    "RAGHighRefusalRate": {
        "title": "RAG High Refusal Rate",
        "severity": "warning",
        "expected_resolution_minutes": 120,
        "steps": [
            {
                "step": 1,
                "title": "Check knowledge base freshness",
                "action": "Verify the index was updated within the expected window.",
                "commands": [
                    "curl -s http://rag-api.ai-platform.svc/v2/index/status | python3 -m json.tool",
                ],
            },
            {
                "step": 2,
                "title": "Analyse refusal query patterns",
                "action": "Query logs to identify the topics that are being refused.",
                "commands": [
                    'kubectl logs -n ai-platform -l app=rag-api --since=1h | python3 -c "import sys,json; [print(json.loads(l).get(\'query_length_chars\',0)) for l in sys.stdin if \'refusal\' in l]"',
                ],
            },
            {
                "step": 3,
                "title": "Trigger knowledge base rebuild",
                "action": "If corpus is stale, trigger a full index rebuild.",
                "commands": [
                    "python3 scripts/trigger_index_rebuild.py --corpus-id main --reason 'high-refusal-rate-incident'",
                ],
                "requires_approval": True,
            },
        ]
    }
}
```

---

### 🧪 Hands-on Lab: Instrumented RAG Service

**Objective:** Build a RAG service with structured logging, Prometheus metrics, and request tracing. Run it and observe all three signals.

**Prerequisites:** `openai`, `sentence-transformers`, `prometheus_client`, `structlog`, Python ≥ 3.11

```python
#!/usr/bin/env python3
"""
instrumented_rag.py — Fully observable RAG service demo.
Demonstrates metrics, structured logs, and request traces in one service.
"""

import time, json, uuid, hashlib, random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from prometheus_client import (
    Counter, Histogram, Gauge, start_http_server, REGISTRY
)
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Prometheus metrics ─────────────────────────────────────────────
requests_total = Counter("rag_requests_total",
                         "Total requests",
                         ["status", "is_refusal"])
request_latency = Histogram("rag_latency_ms",
                             "Request latency",
                             buckets=[50,100,250,500,1000,2000,5000])
retrieval_latency = Histogram("rag_retrieval_ms",
                              "Retrieval latency",
                              buckets=[10,25,50,100,250,500])
tokens_total = Counter("rag_tokens_total",
                       "Tokens consumed",
                       ["token_type"])
cost_total = Counter("rag_cost_usd_total", "Estimated cost USD")

# ── Structured log emitter ─────────────────────────────────────────
def emit_log(event_type: str, level: str = "INFO", **fields):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "event_type": event_type,
        **fields
    }
    print(json.dumps(record))

# ── Minimal knowledge base ─────────────────────────────────────────
DOCS = [
    {"id": "d1", "content": "Enterprise plan: 90-day returns. Standard: 30 days."},
    {"id": "d2", "content": "Professional plan: $150/month, 25 users, priority support."},
    {"id": "d3", "content": "API tokens expire after 3600 seconds (OAuth 2.0)."},
    {"id": "d4", "content": "Refunds processed within 5-7 business days."},
    {"id": "d5", "content": "Enterprise includes dedicated support manager and 99.99% SLA."},
]
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
DOC_EMBEDDINGS = embed_model.encode([d["content"] for d in DOCS])
oai = OpenAI()

# ── RAG pipeline with full instrumentation ─────────────────────────
def instrumented_rag(question: str, team_id: str = "demo") -> dict:
    request_id = uuid.uuid4().hex[:8]
    t_start = time.perf_counter()

    emit_log("request.received", request_id=request_id,
             team_id=team_id, query_length=len(question))

    # Retrieval
    t_ret = time.perf_counter()
    q_emb = embed_model.encode([question])
    sims = cosine_similarity(q_emb, DOC_EMBEDDINGS)[0]
    top_idx = sims.argsort()[-3:][::-1]
    retrieved = [{"id": DOCS[i]["id"], "content": DOCS[i]["content"],
                  "score": float(sims[i])} for i in top_idx]
    ret_ms = (time.perf_counter() - t_ret) * 1000
    retrieval_latency.observe(ret_ms)

    emit_log("retrieval.completed", request_id=request_id,
             retrieved_count=len(retrieved),
             top_score=round(retrieved[0]["score"], 4),
             latency_ms=round(ret_ms, 1))

    # Generation
    context = "\n".join([f"[{i+1}] {r['content']}" for i, r in enumerate(retrieved)])
    messages = [
        {"role": "system",
         "content": "Answer using only the context. Cite sources [N]. "
                    "If not found, say 'I cannot find this information.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    t_gen = time.perf_counter()
    resp = oai.chat.completions.create(
        model="gpt-4o-mini", messages=messages,
        temperature=0, max_tokens=200
    )
    gen_ms = (time.perf_counter() - t_gen) * 1000
    answer = resp.choices[0].message.content
    prompt_tok = resp.usage.prompt_tokens
    compl_tok = resp.usage.completion_tokens
    cost = prompt_tok / 1e6 * 0.15 + compl_tok / 1e6 * 0.60

    # Record token metrics
    tokens_total.labels(token_type="prompt").inc(prompt_tok)
    tokens_total.labels(token_type="completion").inc(compl_tok)
    cost_total.inc(cost)

    is_refusal = "cannot find" in answer.lower()
    total_ms = (time.perf_counter() - t_start) * 1000

    # Record request metrics
    requests_total.labels(status="success",
                          is_refusal=str(is_refusal).lower()).inc()
    request_latency.observe(total_ms)

    emit_log("request.completed", request_id=request_id,
             team_id=team_id,
             is_refusal=is_refusal,
             prompt_tokens=prompt_tok,
             completion_tokens=compl_tok,
             estimated_cost_usd=round(cost, 6),
             retrieval_latency_ms=round(ret_ms, 1),
             generation_latency_ms=round(gen_ms, 1),
             total_latency_ms=round(total_ms, 1),
             answer_length=len(answer))

    return {
        "request_id": request_id,
        "answer": answer,
        "is_refusal": is_refusal,
        "latency_ms": round(total_ms, 1),
        "tokens": prompt_tok + compl_tok,
        "cost_usd": round(cost, 6)
    }

# ── Run demo ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start Prometheus metrics server on port 8001
    start_http_server(8001)
    print("Metrics server started at http://localhost:8001/metrics")
    print("Starting demo requests...\n")

    test_queries = [
        ("product-support", "How long can enterprise customers return products?"),
        ("product-billing",  "What does the Professional plan cost monthly?"),
        ("product-api",      "How do API tokens expire?"),
        ("product-support",  "What is the policy for cryptocurrency payments?"),
        ("product-billing",  "How long does a refund take?"),
    ]

    for team, query in test_queries:
        result = instrumented_rag(query, team_id=team)
        mark = "🚫" if result["is_refusal"] else "✓"
        print(f"{mark} [{team}] {query[:50]}")
        print(f"   → {result['answer'][:80]}")
        print(f"   ⏱ {result['latency_ms']:.0f}ms | "
              f"🪙 {result['tokens']} tokens | "
              f"💰 ${result['cost_usd']:.5f}")
        print()
        time.sleep(0.5)

    print("─" * 60)
    print("Check Prometheus metrics: http://localhost:8001/metrics")
    print("Check structured logs:    above output (JSON lines)")
    print("\nKey metrics to observe:")
    print("  rag_requests_total{status='success'}")
    print("  rag_latency_ms (histogram)")
    print("  rag_tokens_total (by token_type)")
    print("  rag_cost_usd_total")
```

**Run the lab:**
```bash
pip install openai sentence-transformers scikit-learn prometheus-client
python instrumented_rag.py
# In another terminal:
curl -s http://localhost:8001/metrics | grep rag_
```

**Extensions:**
- Add OpenTelemetry tracing to `instrumented_rag`: create a root span `rag.request` with child spans for retrieval and generation
- Connect the Prometheus endpoint to a local Grafana instance and build a 3-panel dashboard (latency, token usage, refusal rate)
- Add `AdaptiveLogSampler` to drop 90% of successful non-refusal requests, verifying that errors and refusals are always logged

---

> ### 📋 Chapter Summary
>
> - Alerts fall into four categories for LLM systems: **availability** (error rate, latency), **quality** (faithfulness, user satisfaction), **cost** (daily spend, cost spikes), and **data freshness** (knowledge base age).
> - Prometheus alerting rules use `for:` clauses to require sustained condition before firing — reducing false positives from transient spikes.
> - `QualityDegradationDetector` detects statistically significant drops in quality metrics against historical baselines, with configurable warning/critical thresholds.
> - **Runbooks** are structured, command-line-executable playbooks attached to each alert — every alert must have a runbook link before being enabled in production.

---

> ### ❓ Comprehension Questions
>
> 1. `RAGHighErrorRate` fires if error rate exceeds 2% for 2 minutes. A 30-second spike to 5% occurs but does not sustain. The alert does not fire. Is this the correct behaviour? Argue both for and against the `for: 2m` clause.
> 2. `RAGCostSpike` uses `rate(... [10m]) > 2 * rate(... offset 1h)`. A batch job runs at 01:00 UTC causing a legitimate cost spike. The alert fires and wakes the on-call engineer at 01:05. How would you exclude scheduled batch jobs from this alert?
> 3. `QualityDegradationDetector` requires at least 10 samples before reporting degradation. At 0.5% sampling rate (from `AdaptiveLogSampler`), how long must a degradation persist before it is detectable at 100 RPS?
> 4. The runbook for `RAGHighRefusalRate` step 3 requires approval before triggering a rebuild. Why is this approval gate important, and what risk does it prevent?
> 5. Design an alert for "embedding model mismatch" — a situation where the index was built with `text-embedding-3-small` but the query path is using `text-embedding-3-large`. What metrics, log fields, or trace attributes would detect this condition?

---

## References

### Documentation
- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/alerting_rules/)
- [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PagerDuty Integration](https://www.pagerduty.com/docs/guides/prometheus-integration-guide/)
- [Grafana Alerting](https://grafana.com/docs/grafana/latest/alerting/)

### Papers
- [Site Reliability Engineering](https://sre.google/sre-book/monitoring-distributed-systems/) — Google SRE on alerting philosophy and on-call practices.
- [Dapper: Google Distributed Tracing](https://research.google/pubs/pub36356/) — Sigelman et al., 2010.

---

> **Navigation**
> [← Part XII — Infrastructure](part_12_infrastructure.md) | [→ Part XIV — Security](part_14_security.md)
