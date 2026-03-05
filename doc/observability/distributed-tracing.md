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

---
[« Back to observability Index](index.md) | [🏠 Home](../../index.md)