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

---
[« Back to observability Index](index.md) | [🏠 Home](../index.md)