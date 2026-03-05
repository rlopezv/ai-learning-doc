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

---
[« Back to observability Index](index.md) | [🏠 Home](../../index.md)