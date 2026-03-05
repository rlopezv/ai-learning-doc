## Chapter 2 — Production RAG Architecture

### 2.1 Architecture Overview

```
Production RAG Architecture — Kubernetes, HA, multi-zone

External                  Ingress                  Platform
────────────────────────────────────────────────────────────────

Users / API Clients
        │
        ▼
[API Gateway / Ingress]   ← TLS termination, rate limiting, auth
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
[RAG API]  ×3 pods         [Ingestion API]  ×2 pods
  FastAPI / Spring Boot      FastAPI async
        │                            │
        ├── [LLM Gateway]  ×2        └── [Task Queue]
        │     Retry, fallback,             Celery + Redis
        │     cost tracking                    │
        │                                      ▼
        ├── [Embedding Svc] ×2         [Ingestion Workers] ×4
        │     all-MiniLM / E5               Chunk, embed, index
        │
        ├── [Prompt Service] ×2
        │     Template registry
        │
        ▼
[Vector DB]  Qdrant ×3     [Cache]  Redis ×3    [Metadata DB]
  StatefulSet, HA,           Semantic cache,      PostgreSQL ×2
  S3 backup daily            session store        RDS / CloudSQL

                    ┌─────────────────────┐
                    │   Observability      │
                    │  Prometheus + Grafana│
                    │  Jaeger (traces)     │
                    │  Loki (logs)         │
                    └─────────────────────┘
```

---

### 2.2 Component Specifications

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ComponentSpec:
    name: str
    technology: str
    replicas_min: int
    replicas_max: int
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    storage: Optional[str]
    on_premise_alt: Optional[str]
    notes: str

PRODUCTION_COMPONENTS = [
    ComponentSpec(
        "RAG API", "FastAPI / Spring Boot",
        replicas_min=3, replicas_max=20,
        cpu_request="500m", cpu_limit="2000m",
        memory_request="512Mi", memory_limit="2Gi",
        storage=None,
        on_premise_alt=None,
        notes="HPA on CPU+RPS. RollingUpdate maxUnavailable=0."
    ),
    ComponentSpec(
        "LLM Gateway", "FastAPI + httpx",
        replicas_min=2, replicas_max=8,
        cpu_request="250m", cpu_limit="1000m",
        memory_request="256Mi", memory_limit="1Gi",
        storage=None,
        on_premise_alt="Ollama/vLLM sidecar",
        notes="Circuit breaker + fallback chain. API keys from Vault."
    ),
    ComponentSpec(
        "Embedding Service", "FastAPI + SentenceTransformers",
        replicas_min=2, replicas_max=6,
        cpu_request="1000m", cpu_limit="4000m",
        memory_request="2Gi", memory_limit="4Gi",
        storage=None,
        on_premise_alt="Same — model loaded from local PVC",
        notes="CPU-optimised. GPU variant available for high-throughput."
    ),
    ComponentSpec(
        "Vector Database", "Qdrant",
        replicas_min=3, replicas_max=3,   # StatefulSet
        cpu_request="1000m", cpu_limit="4000m",
        memory_request="4Gi", memory_limit="16Gi",
        storage="100Gi SSD per node",
        on_premise_alt="Qdrant OSS",
        notes="Distributed mode. Daily S3 snapshot. P2P port 6335."
    ),
    ComponentSpec(
        "Redis Cache", "Redis",
        replicas_min=3, replicas_max=3,   # Sentinel HA
        cpu_request="250m", cpu_limit="1000m",
        memory_request="1Gi", memory_limit="4Gi",
        storage="10Gi",
        on_premise_alt="Redis OSS",
        notes="Semantic cache + session store + rate limit counters."
    ),
    ComponentSpec(
        "Ingestion Workers", "Celery",
        replicas_min=2, replicas_max=10,
        cpu_request="500m", cpu_limit="2000m",
        memory_request="1Gi", memory_limit="4Gi",
        storage=None,
        on_premise_alt=None,
        notes="HPA on queue depth. task_acks_late=True."
    ),
    ComponentSpec(
        "Metadata DB", "PostgreSQL",
        replicas_min=2, replicas_max=2,   # Primary + replica
        cpu_request="500m", cpu_limit="2000m",
        memory_request="1Gi", memory_limit="8Gi",
        storage="50Gi SSD",
        on_premise_alt="PostgreSQL OSS",
        notes="Stores corpus metadata, governance artefacts, audit log."
    ),
]

def print_component_table():
    print(f"{'Component':<22} {'Tech':<28} {'Replicas':<12} {'Memory':<20} {'On-Prem Alt'}")
    print("-" * 100)
    for c in PRODUCTION_COMPONENTS:
        replicas = f"{c.replicas_min}–{c.replicas_max}"
        memory   = f"{c.memory_request}–{c.memory_limit}"
        onprem   = c.on_premise_alt or "—"
        print(f"{c.name:<22} {c.technology:<28} {replicas:<12} {memory:<20} {onprem}")
```

---

### 2.3 Data Flow and Request Lifecycle

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RequestLifecycleStep:
    step: int
    component: str
    action: str
    typical_latency_ms: float
    failure_mode: str
    fallback: str

QUERY_LIFECYCLE = [
    RequestLifecycleStep(1, "API Gateway",
        "TLS termination, JWT validation, rate limit check",
        2.0, "Auth service down → 401/503", "Cached token validation"),
    RequestLifecycleStep(2, "RAG API",
        "Request validation, PII scan, injection detection",
        3.0, "Validation failure → 400", "Return validation error"),
    RequestLifecycleStep(3, "Semantic Cache",
        "Lookup by query embedding similarity ≥ 0.92",
        5.0, "Redis down → cache miss", "Skip cache, proceed to retrieval"),
    RequestLifecycleStep(4, "Embedding Service",
        "Encode query to dense vector (384-dim)",
        30.0, "OOM crash → pod restart", "Fallback embedding model"),
    RequestLifecycleStep(5, "Vector DB",
        "ANN search top-5, filter by tenant_id",
        50.0, "Qdrant node failure → replica serves", "Degraded response if all down"),
    RequestLifecycleStep(6, "Prompt Service",
        "Fetch template by ID+version, render with context",
        2.0, "Template not found → 500", "Default template fallback"),
    RequestLifecycleStep(7, "LLM Gateway",
        "Call LLM provider with retry (3×), circuit breaker",
        700.0, "Provider down → circuit opens", "Fallback to secondary provider"),
    RequestLifecycleStep(8, "Output Scanner",
        "PII scan, injection compliance, schema validation",
        3.0, "Scan failure → log+pass", "Return sanitised response"),
    RequestLifecycleStep(9, "Audit Logger",
        "Write structured log, Prometheus metrics",
        1.0, "Logger failure → drop silently", "Best-effort — never blocks response"),
]

def print_lifecycle():
    total_p50 = sum(s.typical_latency_ms for s in QUERY_LIFECYCLE)
    print(f"{'Step':<5} {'Component':<20} {'Latency (ms)':<14} Action")
    print("-" * 80)
    for s in QUERY_LIFECYCLE:
        print(f"  {s.step:<4} {s.component:<20} {s.typical_latency_ms:<14.0f} {s.action}")
    print(f"\n  Estimated P50 total: {total_p50:.0f}ms")
    print(f"  P99 (2× for tail):   {total_p50 * 2:.0f}ms")
```

---

### 2.4 Failure Modes and Mitigations

```python
from dataclasses import dataclass

@dataclass
class FailureMode:
    scenario: str
    probability: str    # "low" | "medium" | "high"
    impact: str         # "low" | "medium" | "high" | "critical"
    detection: str
    mitigation: str
    recovery_time: str

FAILURE_MODES = [
    FailureMode(
        "LLM provider outage (OpenAI 503)",
        probability="medium", impact="critical",
        detection="Circuit breaker opens after 5 failures; alert fires in <2m",
        mitigation="Fallback chain: OpenAI → Anthropic → Ollama (on-prem)",
        recovery_time="<30s (automatic failover)"
    ),
    FailureMode(
        "Qdrant node failure (1 of 3)",
        probability="low", impact="low",
        detection="Kubernetes pod restart; Prometheus pod count alert",
        mitigation="Qdrant distributed mode: remaining 2 nodes serve queries",
        recovery_time="<5m (pod reschedule + Qdrant peer sync)"
    ),
    FailureMode(
        "Embedding service OOM",
        probability="medium", impact="high",
        detection="Pod OOMKilled; readiness probe fails; removed from LB",
        mitigation="HPA spins new pod; in-flight requests retried by client",
        recovery_time="<60s"
    ),
    FailureMode(
        "Redis cache failure",
        probability="low", impact="medium",
        detection="Redis Sentinel detects primary failure; promotes replica",
        mitigation="Semantic cache disabled; all requests hit RAG pipeline",
        recovery_time="<30s (Sentinel failover)"
    ),
    FailureMode(
        "Corpus poisoning (malicious doc ingested)",
        probability="low", impact="high",
        detection="Injection/secrets scan on ingestion; quality monitor flags refusal spike",
        mitigation="Document quarantine; index rollback to pre-poison snapshot",
        recovery_time="1–4h (investigation + index rebuild)"
    ),
    FailureMode(
        "Prompt regression (new template degrades quality)",
        probability="medium", impact="medium",
        detection="Canary quality gate catches faithfulness drop before full rollout",
        mitigation="Canary rollback; previous prompt version remains active",
        recovery_time="<5m (canary rollback)"
    ),
    FailureMode(
        "Cost runaway (token budget exhausted)",
        probability="medium", impact="medium",
        detection="Cost anomaly detector fires; budget enforcer blocks requests",
        mitigation="Hard cutoff per team; requests return 429 with retry-after",
        recovery_time="Next budget reset (midnight UTC)"
    ),
]
```

---

> ### 📋 Chapter Summary
>
> - The production architecture adds load balancing, horizontal pod autoscaling, HA vector DB, semantic caching, async ingestion, and the full observability stack versus minimal.
> - Component specifications provide concrete starting-point resource requests — every component has a minimum replica count (≥2 for HA, ≥3 for quorum-based components).
> - The 9-step **request lifecycle** accounts for ~800ms P50 end-to-end, with LLM generation consuming ~700ms — every other component is <50ms.
> - Seven failure modes with automatic mitigations cover the most likely production incidents — none require manual intervention for initial recovery.

---

> ### ❓ Comprehension Questions
>
> 1. The production architecture uses `replicas_min=3` for Qdrant but `replicas_min=2` for the RAG API. Justify both choices: why does Qdrant need 3, and why is 2 sufficient for the RAG API?
> 2. The request lifecycle estimates 700ms for LLM generation. A new streaming API is introduced that returns the first token in 100ms and streams the rest over 600ms. How does streaming affect the latency model and the P99 SLO?
> 3. The "Corpus poisoning" failure mode has recovery time 1–4 hours. The attack window is the time between ingestion and detection. At an ingestion rate of 1,000 documents/hour with a 5-minute content scan delay, how many poisoned documents could be ingested before detection?
> 4. The semantic cache threshold is 0.92 cosine similarity. During a Redis Sentinel failover (30s), the cache is unavailable and all requests hit the full RAG pipeline. Calculate the additional LLM cost for 30 seconds at 50 RPS with a 25% normal cache hit rate and $0.001 per request.
> 5. `FAILURE_MODES` lists 7 scenarios. A business continuity review requires a full RTO (Recovery Time Objective) for the entire platform. Given the individual recovery times listed, what is the RTO for a simultaneous Redis failure + Qdrant node failure, and why is this not simply the sum?

---

## References

### Documentation
- [Qdrant Distributed Deployment](https://qdrant.tech/documentation/guides/distributed_deployment/)
- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Redis Sentinel](https://redis.io/docs/manual/sentinel/)
- [Celery Best Practices](https://docs.celeryq.dev/en/stable/userguide/tasks.html#tips-and-best-practices)

---

---
[« Back to reference_architectures Index](index.md) | [🏠 Home](../index.md)