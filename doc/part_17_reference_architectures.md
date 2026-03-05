# Part XVII — Reference Architectures

---

> **Navigation**
> [← Part XVI — Operations](part_16_operations.md) | [→ Part XVIII — Practical Case Studies](part_18_case_studies.md)

---

## Contents

- [Chapter 1 — Minimal RAG Architecture](#chapter-1--minimal-rag-architecture)
  - [1.1 When Minimal is Right](#11-when-minimal-is-right)
  - [1.2 Component Inventory and Sizing](#12-component-inventory-and-sizing)
  - [1.3 Full Implementation Blueprint](#13-full-implementation-blueprint)
  - [1.4 Graduation Checklist](#14-graduation-checklist)
- [Chapter 2 — Production RAG Architecture](#chapter-2--production-rag-architecture)
  - [2.1 Architecture Overview](#21-architecture-overview)
  - [2.2 Component Specifications](#22-component-specifications)
  - [2.3 Data Flow and Request Lifecycle](#23-data-flow-and-request-lifecycle)
  - [2.4 Failure Modes and Mitigations](#24-failure-modes-and-mitigations)
- [Chapter 3 — Multi-Tenant Enterprise Architecture](#chapter-3--multi-tenant-enterprise-architecture)
  - [3.1 Tenancy Model Options](#31-tenancy-model-options)
  - [3.2 Shared Platform with Tenant Isolation](#32-shared-platform-with-tenant-isolation)
  - [3.3 Control Plane and Data Plane Separation](#33-control-plane-and-data-plane-separation)
  - [3.4 Cross-Tenant Governance and Cost Allocation](#34-cross-tenant-governance-and-cost-allocation)
- [Chapter 4 — Air-Gapped and On-Premise Architecture 🔓🧪](#chapter-4--air-gapped-and-on-premise-architecture-)
  - [4.1 Constraints and Design Principles](#41-constraints-and-design-principles)
  - [4.2 Component Selection for Air-Gapped Environments](#42-component-selection-for-air-gapped-environments)
  - [4.3 Full On-Premise Stack](#43-full-on-premise-stack)
  - [🧪 Hands-on Lab: Architecture Decision Record Generator](#-hands-on-lab-architecture-decision-record-generator)

---

## Chapter 1 — Minimal RAG Architecture

### 1.1 When Minimal is Right

Not every RAG system requires a Kubernetes cluster, a streaming ingestion pipeline, and a multi-tier observability stack. Over-engineering early wastes months and produces systems that are harder to reason about, harder to debug, and harder to change. The minimal architecture is the correct starting point when:

- **Daily query volume < 1,000** — a single process handles this comfortably
- **Corpus size < 100,000 documents** — in-memory or single-node vector DB suffices
- **Team size ≤ 3 engineers** — operational complexity must stay low
- **Latency SLO > 2 seconds** — no need to optimise for sub-second response
- **Single tenant** — no isolation requirements
- **Proof of concept or early production** — requirements are still changing

The minimal architecture deliberately omits: separate ingestion workers, distributed caching, multi-zone deployment, and advanced observability. These are added when specific pain points emerge — not pre-emptively.

---

### 1.2 Component Inventory and Sizing

```
Minimal RAG Architecture — single server or managed cloud services

┌─────────────────────────────────────────────────────┐
│                   Single Host or VM                  │
│                                                     │
│  ┌─────────────┐    ┌──────────────┐               │
│  │  FastAPI     │    │  Chroma DB   │               │
│  │  RAG API     │◄──►│  (embedded,  │               │
│  │  port 8080   │    │  local disk) │               │
│  └──────┬───────┘    └──────────────┘               │
│         │                                           │
│         ▼                                           │
│  ┌─────────────┐    ┌──────────────┐               │
│  │ Sentence    │    │  SQLite      │               │
│  │ Transformer │    │  (metadata + │               │
│  │ (in-process)│    │   audit log) │               │
│  └─────────────┘    └──────────────┘               │
└─────────────────────┬───────────────────────────────┘
                      │ HTTPS
                      ▼
               OpenAI API / Anthropic API
               (or Ollama on same host 🔓)
```

```python
# Sizing model for minimal architecture
from dataclasses import dataclass

@dataclass
class MinimalSizingModel:
    daily_queries: int
    corpus_documents: int
    avg_doc_chars: int = 2000
    embedding_dims: int = 384      # all-MiniLM-L6-v2
    top_k: int = 5

    @property
    def corpus_size_gb(self) -> float:
        # Vector storage: dims × 4 bytes per float32
        vector_bytes = self.corpus_documents * self.embedding_dims * 4
        # Metadata: ~500 bytes per document
        meta_bytes = self.corpus_documents * 500
        return (vector_bytes + meta_bytes) / 1e9

    @property
    def recommended_ram_gb(self) -> float:
        # 2× corpus size to hold index in RAM + OS overhead
        return max(self.corpus_size_gb * 2 + 1.0, 2.0)

    @property
    def recommended_vm(self) -> str:
        ram = self.recommended_ram_gb
        if ram <= 4:
            return "t3.medium (2 vCPU, 4 GB RAM) ~$30/month"
        if ram <= 8:
            return "t3.large (2 vCPU, 8 GB RAM) ~$60/month"
        if ram <= 16:
            return "t3.xlarge (4 vCPU, 16 GB RAM) ~$120/month"
        return "m6i.2xlarge (8 vCPU, 32 GB RAM) ~$280/month"

    def print_summary(self):
        print(f"Corpus:          {self.corpus_documents:,} documents")
        print(f"Corpus size:     {self.corpus_size_gb:.2f} GB")
        print(f"Recommended RAM: {self.recommended_ram_gb:.1f} GB")
        print(f"Recommended VM:  {self.recommended_vm}")
        print(f"Daily queries:   {self.daily_queries:,}")
        qps = self.daily_queries / 86400
        print(f"Avg QPS:         {qps:.2f} (peak ~{qps * 5:.1f} assumed)")

# Example: internal knowledge base, 10K docs, 200 queries/day
model = MinimalSizingModel(daily_queries=200, corpus_documents=10_000)
model.print_summary()
```

---

### 1.3 Full Implementation Blueprint

```python
"""
minimal_rag.py — Complete minimal RAG implementation.
Single file, no external infrastructure, ready to deploy on one server.
Dependencies: fastapi, uvicorn, chromadb, sentence-transformers, openai
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import chromadb, uuid, json, sqlite3, time
from datetime import datetime
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────
CHROMA_PATH   = "./data/chroma"
SQLITE_PATH   = "./data/audit.db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL     = "gpt-4o-mini"
COLLECTION    = "knowledge_base"
TOP_K         = 5

# ── Initialise components ─────────────────────────────────────────
app       = FastAPI(title="Minimal RAG API")
embedder  = SentenceTransformer(EMBED_MODEL)
oai       = OpenAI()
chroma    = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(COLLECTION)

# SQLite audit log — no external dependency
Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
db = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS audit (
    id TEXT PRIMARY KEY, ts TEXT, query TEXT,
    answer TEXT, latency_ms REAL, tokens INTEGER, cost_usd REAL
)""")
db.commit()

# ── Request / Response models ─────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K

class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[dict]
    latency_ms: float
    tokens: int
    cost_usd: float

class IngestRequest(BaseModel):
    documents: list[dict]   # [{id, content, metadata}]

# ── Endpoints ─────────────────────────────────────────────────────
@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if len(req.question) > 2000:
        raise HTTPException(400, "Question too long")

    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]

    # Embed and retrieve
    q_emb = embedder.encode([req.question]).tolist()
    results = collection.query(
        query_embeddings=q_emb, n_results=req.top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return QueryResponse(request_id=request_id,
            answer="I cannot find relevant information to answer your question.",
            sources=[], latency_ms=0, tokens=0, cost_usd=0)

    # Build prompt
    context = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    messages = [
        {"role": "system",
         "content": "Answer using only the provided context. Cite sources [N]. "
                    "If not found say: 'I cannot find this information.'"},
        {"role": "user",
         "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
    ]

    # Generate
    resp = oai.chat.completions.create(
        model=LLM_MODEL, messages=messages,
        temperature=0, max_tokens=300
    )
    answer      = resp.choices[0].message.content
    p_tokens    = resp.usage.prompt_tokens
    c_tokens    = resp.usage.completion_tokens
    cost        = p_tokens / 1e6 * 0.15 + c_tokens / 1e6 * 0.60
    latency_ms  = (time.perf_counter() - t0) * 1000

    # Audit log
    db.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?)",
               (request_id, datetime.utcnow().isoformat(),
                req.question[:500], answer[:1000],
                round(latency_ms, 1), p_tokens + c_tokens, round(cost, 6)))
    db.commit()

    sources = [{"id": metadatas[i].get("id", str(i)),
                "relevance": round(1 - distances[i], 3)}
               for i in range(len(docs))]

    return QueryResponse(request_id=request_id, answer=answer,
                         sources=sources, latency_ms=round(latency_ms, 1),
                         tokens=p_tokens + c_tokens, cost_usd=round(cost, 6))

@app.post("/v1/ingest")
def ingest(req: IngestRequest):
    ids       = [d["id"] for d in req.documents]
    texts     = [d["content"] for d in req.documents]
    metas     = [d.get("metadata", {}) | {"id": d["id"]} for d in req.documents]
    embeddings = embedder.encode(texts).tolist()
    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=texts, metadatas=metas)
    return {"ingested": len(ids)}

@app.get("/health")
def health():
    return {"status": "ok", "vectors": collection.count()}
```

---

### 1.4 Graduation Checklist

When the minimal architecture shows these symptoms, it is time to graduate to the production architecture:

```python
GRADUATION_CHECKLIST = {
    "traffic": [
        ("Daily queries > 1,000",        "Add load balancer, scale horizontally"),
        ("Peak QPS > 5",                  "Add response caching layer"),
        ("Concurrent users > 10",         "Multiple API instances behind LB"),
    ],
    "corpus": [
        ("Documents > 100,000",           "Move to dedicated Qdrant cluster"),
        ("Multiple document types",       "Add type-specific chunking strategies"),
        ("Corpus updates > daily",        "Add async ingestion queue (Celery/Kafka)"),
    ],
    "reliability": [
        ("Uptime SLO > 99%",              "Multi-instance, health checks, auto-restart"),
        ("LLM provider outage noticed",   "Add circuit breaker and fallback chain"),
        ("Data loss incident",            "Add vector DB backup strategy"),
    ],
    "quality": [
        ("User complaints about answers", "Add evaluation pipeline and quality metrics"),
        ("Prompt changes break things",   "Add prompt versioning and A/B testing"),
        ("Can't explain why answer wrong","Add tracing and structured logging"),
    ],
    "compliance": [
        ("PII in user queries",           "Add PII detection and redaction"),
        ("Audit trail required",          "Replace SQLite audit with append-only store"),
        ("Multi-team access",             "Add RBAC and tenant isolation"),
    ],
}

for category, items in GRADUATION_CHECKLIST.items():
    print(f"\n{category.upper()}")
    for symptom, action in items:
        print(f"  ☐ {symptom}")
        print(f"    → {action}")
```

---

> ### 📋 Chapter Summary
>
> - The minimal architecture is the correct starting point for low-traffic, single-tenant systems with ≤100K documents — over-engineering early is itself a reliability risk.
> - A single VM with FastAPI, ChromaDB (embedded), SentenceTransformer, and SQLite can serve 200–1,000 daily queries with no infrastructure management overhead.
> - `MinimalSizingModel` derives RAM and VM recommendations from corpus size and query volume, providing concrete capacity planning from first principles.
> - The **graduation checklist** defines objective thresholds at which each minimal component should be replaced — traffic, corpus scale, reliability, quality, and compliance symptoms each trigger specific upgrades.

---

> ### ❓ Comprehension Questions
>
> 1. The minimal architecture stores embeddings in ChromaDB on local disk. A VM restart loses nothing (ChromaDB is persistent). But the disk is on the VM's ephemeral storage. What data loss scenario does this create, and how would you address it without adding complexity?
> 2. `minimal_rag.py` uses a SQLite `db` connection with `check_same_thread=False`. FastAPI runs with multiple worker threads by default. What concurrency issue does this create, and what is the correct fix?
> 3. The graduation threshold for corpus size is 100,000 documents. A company has 80,000 documents but each averages 10,000 characters (5× longer than assumed). Does the document count threshold still apply? Recalculate using `MinimalSizingModel`.
> 4. The minimal architecture omits a separate ingestion worker. An ingest of 50,000 documents takes 45 minutes and blocks the process. How would you add async ingestion as the first upgrade without introducing a message queue?
> 5. The audit log uses SQLite. At 1,000 queries/day with ~500 bytes per record, estimate the SQLite file size after one year. Is this a problem? At what query rate would you migrate to a dedicated log store?

---

## References

### Documentation
- [ChromaDB](https://docs.trychroma.com) — Embedded vector database for minimal setups.
- [FastAPI](https://fastapi.tiangolo.com) — High-performance Python API framework.
- [Sentence Transformers](https://www.sbert.net) — Embedding models.

---

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

## Chapter 3 — Multi-Tenant Enterprise Architecture

### 3.1 Tenancy Model Options

```python
from dataclasses import dataclass
from enum import Enum

class TenancyModel(str, Enum):
    SILO      = "silo"       # Fully isolated stack per tenant
    POOL      = "pool"       # Shared infrastructure, logical isolation
    BRIDGE    = "bridge"     # Shared compute, isolated data stores

@dataclass
class TenancyOption:
    model: TenancyModel
    description: str
    isolation_level: str
    cost_per_tenant: str
    operational_complexity: str
    use_case: str
    data_residency_compliant: bool

TENANCY_OPTIONS = [
    TenancyOption(
        TenancyModel.SILO,
        "Separate Kubernetes namespace, vector DB collection set, and LLM gateway per tenant",
        isolation_level="complete",
        cost_per_tenant="high — dedicated resources",
        operational_complexity="high — N × operational burden",
        use_case="Regulated industries (financial, health), government, highest-value tenants",
        data_residency_compliant=True
    ),
    TenancyOption(
        TenancyModel.POOL,
        "Shared infrastructure; tenant isolation via metadata filters and RBAC",
        isolation_level="logical",
        cost_per_tenant="low — shared resources",
        operational_complexity="medium — single platform to operate",
        use_case="SaaS product with many small-medium tenants, internal enterprise teams",
        data_residency_compliant=False  # Data co-located across tenants
    ),
    TenancyOption(
        TenancyModel.BRIDGE,
        "Shared compute (API, embedding, LLM gateway); isolated vector collections per tenant",
        isolation_level="data-isolated, compute-shared",
        cost_per_tenant="medium",
        operational_complexity="medium",
        use_case="Enterprise with compliance requirements but cost sensitivity",
        data_residency_compliant=True  # Vectors isolated per tenant collection
    ),
]

def recommend_tenancy(
    tenant_count: int,
    has_regulated_tenants: bool,
    data_residency_required: bool,
    cost_sensitive: bool
) -> TenancyModel:
    if has_regulated_tenants and data_residency_required:
        return TenancyModel.SILO
    if tenant_count > 50 and cost_sensitive:
        return TenancyModel.POOL
    return TenancyModel.BRIDGE
```

---

### 3.2 Shared Platform with Tenant Isolation

```
Multi-Tenant Bridge Architecture

                        ┌─────────────────────┐
                        │   Control Plane      │
                        │  Tenant registry     │
                        │  RBAC / Auth         │
                        │  Billing / Quotas    │
                        │  Governance store    │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │         Shared Data Plane (Kubernetes namespace)     │
        │                                                     │
        │  [API Gateway]  [RAG API ×3]  [LLM Gateway ×2]     │
        │       │              │               │              │
        │       │         [Embedding Svc ×2]   │              │
        │       │                              │              │
        └───────┼──────────────────────────────┼──────────────┘
                │                              │
     ┌──────────▼──────────────────────────────▼──────────┐
     │            Qdrant — Multi-collection                │
     │                                                     │
     │   tenant_acme/           tenant_globex/             │
     │   ├── corpus_main        ├── corpus_main            │
     │   ├── corpus_archive     └── corpus_archive         │
     │   └── corpus_hr                                     │
     └─────────────────────────────────────────────────────┘
     
     Every query: filter = {tenant_id: "acme"} — enforced server-side
```

```python
class MultiTenantRAGRouter:
    """
    Routes requests to the correct tenant-isolated data plane.
    Enforces:
    - Tenant namespace isolation
    - Per-tenant token budgets
    - Per-tenant prompt versions
    - Cross-tenant audit logging
    """
    def __init__(self, tenant_registry, budget_enforcer,
                 prompt_client, vector_db, llm_gateway):
        self.registry  = tenant_registry
        self.budgets   = budget_enforcer
        self.prompts   = prompt_client
        self.vdb       = vector_db
        self.gateway   = llm_gateway

    def query(self, tenant_id: str, user_id: str, question: str) -> dict:
        # 1. Resolve tenant configuration
        tenant = self.registry.get(tenant_id)
        if not tenant:
            raise ValueError(f"Unknown tenant: {tenant_id}")

        # 2. Budget check
        allowed, reason = self.budgets.check_and_record(
            tenant_id, estimated_tokens=800, estimated_cost_usd=0.001
        )
        if not allowed:
            return {"error": reason, "code": "BUDGET_EXCEEDED"}

        # 3. Retrieve from tenant-isolated collection
        q_emb = self._embed(question)
        results = self.vdb.search(
            collection=f"tenant_{tenant_id}/corpus_main",
            query_vector=q_emb,
            filter={"tenant_id": tenant_id},   # Mandatory server-side filter
            top_k=tenant.get("top_k", 5)
        )

        # 4. Fetch tenant-specific prompt template
        prompt_id = tenant.get("prompt_template", "default_support")
        messages  = self.prompts.get_messages(prompt_id, {
            "context":  "\n".join(r["content"] for r in results),
            "question": question
        })

        # 5. Generate with tenant model preference
        model_alias = tenant.get("model_alias", "default")
        response    = self.gateway.complete(messages, model_alias)

        return {
            "tenant_id": tenant_id,
            "answer":    response["answer"],
            "model":     response["model"],
            "sources":   [r["id"] for r in results]
        }

    def _embed(self, text: str) -> list[float]:
        return []  # Delegate to embedding service
```

---

### 3.3 Control Plane and Data Plane Separation

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TenantRecord:
    """Tenant registration in the control plane."""
    tenant_id: str
    name: str
    tier: str                        # "starter" | "professional" | "enterprise"
    # Data plane configuration
    vector_collection_prefix: str
    prompt_template_id: str
    model_alias: str
    top_k: int = 5
    # Quotas (from control plane, enforced in data plane)
    daily_token_limit: int = 100_000
    daily_cost_limit_usd: float = 5.0
    # Compliance
    data_residency: str = "any"      # "EU" | "US" | "any"
    eu_ai_act_tier: str = "minimal_risk"
    # Operational
    active: bool = True
    created_at: str = ""

class ControlPlane:
    """
    Manages tenant lifecycle and propagates configuration to data plane.
    Separated from data plane: control plane changes do not require
    data plane restarts.
    """
    def __init__(self, db, config_store):
        self.db     = db            # PostgreSQL — tenant records
        self.config = config_store  # Redis — fast config cache for data plane

    def provision_tenant(self, record: TenantRecord):
        """Provision a new tenant: create DB record + vector collection."""
        # Persist to control plane DB
        self.db.upsert_tenant(record)

        # Create isolated vector collections
        collections = [
            f"tenant_{record.tenant_id}/corpus_main",
            f"tenant_{record.tenant_id}/corpus_archive",
        ]
        for coll in collections:
            self.db.create_vector_collection(coll)

        # Push config to cache for data plane hot-reload
        self.config.set(f"tenant:{record.tenant_id}", record.__dict__)
        print(f"  ✓ Tenant {record.tenant_id} provisioned")

    def update_quota(self, tenant_id: str, daily_tokens: int, daily_cost: float):
        self.db.update_quota(tenant_id, daily_tokens, daily_cost)
        # Live update: data plane reads from cache, no restart needed
        record = self.db.get_tenant(tenant_id)
        record["daily_token_limit"] = daily_tokens
        record["daily_cost_limit_usd"] = daily_cost
        self.config.set(f"tenant:{tenant_id}", record)

    def deprovision_tenant(self, tenant_id: str, retain_data_days: int = 30):
        """Mark tenant inactive; schedule data deletion after retention period."""
        self.db.set_active(tenant_id, False)
        self.config.set(f"tenant:{tenant_id}:active", False)
        self.db.schedule_deletion(tenant_id, retain_data_days)
        print(f"  ✓ Tenant {tenant_id} deprovisioned; data deleted in {retain_data_days}d")
```

---

### 3.4 Cross-Tenant Governance and Cost Allocation

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TenantUsageReport:
    tenant_id: str
    period: str            # ISO date range
    total_queries: int
    total_tokens: int
    total_cost_usd: float
    avg_faithfulness: Optional[float]
    avg_latency_ms: float
    refusal_rate: float
    incidents: int
    compliance_status: str   # "compliant" | "non_compliant" | "review_required"

class CrossTenantGovernanceReport:
    """Generates platform-wide and per-tenant governance reports."""

    def __init__(self, metrics_store, audit_store, compliance_runner):
        self.metrics    = metrics_store
        self.audit      = audit_store
        self.compliance = compliance_runner

    def generate_monthly_report(self, period: str) -> dict:
        tenant_ids = self.metrics.list_active_tenants(period)
        tenant_reports = []

        for tid in tenant_ids:
            usage    = self.metrics.get_usage(tid, period)
            quality  = self.metrics.get_quality(tid, period)
            incidents = self.audit.count_incidents(tid, period)
            comp_result = self.compliance.run({"system_id": tid})

            tenant_reports.append(TenantUsageReport(
                tenant_id=tid,
                period=period,
                total_queries=usage["total_queries"],
                total_tokens=usage["total_tokens"],
                total_cost_usd=usage["total_cost_usd"],
                avg_faithfulness=quality.get("faithfulness"),
                avg_latency_ms=quality.get("avg_latency_ms", 0),
                refusal_rate=quality.get("refusal_rate", 0),
                incidents=incidents,
                compliance_status=comp_result.overall_status
            ))

        total_cost = sum(r.total_cost_usd for r in tenant_reports)
        return {
            "period":         period,
            "tenant_count":   len(tenant_reports),
            "platform_cost":  round(total_cost, 2),
            "tenants":        [r.__dict__ for r in tenant_reports],
            "non_compliant":  [r.tenant_id for r in tenant_reports
                               if r.compliance_status == "non_compliant"],
            "generated_at":   datetime.utcnow().isoformat()
        }
```

---

> ### 📋 Chapter Summary
>
> - Three tenancy models (silo, pool, bridge) trade isolation level for cost and complexity. The bridge model — shared compute, isolated data — is the most common enterprise choice.
> - **Multi-tenant isolation** is enforced at three levels: Qdrant collection namespace, mandatory server-side metadata filters, and RBAC at the API gateway.
> - **Control plane / data plane separation** allows quota and configuration changes to propagate via a config cache without data plane restarts.
> - Cross-tenant governance reports combine usage, quality, incidents, and compliance status per tenant — providing the platform team a single view for SLA management and billing.

---

> ### ❓ Comprehension Questions
>
> 1. The bridge model uses shared compute but isolated Qdrant collections. A "noisy neighbour" tenant submits 100 large requests simultaneously, saturating the embedding service. How would you implement request-level tenant fairness without isolating compute entirely?
> 2. `ControlPlane.deprovision_tenant` retains data for 30 days before deletion. GDPR Art. 17 requires erasure "without undue delay". Is 30 days acceptable under GDPR? What factors determine the acceptable retention period after deprovisioning?
> 3. The cross-tenant report includes `avg_faithfulness` per tenant. Tenant A has 0.91 faithfulness; Tenant B has 0.73. Both use the same RAG platform. What tenant-specific factors (not platform bugs) could explain this 18-point gap?
> 4. `MultiTenantRAGRouter` enforces a server-side `tenant_id` filter on every Qdrant query. A developer bypasses the router and calls Qdrant directly with a forged `tenant_id` filter. What infrastructure-level control prevents this, and how would you implement it?
> 5. A new enterprise tenant requires data residency in the EU. The platform currently runs in `us-east-1`. Design the minimum architecture change to support EU-resident tenants without rebuilding the entire platform.

---

## References

### Documentation
- [Qdrant Multi-tenancy Guide](https://qdrant.tech/documentation/guides/multiple-partitions/)
- [Kubernetes Namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/)
- [AWS Organizations for Multi-Account](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_introduction.html)

---

## Chapter 4 — Air-Gapped and On-Premise Architecture 🔓🧪

### 4.1 Constraints and Design Principles

An air-gapped deployment has zero outbound internet connectivity. Every component must run locally: LLM inference, embeddings, vector DB, object storage, secrets management, container registry, and monitoring. This constraint eliminates cloud-managed services and requires careful component selection.

```python
AIR_GAP_CONSTRAINTS = {
    "network":    "No outbound internet. Internal network only.",
    "models":     "All models must be downloaded before deployment and stored on local registry.",
    "secrets":    "No cloud KMS. Use HashiCorp Vault or SOPS + local key.",
    "storage":    "No S3. Use MinIO (S3-compatible, self-hosted).",
    "registry":   "No Docker Hub or ECR. Use Harbor or Nexus.",
    "updates":    "Software updates require manual air-gap transfer process.",
    "licensing":  "All components must be open-source or have offline licensing.",
}

AIR_GAP_DESIGN_PRINCIPLES = [
    "Prefer OSS components with no telemetry or phone-home behaviour",
    "Pin all container image digests — no 'latest' tags",
    "Store all model weights in local object storage (MinIO)",
    "All secret references point to local Vault — never hardcoded",
    "Monitoring stack is self-contained (Prometheus + Grafana + Loki, no cloud agent)",
    "Document every external dependency that was replaced and its local equivalent",
]
```

---

### 4.2 Component Selection for Air-Gapped Environments

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ComponentMapping:
    cloud_component: str
    on_premise_replacement: str
    notes: str
    license: str
    install_command: str

AIR_GAP_COMPONENT_MAP = [
    ComponentMapping(
        "OpenAI GPT-4o",
        "Ollama + Llama 3.1 / Mistral / Phi-3",
        "Pull models: ollama pull llama3.1:8b. OpenAI-compatible API.",
        "MIT / Meta Llama Community License",
        "curl -fsSL https://ollama.ai/install.sh | sh"
    ),
    ComponentMapping(
        "OpenAI text-embedding-3-small",
        "sentence-transformers/all-MiniLM-L6-v2 (local)",
        "Served via HuggingFace Inference Server or custom FastAPI.",
        "Apache 2.0",
        "pip install sentence-transformers"
    ),
    ComponentMapping(
        "Pinecone / Weaviate Cloud",
        "Qdrant OSS (self-hosted)",
        "Docker or Kubernetes. No internet required after image pull.",
        "Apache 2.0",
        "docker run -p 6333:6333 qdrant/qdrant"
    ),
    ComponentMapping(
        "AWS S3",
        "MinIO",
        "S3-compatible API. Use for model weights, backups, corpus storage.",
        "GNU AGPL v3",
        "docker run -p 9000:9000 minio/minio server /data"
    ),
    ComponentMapping(
        "AWS Secrets Manager",
        "HashiCorp Vault OSS",
        "Full secrets management with dynamic secrets and audit log.",
        "BUSL 1.1 (free for self-hosted)",
        "https://developer.hashicorp.com/vault/install"
    ),
    ComponentMapping(
        "AWS ECR / Docker Hub",
        "Harbor",
        "Container registry with vulnerability scanning and RBAC.",
        "Apache 2.0",
        "helm install harbor harbor/harbor"
    ),
    ComponentMapping(
        "Datadog / CloudWatch",
        "Prometheus + Grafana + Loki + Jaeger",
        "Full observability stack. All OSS, no external endpoints.",
        "Apache 2.0",
        "helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack"
    ),
    ComponentMapping(
        "GitHub Actions / CircleCI",
        "Gitea + Gitea Actions or Jenkins",
        "Self-hosted CI/CD. Gitea is lightweight Git + Actions compatible.",
        "MIT",
        "helm install gitea gitea-charts/gitea"
    ),
]
```

---

### 4.3 Full On-Premise Stack

```yaml
# docker-compose.airgap.yml — Complete air-gapped RAG stack
# All images must be pre-pulled and pushed to Harbor registry
# Run: docker-compose -f docker-compose.airgap.yml up -d

version: "3.9"

services:

  # ── LLM Inference ───────────────────────────────────────────────
  ollama:
    image: harbor.internal/ollama/ollama:0.4.7
    container_name: ollama
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - OLLAMA_HOST=0.0.0.0
    ports: ["11434:11434"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ── Embedding Service ────────────────────────────────────────────
  embedding:
    image: harbor.internal/ai-platform/embedding-service:2.3.1
    container_name: embedding
    volumes:
      - model_cache:/app/models      # Models pre-downloaded and mounted
    environment:
      - MODEL_NAME=all-MiniLM-L6-v2
      - MODEL_PATH=/app/models/all-MiniLM-L6-v2
    ports: ["8001:8001"]
    restart: unless-stopped

  # ── Vector Database ──────────────────────────────────────────────
  qdrant:
    image: harbor.internal/qdrant/qdrant:v1.12.0
    container_name: qdrant
    volumes:
      - qdrant_data:/qdrant/storage
    ports: ["6333:6333"]
    environment:
      - QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 15s

  # ── Object Storage (S3-compatible) ───────────────────────────────
  minio:
    image: harbor.internal/minio/minio:RELEASE.2024-11-07T00-52-20Z
    container_name: minio
    volumes:
      - minio_data:/data
    environment:
      - MINIO_ROOT_USER=admin
      - MINIO_ROOT_PASSWORD_FILE=/run/secrets/minio_password
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    secrets: [minio_password]
    restart: unless-stopped

  # ── Secrets Management ───────────────────────────────────────────
  vault:
    image: harbor.internal/hashicorp/vault:1.18.1
    container_name: vault
    cap_add: [IPC_LOCK]
    volumes:
      - vault_data:/vault/data
      - ./vault/config.hcl:/vault/config/config.hcl
    ports: ["8200:8200"]
    command: vault server -config=/vault/config/config.hcl
    restart: unless-stopped

  # ── RAG API ──────────────────────────────────────────────────────
  rag-api:
    image: harbor.internal/ai-platform/rag-api:2.4.0
    container_name: rag-api
    environment:
      - LLM_BASE_URL=http://ollama:11434/v1
      - EMBEDDING_URL=http://embedding:8001
      - QDRANT_URL=http://qdrant:6333
      - VAULT_ADDR=http://vault:8200
      - VAULT_ROLE=rag-api
    ports: ["8080:8080"]
    depends_on: [ollama, embedding, qdrant, vault]
    restart: unless-stopped

  # ── Observability ────────────────────────────────────────────────
  prometheus:
    image: harbor.internal/prometheus/prometheus:v3.0.1
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]
    restart: unless-stopped

  grafana:
    image: harbor.internal/grafana/grafana:11.3.1
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
    secrets: [grafana_password]
    restart: unless-stopped

volumes:
  ollama_models:
  qdrant_data:
  minio_data:
  vault_data:
  model_cache:
  grafana_data:

secrets:
  minio_password:
    file: ./secrets/minio_password.txt
  grafana_password:
    file: ./secrets/grafana_password.txt
```

---

### 🧪 Hands-on Lab: Architecture Decision Record Generator

**Objective:** Generate a structured ADR (Architecture Decision Record) for a RAG system architecture choice, covering context, options, decision, and consequences.

```python
#!/usr/bin/env python3
"""
adr_generator.py — Architecture Decision Record generator.
Produces a structured ADR.md for a given architecture choice.
No external dependencies required.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class ArchitectureOption:
    name: str
    description: str
    pros: list[str]
    cons: list[str]
    cost_estimate: str
    implementation_effort: str    # "low" | "medium" | "high"

@dataclass
class ADR:
    adr_id: str
    title: str
    status: str                   # "proposed" | "accepted" | "deprecated" | "superseded"
    context: str
    decision_drivers: list[str]
    options: list[ArchitectureOption]
    chosen_option: str            # Name of chosen option
    decision_rationale: str
    consequences: dict            # {"positive": [...], "negative": [...], "neutral": [...]}
    links: list[str] = field(default_factory=list)
    date: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    authors: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        chosen = next((o for o in self.options if o.name == self.chosen_option), None)
        lines = [
            f"# ADR {self.adr_id}: {self.title}",
            f"",
            f"**Status:** {self.status.upper()}  ",
            f"**Date:** {self.date}  ",
            f"**Authors:** {', '.join(self.authors) or 'TBD'}",
            f"",
            f"---",
            f"",
            f"## Context",
            f"",
            self.context,
            f"",
            f"## Decision Drivers",
            f"",
        ] + [f"- {d}" for d in self.decision_drivers] + [
            f"",
            f"## Options Considered",
            f"",
        ]
        for opt in self.options:
            marker = "✅ **CHOSEN**" if opt.name == self.chosen_option else ""
            lines += [
                f"### {opt.name} {marker}",
                f"",
                f"{opt.description}",
                f"",
                f"**Pros:**",
            ] + [f"- {p}" for p in opt.pros] + [
                f"",
                f"**Cons:**",
            ] + [f"- {c}" for c in opt.cons] + [
                f"",
                f"*Cost: {opt.cost_estimate} | Effort: {opt.implementation_effort}*",
                f"",
            ]
        lines += [
            f"## Decision",
            f"",
            f"**Chosen: {self.chosen_option}**",
            f"",
            self.decision_rationale,
            f"",
            f"## Consequences",
            f"",
            f"**Positive:**",
        ] + [f"- {c}" for c in self.consequences.get("positive", [])] + [
            f"",
            f"**Negative:**",
        ] + [f"- {c}" for c in self.consequences.get("negative", [])] + [
            f"",
            f"**Neutral:**",
        ] + [f"- {c}" for c in self.consequences.get("neutral", [])]

        if self.links:
            lines += ["", "## Links", ""] + [f"- {l}" for l in self.links]
        return "\n".join(lines)


# ── Example ADR: Vector Database Selection ────────────────────────
vector_db_adr = ADR(
    adr_id="0003",
    title="Vector Database Selection for Production RAG Platform",
    status="accepted",
    context=(
        "The RAG platform requires a vector database to store and query document "
        "embeddings. The system must support 500K+ vectors, multi-tenancy via "
        "metadata filters, horizontal scaling, and HA. We evaluated three options."
    ),
    decision_drivers=[
        "Support for 500K+ vectors with sub-100ms P99 retrieval",
        "Multi-tenancy via payload filters enforced server-side",
        "Kubernetes StatefulSet deployment with HA (≥3 replicas)",
        "Active OSS community and recent releases",
        "On-premise deployment support (air-gap compatible)",
        "S3-compatible snapshot backup",
    ],
    options=[
        ArchitectureOption(
            "Qdrant",
            "Written in Rust. Distributed mode with P2P replication. "
            "Rich payload filtering. REST + gRPC API.",
            pros=[
                "Excellent filtering performance on payload metadata",
                "Distributed mode with shard replication",
                "Active development (weekly releases)",
                "OpenAI-compatible REST API",
                "Built-in snapshot to S3",
            ],
            cons=[
                "Smaller ecosystem than Weaviate",
                "Distributed mode requires 3+ nodes for quorum",
            ],
            cost_estimate="Free OSS; ~$0.10/GB/month self-hosted storage",
            implementation_effort="medium"
        ),
        ArchitectureOption(
            "Weaviate",
            "Go-based. GraphQL + REST API. Strong hybrid search (BM25 + vector). "
            "Module ecosystem for automatic vectorisation.",
            pros=[
                "Best-in-class hybrid search (BM25 + vector)",
                "GraphQL API for complex filtered queries",
                "Large community and extensive documentation",
            ],
            cons=[
                "Higher memory footprint than Qdrant",
                "Cluster setup more complex",
                "Module system adds operational complexity",
            ],
            cost_estimate="Free OSS; Cloud from $25/month",
            implementation_effort="medium"
        ),
        ArchitectureOption(
            "pgvector (PostgreSQL extension)",
            "Adds vector similarity search to PostgreSQL. "
            "No separate infrastructure required.",
            pros=[
                "No new infrastructure — extends existing PostgreSQL",
                "ACID transactions; combine vector and relational queries",
                "Familiar operational model for teams with PG expertise",
            ],
            cons=[
                "Poor performance at scale (>500K vectors, high QPS)",
                "No native distributed mode",
                "ANN accuracy lower than dedicated vector DBs at scale",
            ],
            cost_estimate="Free (uses existing PostgreSQL)",
            implementation_effort="low"
        ),
    ],
    chosen_option="Qdrant",
    decision_rationale=(
        "Qdrant is selected because it provides the best balance of filtering "
        "performance, distributed mode for HA, and operational simplicity. "
        "Its Rust implementation delivers consistent sub-50ms P99 retrieval at "
        "our target corpus size. The payload filter system maps directly to our "
        "multi-tenant isolation requirement. Weaviate's hybrid search is superior "
        "but our retrieval pipeline handles BM25 at the application layer. "
        "pgvector is rejected due to scale limitations — expected corpus growth "
        "to 2M vectors within 12 months exceeds pgvector's performance envelope."
    ),
    consequences={
        "positive": [
            "Sub-50ms P99 vector retrieval at 500K vectors, validated in benchmarks",
            "Tenant isolation enforced server-side via payload filters",
            "Daily S3 snapshots satisfy data durability requirements",
        ],
        "negative": [
            "Teams must learn Qdrant-specific client library and filter syntax",
            "Distributed mode requires minimum 3 nodes — minimum infrastructure cost",
        ],
        "neutral": [
            "Migration from Chroma (used in development) requires re-embedding ~10K dev corpus",
            "Qdrant version upgrades must be coordinated across all 3 nodes",
        ]
    },
    links=[
        "https://qdrant.tech/documentation/",
        "Benchmark: qdrant.tech/benchmarks/",
        "ADR-0001: Kubernetes as deployment platform",
        "ADR-0002: Embedding model selection",
    ],
    authors=["platform-team@company.com"]
)

# Generate and save ADR
adr_md = vector_db_adr.to_markdown()
output_path = Path("docs/adr/ADR-0003-vector-database.md")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(adr_md)
print(adr_md[:2000])
print(f"\n... (full ADR saved to {output_path})")

# ── Architecture comparison summary ───────────────────────────────
print("\n" + "=" * 55)
print("  Architecture Tier Comparison")
print("=" * 55)

COMPARISON = [
    ("Component",           "Minimal",         "Production",          "Air-Gapped"),
    ("LLM",                 "OpenAI API",       "OpenAI + fallback",   "Ollama/vLLM"),
    ("Embeddings",          "In-process",       "Svc ×2",              "In-process (local)"),
    ("Vector DB",           "ChromaDB local",   "Qdrant ×3",           "Qdrant OSS"),
    ("Cache",               "None",             "Redis ×3",            "Redis OSS"),
    ("Object Storage",      "Local disk",       "S3",                  "MinIO"),
    ("Secrets",             "Env vars",         "Vault/Secrets Mgr",   "Vault OSS"),
    ("Container Registry",  "Docker Hub",       "ECR",                 "Harbor"),
    ("CI/CD",               "GitHub Actions",   "GitHub Actions",      "Gitea"),
    ("Monitoring",          "Logs to stdout",   "Prometheus+Grafana",  "Prometheus+Grafana"),
    ("Monthly cost (est.)", "~$50–150",         "~$800–2000",          "Capex only"),
]

col_w = [24, 18, 22, 22]
header = COMPARISON[0]
print("  " + "".join(f"{h:<{w}}" for h, w in zip(header, col_w)))
print("  " + "-" * sum(col_w))
for row in COMPARISON[1:]:
    print("  " + "".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
```

**Run the lab:**
```bash
python adr_generator.py
```

**Extensions:**
- Add a fourth tier "Enterprise SaaS" to the comparison table with multi-tenant components
- Generate ADRs for two more decisions: (a) embedding model selection, (b) LLM gateway retry strategy
- Build a simple ADR index (`docs/adr/README.md`) that lists all ADRs with status and date, auto-generated from the ADR files in the directory

---

> ### 📋 Chapter Summary
>
> - The air-gapped architecture replaces every cloud-managed service with a self-hosted OSS equivalent: Ollama for LLM, Qdrant for vectors, MinIO for object storage, Vault for secrets, Harbor for container registry.
> - All three tiers (minimal, production, air-gapped) share the same application logic — only the infrastructure layer changes, validating the abstraction design from Part XI.
> - **ADRs** document architecture decisions with options, rationale, and consequences — they are living governance artefacts that must be updated when decisions change.
> - The architecture comparison table provides a concise decision matrix across tiers, enabling teams to plan their migration path from minimal to production to air-gapped.

---

> ### ❓ Comprehension Questions
>
> 1. The air-gapped stack uses Ollama for LLM inference. Ollama runs a single model at a time by default. A production workload needs to serve `llama3.1:8b` for general queries and `codellama:13b` for code questions. How would you extend the on-premise LLM platform to support multiple concurrent models?
> 2. The ADR for vector database selection lists "migration from Chroma requires re-embedding ~10K dev corpus" as a neutral consequence. In production, the corpus has 500K documents. Estimate the re-embedding time using `all-MiniLM-L6-v2` on CPU (approximate: 500 docs/minute on a 4-core server), and identify what else must be migrated beyond embeddings.
> 3. The `docker-compose.airgap.yml` uses Docker secrets for MinIO and Grafana passwords. In a true air-gapped environment, how would you bootstrap Vault without internet access, and how would you rotate secrets after initial deployment?
> 4. The multi-tenant architecture provision creates two collections per tenant: `corpus_main` and `corpus_archive`. A tenant with 3 years of history has 50 corpora variants. How would you design a corpus versioning scheme that keeps the collection namespace manageable?
> 5. An enterprise customer requires a hybrid architecture: standard queries use the cloud production stack, but queries containing specific keywords (patient names, account numbers) must be routed to an on-premise air-gapped deployment. Design the routing logic and describe the dual-write corpus maintenance challenge this creates.

---

## References

### Documentation
- [Ollama Model Library](https://ollama.ai/library) — Available models for local deployment.
- [vLLM Documentation](https://docs.vllm.ai) — GPU-optimised LLM serving.
- [MinIO Documentation](https://min.io/docs/minio/container/index.html)
- [HashiCorp Vault](https://developer.hashicorp.com/vault/docs)
- [Harbor Container Registry](https://goharbor.io/docs/)
- [Architecture Decision Records](https://adr.github.io) — ADR format and tooling.

### Papers
- [Architectural Decision Records in Practice](https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — Nygard, 2011. Original ADR proposal.

---

> **Navigation**
> [← Part XVI — Operations](part_16_operations.md) | [→ Part XVIII — Practical Case Studies](part_18_case_studies.md)
