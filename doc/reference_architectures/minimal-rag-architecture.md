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

---
[« Back to reference_architectures Index](index.md) | [🏠 Home](../../index.md)