## Chapter 4 — Documentation as Code 🧪

### 4.1 Living Documentation for AI Systems

AI systems suffer from a documentation anti-pattern: decisions are made verbally in meetings, committed as code without explanation, and the reasoning is lost within weeks. Six months later, no one can answer "why is the chunk size 512?" or "why did we choose Qdrant over Weaviate?"

Documentation as code treats documentation with the same version control, review, and quality discipline applied to source code. It has three practical components:

1. **Architecture Decision Records (ADRs)** — document significant decisions and their rationale
2. **Runbooks** — executable, testable operational procedures
3. **API documentation** — auto-generated from code, always current

The key discipline: documentation lives in the repository alongside the code it describes. It is updated in the same PR as the code change. It is reviewed by the same reviewers. It becomes outdated for the same reason code becomes outdated — and is visible in the same git blame.

---

### 4.2 Architecture Decision Records

An ADR records a significant architectural decision: its context, the options considered, the decision made, and the consequences.

```markdown
<!-- docs/adr/ADR-007-vector-database-selection.md -->

# ADR-007: Vector Database Selection

**Date:** 2024-11-15
**Status:** Accepted
**Deciders:** Platform Team (Alice, Bob, Carlos)
**Supersedes:** ADR-003 (FAISS in-process index)

## Context

The RAG system needs a vector database for semantic search over 500K+ documents.
Current FAISS in-process solution cannot support:

- Multi-node horizontal scaling
- Real-time document updates without full re-index
- Metadata filtering alongside vector similarity
- Production-grade operational tooling (monitoring, backup, access control)

## Decision Drivers

- Must support hybrid search (dense + sparse) natively
- Must support metadata filtering at query time
- Must have a managed cloud offering AND a self-hosted option (EU data residency requirement)
- Must support Java client SDK for main API service
- Must be stable with > 5,000 GitHub stars and active maintenance

## Options Considered

| Criterion                                 | Qdrant | Weaviate | Pinecone | Milvus |
| ----------------------------------------- | ------ | -------- | -------- | ------ |
| Hybrid search native                      | ✓      | ✓        | ✗        | ✓      |
| Metadata filtering                        | ✓      | ✓        | ✓        | ✓      |
| Self-hosted option                        | ✓      | ✓        | ✗        | ✓      |
| Java SDK                                  | ✓      | ✓        | ✓        | ✓      |
| Operational maturity                      | High   | High     | High     | Medium |
| Benchmark (1M vectors, top-5 latency p99) | 8ms    | 12ms     | 15ms     | 10ms   |

Pinecone was eliminated due to no self-hosted option (EU data residency requirement).
Milvus was eliminated due to higher operational complexity for the team's Kubernetes experience level.

## Decision

**Qdrant** — best combination of hybrid search support, performance, operational maturity,
and alignment with our self-hosted requirement.

## Consequences

### Positive

- Native hybrid search eliminates need for separate BM25 index
- Rust-based core provides low-latency P99 performance
- Docker Compose deployment for development, Kubernetes Operator for production

### Negative

- Team must learn Qdrant-specific APIs (Java SDK is less mature than Python)
- Migration from FAISS requires re-embedding entire corpus (~2 hours)

### Risks

- Qdrant Java SDK lags Python SDK in feature parity — mitigated by wrapping in internal `java-rag` library

## Links

- [Qdrant Documentation](https://qdrant.tech/documentation)
- [Benchmark Script](../scripts/vector_db_benchmark.py)
- [Migration Plan](ADR-007-migration-plan.md)
```

**Benchmark Script**

```python
#!/usr/bin/env python3
"""
vector_db_benchmark.py
======================
Benchmark script referenced in ADR-007 (Vector Database Selection).

Measures latency and throughput for top-K semantic search across:
  - Qdrant       (local via Docker or qdrant-client)
  - Weaviate     (local via Docker or weaviate-client)
  - Milvus       (local via Docker or pymilvus)

Usage
-----
  # Run all backends (requires each running locally):
  python vector_db_benchmark.py --backends qdrant weaviate milvus

  # Qdrant only, 100K vectors, 500 queries:
  python vector_db_benchmark.py --backends qdrant --n-vectors 100000 --n-queries 500

  # Dry-run with synthetic data only (no real DB connections):
  python vector_db_benchmark.py --dry-run

Dependencies
------------
  pip install qdrant-client weaviate-client pymilvus numpy tqdm

Docker quick-start
------------------
  # Qdrant
  docker run -d -p 6333:6333 qdrant/qdrant

  # Weaviate
  docker run -d -p 8080:8080 -e QUERY_DEFAULTS_LIMIT=25 \
    -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
    -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
    semitechnologies/weaviate:latest

  # Milvus (standalone)
  docker run -d -p 19530:19530 milvusdb/milvus:v2.4.0-standalone
"""

import argparse
import time
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Configuration ─────────────────────────────────────────────────

DIMENSION       = 1536          # OpenAI text-embedding-3-small dimensions
COLLECTION_NAME = "benchmark"
TOP_K           = 5             # Match ADR-007 benchmark: top-5 latency p99


@dataclass
class BenchmarkConfig:
    n_vectors:  int   = 1_000_000   # Number of vectors to index
    n_queries:  int   = 1_000       # Number of search queries
    top_k:      int   = TOP_K
    dimension:  int   = DIMENSION
    batch_size: int   = 1_000       # Upsert batch size
    warmup_queries: int = 50        # Warm-up before timing


@dataclass
class BenchmarkResult:
    backend:        str
    n_vectors:      int
    n_queries:      int
    top_k:          int
    index_time_s:   float
    latencies_ms:   list[float] = field(default_factory=list)
    error:          Optional[str] = None

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def qps(self) -> float:
        total_s = sum(self.latencies_ms) / 1000.0
        return len(self.latencies_ms) / total_s if total_s > 0 else 0.0


# ── Data generation ────────────────────────────────────────────────

def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate normalised random vectors simulating text embeddings."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def generate_payloads(n: int) -> list[dict]:
    """Generate metadata payloads simulating document chunks."""
    sources  = ["confluence", "github_wiki", "runbook", "postmortem"]
    return [
        {
            "doc_id":    f"doc_{i:07d}",
            "source":    sources[i % len(sources)],
            "chunk_idx": i % 10,
            "version":   "v1",
        }
        for i in range(n)
    ]


# ── Qdrant backend ─────────────────────────────────────────────────

def benchmark_qdrant(cfg: BenchmarkConfig, vectors: np.ndarray,
                     queries: np.ndarray) -> BenchmarkResult:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct, Filter,
        )
    except ImportError:
        return BenchmarkResult("qdrant", cfg.n_vectors, cfg.n_queries, cfg.top_k,
                               0.0, error="qdrant-client not installed")

    result = BenchmarkResult("qdrant", cfg.n_vectors, cfg.n_queries,
                             cfg.top_k, 0.0)
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=30)

        # Recreate collection
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=cfg.dimension, distance=Distance.COSINE),
        )

        # Index
        payloads = generate_payloads(cfg.n_vectors)
        t0 = time.perf_counter()
        for start in range(0, cfg.n_vectors, cfg.batch_size):
            end = min(start + cfg.batch_size, cfg.n_vectors)
            points = [
                PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i])
                for i in range(start, end)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
        result.index_time_s = time.perf_counter() - t0
        print(f"  [qdrant] indexed {cfg.n_vectors:,} vectors in {result.index_time_s:.1f}s")

        # Warm-up
        for q in queries[:cfg.warmup_queries]:
            client.search(COLLECTION_NAME, query_vector=q.tolist(), limit=cfg.top_k)

        # Benchmark
        for q in queries:
            t0 = time.perf_counter()
            client.search(COLLECTION_NAME, query_vector=q.tolist(), limit=cfg.top_k)
            result.latencies_ms.append((time.perf_counter() - t0) * 1000)

        client.delete_collection(COLLECTION_NAME)

    except Exception as e:
        result.error = str(e)
    return result


# ── Weaviate backend ────────────────────────────────────────────────

def benchmark_weaviate(cfg: BenchmarkConfig, vectors: np.ndarray,
                       queries: np.ndarray) -> BenchmarkResult:
    try:
        import weaviate
    except ImportError:
        return BenchmarkResult("weaviate", cfg.n_vectors, cfg.n_queries, cfg.top_k,
                               0.0, error="weaviate-client not installed")

    result = BenchmarkResult("weaviate", cfg.n_vectors, cfg.n_queries,
                             cfg.top_k, 0.0)
    try:
        client = weaviate.Client("http://localhost:8080")

        # Recreate class
        class_name = "Benchmark"
        if client.schema.exists(class_name):
            client.schema.delete_class(class_name)
        client.schema.create_class({
            "class": class_name,
            "vectorizer": "none",
            "properties": [
                {"name": "doc_id",    "dataType": ["text"]},
                {"name": "source",    "dataType": ["text"]},
                {"name": "chunk_idx", "dataType": ["int"]},
            ],
        })

        # Index
        payloads = generate_payloads(cfg.n_vectors)
        t0 = time.perf_counter()
        with client.batch(batch_size=cfg.batch_size) as batch:
            for i in range(cfg.n_vectors):
                batch.add_data_object(
                    data_object=payloads[i],
                    class_name=class_name,
                    vector=vectors[i].tolist(),
                )
        result.index_time_s = time.perf_counter() - t0
        print(f"  [weaviate] indexed {cfg.n_vectors:,} vectors in {result.index_time_s:.1f}s")

        # Warm-up
        for q in queries[:cfg.warmup_queries]:
            client.query.get(class_name).with_near_vector(
                {"vector": q.tolist()}
            ).with_limit(cfg.top_k).do()

        # Benchmark
        for q in queries:
            t0 = time.perf_counter()
            client.query.get(class_name).with_near_vector(
                {"vector": q.tolist()}
            ).with_limit(cfg.top_k).do()
            result.latencies_ms.append((time.perf_counter() - t0) * 1000)

        client.schema.delete_class(class_name)

    except Exception as e:
        result.error = str(e)
    return result


# ── Milvus backend ──────────────────────────────────────────────────

def benchmark_milvus(cfg: BenchmarkConfig, vectors: np.ndarray,
                     queries: np.ndarray) -> BenchmarkResult:
    try:
        from pymilvus import (
            connections, Collection, CollectionSchema,
            FieldSchema, DataType, utility,
        )
    except ImportError:
        return BenchmarkResult("milvus", cfg.n_vectors, cfg.n_queries, cfg.top_k,
                               0.0, error="pymilvus not installed")

    result = BenchmarkResult("milvus", cfg.n_vectors, cfg.n_queries,
                             cfg.top_k, 0.0)
    try:
        connections.connect(host="localhost", port="19530")

        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        schema = CollectionSchema(fields=[
            FieldSchema("id",        DataType.INT64,       is_primary=True),
            FieldSchema("vector",    DataType.FLOAT_VECTOR, dim=cfg.dimension),
            FieldSchema("source",    DataType.VARCHAR,     max_length=64),
            FieldSchema("chunk_idx", DataType.INT64),
        ])
        collection = Collection(COLLECTION_NAME, schema)

        # Index
        payloads = generate_payloads(cfg.n_vectors)
        t0 = time.perf_counter()
        for start in range(0, cfg.n_vectors, cfg.batch_size):
            end = min(start + cfg.batch_size, cfg.n_vectors)
            collection.insert([
                list(range(start, end)),
                vectors[start:end].tolist(),
                [p["source"] for p in payloads[start:end]],
                [p["chunk_idx"] for p in payloads[start:end]],
            ])
        collection.create_index("vector", {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024},
        })
        collection.load()
        result.index_time_s = time.perf_counter() - t0
        print(f"  [milvus] indexed {cfg.n_vectors:,} vectors in {result.index_time_s:.1f}s")

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}

        # Warm-up
        for q in queries[:cfg.warmup_queries]:
            collection.search([q.tolist()], "vector", search_params, limit=cfg.top_k)

        # Benchmark
        for q in queries:
            t0 = time.perf_counter()
            collection.search([q.tolist()], "vector", search_params, limit=cfg.top_k)
            result.latencies_ms.append((time.perf_counter() - t0) * 1000)

        utility.drop_collection(COLLECTION_NAME)

    except Exception as e:
        result.error = str(e)
    return result


# ── Dry-run backend (no real DB) ────────────────────────────────────

def benchmark_dryrun(name: str, cfg: BenchmarkConfig,
                     latency_p50: float, jitter: float = 0.3) -> BenchmarkResult:
    """Simulate benchmark results for testing without a running database."""
    rng = np.random.default_rng(42)
    latencies = rng.lognormal(
        mean=np.log(latency_p50),
        sigma=jitter,
        size=cfg.n_queries,
    ).tolist()
    return BenchmarkResult(
        backend=f"{name} (simulated)",
        n_vectors=cfg.n_vectors,
        n_queries=cfg.n_queries,
        top_k=cfg.top_k,
        index_time_s=cfg.n_vectors / 50_000,
        latencies_ms=latencies,
    )


# ── Reporting ──────────────────────────────────────────────────────

def print_report(results: list[BenchmarkResult], cfg: BenchmarkConfig) -> None:
    print("\n" + "=" * 72)
    print(f"  Vector DB Benchmark — {cfg.n_vectors:,} vectors · "
          f"{cfg.n_queries:,} queries · top-{cfg.top_k} · dim={cfg.dimension}")
    print("=" * 72)
    header = f"  {'Backend':<24} {'Index(s)':>8}  {'Mean':>7}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'QPS':>7}"
    print(header)
    print("  " + "-" * 68)

    # Sort by p99 ascending
    valid = [r for r in results if not r.error]
    errored = [r for r in results if r.error]

    for r in sorted(valid, key=lambda x: x.p99_ms):
        print(f"  {r.backend:<24} {r.index_time_s:>8.1f}  "
              f"{r.mean_ms:>6.1f}ms  {r.p50_ms:>6.1f}ms  "
              f"{r.p95_ms:>6.1f}ms  {r.p99_ms:>6.1f}ms  {r.qps:>6.0f}")

    for r in errored:
        print(f"  {r.backend:<24}  ERROR: {r.error}")

    if valid:
        winner = min(valid, key=lambda x: x.p99_ms)
        print(f"\n  ✓ Lowest P99 latency: {winner.backend} ({winner.p99_ms:.1f}ms)")
        print(f"    → ADR-007 benchmark target: ≤10ms P99 for top-{cfg.top_k}")

    print("=" * 72)


def export_csv(results: list[BenchmarkResult], path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "n_vectors", "n_queries", "top_k",
                         "index_time_s", "mean_ms", "p50_ms", "p95_ms",
                         "p99_ms", "qps", "error"])
        for r in results:
            writer.writerow([
                r.backend, r.n_vectors, r.n_queries, r.top_k,
                f"{r.index_time_s:.3f}",
                f"{r.mean_ms:.3f}", f"{r.p50_ms:.3f}",
                f"{r.p95_ms:.3f}", f"{r.p99_ms:.3f}",
                f"{r.qps:.1f}", r.error or "",
            ])
    print(f"\n  Results exported to: {path}")


# ── CLI ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vector database benchmark for ADR-007 evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--backends", nargs="+",
                        choices=["qdrant", "weaviate", "milvus"],
                        default=["qdrant", "weaviate", "milvus"],
                        help="Backends to benchmark (default: all three)")
    parser.add_argument("--n-vectors", type=int, default=1_000_000,
                        help="Number of vectors to index (default: 1_000_000)")
    parser.add_argument("--n-queries", type=int, default=1_000,
                        help="Number of search queries (default: 1_000)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Top-K results per query (default: {TOP_K})")
    parser.add_argument("--batch-size", type=int, default=1_000,
                        help="Upsert batch size (default: 1_000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate results without real DB connections")
    parser.add_argument("--export-csv", metavar="PATH",
                        help="Export results to CSV file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        n_vectors=args.n_vectors,
        n_queries=args.n_queries,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    if args.dry_run:
        print("Dry-run mode — generating simulated results\n")
        # Simulated P99 values match ADR-007 comparison table
        results = [
            benchmark_dryrun("qdrant",   cfg, latency_p50=5.0),
            benchmark_dryrun("weaviate", cfg, latency_p50=8.0),
            benchmark_dryrun("milvus",   cfg, latency_p50=7.0),
        ]
    else:
        print(f"Generating {cfg.n_vectors:,} vectors (dim={cfg.dimension})...")
        vectors = generate_vectors(cfg.n_vectors, cfg.dimension)
        queries = generate_vectors(cfg.n_queries, cfg.dimension, seed=99)
        print(f"Running benchmarks: {args.backends}\n")

        results = []
        backend_fns = {
            "qdrant":   lambda: benchmark_qdrant(cfg, vectors, queries),
            "weaviate": lambda: benchmark_weaviate(cfg, vectors, queries),
            "milvus":   lambda: benchmark_milvus(cfg, vectors, queries),
        }
        for name in args.backends:
            print(f"Benchmarking {name}...")
            results.append(backend_fns[name]())

    print_report(results, cfg)

    if args.export_csv:
        export_csv(results, args.export_csv)


if __name__ == "__main__":
    main()
```

**ADR management tooling:**

```python
# scripts/adr_manager.py — Create and list ADRs

import re
from pathlib import Path
from datetime import datetime

ADR_DIR = Path("docs/adr")
ADR_TEMPLATE = """# ADR-{number:03d}: {title}

**Date:** {date}
**Status:** Proposed
**Deciders:** [Add decision makers]

## Context

[Describe the situation and problem that requires a decision]

## Decision Drivers

- [Driver 1]
- [Driver 2]

## Options Considered

| Criterion | Option A | Option B |
|---|---|---|
| [Criterion 1] | | |

## Decision

[State the decision]

## Consequences

### Positive
- [Positive consequence]

### Negative
- [Negative consequence]

## Links
- [Relevant documentation or tickets]
"""

def next_adr_number() -> int:
    existing = list(ADR_DIR.glob("ADR-*.md"))
    if not existing:
        return 1
    numbers = [int(re.search(r'ADR-(\d+)', f.stem).group(1)) for f in existing
               if re.search(r'ADR-(\d+)', f.stem)]
    return max(numbers) + 1 if numbers else 1

def create_adr(title: str) -> Path:
    ADR_DIR.mkdir(parents=True, exist_ok=True)
    number = next_adr_number()
    slug = title.lower().replace(" ", "-").replace("/", "-")
    filename = ADR_DIR / f"ADR-{number:03d}-{slug}.md"
    content = ADR_TEMPLATE.format(
        number=number,
        title=title,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    )
    filename.write_text(content)
    print(f"Created: {filename}")
    return filename

def list_adrs() -> list[dict]:
    adrs = []
    for f in sorted(ADR_DIR.glob("ADR-*.md")):
        content = f.read_text()
        status_match = re.search(r'\*\*Status:\*\*\s*(\w+)', content)
        title_match = re.search(r'# ADR-\d+: (.+)', content)
        adrs.append({
            "file": f.name,
            "title": title_match.group(1) if title_match else "Unknown",
            "status": status_match.group(1) if status_match else "Unknown"
        })
    return adrs
```

---

### 4.3 Runbooks as Code

A runbook describes how to perform an operational procedure. As code, it is versioned, testable, and automatically validated.

```python
# docs/runbooks/reindex_knowledge_base.py
"""
Runbook: Re-index Knowledge Base
=================================
Use when: Embedding model changed, corpus updated, index corrupted.
Owner: Platform Team
Last tested: 2024-11-01
SLA: Complete within 4 hours for corpus < 100K documents.

Steps:
1. Build new index in staging collection
2. Validate recall against eval dataset
3. Switch active collection (blue-green)
4. Deprecate old collection
5. Notify #platform-alerts
"""

import argparse
import sys
from pathlib import Path

def step_1_build_staging_index(corpus_path: str, config: dict) -> str:
    """Build new index in an isolated staging collection. Returns collection name."""
    print("[STEP 1] Building staging index...")
    # Implementation calls BlueGreenIndexManager.build_new_version()
    collection_name = f"kb_staging_{int(__import__('time').time())}"
    print(f"  ✓ Staging collection created: {collection_name}")
    return collection_name

def step_2_validate_recall(collection_name: str, eval_dataset_path: str) -> float:
    """Validate retrieval quality meets minimum threshold."""
    print("[STEP 2] Validating recall on staging index...")
    # Load eval dataset, run queries, compute Recall@5
    recall = 0.87  # Example
    print(f"  ✓ Recall@5 = {recall:.2%}")
    if recall < 0.80:
        print(f"  ✗ Recall below threshold (0.80). Aborting.", file=sys.stderr)
        sys.exit(1)
    return recall

def step_3_promote_index(collection_name: str):
    """Switch active collection atomically."""
    print(f"[STEP 3] Promoting {collection_name} to active...")
    print("  ✓ Active collection updated")

def step_4_deprecate_old(old_collection: str):
    """Mark old collection as deprecated (retain for 7 days for rollback)."""
    print(f"[STEP 4] Deprecating {old_collection} (7-day retention for rollback)...")
    print("  ✓ Old collection deprecated")

def step_5_notify(recall: float, collection_name: str, slack_webhook: str = None):
    """Post completion notification."""
    message = f"Knowledge base re-indexed. New collection: {collection_name}. Recall@5: {recall:.2%}"
    print(f"[STEP 5] Notification: {message}")
    if slack_webhook:
        import urllib.request, json
        data = json.dumps({"text": message}).encode()
        urllib.request.urlopen(urllib.request.Request(slack_webhook, data=data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index knowledge base")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--old-collection", required=True)
    parser.add_argument("--slack-webhook")
    args = parser.parse_args()

    config = {"embedding_model": "text-embedding-3-small", "chunk_size": 512}

    new_collection = step_1_build_staging_index(args.corpus, config)
    recall = step_2_validate_recall(new_collection, args.eval_dataset)
    step_3_promote_index(new_collection)
    step_4_deprecate_old(args.old_collection)
    step_5_notify(recall, new_collection, args.slack_webhook)

    print("\n✓ Re-indexing complete")
```

---

### 4.4 API Documentation with OpenAPI

```python
# apps/api/src/main.py — FastAPI with auto-generated OpenAPI docs
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="AI Knowledge Assistant API",
    version="2.0.0",
    description="""
Enterprise RAG API for querying the internal knowledge base.

## Authentication
All endpoints require an `Authorization: Bearer <token>` header.

## Rate Limits
- Standard: 100 req/min
- Enterprise: 1000 req/min
""",
    contact={"name": "Platform Team", "email": "platform@company.com"},
    openapi_tags=[
        {"name": "query", "description": "Knowledge base query operations"},
        {"name": "admin", "description": "Administrative operations"},
    ]
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="The question to answer from the knowledge base",
                          example="What is the enterprise refund policy?")
    language: Optional[str] = Field("en", description="ISO 639-1 language code for response",
                                    example="en")
    top_k: Optional[int] = Field(5, ge=1, le=20,
                                 description="Number of source documents to retrieve")

class Source(BaseModel):
    document_id: str
    title: str
    excerpt: str = Field(..., description="Relevant excerpt from the source document")
    relevance_score: float = Field(..., ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    query_id: str = Field(..., description="Unique identifier for this query, for feedback")
    latency_ms: int

@app.post(
    "/v2/query",
    response_model=QueryResponse,
    tags=["query"],
    summary="Query the knowledge base",
    response_description="Answer with supporting source documents"
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    Submit a natural language question and receive an answer grounded in
    the internal knowledge base with source citations.

    The response includes the answer text and up to `top_k` source documents
    that were used to generate the answer.
    """
    # Implementation here
    raise HTTPException(status_code=501, detail="Not implemented in example")
```

---

### 4.5 Automated Documentation Pipelines

```yaml
# .github/workflows/docs.yml
name: Documentation Pipeline

on:
  push:
    branches: [main]
  pull_request:
    paths:
      - "docs/**"
      - "apps/**/*.py"
      - "prompts/**"

jobs:
  validate-adrs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate ADR format
        run: python scripts/adr_manager.py --validate
      - name: Check ADRs have status
        run: |
          python3 -c "
          import re
          from pathlib import Path
          errors = []
          for f in Path('docs/adr').glob('ADR-*.md'):
              if not re.search(r'\*\*Status:\*\*\s*(Accepted|Proposed|Deprecated|Superseded)', f.read_text()):
                  errors.append(str(f))
          if errors:
              print('ADRs missing valid status:', errors)
              exit(1)
          print(f'All {len(list(Path(\"docs/adr\").glob(\"ADR-*.md\")))} ADRs have valid status')
          "

  generate-api-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install fastapi[all]
      - name: Export OpenAPI spec
        run: |
          python3 -c "
          import json
          from apps.api.src.main import app
          spec = app.openapi()
          open('docs/api/openapi.json', 'w').write(json.dumps(spec, indent=2))
          "
      - name: Check for breaking API changes
        run: |
          # Compare current spec with last released spec
          diff docs/api/openapi.json docs/api/openapi.last-release.json || \
            echo "::warning::API spec changed — review for breaking changes"
```

---

### 🧪 Hands-on Lab: ADR Pipeline

**Objective:** Build an ADR creation and validation CLI. Demonstrate the complete documentation-as-code workflow.

**Prerequisites:** Python standard library only.

```python
#!/usr/bin/env python3
# scripts/adr.py — ADR management CLI

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

ADR_DIR = Path("docs/adr")

TEMPLATE = """# ADR-{number:03d}: {title}

**Date:** {date}
**Status:** Proposed
**Deciders:** [List decision makers]

## Context

[What is the problem? What forces are at play?]

## Decision Drivers

- [Driver 1]
- [Driver 2]

## Options Considered

| Criterion | Option A | Option B | Option C |
|---|---|---|---|
| [Performance] | | | |
| [Cost] | | | |
| [Operational complexity] | | | |

## Decision

**[Chosen Option]** because [primary reason].

## Consequences

### Positive
- [Benefit 1]

### Negative
- [Trade-off 1]

### Neutral
- [Neutral consequence]

## Links
- [Related ADR or documentation]
"""

VALID_STATUSES = {"Proposed", "Accepted", "Deprecated", "Superseded"}

def cmd_new(title: str) -> Path:
    ADR_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(ADR_DIR.glob("ADR-*.md"))
    number = 1
    if existing:
        nums = [int(re.search(r'ADR-(\d+)', f.stem).group(1)) for f in existing
                if re.search(r'ADR-(\d+)', f.stem)]
        number = max(nums) + 1 if nums else 1
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    filepath = ADR_DIR / f"ADR-{number:03d}-{slug}.md"
    filepath.write_text(TEMPLATE.format(
        number=number, title=title,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    ))
    print(f"Created: {filepath}")
    return filepath

def cmd_list() -> list[dict]:
    adrs = []
    for f in sorted(ADR_DIR.glob("ADR-*.md")):
        text = f.read_text()
        title = re.search(r'# ADR-\d+: (.+)', text)
        status = re.search(r'\*\*Status:\*\*\s*(\w+)', text)
        date = re.search(r'\*\*Date:\*\*\s*([\d-]+)', text)
        adrs.append({
            "file": f.name,
            "title": title.group(1) if title else "—",
            "status": status.group(1) if status else "MISSING",
            "date": date.group(1) if date else "—",
        })
    if adrs:
        print(f"{'File':<35} {'Status':<12} {'Date':<12} {'Title'}")
        print("─" * 90)
        for a in adrs:
            print(f"{a['file']:<35} {a['status']:<12} {a['date']:<12} {a['title']}")
    else:
        print("No ADRs found in docs/adr/")
    return adrs

def cmd_validate() -> int:
    errors = []
    warnings = []
    files = list(ADR_DIR.glob("ADR-*.md"))
    if not files:
        print("No ADRs to validate.")
        return 0

    for f in sorted(files):
        text = f.read_text()
        # Required sections
        required = ["## Context", "## Decision", "## Consequences"]
        for section in required:
            if section not in text:
                errors.append(f"{f.name}: missing section '{section}'")
        # Valid status
        status_match = re.search(r'\*\*Status:\*\*\s*(\w+)', text)
        if not status_match:
            errors.append(f"{f.name}: missing **Status:**")
        elif status_match.group(1) not in VALID_STATUSES:
            errors.append(f"{f.name}: invalid status '{status_match.group(1)}' — must be one of {VALID_STATUSES}")
        # Placeholder detection
        if "[List decision makers]" in text or "[What is the problem?" in text:
            warnings.append(f"{f.name}: contains unfilled template placeholders")

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR:   {e}", file=sys.stderr)

    total = len(files)
    print(f"\nValidated {total} ADR(s): {len(errors)} error(s), {len(warnings)} warning(s)")
    return len(errors)

def cmd_accept(adr_file: str):
    filepath = ADR_DIR / adr_file
    if not filepath.exists():
        print(f"Not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    text = filepath.read_text()
    updated = re.sub(r'\*\*Status:\*\*\s*\w+', '**Status:** Accepted', text)
    filepath.write_text(updated)
    print(f"Updated status to Accepted: {filepath.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADR management CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_new = sub.add_parser("new", help="Create a new ADR")
    p_new.add_argument("title", help="ADR title")

    sub.add_parser("list", help="List all ADRs")
    sub.add_parser("validate", help="Validate ADR format (for CI)")

    p_accept = sub.add_parser("accept", help="Mark ADR as Accepted")
    p_accept.add_argument("file", help="ADR filename (e.g. ADR-007-vector-database.md)")

    args = parser.parse_args()

    if args.cmd == "new":
        cmd_new(args.title)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "validate":
        sys.exit(cmd_validate())
    elif args.cmd == "accept":
        cmd_accept(args.file)
    else:
        parser.print_help()
```

**Run the lab:**

```bash
# Create a new ADR
python scripts/adr.py new "Embedding Model Selection"

# List ADRs
python scripts/adr.py list

# Validate format (CI gate)
python scripts/adr.py validate

# Accept a decision
python scripts/adr.py accept ADR-001-embedding-model-selection.md
```

**Extensions:**

- Add a `supersede` command that marks an existing ADR as Superseded and links to the new one
- Add a `--check-links` flag to `validate` that verifies all `[text](url)` links in ADRs are reachable
- Integrate `validate` into the CI workflow so that invalid ADRs block merges

---

> ### 📋 Chapter Summary
>
> - **Documentation as code** applies version control, review, and quality discipline to architectural documentation.
> - **ADRs** record significant decisions with context, options considered, rationale, and consequences — answering "why" when the code only shows "what".
> - **Runbooks as code** are executable, testable operational procedures versioned alongside the systems they describe.
> - **OpenAPI specs** auto-generated from code remain always current and enable breaking-change detection in CI.
> - **Automated documentation pipelines** enforce ADR format, detect API breaking changes, and publish documentation on every merge.

---

> ### ❓ Comprehension Questions
>
> 1. Six months after deployment, the team debates changing the chunk size from 512 to 1024 tokens. With an ADR for the original decision, what information is immediately available that would otherwise be lost?
> 2. A runbook is written as a Markdown document with manual steps. Compare this with the Python runbook approach in section 4.3. What are the operational advantages of executable runbooks?
> 3. The API documentation pipeline detects that a field was removed from `QueryResponse`. How should the CI pipeline handle this, and what process should follow?
> 4. An ADR status is "Proposed" for six months with no decision made. What organisational anti-pattern does this indicate, and how would you address it?
> 5. Describe a strategy for keeping ADRs current as the system evolves. When should an ADR be superseded vs updated in place?

---

## References

### Documentation

- [ADR GitHub Organisation](https://adr.github.io) — Architecture Decision Records tooling and examples.
- [Michael Nygard's ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — Original ADR blog post.
- [FastAPI Documentation](https://fastapi.tiangolo.com) — Auto-generated OpenAPI docs.
- [Swagger/OpenAPI Specification](https://swagger.io/specification/) — OpenAPI 3.1 reference.
- [adr-tools](https://github.com/npryce/adr-tools) — Shell-based ADR management.

### Books

- _Documenting Architecture Decisions_ — Michael Keeling (O'Reilly). ADR patterns and practices.
- _The DevOps Handbook_ — Kim, Humble, Debois, Willis (IT Revolution). Documentation in DevOps culture.

---

[« Back to layouts-repositories Index](index.md) | [🏠 Home](../index.md)
