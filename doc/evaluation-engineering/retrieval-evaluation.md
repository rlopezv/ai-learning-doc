## Chapter 2 — Retrieval Evaluation

### 2.1 Retrieval Metrics Taxonomy

Retrieval evaluation measures whether the retrieval component surfaces the documents needed to answer a query correctly. It is independent of generation quality — a system can retrieve perfectly and generate poorly, or vice versa. Evaluating them separately enables precise attribution of quality issues.

The core principle: **for each query, we know which documents are relevant** (from our evaluation dataset). We measure whether the retrieval system returns those documents in the top-K results.

---

### 2.2 Recall@K, Precision@K, MRR, NDCG

```python
from typing import Optional
import math

def recall_at_k(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    k: int
) -> float:
    """
    Fraction of relevant documents found in top-K results.
    Primary metric for RAG: did we retrieve what we need?
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)

def precision_at_k(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    k: int
) -> float:
    """
    Fraction of top-K results that are relevant.
    Measures result list quality (fewer irrelevant results).
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for doc_id in top_k if doc_id in relevant) / k

def mean_reciprocal_rank(
    relevant_ids: list[str],
    retrieved_ids: list[str]
) -> float:
    """
    MRR: 1/rank of the first relevant result.
    Measures how quickly the first relevant document appears.
    MRR=1.0 means relevant doc is always first; 0.5 means second.
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    k: int
) -> float:
    """
    NDCG@K: Normalised Discounted Cumulative Gain.
    Measures ranking quality — relevant docs ranked higher get more credit.
    """
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # Actual DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant
    )

    # Ideal DCG (all relevant docs at top)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return dcg / idcg if idcg > 0 else 0.0

def hit_rate(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    k: int
) -> float:
    """
    Binary: 1 if ANY relevant document is in top-K, 0 otherwise.
    Simple and interpretable — did the retriever find at least one answer?
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if any(rid in top_k for rid in relevant_ids) else 0.0
```

**Metric selection guidance:**

| Metric | Use when | Avoid when |
|---|---|---|
| **Recall@K** | Each query has 1–3 relevant docs; need all of them | Queries have many relevant docs |
| **Hit Rate@K** | Single-hop queries with one ground-truth doc | Multi-doc synthesis required |
| **MRR** | First-rank position matters (UI shows top result prominently) | All K results weighted equally |
| **NDCG@K** | Ranking quality matters beyond binary relevance | Simple RAG where all retrieved docs are passed to LLM |
| **Precision@K** | Minimising noise in context is important | Recall is the primary concern |

For most RAG systems: **Recall@5** is the primary metric, **Hit Rate@5** is a useful secondary signal, **MRR** is relevant when results are displayed to users.

---

### 2.3 Building a Retrieval Evaluator

```python
import json
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

@dataclass
class RetrievalEvalResult:
    recall_at_k: dict[int, float]   # k → mean recall
    precision_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    total_queries: int
    failed_queries: int
    # Per-category breakdown
    by_category: dict[str, dict] = field(default_factory=dict)

class RetrievalEvaluator:
    def __init__(
        self,
        retriever: Callable,     # fn(query: str, top_k: int) -> list[dict with 'id' field]
        k_values: list[int] = None
    ):
        self.retriever = retriever
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        eval_records: list[EvalRecord],
        verbose: bool = False
    ) -> RetrievalEvalResult:
        results = {k: {"recall": [], "precision": [], "hit_rate": [], "ndcg": []}
                   for k in self.k_values}
        mrr_scores = []
        latencies_ms = []
        failed = 0
        by_category: dict[str, list] = {}

        for record in eval_records:
            try:
                t0 = time.perf_counter()
                retrieved = self.retriever(record.query, top_k=max(self.k_values))
                latency_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(latency_ms)

                retrieved_ids = [r["id"] for r in retrieved]

                for k in self.k_values:
                    results[k]["recall"].append(
                        recall_at_k(record.relevant_document_ids, retrieved_ids, k))
                    results[k]["precision"].append(
                        precision_at_k(record.relevant_document_ids, retrieved_ids, k))
                    results[k]["hit_rate"].append(
                        hit_rate(record.relevant_document_ids, retrieved_ids, k))
                    results[k]["ndcg"].append(
                        ndcg_at_k(record.relevant_document_ids, retrieved_ids, k))

                mrr_scores.append(mean_reciprocal_rank(record.relevant_document_ids, retrieved_ids))

                # Category tracking
                cat = record.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(
                    hit_rate(record.relevant_document_ids, retrieved_ids, 5))

                if verbose:
                    hr = hit_rate(record.relevant_document_ids, retrieved_ids, 5)
                    print(f"  {'✓' if hr else '✗'} [{record.category.value}] {record.query[:60]}")

            except Exception as e:
                failed += 1
                if verbose:
                    print(f"  ✗ FAILED: {record.id} — {e}")

        def mean(lst): return sum(lst) / len(lst) if lst else 0.0
        def pct(lst, p):
            if not lst: return 0.0
            idx = int(len(sorted(lst)) * p / 100)
            return sorted(lst)[min(idx, len(lst)-1)]

        return RetrievalEvalResult(
            recall_at_k={k: mean(results[k]["recall"]) for k in self.k_values},
            precision_at_k={k: mean(results[k]["precision"]) for k in self.k_values},
            mrr=mean(mrr_scores),
            ndcg_at_k={k: mean(results[k]["ndcg"]) for k in self.k_values},
            hit_rate_at_k={k: mean(results[k]["hit_rate"]) for k in self.k_values},
            latency_p50_ms=pct(latencies_ms, 50),
            latency_p95_ms=pct(latencies_ms, 95),
            latency_p99_ms=pct(latencies_ms, 99),
            total_queries=len(eval_records),
            failed_queries=failed,
            by_category={cat: {"hit_rate_at_5": mean(scores)}
                        for cat, scores in by_category.items()}
        )

    def print_report(self, result: RetrievalEvalResult):
        print("\n── Retrieval Evaluation Report ─────────────────────")
        print(f"  Total queries  : {result.total_queries} ({result.failed_queries} failed)")
        print(f"  MRR            : {result.mrr:.4f}")
        print()
        for k in sorted(result.recall_at_k.keys()):
            print(f"  Recall@{k:<3}     : {result.recall_at_k[k]:.4f}  "
                  f"Precision@{k}: {result.precision_at_k[k]:.4f}  "
                  f"NDCG@{k}: {result.ndcg_at_k[k]:.4f}  "
                  f"Hit@{k}: {result.hit_rate_at_k[k]:.4f}")
        print()
        print(f"  Latency P50    : {result.latency_p50_ms:.0f}ms")
        print(f"  Latency P95    : {result.latency_p95_ms:.0f}ms")
        print(f"  Latency P99    : {result.latency_p99_ms:.0f}ms")
        if result.by_category:
            print()
            print("  By category:")
            for cat, metrics in sorted(result.by_category.items()):
                print(f"    {cat:<20}: Hit@5={metrics['hit_rate_at_5']:.4f}")
        print()
```

---

### 2.4 Retrieval Baseline Tracking

Baselines record the reference quality score for a known-good system configuration. Every evaluation run is compared to the baseline to detect regressions.

```python
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class RetrievalBaseline:
    baseline_id: str
    created_at: str
    git_sha: str
    embedding_model: str
    chunk_size: int
    collection_name: str
    eval_dataset_version: str
    eval_dataset_checksum: str
    # Pinned scores
    recall_at_5: float
    mrr: float
    hit_rate_at_5: float
    latency_p99_ms: float
    total_queries: int

class BaselineRegistry:
    def __init__(self, path: str = "eval/baselines.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, baseline: RetrievalBaseline):
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(baseline)) + "\n")
        print(f"Baseline saved: {baseline.baseline_id} (Recall@5={baseline.recall_at_5:.4f})")

    def get_latest(self, embedding_model: str = None) -> RetrievalBaseline | None:
        baselines = self._load_all()
        if embedding_model:
            baselines = [b for b in baselines if b.embedding_model == embedding_model]
        if not baselines:
            return None
        return max(baselines, key=lambda b: b.created_at)

    def compare(self, result: RetrievalEvalResult, baseline: RetrievalBaseline) -> dict:
        """Compare current result against baseline. Returns delta report."""
        deltas = {
            "recall_at_5_delta":   result.recall_at_k.get(5, 0) - baseline.recall_at_5,
            "mrr_delta":           result.mrr - baseline.mrr,
            "hit_rate_at_5_delta": result.hit_rate_at_k.get(5, 0) - baseline.hit_rate_at_5,
            "latency_p99_delta_ms": result.latency_p99_ms - baseline.latency_p99_ms,
        }
        regressions = {k: v for k, v in deltas.items() if
                       ("latency" in k and v > 50) or ("latency" not in k and v < -0.02)}
        return {
            "deltas": {k: round(v, 4) for k, v in deltas.items()},
            "regressions": regressions,
            "has_regression": len(regressions) > 0,
        }

    def _load_all(self) -> list[RetrievalBaseline]:
        if not self.path.exists():
            return []
        results = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(RetrievalBaseline(**json.loads(line)))
                    except Exception:
                        pass
        return results
```

---

### 2.5 Java Retrieval Evaluation

```java
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import java.util.*;

public class RetrievalEvaluator {

    private final EmbeddingStore<TextSegment> store;
    private final EmbeddingModel embeddingModel;

    public RetrievalMetrics evaluate(List<EvalRecord> records, int k) {
        List<Double> recalls = new ArrayList<>();
        List<Double> hitRates = new ArrayList<>();
        List<Double> mrrs = new ArrayList<>();

        for (EvalRecord record : records) {
            var embedding = embeddingModel.embed(record.query()).content();
            var request = EmbeddingSearchRequest.builder()
                .queryEmbedding(embedding)
                .maxResults(k)
                .build();
            var results = store.search(request);

            List<String> retrievedIds = results.matches().stream()
                .map(m -> m.embeddingId())
                .toList();

            Set<String> relevant = new HashSet<>(record.relevantDocumentIds());

            // Recall@K
            long hits = retrievedIds.stream().filter(relevant::contains).count();
            recalls.add((double) hits / relevant.size());

            // Hit Rate@K
            hitRates.add(retrievedIds.stream().anyMatch(relevant::contains) ? 1.0 : 0.0);

            // MRR
            double mrr = 0.0;
            for (int i = 0; i < retrievedIds.size(); i++) {
                if (relevant.contains(retrievedIds.get(i))) {
                    mrr = 1.0 / (i + 1);
                    break;
                }
            }
            mrrs.add(mrr);
        }

        return new RetrievalMetrics(
            average(recalls),
            average(hitRates),
            average(mrrs),
            records.size()
        );
    }

    private double average(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    public record RetrievalMetrics(
        double recallAtK, double hitRateAtK, double mrr, int totalQueries
    ) {}

    public record EvalRecord(
        String id, String query, List<String> relevantDocumentIds, String category
    ) {}
}
```

---

> ### 📋 Chapter Summary
>
> - Retrieval evaluation measures the retrieval component **independently** from generation — enabling precise attribution of quality issues.
> - **Recall@K** is the primary RAG metric: did we retrieve all documents needed to answer the query?
> - The `RetrievalEvaluator` computes Recall@K, Precision@K, MRR, NDCG@K, Hit Rate, and latency percentiles in one pass.
> - **Baseline tracking** enables regression detection: every evaluation run is compared to a pinned reference score, and a configurable threshold triggers alerts.

---

> ### ❓ Comprehension Questions
>
> 1. Explain the difference between Recall@5 and Hit Rate@5. Give an example where they produce different values for the same query.
> 2. A RAG system's Recall@5 is 0.85 but MRR is 0.42. What does this combination tell you about the retrieval system's behaviour?
> 3. After upgrading the embedding model, Recall@5 improves from 0.81 to 0.88 but P99 latency increases from 120ms to 340ms. How would you evaluate whether this trade-off is acceptable?
> 4. Your evaluation dataset has 40% "unanswerable" queries. How should these be handled in Recall@K computation, and what specific metric would you use to evaluate unanswerable query handling?
> 5. The `RetrievalEvaluator` reports Recall@5 = 0.87 overall, but the by-category breakdown shows Recall@5 = 0.52 for `procedural` queries. What does this indicate and what would you investigate?

---

## References

### Papers
- [BEIR: Zero-shot Evaluation of Information Retrieval](https://arxiv.org/abs/2104.08663) — Thakur et al., 2021. Standard retrieval evaluation benchmark.
- [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — Karpukhin et al., 2020. DPR evaluation methodology.
- [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268) — Bajaj et al., 2016.

### Documentation
- [RAGAS Retrieval Metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/) — Context recall and precision.
- [LangChain4j Embedding Search](https://docs.langchain4j.dev/tutorials/rag#embedding-store) — Java retrieval implementation.
- [ir_measures](https://ir-measur.es/en/latest/) — Python library for standard IR metrics.

---

---
[« Back to evaluation-engineering Index](index.md) | [🏠 Home](../index.md)