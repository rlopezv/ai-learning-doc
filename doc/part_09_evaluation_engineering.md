# Part IX — Evaluation Engineering

---

> **Navigation**
> [← Part VIII — AI Systems SDLC](part_08_sdlc.md) | [→ Part X — Testing LLM Systems](part_10_testing.md)

---

## Contents

- [Chapter 1 — Evaluation Strategy](#chapter-1--evaluation-strategy)
  - [1.1 Why Evaluation Is an Engineering Discipline](#11-why-evaluation-is-an-engineering-discipline)
  - [1.2 The Evaluation Pyramid](#12-the-evaluation-pyramid)
  - [1.3 Offline vs Online Evaluation](#13-offline-vs-online-evaluation)
  - [1.4 Evaluation Dataset Strategy](#14-evaluation-dataset-strategy)
  - [1.5 The Evaluation Infrastructure Stack](#15-the-evaluation-infrastructure-stack)
- [Chapter 2 — Retrieval Evaluation](#chapter-2--retrieval-evaluation)
  - [2.1 Retrieval Metrics Taxonomy](#21-retrieval-metrics-taxonomy)
  - [2.2 Recall@K, Precision@K, MRR, NDCG](#22-recallk-precisionk-mrr-ndcg)
  - [2.3 Building a Retrieval Evaluator](#23-building-a-retrieval-evaluator)
  - [2.4 Retrieval Baseline Tracking](#24-retrieval-baseline-tracking)
  - [2.5 Java Retrieval Evaluation](#25-java-retrieval-evaluation)
- [Chapter 3 — Generation Evaluation](#chapter-3--generation-evaluation)
  - [3.1 The Generation Quality Problem](#31-the-generation-quality-problem)
  - [3.2 RAGAS: Automated RAG Evaluation](#32-ragas-automated-rag-evaluation)
  - [3.3 LLM-as-Judge](#33-llm-as-judge)
  - [3.4 Faithfulness Checking](#34-faithfulness-checking)
  - [3.5 Evaluating Refusal Behaviour](#35-evaluating-refusal-behaviour)
- [Chapter 4 — Human Evaluation](#chapter-4--human-evaluation)
  - [4.1 When Human Evaluation Is Required](#41-when-human-evaluation-is-required)
  - [4.2 Annotation Guidelines](#42-annotation-guidelines)
  - [4.3 Inter-Annotator Agreement](#43-inter-annotator-agreement)
  - [4.4 Human Feedback Collection in Production](#44-human-feedback-collection-in-production)
- [Chapter 5 — Evaluation Pipelines 🧪](#chapter-5--evaluation-pipelines-)
  - [5.1 The Evaluation Pipeline Architecture](#51-the-evaluation-pipeline-architecture)
  - [5.2 Continuous Evaluation with Baselines](#52-continuous-evaluation-with-baselines)
  - [5.3 Regression Detection](#53-regression-detection)
  - [5.4 Evaluation Reporting](#54-evaluation-reporting)
  - [🧪 Hands-on Lab: Full Evaluation Pipeline](#-hands-on-lab-full-evaluation-pipeline)

---

## Chapter 1 — Evaluation Strategy

### 1.1 Why Evaluation Is an Engineering Discipline

Evaluation in LLM systems is not a QA afterthought — it is a core engineering activity that must be designed, built, and maintained with the same rigour as the system it measures. Treating evaluation as secondary leads to a predictable set of failures:

- Quality regressions are discovered by users, not by the team
- Prompt experiments cannot be compared because evaluation conditions differ between runs
- A model upgrade shows "improvement" because the evaluation dataset was changed between measurements
- The team has no early warning when knowledge base staleness degrades answer quality

Well-engineered evaluation provides three capabilities: **detection** (we know when quality degrades), **attribution** (we know which component caused the degradation), and **comparison** (we can reliably compare two versions of the system).

These three capabilities require evaluation to be: reproducible (same inputs, same evaluation method), independent (measuring retrieval separately from generation), versioned (evaluation datasets are immutable after sealing), and continuous (running on every change, not just at release time).

---

### 1.2 The Evaluation Pyramid

Analogous to the software testing pyramid, the evaluation pyramid defines a hierarchy of evaluation types ordered by cost, coverage, and feedback speed:

```
        ┌──────────────────────────┐
        │    Human evaluation      │  Slow, expensive, highest signal
        │  (10–100 examples/cycle) │
        ├──────────────────────────┤
        │  LLM-as-Judge evaluation │  Medium speed, medium cost, good signal
        │  (100–1000 examples)     │
        ├──────────────────────────┤
        │   RAGAS / automated      │  Fast, cheap, reasonable signal
        │   metrics evaluation     │
        │   (100–10K examples)     │
        ├──────────────────────────┤
        │   Retrieval metrics      │  Very fast, cheap, precise signal
        │   Recall@K, MRR, NDCG   │  for retrieval component only
        │   (100–10K examples)     │
        ├──────────────────────────┤
        │   Prompt regression      │  Instantaneous, deterministic
        │   unit tests             │
        │   (10–100 examples)      │
        └──────────────────────────┘
           Broad ◄──────────► Narrow
```

Run the bottom layers on every PR. Run RAGAS on every merge to main. Run LLM-as-Judge weekly or before major releases. Run human evaluation quarterly or when making high-stakes decisions (new model, major prompt rewrite).

---

### 1.3 Offline vs Online Evaluation

| Dimension | Offline evaluation | Online evaluation |
|---|---|---|
| **When** | Before deployment | After deployment |
| **Data** | Curated evaluation dataset | Real user queries |
| **Signal** | Controlled, reproducible | Authentic, noisy |
| **Latency** | Minutes to hours | Real-time |
| **Coverage** | Defined by dataset | Full distribution |
| **Cost** | Fixed per run | Per user query |
| **Limitation** | Dataset may not reflect real distribution | Requires production traffic |

Both are necessary. Offline evaluation gates deployments; online evaluation monitors quality post-deployment and feeds back into offline evaluation dataset updates.

```python
from dataclasses import dataclass
from enum import Enum

class EvaluationMode(str, Enum):
    OFFLINE = "offline"     # Against curated dataset
    ONLINE = "online"       # Against production traffic sample
    SHADOW = "shadow"       # Offline eval on sampled production queries

@dataclass
class EvaluationConfig:
    mode: EvaluationMode
    dataset_path: str = None          # For offline
    sample_rate: float = 0.01         # For online: fraction of traffic to evaluate
    min_sample_size: int = 100
    evaluation_model: str = "gpt-4o"  # LLM judge model
    # Thresholds
    min_recall_at_5: float = 0.80
    min_faithfulness: float = 0.85
    min_answer_relevancy: float = 0.80
    max_latency_p99_ms: float = 500.0
    # Storage
    results_store: str = "mlflow"     # mlflow | wandb | local
```

---

### 1.4 Evaluation Dataset Strategy

A well-designed evaluation dataset covers the full distribution of query types, difficulty levels, and edge cases the system will encounter in production.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class QueryCategory(str, Enum):
    FACTUAL = "factual"               # Direct fact lookup
    INFERENTIAL = "inferential"       # Requires reasoning
    COMPARATIVE = "comparative"       # Compares options
    PROCEDURAL = "procedural"         # How-to queries
    UNANSWERABLE = "unanswerable"     # Not in knowledge base
    ADVERSARIAL = "adversarial"       # Edge cases, tricky phrasing
    MULTI_HOP = "multi_hop"           # Requires combining multiple docs

@dataclass
class EvalRecord:
    id: str
    query: str
    ground_truth_answer: str
    relevant_document_ids: list[str]
    category: QueryCategory
    difficulty: str                     # easy | medium | hard
    persona: Optional[str] = None      # technical | business | new_user
    requires_multi_hop: bool = False
    is_adversarial: bool = False
    source: str = "synthetic"          # synthetic | human | production_log

@dataclass
class EvalDatasetSpec:
    """Target composition for a balanced evaluation dataset."""
    total_size: int = 300
    # Category distribution
    factual_pct: float = 0.30
    inferential_pct: float = 0.20
    comparative_pct: float = 0.10
    procedural_pct: float = 0.10
    unanswerable_pct: float = 0.15
    adversarial_pct: float = 0.10
    multi_hop_pct: float = 0.05
    # Difficulty distribution
    easy_pct: float = 0.30
    medium_pct: float = 0.50
    hard_pct: float = 0.20

    def validate_percentages(self) -> bool:
        cat_total = (self.factual_pct + self.inferential_pct + self.comparative_pct +
                     self.procedural_pct + self.unanswerable_pct +
                     self.adversarial_pct + self.multi_hop_pct)
        diff_total = self.easy_pct + self.medium_pct + self.hard_pct
        return abs(cat_total - 1.0) < 0.01 and abs(diff_total - 1.0) < 0.01

def audit_dataset_coverage(records: list[EvalRecord]) -> dict:
    """Check if a dataset meets the target composition."""
    n = len(records)
    if n == 0:
        return {}
    by_category = {}
    by_difficulty = {}
    by_source = {}
    for r in records:
        by_category[r.category.value] = by_category.get(r.category.value, 0) + 1
        by_difficulty[r.difficulty] = by_difficulty.get(r.difficulty, 0) + 1
        by_source[r.source] = by_source.get(r.source, 0) + 1
    return {
        "total": n,
        "by_category": {k: f"{v/n:.1%}" for k, v in by_category.items()},
        "by_difficulty": {k: f"{v/n:.1%}" for k, v in by_difficulty.items()},
        "by_source": {k: f"{v/n:.1%}" for k, v in by_source.items()},
        "unanswerable_count": by_category.get("unanswerable", 0),
        "adversarial_count": by_category.get("adversarial", 0),
    }
```

---

### 1.5 The Evaluation Infrastructure Stack

```
Evaluation infrastructure components:

┌─────────────────────────────────────────────────────┐
│  Evaluation datasets (versioned, sealed, DVC)       │
├─────────────────────────────────────────────────────┤
│  Evaluation runner (run_eval.py, CI-callable)       │
├─────────────────────────────────────────────────────┤
│  Metrics computation                                 │
│  ├── Retrieval: Recall@K, MRR, NDCG                │
│  ├── Generation: RAGAS, LLM-judge, faithfulness     │
│  └── System: latency percentiles, cost per query    │
├─────────────────────────────────────────────────────┤
│  Results store (MLflow, W&B, or local JSONL)        │
├─────────────────────────────────────────────────────┤
│  Baseline registry (pinned reference scores)        │
├─────────────────────────────────────────────────────┤
│  Regression detector (alerts when score drops)      │
└─────────────────────────────────────────────────────┘
```

---

> ### 📋 Chapter Summary
>
> - Evaluation engineering provides three capabilities: **detection**, **attribution**, and **comparison** — none of which are available without deliberate investment.
> - The **evaluation pyramid** orders evaluation types by cost and speed: prompt unit tests (fast) → retrieval metrics → RAGAS → LLM-as-Judge → human evaluation (slow).
> - **Offline evaluation** gates deployments against curated datasets; **online evaluation** monitors quality over real production traffic.
> - A well-designed evaluation dataset explicitly balances categories, difficulty, and adversarial examples — a balanced dataset is an engineered artifact.

---

> ### ❓ Comprehension Questions
>
> 1. A team's evaluation dataset contains only factual queries because those were easiest to generate synthetically. The system is deployed and fails on unanswerable queries. What evaluation dataset design failure led to this?
> 2. Explain the difference between offline and online evaluation. Why is offline evaluation alone insufficient to detect all quality issues?
> 3. An evaluation run takes 45 minutes. A developer argues it should only run on release branches, not every PR. What is the risk of this decision?
> 4. The evaluation pyramid suggests running human evaluation quarterly. A product manager wants monthly human evaluation. What are the cost implications and what alternative might provide equivalent signal at lower cost?
> 5. Design an evaluation dataset composition for a RAG system used by medical professionals. How would the category distribution differ from a general-purpose enterprise assistant?

---

## References

### Documentation
- [RAGAS Documentation](https://docs.ragas.io) — Automated RAG evaluation framework.
- [MLflow Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html) — LLM evaluation tracking.
- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation) — Evaluation pipelines with LangChain.
- [Weights & Biases Prompts](https://docs.wandb.ai/guides/prompts) — LLM evaluation with W&B.

### Papers
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Es et al., 2023.
- [Evaluating LLMs: A Survey](https://arxiv.org/abs/2307.03109) — Chang et al., 2023. Comprehensive survey of LLM evaluation methods.

---

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

## Chapter 3 — Generation Evaluation

### 3.1 The Generation Quality Problem

Generation evaluation is harder than retrieval evaluation because there is no single correct output for most queries. "What is the refund policy?" can be answered correctly in dozens of ways. Evaluating whether a specific answer is correct, grounded, and appropriate requires judgement — not exact matching.

Three properties define a high-quality generated answer in a RAG system:

**Faithfulness:** Every factual claim in the answer is supported by the retrieved context. A faithful answer does not add information from model weights or hallucinate details not in the source documents.

**Answer relevancy:** The answer directly addresses the question asked. A relevant answer does not drift into tangential information or fail to address the core query.

**Groundedness:** The answer cites or references the source documents it draws from, enabling verification. This is a requirement distinct from faithfulness — a faithful answer may still be ungrounded if it provides no attribution.

---

### 3.2 RAGAS: Automated RAG Evaluation

[RAGAS](https://docs.ragas.io) provides automated metrics for all three generation quality dimensions.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from openai import OpenAI

def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],    # Retrieved contexts per query
    ground_truths: list[str],
    evaluation_model: str = "gpt-4o"
) -> dict:
    """
    Run RAGAS evaluation suite.

    context_recall:    Were the retrieved contexts sufficient to answer?
    context_precision: Were the retrieved contexts relevant (low noise)?
    faithfulness:      Is the answer grounded in the contexts?
    answer_relevancy:  Does the answer address the question?
    """
    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluation_model,
        raise_exceptions=False
    )

    return {
        "faithfulness":       float(result["faithfulness"]),
        "answer_relevancy":   float(result["answer_relevancy"]),
        "context_precision":  float(result["context_precision"]),
        "context_recall":     float(result["context_recall"]),
        "ragas_score":        float(result["ragas_score"]) if "ragas_score" in result else None,
    }

class RAGPipelineEvaluator:
    """
    Evaluates a complete RAG pipeline end-to-end.
    Collects inputs for RAGAS automatically.
    """
    def __init__(self, retriever, generator, k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def run(self, eval_records: list[EvalRecord]) -> dict:
        questions, answers, contexts, ground_truths = [], [], [], []

        for record in eval_records:
            retrieved = self.retriever(record.query, top_k=self.k)
            context_texts = [r["content"] for r in retrieved]
            answer = self.generator(record.query, context_texts)

            questions.append(record.query)
            answers.append(answer)
            contexts.append(context_texts)
            ground_truths.append(record.ground_truth_answer)

        return run_ragas_evaluation(questions, answers, contexts, ground_truths)
```

---

### 3.3 LLM-as-Judge

LLM-as-Judge uses a capable LLM to evaluate answer quality on dimensions that automated metrics cannot reliably measure: tone appropriateness, completeness, safety compliance, and domain-specific accuracy.

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json

client = OpenAI()

class JudgeScore(BaseModel):
    overall_score: float            # 0.0 – 1.0
    faithfulness_score: float
    relevancy_score: float
    completeness_score: float
    tone_score: float
    reasoning: str
    issues: list[str]

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI-generated answers
in a RAG (Retrieval-Augmented Generation) system.

Evaluate the answer on four dimensions, each scored 0.0 to 1.0:

1. FAITHFULNESS (0–1): Every claim in the answer must be directly supported by the context.
   - 1.0 = All claims supported by context
   - 0.5 = Some claims lack context support
   - 0.0 = Answer contradicts context or invents information

2. RELEVANCY (0–1): Does the answer address what was asked?
   - 1.0 = Directly and completely answers the question
   - 0.5 = Partially answers or includes tangential information
   - 0.0 = Does not address the question

3. COMPLETENESS (0–1): Is all available information from context included?
   - 1.0 = All relevant context information used
   - 0.5 = Some relevant information omitted
   - 0.0 = Significant relevant information missing

4. TONE (0–1): Is the tone appropriate for the context?
   - 1.0 = Professional, clear, and appropriately formal
   - 0.5 = Acceptable but could be improved
   - 0.0 = Inappropriate, condescending, or confusing

Return ONLY valid JSON matching the schema. No preamble."""

def judge_answer(
    query: str,
    context: list[str],
    answer: str,
    ground_truth: str,
    model: str = "gpt-4o"
) -> JudgeScore:
    context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context)])
    user_content = f"""QUERY: {query}

CONTEXT:
{context_text}

GROUND TRUTH: {ground_truth}

ANSWER TO EVALUATE:
{answer}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    data = json.loads(response.choices[0].message.content)
    overall = (data.get("faithfulness_score", 0) * 0.35 +
               data.get("relevancy_score", 0) * 0.35 +
               data.get("completeness_score", 0) * 0.20 +
               data.get("tone_score", 0) * 0.10)
    data["overall_score"] = round(overall, 4)
    return JudgeScore(**data)

def batch_judge(
    eval_records: list[EvalRecord],
    system_outputs: list[dict],   # {"answer": str, "contexts": list[str]}
    judge_model: str = "gpt-4o",
    max_workers: int = 5
) -> dict:
    """Evaluate a batch of answers and aggregate scores."""
    from concurrent.futures import ThreadPoolExecutor
    scores = []

    def evaluate_one(pair):
        record, output = pair
        try:
            score = judge_answer(
                query=record.query,
                context=output["contexts"],
                answer=output["answer"],
                ground_truth=record.ground_truth_answer,
                model=judge_model
            )
            return score
        except Exception as e:
            print(f"Judge failed for {record.id}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(evaluate_one, zip(eval_records, system_outputs)))
        scores = [r for r in results if r is not None]

    if not scores:
        return {}
    return {
        "mean_overall":      round(sum(s.overall_score for s in scores) / len(scores), 4),
        "mean_faithfulness": round(sum(s.faithfulness_score for s in scores) / len(scores), 4),
        "mean_relevancy":    round(sum(s.relevancy_score for s in scores) / len(scores), 4),
        "mean_completeness": round(sum(s.completeness_score for s in scores) / len(scores), 4),
        "scored_count":      len(scores),
        "common_issues":     _count_issues([i for s in scores for i in s.issues])
    }

def _count_issues(issues: list[str]) -> list[dict]:
    from collections import Counter
    counts = Counter(issues)
    return [{"issue": k, "count": v} for k, v in counts.most_common(5)]
```

---

### 3.4 Faithfulness Checking

Faithfulness checking verifies that every claim in a generated answer is supported by the retrieved context.

```python
def check_faithfulness(
    answer: str,
    context: list[str],
    model: str = "gpt-4o"
) -> dict:
    """
    Decompose the answer into atomic claims and verify each against context.
    Returns faithfulness score and per-claim verdicts.
    """
    context_text = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(context)])

    # Step 1: Extract atomic claims from the answer
    claims_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Extract all factual claims from the answer as a JSON list. "
             "Each claim must be a single, verifiable statement. "
             'Return JSON: {"claims": ["claim1", "claim2"]}'},
            {"role": "user", "content": f"Answer: {answer}"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    claims = json.loads(claims_response.choices[0].message.content).get("claims", [])

    # Step 2: Verify each claim against context
    verdicts = []
    for claim in claims:
        verdict_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": 'Determine if the claim is supported by the context. '
                 'Return JSON: {"supported": true/false, "evidence": "quote from context or null"}'},
                {"role": "user", "content": f"Context:\n{context_text}\n\nClaim: {claim}"}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        verdict = json.loads(verdict_response.choices[0].message.content)
        verdicts.append({
            "claim": claim,
            "supported": verdict.get("supported", False),
            "evidence": verdict.get("evidence")
        })

    supported = sum(1 for v in verdicts if v["supported"])
    faithfulness_score = supported / len(verdicts) if verdicts else 1.0

    return {
        "faithfulness_score": round(faithfulness_score, 4),
        "supported_claims": supported,
        "total_claims": len(verdicts),
        "verdicts": verdicts,
        "unsupported_claims": [v["claim"] for v in verdicts if not v["supported"]]
    }
```

---

### 3.5 Evaluating Refusal Behaviour

A RAG system must refuse to answer queries that cannot be answered from the knowledge base. Evaluating refusal behaviour is as important as evaluating answer quality.

```python
from enum import Enum

class RefusalVerdict(str, Enum):
    CORRECT_REFUSAL = "correct_refusal"       # Correctly refused an unanswerable query
    INCORRECT_REFUSAL = "incorrect_refusal"   # Refused an answerable query
    CORRECT_ANSWER = "correct_answer"         # Correctly answered an answerable query
    HALLUCINATED = "hallucinated"             # Answered with info not in context

REFUSAL_PHRASES = [
    "cannot find", "not in the", "don't have information",
    "unable to find", "not mentioned", "no information available"
]

def classify_response(
    query: str,
    answer: str,
    context: list[str],
    is_answerable: bool
) -> RefusalVerdict:
    """
    Classify a response as correct refusal, incorrect refusal, correct answer, or hallucination.
    """
    is_refusal = any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)

    if is_answerable:
        if is_refusal:
            return RefusalVerdict.INCORRECT_REFUSAL
        # Check faithfulness to determine if answer or hallucination
        faith = check_faithfulness(answer, context)
        return (RefusalVerdict.CORRECT_ANSWER
                if faith["faithfulness_score"] >= 0.8
                else RefusalVerdict.HALLUCINATED)
    else:
        return (RefusalVerdict.CORRECT_REFUSAL
                if is_refusal
                else RefusalVerdict.HALLUCINATED)

def evaluate_refusal_behaviour(
    eval_records: list[EvalRecord],
    system_outputs: list[dict]
) -> dict:
    """Compute refusal evaluation metrics."""
    verdicts = []
    for record, output in zip(eval_records, system_outputs):
        is_answerable = record.category != QueryCategory.UNANSWERABLE
        verdict = classify_response(
            record.query, output["answer"],
            output.get("contexts", []), is_answerable
        )
        verdicts.append(verdict)

    total = len(verdicts)
    counts = {v.value: verdicts.count(v) for v in RefusalVerdict}

    unanswerable = sum(1 for r in eval_records if r.category == QueryCategory.UNANSWERABLE)
    answerable = total - unanswerable

    return {
        "verdict_counts": counts,
        "refusal_precision": (counts.get("correct_refusal", 0) /
                             (counts.get("correct_refusal", 0) + counts.get("incorrect_refusal", 0) + 1e-9)),
        "refusal_recall": (counts.get("correct_refusal", 0) / unanswerable if unanswerable else 1.0),
        "hallucination_rate": counts.get("hallucinated", 0) / total if total else 0.0,
        "correct_answer_rate": counts.get("correct_answer", 0) / answerable if answerable else 0.0,
    }
```

---

> ### 📋 Chapter Summary
>
> - Generation quality has three core dimensions: **faithfulness** (grounded in context), **answer relevancy** (addresses the question), and **groundedness** (cites sources).
> - [RAGAS](https://docs.ragas.io) provides automated metrics for all dimensions — context recall, context precision, faithfulness, and answer relevancy.
> - **LLM-as-Judge** evaluates quality dimensions that automated metrics miss: tone, completeness, and domain-specific accuracy.
> - **Faithfulness checking** decomposes answers into atomic claims and verifies each against the retrieved context — the most reliable automated hallucination detection method.
> - **Refusal evaluation** is a first-class concern: incorrect refusals and hallucinations on unanswerable queries are distinct failure modes requiring separate metrics.

---

> ### ❓ Comprehension Questions
>
> 1. RAGAS faithfulness score is 0.92. A domain expert reviews 20 answers and finds 3 that contain hallucinations. How is this possible and what does it reveal about automated faithfulness metrics?
> 2. LLM-as-Judge uses `gpt-4o` to evaluate answers generated by `gpt-4o`. What self-evaluation bias does this introduce and how would you mitigate it?
> 3. Context precision measures the fraction of retrieved contexts that are relevant. It is 0.45 (many irrelevant contexts retrieved). How does low context precision affect faithfulness and answer quality?
> 4. A system has refusal_recall = 0.95 but refusal_precision = 0.40. What does this mean in practice, and which failure mode is more harmful for a customer-facing product?
> 5. Design a faithfulness checking pipeline for a medical RAG system where hallucinated medical information is a patient safety risk. How would you increase reliability beyond the standard claim-verification approach?

---

## References

### Papers
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Es et al., 2023.
- [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023.
- [FActScoring: Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2305.14251) — Min et al., 2023.
- [Hallucination in LLMs: A Survey](https://arxiv.org/abs/2311.05232) — Ji et al., 2023.

### Documentation
- [RAGAS Documentation](https://docs.ragas.io)
- [DeepEval](https://docs.confident-ai.com) — LLM evaluation framework with faithfulness metrics.
- [TruLens](https://www.trulens.org/docs/) — RAG triad evaluation (context relevance, groundedness, answer relevance).

---

## Chapter 4 — Human Evaluation

### 4.1 When Human Evaluation Is Required

Automated metrics are proxies. They measure properties that correlate with quality — not quality itself. Human evaluation is required when:

- **Making high-stakes decisions:** Selecting a new base model, rewriting the system prompt, changing the embedding model. Automated metrics may not capture the full quality impact.
- **Calibrating automated metrics:** Before trusting RAGAS or LLM-judge scores, verify they correlate with human judgement on a sample of 50–100 examples.
- **Evaluating subjective dimensions:** Tone appropriateness, cultural sensitivity, explanation clarity — dimensions that resist algorithmic measurement.
- **Investigating automated metric disagreements:** When RAGAS faithfulness is high but user satisfaction is low, human review uncovers what the automated metric missed.
- **Compliance and audit requirements:** Some regulated industries require human sign-off on model quality.

---

### 4.2 Annotation Guidelines

Clear annotation guidelines are essential for consistent human evaluation. Ambiguous guidelines produce noisy labels and low inter-annotator agreement.

```python
ANNOTATION_GUIDELINES = """
# RAG Answer Quality Annotation Guidelines

## Task
You will evaluate AI-generated answers against the question asked and the context provided.
Score each dimension from 1 to 5.

## Dimensions

### 1. FAITHFULNESS (1–5)
Does every factual claim in the answer appear in the provided context?
5 — All claims are directly supported by context. No invented information.
4 — Most claims supported; minor paraphrase that doesn't change meaning.
3 — Majority supported but 1–2 claims lack clear context support.
2 — Several claims not in context or ambiguous support.
1 — Answer contradicts context or introduces significant information not present.

### 2. ANSWER RELEVANCY (1–5)
Does the answer address what was asked?
5 — Directly and completely answers the question.
4 — Answers the main question, minor tangential information.
3 — Partially addresses the question; misses some aspects.
2 — Tangentially related but does not answer the core question.
1 — Does not address the question.

### 3. COMPLETENESS (1–5)
Does the answer include all relevant information from the context?
5 — All relevant context information included.
4 — Most relevant information included; minor omissions.
3 — Some relevant information omitted.
2 — Significant relevant information missing.
1 — Answer is a fragment; most relevant information omitted.

### 4. REFUSAL APPROPRIATENESS (for unanswerable queries only)
If the answer is a refusal ("I cannot find this information"):
5 — Correctly refused because information genuinely not in context.
1 — Incorrectly refused when context contained the answer.
N/A — Query was answerable (not a refusal).

## Calibration Examples

Example A:
Context: "Enterprise customers have a 90-day return window."
Query: "How long can enterprise customers return products?"
Answer: "Enterprise customers can return products within 90 days."
→ Faithfulness: 5, Relevancy: 5, Completeness: 5

Example B:
Context: "The refund takes 5–7 business days."
Query: "When will my refund arrive?"
Answer: "Your refund will arrive in 5–7 business days, and you'll receive an email confirmation."
→ Faithfulness: 3 (email confirmation not in context), Relevancy: 5, Completeness: 5
"""
```

---

### 4.3 Inter-Annotator Agreement

When multiple annotators label the same examples, their agreement is measured to validate annotation consistency.

```python
from collections import defaultdict
import statistics

def cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """
    Cohen's Kappa — measures inter-annotator agreement beyond chance.
    κ > 0.8: near-perfect; 0.6–0.8: substantial; 0.4–0.6: moderate; < 0.4: poor.
    """
    if len(rater_a) != len(rater_b) or not rater_a:
        return 0.0

    labels = sorted(set(rater_a + rater_b))
    n = len(rater_a)

    # Observed agreement
    p_o = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    # Expected agreement
    p_e = sum(
        (rater_a.count(l) / n) * (rater_b.count(l) / n)
        for l in labels
    )

    return (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0

def compute_agreement_report(
    annotations: dict[str, list[int]]  # annotator_id → list of scores
) -> dict:
    """Compute pairwise agreement between all annotator pairs."""
    annotator_ids = list(annotations.keys())
    pairwise_kappas = []
    pair_reports = []

    for i in range(len(annotator_ids)):
        for j in range(i + 1, len(annotator_ids)):
            a_id, b_id = annotator_ids[i], annotator_ids[j]
            kappa = cohens_kappa(annotations[a_id], annotations[b_id])
            pairwise_kappas.append(kappa)
            pair_reports.append({
                "pair": f"{a_id} vs {b_id}",
                "kappa": round(kappa, 4),
                "interpretation": _interpret_kappa(kappa)
            })

    mean_kappa = statistics.mean(pairwise_kappas) if pairwise_kappas else 0.0
    return {
        "mean_kappa": round(mean_kappa, 4),
        "interpretation": _interpret_kappa(mean_kappa),
        "pairs": pair_reports,
        "is_acceptable": mean_kappa >= 0.6
    }

def _interpret_kappa(kappa: float) -> str:
    if kappa >= 0.80: return "near-perfect"
    if kappa >= 0.60: return "substantial"
    if kappa >= 0.40: return "moderate"
    if kappa >= 0.20: return "fair"
    return "poor"

def find_disagreements(
    annotations: dict[str, list[int]],
    example_ids: list[str],
    disagreement_threshold: int = 2
) -> list[dict]:
    """Find examples where annotators disagree significantly."""
    disagreements = []
    n_examples = len(example_ids)
    for i in range(n_examples):
        scores = [annotations[ann][i] for ann in annotations]
        spread = max(scores) - min(scores)
        if spread >= disagreement_threshold:
            disagreements.append({
                "example_id": example_ids[i],
                "scores": {ann: annotations[ann][i] for ann in annotations},
                "spread": spread
            })
    return disagreements
```

---

### 4.4 Human Feedback Collection in Production

Production feedback — thumbs up/down, explicit ratings — provides high-signal labels for continuous evaluation.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class UserFeedback:
    feedback_id: str
    query_id: str           # Links to the original query
    session_id: str
    feedback_type: str      # thumbs_up | thumbs_down | rating | comment
    value: Optional[float] = None    # 1.0 for up, 0.0 for down, or 1–5 rating
    comment: Optional[str] = None
    query_text: str = ""    # Stored for analysis (PII-scrubbed)
    answer_text: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class FeedbackStore:
    def __init__(self, storage_path: str = "feedback/feedback.jsonl"):
        from pathlib import Path
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, feedback: UserFeedback):
        with open(self.path, "a") as f:
            f.write(json.dumps(feedback.__dict__) + "\n")

    def get_satisfaction_rate(self, window_hours: int = 24) -> float:
        """Compute thumbs-up rate over recent window."""
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()
        records = self._load_window(cutoff)
        binary = [r for r in records if r["feedback_type"] in ("thumbs_up", "thumbs_down")]
        if not binary:
            return 0.0
        positive = sum(1 for r in binary if r["value"] == 1.0)
        return positive / len(binary)

    def get_low_rated_examples(
        self,
        max_rating: float = 2.0,
        limit: int = 100
    ) -> list[dict]:
        """Get recent low-rated examples for analysis and dataset augmentation."""
        all_records = self._load_all()
        low_rated = [r for r in all_records
                     if r.get("feedback_type") == "rating"
                     and r.get("value", 5) <= max_rating]
        return sorted(low_rated, key=lambda r: r["timestamp"], reverse=True)[:limit]

    def _load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _load_window(self, cutoff_iso: str) -> list[dict]:
        return [r for r in self._load_all() if r.get("timestamp", "") >= cutoff_iso]
```

---

> ### 📋 Chapter Summary
>
> - Human evaluation is required for high-stakes decisions, automated metric calibration, subjective quality dimensions, and compliance requirements.
> - **Annotation guidelines** with calibration examples are essential — ambiguous guidelines produce unusable labels.
> - **Cohen's Kappa** measures inter-annotator agreement; κ ≥ 0.6 is required before labels are trusted for evaluation.
> - **Production feedback** (thumbs up/down, ratings) provides continuous high-signal quality monitoring at low cost.

---

> ### ❓ Comprehension Questions
>
> 1. RAGAS faithfulness = 0.91 but human annotators rate faithfulness at 3.2/5 (moderate). What does this discrepancy indicate about the automated metric, and how would you investigate?
> 2. Two annotators score the same 50 examples. Cohen's Kappa = 0.35 (fair agreement). What are the likely causes, and what process would you use to improve agreement before collecting more labels?
> 3. Production thumbs-down rate is 12% on average but spikes to 28% on questions about "pricing". What actions would you take based on this signal?
> 4. A team collects user comments as feedback. Comments are unstructured ("this is wrong", "helpful!", "why did it say that?"). Design a pipeline to extract structured quality signals from unstructured feedback.
> 5. Explain why low-rated production examples are valuable for evaluation dataset expansion. What quality checks would you apply before adding them to the sealed evaluation dataset?

---

## References

### Papers
- [Chatbot Arena: An Open Platform for Evaluating LLMs](https://arxiv.org/abs/2403.04132) — Chiang et al., 2024. Human preference evaluation at scale.
- [RLHF: Learning to Summarise from Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., 2020.
- [Inter-Rater Reliability in NLP Annotation](https://arxiv.org/abs/2010.11981) — Berzak et al., 2020.

### Documentation
- [Label Studio](https://labelstud.io/guide/) — Open-source annotation platform.
- [Scale AI](https://scale.com/docs) — Managed annotation platform.
- [Argilla](https://docs.argilla.io) — Open-source data annotation for NLP.

---

## Chapter 5 — Evaluation Pipelines 🧪

### 5.1 The Evaluation Pipeline Architecture

An evaluation pipeline orchestrates all evaluation components — dataset loading, RAG execution, metric computation, baseline comparison, and reporting — into a reproducible, CI-callable workflow.

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib

@dataclass
class EvaluationPipelineConfig:
    # Dataset
    eval_dataset_path: str
    dataset_sample_size: int = None    # None = full dataset
    # Retrieval
    retrieval_k: int = 5
    # Generation
    run_ragas: bool = True
    run_llm_judge: bool = False        # Expensive — off by default in CI
    judge_model: str = "gpt-4o"
    judge_sample_size: int = 50        # LLM judge on subset
    # Thresholds
    min_recall_at_5: float = 0.80
    min_faithfulness: float = 0.85
    min_answer_relevancy: float = 0.80
    max_latency_p99_ms: float = 500.0
    # Outputs
    results_dir: str = "eval/results"
    compare_to_baseline: bool = True

@dataclass
class PipelineRun:
    run_id: str
    started_at: str
    git_sha: str
    config: EvaluationPipelineConfig
    # Results
    retrieval: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)
    # Gates
    gates_passed: bool = False
    gate_failures: list[str] = field(default_factory=list)
    # Baseline comparison
    baseline_delta: dict = field(default_factory=dict)
    regression_detected: bool = False

class EvaluationPipeline:
    def __init__(
        self,
        config: EvaluationPipelineConfig,
        retriever,
        generator,
        baseline_registry: BaselineRegistry
    ):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.baselines = baseline_registry

    def run(self, git_sha: str = "unknown") -> PipelineRun:
        import time
        run_id = hashlib.md5(f"{git_sha}{time.time()}".encode()).hexdigest()[:10]
        run = PipelineRun(
            run_id=run_id,
            started_at=datetime.utcnow().isoformat(),
            git_sha=git_sha,
            config=self.config
        )

        print(f"\n{'='*55}")
        print(f"Evaluation Pipeline — Run {run_id}")
        print(f"{'='*55}\n")

        # Load dataset
        records = self._load_dataset()
        print(f"Dataset: {len(records)} records from {self.config.eval_dataset_path}")

        # 1. Retrieval evaluation
        print("\n[1/4] Retrieval evaluation...")
        evaluator = RetrievalEvaluator(
            retriever=lambda q, top_k: self.retriever(q, top_k=top_k),
            k_values=[1, 3, 5, 10]
        )
        retrieval_result = evaluator.evaluate(records)
        evaluator.print_report(retrieval_result)
        run.retrieval = {
            "recall_at_5":   retrieval_result.recall_at_k.get(5, 0),
            "mrr":           retrieval_result.mrr,
            "hit_rate_at_5": retrieval_result.hit_rate_at_k.get(5, 0),
            "latency_p99_ms": retrieval_result.latency_p99_ms,
            "by_category":   retrieval_result.by_category
        }

        # 2. Generate answers
        print("\n[2/4] Generating answers...")
        questions, answers, contexts, ground_truths = [], [], [], []
        for record in records:
            retrieved = self.retriever(record.query, top_k=self.config.retrieval_k)
            ctx = [r["content"] for r in retrieved]
            answer = self.generator(record.query, ctx)
            questions.append(record.query)
            answers.append(answer)
            contexts.append(ctx)
            ground_truths.append(record.ground_truth_answer)
        print(f"  Generated {len(answers)} answers")

        # 3. RAGAS evaluation
        if self.config.run_ragas:
            print("\n[3/4] RAGAS evaluation...")
            ragas_scores = run_ragas_evaluation(questions, answers, contexts, ground_truths)
            run.generation = ragas_scores
            print(f"  Faithfulness     : {ragas_scores['faithfulness']:.4f}")
            print(f"  Answer relevancy : {ragas_scores['answer_relevancy']:.4f}")
            print(f"  Context recall   : {ragas_scores['context_recall']:.4f}")
            print(f"  Context precision: {ragas_scores['context_precision']:.4f}")
        else:
            print("\n[3/4] RAGAS — skipped")
            run.generation = {}

        # 4. LLM judge (optional)
        if self.config.run_llm_judge:
            print(f"\n[4/4] LLM judge (sample={self.config.judge_sample_size})...")
            import random
            sample_idx = random.sample(range(len(records)), min(self.config.judge_sample_size, len(records)))
            sample_records = [records[i] for i in sample_idx]
            sample_outputs = [{"answer": answers[i], "contexts": contexts[i]} for i in sample_idx]
            run.judge = batch_judge(sample_records, sample_outputs, self.config.judge_model)
            print(f"  Overall score   : {run.judge.get('mean_overall', 0):.4f}")
        else:
            print("\n[4/4] LLM judge — skipped (set run_llm_judge=True to enable)")

        # Gate evaluation
        run.gates_passed, run.gate_failures = self._check_gates(run)

        # Baseline comparison
        if self.config.compare_to_baseline:
            baseline = self.baselines.get_latest()
            if baseline:
                run.baseline_delta = self.baselines.compare(retrieval_result, baseline)
                run.regression_detected = run.baseline_delta.get("has_regression", False)

        # Save results
        self._save_results(run)
        self._print_summary(run)
        return run

    def _load_dataset(self) -> list[EvalRecord]:
        import random
        path = Path(self.config.eval_dataset_path)
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    records.append(EvalRecord(
                        id=d["id"], query=d["query"],
                        ground_truth_answer=d["ground_truth_answer"],
                        relevant_document_ids=d.get("relevant_document_ids", []),
                        category=QueryCategory(d.get("category", "factual")),
                        difficulty=d.get("difficulty", "medium"),
                        source=d.get("source", "synthetic")
                    ))
        if self.config.dataset_sample_size and len(records) > self.config.dataset_sample_size:
            records = random.sample(records, self.config.dataset_sample_size)
        return records

    def _check_gates(self, run: PipelineRun) -> tuple[bool, list[str]]:
        failures = []
        r5 = run.retrieval.get("recall_at_5", 0)
        if r5 < self.config.min_recall_at_5:
            failures.append(f"Recall@5 {r5:.4f} < {self.config.min_recall_at_5}")
        if run.generation:
            faith = run.generation.get("faithfulness", 0)
            if faith < self.config.min_faithfulness:
                failures.append(f"Faithfulness {faith:.4f} < {self.config.min_faithfulness}")
            relevancy = run.generation.get("answer_relevancy", 0)
            if relevancy < self.config.min_answer_relevancy:
                failures.append(f"Answer relevancy {relevancy:.4f} < {self.config.min_answer_relevancy}")
        p99 = run.retrieval.get("latency_p99_ms", 0)
        if p99 > self.config.max_latency_p99_ms:
            failures.append(f"P99 latency {p99:.0f}ms > {self.config.max_latency_p99_ms:.0f}ms")
        return len(failures) == 0, failures

    def _save_results(self, run: PipelineRun):
        path = Path(self.config.results_dir) / f"run_{run.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run.__dict__, indent=2, default=str))

    def _print_summary(self, run: PipelineRun):
        print(f"\n{'─'*55}")
        status = "✓ GATES PASSED" if run.gates_passed else "✗ GATES FAILED"
        print(f"  {status}")
        for f in run.gate_failures:
            print(f"    ✗ {f}")
        if run.regression_detected:
            print(f"  ⚠ REGRESSION DETECTED:")
            for k, v in run.baseline_delta.get("regressions", {}).items():
                print(f"    {k}: {v:+.4f}")
        print(f"{'─'*55}\n")
```

---

### 5.2 Continuous Evaluation with Baselines

```python
import json
from pathlib import Path
from datetime import datetime

class ContinuousEvaluationTracker:
    """
    Tracks evaluation results over time and detects trends.
    """
    def __init__(self, history_path: str = "eval/history.jsonl"):
        self.path = Path(history_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, run: PipelineRun):
        entry = {
            "run_id":       run.run_id,
            "git_sha":      run.git_sha,
            "timestamp":    run.started_at,
            "recall_at_5":  run.retrieval.get("recall_at_5"),
            "faithfulness": run.generation.get("faithfulness"),
            "p99_ms":       run.retrieval.get("latency_p99_ms"),
            "gates_passed": run.gates_passed,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_trend(self, metric: str, last_n: int = 10) -> dict:
        """Compute metric trend over last N runs."""
        history = self._load()[-last_n:]
        values = [h.get(metric) for h in history if h.get(metric) is not None]
        if len(values) < 2:
            return {"trend": "insufficient_data", "values": values}
        delta = values[-1] - values[0]
        direction = "improving" if delta > 0.01 else "degrading" if delta < -0.01 else "stable"
        return {
            "trend":     direction,
            "values":    [round(v, 4) for v in values],
            "delta":     round(delta, 4),
            "latest":    round(values[-1], 4),
        }

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]
```

---

### 5.3 Regression Detection

```python
from dataclasses import dataclass

@dataclass
class RegressionAlert:
    metric: str
    current_value: float
    baseline_value: float
    delta: float
    threshold: float
    severity: str   # warning | critical

class RegressionDetector:
    def __init__(
        self,
        warning_threshold: float = -0.02,   # 2% drop → warning
        critical_threshold: float = -0.05   # 5% drop → critical
    ):
        self.warning = warning_threshold
        self.critical = critical_threshold

    def detect(self, current: dict, baseline: dict) -> list[RegressionAlert]:
        alerts = []
        metrics_to_check = ["recall_at_5", "faithfulness", "answer_relevancy"]
        for metric in metrics_to_check:
            cur = current.get(metric)
            base = baseline.get(metric)
            if cur is None or base is None:
                continue
            delta = cur - base
            if delta <= self.critical:
                alerts.append(RegressionAlert(
                    metric=metric, current_value=cur, baseline_value=base,
                    delta=delta, threshold=self.critical, severity="critical"
                ))
            elif delta <= self.warning:
                alerts.append(RegressionAlert(
                    metric=metric, current_value=cur, baseline_value=base,
                    delta=delta, threshold=self.warning, severity="warning"
                ))
        # Latency regression (opposite direction)
        p99_cur = current.get("latency_p99_ms", 0)
        p99_base = baseline.get("latency_p99_ms", 0)
        if p99_base > 0:
            p99_delta_pct = (p99_cur - p99_base) / p99_base
            if p99_delta_pct > 0.30:
                alerts.append(RegressionAlert(
                    metric="latency_p99_ms", current_value=p99_cur,
                    baseline_value=p99_base, delta=p99_cur - p99_base,
                    threshold=p99_base * 0.30, severity="critical"
                ))
            elif p99_delta_pct > 0.15:
                alerts.append(RegressionAlert(
                    metric="latency_p99_ms", current_value=p99_cur,
                    baseline_value=p99_base, delta=p99_cur - p99_base,
                    threshold=p99_base * 0.15, severity="warning"
                ))
        return alerts
```

---

### 5.4 Evaluation Reporting

```python
def generate_eval_report(run: PipelineRun, baseline: dict = None) -> str:
    lines = [
        f"# Evaluation Report — Run {run.run_id}",
        f"**Date:** {run.started_at[:10]}  **Git SHA:** `{run.git_sha[:8]}`",
        "",
        "## Retrieval Metrics",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall@5 | {run.retrieval.get('recall_at_5', 0):.4f} |",
        f"| MRR | {run.retrieval.get('mrr', 0):.4f} |",
        f"| Hit Rate@5 | {run.retrieval.get('hit_rate_at_5', 0):.4f} |",
        f"| P99 Latency | {run.retrieval.get('latency_p99_ms', 0):.0f}ms |",
    ]
    if run.generation:
        lines += [
            "",
            "## Generation Metrics (RAGAS)",
            "| Metric | Value |",
            "|---|---|",
            f"| Faithfulness | {run.generation.get('faithfulness', 0):.4f} |",
            f"| Answer Relevancy | {run.generation.get('answer_relevancy', 0):.4f} |",
            f"| Context Recall | {run.generation.get('context_recall', 0):.4f} |",
            f"| Context Precision | {run.generation.get('context_precision', 0):.4f} |",
        ]
    lines += [
        "",
        "## Quality Gates",
        f"**Status: {'✅ PASSED' if run.gates_passed else '❌ FAILED'}**",
    ]
    if run.gate_failures:
        for f in run.gate_failures:
            lines.append(f"- ✗ {f}")
    if run.regression_detected and run.baseline_delta:
        lines += ["", "## ⚠️ Regressions Detected"]
        for k, v in run.baseline_delta.get("regressions", {}).items():
            lines.append(f"- **{k}**: {v:+.4f}")
    return "\n".join(lines)
```

---

### 🧪 Hands-on Lab: Full Evaluation Pipeline

**Objective:** Build and run a complete evaluation pipeline from dataset to report.

**Prerequisites:** `openai`, `sentence-transformers`, `scikit-learn`, `pathlib`

```python
import json
import hashlib
import time
import random
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Sample knowledge base ──────────────────────────────────
DOCS = [
    {"id": "d1", "content": "Enterprise customers have a 90-day return window. Standard customers have 30 days."},
    {"id": "d2", "content": "Professional plan costs $150/month with 25 users and email support."},
    {"id": "d3", "content": "API rate limits: Free 100 req/min, Professional 1000 req/min, Enterprise 10000 req/min."},
    {"id": "d4", "content": "Refunds are processed in 5-7 business days for card payments."},
    {"id": "d5", "content": "The REST API uses OAuth 2.0 for authentication. Tokens expire after 3600 seconds."},
]
DOC_EMBEDDINGS = embed_model.encode([d["content"] for d in DOCS])

# ── Simple retriever and generator ────────────────────────
def simple_retriever(query: str, top_k: int = 5) -> list[dict]:
    q_emb = embed_model.encode(query)
    sims = cosine_similarity([q_emb], DOC_EMBEDDINGS)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [{"id": DOCS[i]["id"], "content": DOCS[i]["content"], "score": float(sims[i])} for i in top_idx]

def simple_generator(query: str, contexts: list[str]) -> str:
    ctx = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the context. Cite sources [N]. If not found, say 'I cannot find this information.'"},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}
        ],
        temperature=0, max_tokens=200
    )
    return resp.choices[0].message.content

# ── Minimal eval dataset ───────────────────────────────────
EVAL_RECORDS = [
    {"id": "e1", "query": "How long do enterprise customers have to return items?",
     "ground_truth_answer": "90 days", "relevant_document_ids": ["d1"],
     "category": "factual", "difficulty": "easy"},
    {"id": "e2", "query": "What are the API rate limits for Professional plan?",
     "ground_truth_answer": "1000 requests per minute", "relevant_document_ids": ["d3"],
     "category": "factual", "difficulty": "easy"},
    {"id": "e3", "query": "How long does a refund take and what are the Professional plan costs?",
     "ground_truth_answer": "5-7 business days; $150/month", "relevant_document_ids": ["d4", "d2"],
     "category": "multi_hop", "difficulty": "medium", "requires_multi_hop": True},
    {"id": "e4", "query": "What is the pricing for the government procurement plan?",
     "ground_truth_answer": "NOT_IN_DOCUMENT", "relevant_document_ids": [],
     "category": "unanswerable", "difficulty": "medium"},
    {"id": "e5", "query": "How do tokens work in the API authentication?",
     "ground_truth_answer": "Tokens expire after 3600 seconds", "relevant_document_ids": ["d5"],
     "category": "factual", "difficulty": "easy"},
]

# Save eval dataset
eval_path = Path("/tmp/lab_eval.jsonl")
eval_path.write_text("\n".join(json.dumps(r) for r in EVAL_RECORDS))
print(f"Eval dataset: {len(EVAL_RECORDS)} records → {eval_path}")

# ── Run retrieval evaluation ───────────────────────────────
print("\n── Retrieval Evaluation ──────────────────────────")
answerable = [r for r in EVAL_RECORDS if r["category"] != "unanswerable"]
for record in answerable:
    retrieved = simple_retriever(record["query"], top_k=5)
    r_ids = [r["id"] for r in retrieved]
    r5 = recall_at_k(record["relevant_document_ids"], r_ids, 5)
    hit = hit_rate(record["relevant_document_ids"], r_ids, 5)
    mrr = mean_reciprocal_rank(record["relevant_document_ids"], r_ids)
    print(f"  [{record['category']:<12}] Recall@5={r5:.2f} Hit@5={hit:.0f} MRR={mrr:.2f}  {record['query'][:50]}")

# ── Run generation evaluation ──────────────────────────────
print("\n── Generation Evaluation ─────────────────────────")
results = []
for record in EVAL_RECORDS:
    retrieved = simple_retriever(record["query"], top_k=5)
    ctx = [r["content"] for r in retrieved]
    answer = simple_generator(record["query"], ctx)
    is_refusal = "cannot find" in answer.lower()
    is_unanswerable = record["category"] == "unanswerable"
    status = "✓" if (is_unanswerable == is_refusal) else "✗"
    print(f"  {status} [{record['category']:<12}] {record['query'][:45]}")
    print(f"      → {answer[:80]}")
    results.append({"record": record, "answer": answer, "contexts": ctx})

# ── Gate evaluation ────────────────────────────────────────
print("\n── Quality Gate Check ────────────────────────────")
answerable_results = [r for r in results if r["record"]["category"] != "unanswerable"]
recalls = []
for r in answerable_results:
    retrieved_ids = [d["id"] for d in simple_retriever(r["record"]["query"], top_k=5)]
    recalls.append(recall_at_k(r["record"]["relevant_document_ids"], retrieved_ids, 5))
mean_recall = sum(recalls) / len(recalls) if recalls else 0
print(f"  Mean Recall@5 : {mean_recall:.4f} (threshold: 0.80) {'✓' if mean_recall >= 0.80 else '✗'}")
refusals_correct = sum(1 for r in results
                       if (r["record"]["category"] == "unanswerable") == ("cannot find" in r["answer"].lower()))
print(f"  Refusal accuracy: {refusals_correct}/{len(results)} ({refusals_correct/len(results):.0%})")
print(f"\n{'✓ GATES PASSED' if mean_recall >= 0.80 else '✗ GATES FAILED'}")
```

**Extensions:**
- Add `RegressionDetector` with a baseline captured from the first run and compare subsequent runs against it
- Extend the generator to use the full `RAG_PROMPT` template from Part VI and compare quality scores
- Add LLM-as-Judge scoring for the 5 answers and compare automated vs. manual quality ranking

---

> ### 📋 Chapter Summary
>
> - The `EvaluationPipeline` orchestrates dataset loading, retrieval evaluation, answer generation, RAGAS metrics, LLM judge, gate checking, baseline comparison, and result persistence.
> - **Continuous evaluation** tracks metric trends over time — detecting gradual degradation that single-run comparisons miss.
> - **Regression detection** uses configurable thresholds (warning/critical) with separate logic for quality metrics (lower is worse) and latency (higher is worse).
> - A structured **evaluation report** communicates gate status, regressions, and per-category breakdowns for both CI and human review.

---

> ### ❓ Comprehension Questions
>
> 1. The `EvaluationPipeline` runs RAGAS by default but skips LLM judge. Justify this design choice and describe the conditions under which you would enable LLM judge in CI.
> 2. The trend tracker shows Recall@5 values over 10 runs: `[0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79]`. No individual run drops below the 0.80 threshold. What process failure does this trend represent?
> 3. A regression alert fires with `severity="critical"` for `faithfulness` delta = -0.06. Investigation reveals the evaluation dataset was updated with 50 new hard examples. Is this a true regression or a dataset change effect? How would you determine which it is?
> 4. The evaluation pipeline saves results to `eval/results/run_*.json`. After 6 months, the results directory contains 180 files. Design a retention and archival policy for evaluation results.
> 5. Design an evaluation pipeline that supports A/B comparison: given two system configurations (A = current production, B = proposed change), run both against the same dataset and produce a side-by-side comparison report.

---

## References

### Papers
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Es et al., 2023.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023.
- [What's Your ML Test Score? A Rubric for ML Production Readiness](https://research.google/pubs/pub46555/) — Breck et al., 2017.
- [Continuous Integration for ML](https://arxiv.org/abs/2107.03498) — Shankar et al., 2021.

### Documentation
- [RAGAS Documentation](https://docs.ragas.io)
- [DeepEval Evaluation Framework](https://docs.confident-ai.com)
- [MLflow Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [TruLens](https://www.trulens.org/docs/) — RAG triad evaluation (context relevance, groundedness, answer relevance).
- [Weights & Biases](https://docs.wandb.ai) — Experiment and evaluation tracking.

---

> **Navigation**
> [← Part VIII — AI Systems SDLC](part_08_sdlc.md) | [→ Part X — Testing LLM Systems](part_10_testing.md)
