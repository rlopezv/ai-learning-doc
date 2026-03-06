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

---
[« Back to evaluation-engineering Index](index.md) | [🏠 Home](../index.md)