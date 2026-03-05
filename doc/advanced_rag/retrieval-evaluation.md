## Chapter 6 — Retrieval Evaluation

### 6.1 Why Retrieval Must Be Measured Separately

A common mistake in RAG system development is to evaluate only end-to-end answer quality — asking "did the LLM produce a correct answer?" without asking "did retrieval surface the right context?"

This conflates two independent failure modes:

**Retrieval failure:** The correct chunks are not returned. No amount of prompt engineering or model upgrading can fix an answer that cannot be generated because the evidence was never retrieved.

**Generation failure:** The correct chunks were retrieved but the model failed to use them correctly — misunderstood, hallucinated over, or ignored the context.

Measuring retrieval independently enables precise diagnosis and targeted improvement. It also enables fast iteration: retrieval metrics can be computed without calling a generation model.

---

### 6.2 Retrieval Metrics

**Recall@K** — the proportion of relevant documents found in the top-K results. The primary metric for RAG retrieval.

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
```

**Precision@K** — the proportion of top-K results that are relevant.

```python
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    return sum(1 for r in top_k if r in relevant_ids) / k
```

**MRR (Mean Reciprocal Rank)** — rewards systems that rank the first relevant document higher.

```python
def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
```

**NDCG@K (Normalised Discounted Cumulative Gain)** — accounts for graded relevance (highly relevant vs. partially relevant).

```python
import math

def ndcg_at_k(retrieved_ids: list[str], relevance_scores: dict[str, float], k: int) -> float:
    dcg = sum(
        relevance_scores.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
    )
    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(s / math.log2(rank + 1) for rank, s in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
```

**Context Precision and Context Recall** — [RAGAS](https://docs.ragas.io) metrics specific to RAG:

- **Context Precision:** Are the retrieved chunks relevant to the question?
- **Context Recall:** Do the retrieved chunks contain the information needed to answer?

---

### 6.3 Building a Retrieval Evaluation Dataset

A retrieval evaluation dataset consists of `(query, relevant_document_ids)` pairs. Three construction approaches:

**Manual annotation** — Human experts label relevant documents for each query. Highest quality, slow and expensive.

**Synthetic generation from documents** — LLM generates questions from document chunks; those chunks are the ground truth:

```python
def generate_eval_pairs(chunks: list[dict], n_per_chunk: int = 2) -> list[dict]:
    """
    Generate (question, relevant_chunk_id) pairs from corpus chunks.
    """
    eval_pairs = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""Generate {n_per_chunk} distinct questions that can be answered 
using ONLY the provided text. Questions should be natural user questions.
Return JSON: {{"questions": ["q1", "q2"]}}"""
                },
                {"role": "user", "content": chunk["content"]}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        questions = json.loads(response.choices[0].message.content).get("questions", [])
        for q in questions:
            eval_pairs.append({
                "query": q,
                "relevant_ids": [chunk["id"]],
                "source_text": chunk["content"]
            })

    return eval_pairs
```

**Production log mining** — sample real user queries from production logs and label them. Reflects actual usage but requires a running system.

---

### 6.4 Automated Evaluation with RAGAS

[RAGAS](https://docs.ragas.io) provides automated retrieval and generation evaluation metrics without requiring human annotation:

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

# Build evaluation dataset in RAGAS format
eval_data = {
    "question":  ["What is the refund policy?", "How long does processing take?"],
    "answer":    ["Refunds are processed...", "Processing takes 5-7 days..."],
    "contexts":  [
        ["Standard refund policy allows returns within 30 days..."],
        ["Refunds are processed within 5-7 business days..."]
    ],
    "ground_truth": [
        "Returns are allowed within 30 days of purchase.",
        "Standard refunds take 5-7 business days."
    ]
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
)
print(results)
# {'context_precision': 0.83, 'context_recall': 0.91,
#  'faithfulness': 0.95, 'answer_relevancy': 0.88}
```

---

### 6.5 Continuous Retrieval Monitoring

Retrieval quality degrades over time as the knowledge base evolves, query patterns shift, and user language changes. Continuous monitoring detects regressions before they impact users.

```python
import logging
from datetime import datetime

logger = logging.getLogger("retrieval_monitor")

class RetrievalMonitor:
    def __init__(self, evaluation_set: list[dict], retriever, k: int = 5):
        self.eval_set = evaluation_set
        self.retriever = retriever
        self.k = k
        self.baseline_recall: Optional[float] = None

    def run_evaluation(self) -> dict:
        recalls = []
        for item in self.eval_set:
            retrieved = self.retriever.retrieve(item["query"])
            retrieved_ids = [r.get("id", r["content"][:80]) for r in retrieved[:self.k]]
            relevant = set(item["relevant_ids"])
            recall = recall_at_k(retrieved_ids, relevant, self.k)
            recalls.append(recall)

        metrics = {
            "recall_at_k": sum(recalls) / len(recalls),
            "k": self.k,
            "eval_size": len(self.eval_set),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info("retrieval_eval", extra=metrics)
        return metrics

    def check_regression(self, threshold: float = 0.05) -> bool:
        """Returns True if recall has degraded by more than threshold vs baseline."""
        current = self.run_evaluation()
        if self.baseline_recall is None:
            self.baseline_recall = current["recall_at_k"]
            return False
        degradation = self.baseline_recall - current["recall_at_k"]
        if degradation > threshold:
            logger.warning(
                f"Retrieval regression detected: {self.baseline_recall:.2%} → "
                f"{current['recall_at_k']:.2%} (Δ={degradation:.2%})"
            )
            return True
        return False
```

---

> ### 📋 Chapter Summary
>
> - Retrieval must be evaluated **independently** from generation to diagnose whether failures are in the retrieval or generation stage.
> - Core metrics: **Recall@K** (primary), **Precision@K**, **MRR**, **NDCG@K** for graded relevance.
> - Evaluation datasets are built via manual annotation, synthetic LLM-generated pairs, or production log mining.
> - [RAGAS](https://docs.ragas.io) provides automated RAG-specific metrics: context precision, context recall, faithfulness, and answer relevancy.
> - **Continuous monitoring** detects retrieval regression as the knowledge base and query distribution evolve.

---

> ### ❓ Comprehension Questions
>
> 1. Your RAG system answers 90% of questions correctly in manual testing but drops to 60% when the knowledge base is updated. You suspect retrieval quality has degraded. What metric would you measure first and how?
> 2. Explain the difference between Recall@K and Precision@K. In a RAG system, which is generally more important and why?
> 3. LLM-generated evaluation pairs (synthetic dataset) have a known bias: the LLM generates questions that are easy to answer from the corpus. How would this bias affect the reliability of your evaluation results?
> 4. [RAGAS](https://docs.ragas.io) measures context recall. What does a context recall of 0.65 tell you about your retrieval pipeline, and what components would you investigate?
> 5. Design a continuous monitoring system that runs retrieval evaluation nightly and sends an alert when Recall@5 drops below 0.80. What infrastructure components does this require?

---

## References

### Papers
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — Es et al., 2023.
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR](https://arxiv.org/abs/2104.08663) — Thakur et al., 2021.
- [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431) — Chen et al., 2023.
- [ARES: An Automated Evaluation Framework for RAG Systems](https://arxiv.org/abs/2311.09476) — Saad-Falcon et al., 2023.

### Documentation
- [RAGAS Documentation](https://docs.ragas.io) — Automated RAG evaluation framework.
- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation) — LangChain's evaluation and tracing platform.
- [Arize Phoenix](https://docs.arize.com/phoenix) — Open-source LLM observability and evaluation.
- [TruLens](https://www.trulens.org/docs/) — RAG evaluation and tracking.

---

> **Navigation**
> [← Part III — RAG Engineering](../rag_engineering/index.md) | [→ Part V — Dataset Engineering](../dataset_engineering/index.md)

---
[« Back to advanced_rag Index](index.md) | [🏠 Home](../index.md)