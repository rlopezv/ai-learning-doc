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
> [← Part VIII — AI Systems SDLC](../sdlc/index.md) | [→ Part X — Testing LLM Systems](../testing/index.md)

---
[« Back to evaluation_engineering Index](index.md) | [🏠 Home](../index.md)