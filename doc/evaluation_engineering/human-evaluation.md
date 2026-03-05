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

[« Back to evaluation_engineering Index](index.md) | [🏠 Home](../../index.md)
