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

---
[« Back to evaluation-engineering Index](index.md) | [🏠 Home](../index.md)