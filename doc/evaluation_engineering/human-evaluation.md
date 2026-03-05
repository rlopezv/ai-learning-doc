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

---
[« Back to evaluation_engineering Index](index.md) | [🏠 Home](../../index.md)