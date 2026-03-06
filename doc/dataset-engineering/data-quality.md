## Chapter 4 — Data Quality

### 4.1 Quality Dimensions

Data quality for LLM datasets spans multiple dimensions:

| Dimension | Definition | Check method |
|---|---|---|
| **Completeness** | All required fields present and non-empty | Schema validation |
| **Accuracy** | Content is factually correct | LLM-as-judge or human review |
| **Consistency** | No contradictions within or across records | Cross-record validation |
| **Uniqueness** | No duplicate or near-duplicate records | Hash + embedding dedup |
| **Timeliness** | Content is not outdated | Metadata date checks |
| **Privacy** | No PII in datasets not intended to contain it | NER-based detection |
| **Format** | Records conform to schema | [Pydantic](https://docs.pydantic.dev) validation |
| **Toxicity** | No harmful or biased content | Content classification |

Quality checks must run as part of the data pipeline — not as one-time manual reviews.

---

### 4.2 Automated Quality Checks

```python
from pydantic import ValidationError
from typing import Optional

class QualityReport:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failure_reasons: dict[str, int] = {}

    def add_failure(self, reason: str):
        self.failed += 1
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

    def add_pass(self):
        self.passed += 1

    @property
    def total(self):
        return self.passed + self.failed

    def summary(self) -> str:
        return f"{self.passed}/{self.total} passed | failures: {self.failure_reasons}"

class DatasetQualityChecker:
    def __init__(self, schema_class, checks: list = None):
        self.schema = schema_class
        self.checks = checks or []

    def run(self, records: list[dict]) -> QualityReport:
        report = QualityReport()
        for record in records:
            errors = self._check(record)
            if errors:
                for err in errors:
                    report.add_failure(err)
            else:
                report.add_pass()
        return report

    def _check(self, record: dict) -> list[str]:
        errors = []
        try:
            self.schema(**record)
        except (ValidationError, TypeError) as e:
            errors.append(f"schema_error")
            return errors
        for check_fn in self.checks:
            error = check_fn(record)
            if error:
                errors.append(error)
        return errors

# Check functions
def check_question_length(record: dict) -> Optional[str]:
    q = record.get("question", "")
    if len(q) < 15:
        return "question_too_short"
    if len(q) > 500:
        return "question_too_long"
    return None

def check_answer_not_trivial(record: dict) -> Optional[str]:
    answer = record.get("answer", "")
    if len(answer.split()) < 5:
        return "answer_too_brief"
    return None

checker = DatasetQualityChecker(
    schema_class=EvalRecord,
    checks=[check_question_length, check_answer_not_trivial]
)
report = checker.run(synthetic_pairs)
print(report.summary())
```

---

### 4.3 Deduplication

Duplicate records inflate dataset size, bias evaluation, and waste fine-tuning compute.

```python
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def exact_dedup(records: list[dict], key_field: str = "question") -> list[dict]:
    """Remove exact duplicates by content hash."""
    seen = set()
    result = []
    for record in records:
        key = hashlib.md5(record[key_field].strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(record)
    removed = len(records) - len(result)
    print(f"Exact dedup: {len(records)} → {len(result)} ({removed} removed)")
    return result

def semantic_dedup(
    records: list[dict],
    key_field: str = "question",
    threshold: float = 0.92,
    model_name: str = "all-MiniLM-L6-v2"
) -> list[dict]:
    """Remove near-duplicate records using embedding similarity."""
    model = SentenceTransformer(model_name)
    texts = [r[key_field] for r in records]
    embeddings = model.encode(texts, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    keep = np.ones(len(records), dtype=bool)
    for i in range(len(records)):
        if keep[i]:
            dups = np.where((sim_matrix[i] >= threshold) & (np.arange(len(records)) > i))[0]
            keep[dups] = False
    result = [r for r, k in zip(records, keep) if k]
    removed = len(records) - len(result)
    print(f"Semantic dedup ({threshold}): {len(records)} → {len(result)} ({removed} removed)")
    return result
```

---

### 4.4 PII Detection and Redaction

Datasets derived from user content must be scanned for PII before use in evaluation or fine-tuning.

```python
import re
from typing import NamedTuple

class PIIMatch(NamedTuple):
    pii_type: str
    value: str
    start: int
    end: int

class PIIDetector:
    """Rule-based PII detection. Augment with spaCy or Presidio for production."""
    PATTERNS = {
        "email":       r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_es":    r'\b(?:\+34\s?)?[6789]\d{2}[\s-]?\d{3}[\s-]?\d{3}\b',
        "phone_us":    r'\b(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[\s-]?){3}\d{4}\b',
        "dni_spain":   r'\b\d{8}[A-Za-z]\b',
        "iban":        r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b',
        "ip_address":  r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }

    def detect(self, text: str) -> list[PIIMatch]:
        matches = []
        for pii_type, pattern in self.PATTERNS.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(PIIMatch(pii_type, m.group(), m.start(), m.end()))
        return sorted(matches, key=lambda x: x.start)

    def redact(self, text: str) -> str:
        for match in reversed(self.detect(text)):
            text = text[:match.start] + f"[{match.pii_type.upper()}]" + text[match.end:]
        return text

def scan_dataset_for_pii(records: list[dict], fields: list[str]) -> dict:
    detector = PIIDetector()
    pii_counts: dict[str, int] = {}
    flagged = []
    for record in records:
        has_pii = False
        for field in fields:
            matches = detector.detect(str(record.get(field, "")))
            if matches:
                has_pii = True
                for m in matches:
                    pii_counts[m.pii_type] = pii_counts.get(m.pii_type, 0) + 1
        if has_pii:
            flagged.append(record.get("id", "unknown"))
    return {"pii_counts": pii_counts, "flagged_count": len(flagged), "flagged_sample": flagged[:10]}
```

🔓 **Production PII detection with [Microsoft Presidio](https://microsoft.github.io/presidio/):**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def presidio_redact(text: str, language: str = "en") -> str:
    results = analyzer.analyze(text=text, language=language)
    return anonymizer.anonymize(text=text, analyzer_results=results).text
```

---

### 4.5 Quality Scoring Pipelines

Rather than binary pass/fail, assign a continuous quality score to each record.

```python
from dataclasses import dataclass

@dataclass
class QualityScore:
    record_id: str
    overall_score: float
    completeness_score: float
    grounding_score: float
    length_score: float
    uniqueness_score: float
    passes_threshold: bool

def score_record(
    record: dict,
    doc_text: str,
    existing_questions: list[str],
    embed_model,
    threshold: float = 0.70
) -> QualityScore:
    # Completeness
    required = ["question", "answer", "source_document_id"]
    completeness = sum(1 for f in required if record.get(f)) / len(required)

    # Grounding (answer similarity to source)
    if doc_text and record.get("answer"):
        a_emb = embed_model.encode(record["answer"])
        d_emb = embed_model.encode(doc_text[:1000])
        grounding = float(cosine_similarity([a_emb], [d_emb])[0][0])
    else:
        grounding = 0.0

    # Length score
    q_len = len(record.get("question", "").split())
    a_len = len(record.get("answer", "").split())
    length = min(1.0, q_len / 10) * min(1.0, a_len / 15)

    # Uniqueness vs recent questions
    if existing_questions:
        q_emb = embed_model.encode(record.get("question", ""))
        ex_embs = embed_model.encode(existing_questions[-50:])
        max_sim = float(cosine_similarity([q_emb], ex_embs).max())
        uniqueness = 1.0 - min(1.0, max(0.0, (max_sim - 0.7) / 0.3))
    else:
        uniqueness = 1.0

    overall = 0.25 * completeness + 0.35 * grounding + 0.20 * length + 0.20 * uniqueness
    return QualityScore(
        record_id=record.get("id", ""),
        overall_score=round(overall, 4),
        completeness_score=round(completeness, 4),
        grounding_score=round(grounding, 4),
        length_score=round(length, 4),
        uniqueness_score=round(uniqueness, 4),
        passes_threshold=overall >= threshold
    )
```

---

> ### 📋 Chapter Summary
>
> - Data quality has eight dimensions; all require automated checks in the pipeline.
> - **Exact dedup** (hash) and **semantic dedup** (embeddings) address different duplicate patterns.
> - **PII detection** is mandatory before using any dataset derived from user content.
> - **Quality scoring** enables threshold-based filtering with configurable stringency.

---

> ### ❓ Comprehension Questions
>
> 1. A fine-tuning dataset passes all automated checks but contains customer email addresses. What check is missing?
> 2. Exact dedup removes 5%; semantic dedup at 0.92 removes an additional 18%. What does this tell you about the dataset?
> 3. Design a quality gate for fine-tuning: zero PII, ≥0.75 quality score for 90% of records, ≥20% difficulty=hard.
> 4. The grounding filter at 0.4 removes 30% of pairs. How do you determine the right threshold empirically?
> 5. A dataset contains English and Spanish. How does your quality pipeline need to adapt?

---

## References

### Documentation
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection and anonymisation.
- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) — Named entity recognition.
- [Great Expectations](https://docs.greatexpectations.io) — Data quality testing framework.
- [Pydantic Documentation](https://docs.pydantic.dev) — Schema validation.

### Papers
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — Lee et al., 2021.
- [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758) — Dodge et al., 2021.

---

---
[« Back to dataset-engineering Index](index.md) | [🏠 Home](../index.md)