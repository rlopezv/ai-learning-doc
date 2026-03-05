## Chapter 5 — Healthcare Knowledge Assistant 🧪

### 5.1 Clinical Use Case Constraints

A hospital network deployed an internal RAG system to help clinical staff query treatment protocols, drug interaction databases, and discharge planning guidelines. The system is NOT patient-facing and is designed to assist clinicians who retain full professional responsibility.

```python
CLINICAL_CONSTRAINTS = {
    "system_classification": "Clinical Decision Support Tool (CDST)",
    "eu_ai_act_tier":        "high_risk (medical device adjacent)",
    "intended_users":        ["registered nurses", "junior doctors", "pharmacists"],
    "NOT_intended_for":      ["patients", "self-diagnosis", "replacing clinical judgment"],
    "mandatory_disclaimer":  (
        "This system provides reference information from approved clinical guidelines only. "
        "All clinical decisions remain the responsibility of the licensed clinician. "
        "Never substitute for direct patient assessment or specialist consultation."
    ),
    "prohibited_outputs": [
        "Specific drug dosing for individual patients",
        "Diagnosis of specific patient conditions",
        "Recommendations that contradict the retrieved guideline",
        "Answers about patients by name or ID (HIPAA/GDPR protection)",
    ],
    "required_outputs": [
        "Explicit citation of source guideline with version and date",
        "Explicit uncertainty acknowledgement when guideline is ambiguous",
        "Escalation prompt when question is outside guideline scope",
    ],
    "data_requirements": {
        "corpus_retention": "Current version + 1 prior version",
        "audit_retention_years": 7,
        "pii_in_corpus":    False,    # Clinical guidelines only — no patient data
        "access_control":   "Staff ID + ward/role verification",
    }
}
```

---

### 5.2 Medical Corpus Management

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class ClinicalGuideline:
    guideline_id: str
    title: str
    issuing_body: str          # "NICE" | "WHO" | "NHS" | "hospital_internal"
    version: str
    published_date: str
    effective_date: str
    expiry_date: Optional[str]  # Some guidelines have explicit expiry
    supersedes: Optional[str]   # ID of previous version this replaces
    specialty: str
    evidence_grade: str         # "A" | "B" | "C" | "D" | "GPP" (Good Practice Point)
    status: str                 # "active" | "archived" | "under_review"
    content_hash: str           # Detect unauthorised modifications

class ClinicalCorpusManager:
    """
    Manages clinical guideline corpus with strict version control.
    Only ACTIVE guidelines are indexed; ARCHIVED versions are retained
    for audit but excluded from retrieval.
    """
    def __init__(self, vector_db, metadata_store, audit_logger):
        self.vdb    = vector_db
        self.meta   = metadata_store
        self.audit  = audit_logger

    def publish_guideline(self, guideline: ClinicalGuideline,
                          content: str, approved_by: str):
        """Publish a new guideline and supersede previous version."""
        # Verify content integrity
        import hashlib
        computed_hash = hashlib.sha256(content.encode()).hexdigest()
        assert computed_hash == guideline.content_hash, "Content hash mismatch"

        # Archive previous version if superseding
        if guideline.supersedes:
            self._archive_version(guideline.supersedes)

        # Index new version
        chunks = self._chunk_guideline(content, guideline)
        self.vdb.upsert_batch(
            ids=[c["id"] for c in chunks],
            embeddings=self._embed([c["content"] for c in chunks]),
            payloads=[{
                **c["metadata"],
                "guideline_id":   guideline.guideline_id,
                "version":        guideline.version,
                "issuing_body":   guideline.issuing_body,
                "evidence_grade": guideline.evidence_grade,
                "status":         "active",
                "effective_date": guideline.effective_date,
            } for c in chunks]
        )
        self.audit.log({
            "event": "guideline_published",
            "guideline_id": guideline.guideline_id,
            "version": guideline.version,
            "approved_by": approved_by,
            "timestamp": datetime.utcnow().isoformat()
        })

    def _archive_version(self, guideline_id: str):
        """Mark all chunks for this guideline as archived — removed from retrieval."""
        self.vdb.update_payload(
            filter={"guideline_id": guideline_id},
            payload={"status": "archived"}
        )
        print(f"  Archived guideline: {guideline_id}")

    def check_expired(self) -> list[str]:
        """Return guidelines past their expiry date still marked active."""
        today = datetime.utcnow().isoformat()[:10]
        return [g.guideline_id for g in self.meta.get_active_guidelines()
                if g.expiry_date and g.expiry_date < today]

    def _chunk_guideline(self, content: str, guideline: ClinicalGuideline) -> list[dict]:
        # Clinical guidelines: chunk at section level, preserve recommendation boxes
        import re
        sections = re.split(r'\n(?=\d+\.\s+[A-Z])', content)
        return [{"id": f"{guideline.guideline_id}_v{guideline.version}_{i}",
                 "content": sec, "metadata": {"specialty": guideline.specialty}}
                for i, sec in enumerate(sections) if len(sec.strip()) > 100]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]  # Placeholder
```

---

### 5.3 Safety-First Generation Design

```python
import re

class ClinicalRAGPipeline:
    """
    Safety-first clinical RAG with:
    - Patient PII detection (block queries about specific patients)
    - Dosing question detection (require specialist escalation)
    - Mandatory evidence grade and disclaimer in every answer
    - Active-only guideline retrieval
    """
    DOSING_PATTERNS = [
        r'\b(\d+\s*mg|\d+\s*ml|\d+\s*mcg)\b',
        r'\b(dose|dosing|dosage)\s+for\b',
        r'\bhow\s+much\s+to\s+(give|administer|prescribe)\b',
    ]
    PATIENT_PATTERNS = [
        r'\bpatient\s+[A-Z][a-z]+\b',
        r'\bMr\.\s*[A-Z]\b',
        r'\bNHS\s+number\b',
        r'\bDOB\b|\bdate\s+of\s+birth\b',
    ]

    def __init__(self, retriever, llm_gateway, audit_logger):
        self.retriever    = retriever
        self.gateway      = llm_gateway
        self.audit        = audit_logger
        self._dosing_re   = [re.compile(p, re.IGNORECASE) for p in self.DOSING_PATTERNS]
        self._patient_re  = [re.compile(p, re.IGNORECASE) for p in self.PATIENT_PATTERNS]

    def query(self, staff_id: str, question: str, specialty: str = None) -> dict:
        # Safety check 1: specific patient data
        if any(p.search(question) for p in self._patient_re):
            return self._safety_refusal("PATIENT_PII_QUERY",
                "Questions about individual patients cannot be processed. "
                "Please consult the patient's record system directly.")

        # Safety check 2: specific dosing request
        is_dosing = any(p.search(question) for p in self._dosing_re)

        # Retrieve from ACTIVE guidelines only
        results = self.retriever.search(
            question,
            filter={"status": "active",
                    **({"specialty": specialty} if specialty else {})},
            top_k=5
        )

        if not results or max(r.get("score", 0) for r in results) < 0.60:
            return self._safety_refusal("LOW_CONFIDENCE",
                "The question falls outside available clinical guidelines. "
                "Please consult a senior clinician or specialist.")

        # Build context with evidence grades visible
        context = "\n".join(
            f"[{i+1}] {r['metadata'].get('issuing_body','?')} Guideline "
            f"v{r['metadata'].get('version','?')} "
            f"(Evidence Grade: {r['metadata'].get('evidence_grade','?')}) — "
            f"{r['content']}"
            for i, r in enumerate(results)
        )

        dosing_instruction = (
            "\n\nIMPORTANT: This question relates to dosing. "
            "Do NOT provide specific doses. State that dosing must be confirmed "
            "with a pharmacist or prescriber per local protocol."
        ) if is_dosing else ""

        messages = [
            {"role": "system", "content":
             "You are a clinical reference assistant. "
             "Answer ONLY using information from the provided clinical guidelines. "
             "Every answer MUST include: (1) source guideline and version cited as [N], "
             "(2) evidence grade from the source, "
             "(3) the mandatory disclaimer at the end. "
             "If the guideline is ambiguous or the answer is unclear, say so explicitly."
             + dosing_instruction},
            {"role": "user",
             "content": (f"Clinical guidelines context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Required ending: {CLINICAL_CONSTRAINTS['mandatory_disclaimer']}")}
        ]
        response = self.gateway.complete(messages, model_alias="default")
        self.audit.log_query(staff_id, question, [r["id"] for r in results])
        return {
            "answer":     response["answer"],
            "sources":    results[:3],
            "is_dosing":  is_dosing,
            "confidence": "high" if results[0].get("score", 0) > 0.80 else "medium"
        }

    def _safety_refusal(self, reason: str, message: str) -> dict:
        return {"answer": message, "safety_refusal": reason,
                "sources": [], "confidence": "refused"}
```

---

### 5.4 Evaluation for Clinical Settings

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClinicalEvalCase:
    case_id: str
    question: str
    expected_guideline_id: str   # Which guideline should be retrieved
    expected_evidence_grade: str
    is_in_scope: bool            # Should system answer or refuse?
    is_dosing_question: bool
    contains_patient_pii: bool
    clinical_reviewer_approved: bool   # Human expert approved this test case
    gold_standard_answer: Optional[str] = None

CLINICAL_EVAL_METRICS = {
    "guideline_recall":         "Correct guideline retrieved in top-3 (%)",
    "evidence_grade_cited":     "Response cites correct evidence grade (%)",
    "disclaimer_present":       "Mandatory disclaimer present in response (%)",
    "refusal_precision":        "Out-of-scope questions correctly refused (%)",
    "refusal_recall":           "All out-of-scope questions refused (%) — critical",
    "dosing_refusal_rate":      "Dosing questions refused or escalated (%)",
    "patient_pii_refusal":      "Patient PII queries refused (%) — must be 100%",
    "factual_accuracy":         "Clinical expert agreement with answer (%)",
    "hallucination_rate":       "Claims not in source guideline (%) — must be near 0",
}

class ClinicalEvaluator:
    def __init__(self, pipeline, eval_cases: list[ClinicalEvalCase]):
        self.pipeline = pipeline
        self.cases    = eval_cases

    def run(self) -> dict:
        results = {m: [] for m in CLINICAL_EVAL_METRICS}
        for case in self.cases:
            resp = self.pipeline.query("eval_runner", case.question)

            # Patient PII: must refuse (100% target)
            if case.contains_patient_pii:
                results["patient_pii_refusal"].append(
                    1 if resp.get("safety_refusal") else 0
                )
                continue

            # Out-of-scope: must refuse
            if not case.is_in_scope:
                results["refusal_recall"].append(
                    1 if resp.get("safety_refusal") else 0
                )
                continue

            # Dosing: must escalate or refuse dosing
            if case.is_dosing_question:
                results["dosing_refusal_rate"].append(
                    1 if resp.get("is_dosing") else 0
                )

            # Guideline retrieval
            source_ids = [s.get("id", "") for s in resp.get("sources", [])]
            guideline_retrieved = any(
                case.expected_guideline_id in sid for sid in source_ids[:3]
            )
            results["guideline_recall"].append(1 if guideline_retrieved else 0)

            # Disclaimer
            disclaimer_present = (
                CLINICAL_CONSTRAINTS["mandatory_disclaimer"][:50].lower()
                in resp.get("answer", "").lower()
            )
            results["disclaimer_present"].append(1 if disclaimer_present else 0)

        return {
            metric: round(sum(vals) / len(vals), 4) if vals else None
            for metric, vals in results.items()
        }
```

---

### 🧪 Hands-on Lab: End-to-End Case Study Simulator

**Objective:** Simulate key patterns from all five case studies in a unified runnable script, demonstrating: escalation logic, hybrid RRF fusion, compliance refusal, code query classification, and clinical safety checks.

```python
#!/usr/bin/env python3
"""
case_study_simulator.py — Patterns from all 5 case studies.
No external dependencies required.
"""
import re, math
from typing import Optional

print("=" * 60)
print("  Case Study Pattern Simulator")
print("=" * 60)

# ── Case 1: Support escalation classifier ─────────────────────
print("\n[CS-1] Customer Support — Escalation Classifier")
ESCALATION_TRIGGERS = ["billing dispute", "data breach", "legal", "refund",
                       "outage", "account suspension", "data loss"]
def should_escalate(question: str, top_score: float,
                    tier: str = "standard") -> tuple[bool, str]:
    for t in ESCALATION_TRIGGERS:
        if t in question.lower():
            return True, f"trigger:{t}"
    if top_score < 0.55:
        return True, f"low_confidence:{top_score:.2f}"
    if tier == "enterprise" and top_score < 0.70:
        return True, f"enterprise_low_confidence:{top_score:.2f}"
    return False, "ok"

support_cases = [
    ("How do I reset my password?",             0.91, "standard"),
    ("There is a data breach in my account",    0.88, "standard"),
    ("I want a refund for my subscription",     0.72, "professional"),
    ("How do I export data?",                   0.50, "standard"),
    ("What is the SLA for enterprise tier?",    0.65, "enterprise"),
]
for q, score, tier in support_cases:
    escalate, reason = should_escalate(q, score, tier)
    icon = "↑ ESCALATE" if escalate else "✓ ANSWER"
    print(f"  {icon}  [{tier}] {q[:45]} ({reason})")

# ── Case 2: Hybrid RRF Fusion ──────────────────────────────────
print("\n[CS-2] Internal KB — RRF Fusion Demo")
def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank + 1)

# Simulated dense and BM25 results
dense_results = [
    {"id": "doc_kubernetes_001", "score": 0.91},
    {"id": "doc_kubernetes_002", "score": 0.85},
    {"id": "doc_helm_003",       "score": 0.80},
]
bm25_results = [
    {"id": "doc_helm_003",       "score": 12.4},   # BM25 boosts exact match
    {"id": "doc_kubernetes_001", "score": 9.1},
    {"id": "doc_runbook_k8s",    "score": 8.7},
]
rrf: dict = {}
for rank, doc in enumerate(dense_results):
    rrf[doc["id"]] = rrf.get(doc["id"], 0) + rrf_score(rank)
for rank, doc in enumerate(bm25_results):
    rrf[doc["id"]] = rrf.get(doc["id"], 0) + rrf_score(rank)
fused = sorted(rrf.items(), key=lambda x: -x[1])
print(f"  {'Doc ID':<30} {'RRF Score'}")
for doc_id, score in fused:
    print(f"  {doc_id:<30} {score:.5f}")

# ── Case 3: Financial compliance guardrails ────────────────────
print("\n[CS-3] Financial RAG — Compliance Guardrails")
ADVICE_PATTERNS = [r'\b(buy|sell|recommend)\b.*\bstock\b',
                   r'\bprice\s+target\b', r'\bshould\s+i\s+(invest|buy)\b']
MNPI_PATTERNS   = [r'\bunpublished\b', r'\bnon-?public\b', r'\binside\s+info\b']

def financial_compliance_check(q: str) -> tuple[str, Optional[str]]:
    for p in ADVICE_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "REFUSED", "INVESTMENT_ADVICE"
    for p in MNPI_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "REFUSED", "MNPI_QUERY"
    return "ALLOWED", None

fin_queries = [
    "What was AAPL revenue in Q3 2024?",
    "Should I buy MSFT stock given their earnings?",
    "What are the disclosed risk factors for Tesla?",
    "What is the unpublished forecast for Q4?",
    "Compare Amazon and Alphabet operating margins in 2023",
]
for q in fin_queries:
    status, reason = financial_compliance_check(q)
    icon = "✗" if status == "REFUSED" else "✓"
    print(f"  {icon} {status:<8} {reason or '':<22} {q[:50]}")

# ── Case 4: Code query classification ─────────────────────────
print("\n[CS-4] Code Intelligence — Query Type Classification")
def classify_code_query(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["how do", "how to", "implement", "example"]):
        return "HOW_TO"
    if any(w in q for w in ["where", "used", "usages", "references"]):
        return "FIND_USAGE"
    if any(w in q for w in ["explain", "what does", "describe"]):
        return "EXPLAIN"
    if any(w in q for w in ["error", "bug", "fail", "exception"]):
        return "DEBUG"
    if any(w in q for w in ["refactor", "improve", "better"]):
        return "REFACTOR"
    return "HOW_TO"

code_queries = [
    "How do I implement pagination with cursor?",
    "Where is TokenBudget class used in the codebase?",
    "Explain what the retry_with_backoff function does",
    "Why does the CircuitBreaker fail after 5 errors?",
    "How should I refactor the FallbackChain class?",
]
for q in code_queries:
    qtype = classify_code_query(q)
    print(f"  [{qtype:<12}] {q}")

# ── Case 5: Clinical safety checks ────────────────────────────
print("\n[CS-5] Healthcare RAG — Clinical Safety Checks")
DOSING_PATTERNS  = [r'\b(\d+\s*mg|\d+\s*ml)', r'\b(dose|dosing)\s+for', r'how\s+much\s+to\s+give']
PATIENT_PATTERNS = [r'\bpatient\s+[A-Z][a-z]+', r'\bNHS\s+number\b', r'\bDOB\b']

def clinical_safety_check(q: str, top_score: float = 0.75) -> tuple[str, str]:
    for p in PATIENT_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "REFUSED", "PATIENT_PII"
    for p in DOSING_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "ESCALATE", "DOSING_QUESTION"
    if top_score < 0.60:
        return "REFUSED", "LOW_CONFIDENCE"
    return "ANSWER", "OK"

clinical_cases = [
    ("What is the NICE guideline for sepsis management?",    0.88),
    ("What dose of amoxicillin 500mg should I give?",        0.82),
    ("Patient John Smith has a rash — what is the cause?",   0.70),
    ("What are the contraindications for metformin?",         0.85),
    ("What is the evidence for CPAP in sleep apnoea?",        0.45),
]
for q, score in clinical_cases:
    status, reason = clinical_safety_check(q, score)
    icon = {"ANSWER": "✓", "ESCALATE": "⚠", "REFUSED": "✗"}[status]
    print(f"  {icon} {status:<8} [{reason:<18}] {q[:55]}")

print("\n" + "=" * 60)
print("  All 5 case study patterns demonstrated")
```

**Run the lab:**
```bash
python case_study_simulator.py
```

**Extensions:**
- Add a **Case 6**: multi-language RAG for a global product where the same question arrives in EN, ES, and FR — implement a language detection + query translation step before retrieval
- Extend the clinical safety check to also flag questions referencing drug names and output a "PHARMACIST_REVIEW" status instead of a hard refusal
- Add a metrics summary at the end of the simulator showing: escalation rate (CS-1), RRF rank improvement (CS-2), compliance refusal rate (CS-3)

---

> ### 📋 Chapter Summary
>
> - Clinical RAG requires three safety layers unique to healthcare: patient PII detection (100% refusal required), dosing question escalation, and active-only guideline retrieval with evidence grade citation.
> - **Clinical evaluation** treats `patient_pii_refusal` and `refusal_recall` as must-be-100% metrics — a single incorrect answer to an out-of-scope question is an unacceptable safety failure.
> - Version-controlled corpus management is critical in clinical settings: archived guidelines must be excluded from retrieval but retained for audit, and expired guidelines must be detected automatically.
> - The case study simulator validates all five domain-specific patterns — support escalation, hybrid fusion, financial compliance, code query routing, and clinical safety — in a single runnable lab.

---

> ### ❓ Comprehension Questions
>
> 1. `ClinicalRAGPipeline` refuses queries with `top_score < 0.60`. A nurse asks about a treatment not covered by the indexed NICE guidelines but covered by a WHO guideline not yet ingested. The system refuses. What process should exist for nurses to report gaps in corpus coverage, and how quickly should they be addressed?
> 2. `ClinicalEvaluator` tests `patient_pii_refusal` — but the test cases must be created by someone. Generating test cases with fake patient names requires careful handling to avoid them leaking into the knowledge base. What controls would you put around the clinical test case generation process?
> 3. The `mandatory_disclaimer` is appended to every answer. A clinician complains that it appears even on straightforward factual questions (e.g. "What is the definition of sepsis?"), reducing trust in the system. How would you preserve safety while reducing disclaimer fatigue?
> 4. `ClinicalCorpusManager.check_expired()` identifies guidelines past their expiry date. What should happen automatically versus requiring manual review, and who must approve removal of an expired guideline from the active index?
> 5. The financial RAG system stores `query_hash` instead of the full query for privacy. The clinical system stores `staff_id` alongside query content (for audit purposes). Justify this difference: why is it acceptable to retain clinical query content but not financial query content in plaintext?

---

## References

### Documentation
- [NICE Evidence Standards](https://www.nice.org.uk/standards-and-indicators) — Clinical guideline quality standards.
- [MiFID II Article 25](https://www.esma.europa.eu/regulation/post-trading/mifid-ii-and-mifir) — Investment suitability requirements.
- [Python `ast` module](https://docs.python.org/3/library/ast.html) — AST parsing for code chunking.
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection for healthcare data.

### Papers
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Cormack et al., 2009.
- [RAG for Clinical Decision Support](https://arxiv.org/abs/2402.01030) — Clinical RAG evaluation challenges, 2024.

---

> **Navigation**
> [← Part XVII — Reference Architectures](part_17_reference_architectures.md) | [→ Part XIX — The Future](part_19_future.md)

---
[« Back to case_studies Index](index.md) | [🏠 Home](../../index.md)