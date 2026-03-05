## Chapter 3 — Responsible AI Engineering

### 3.1 Fairness and Bias Detection

```python
from dataclasses import dataclass
from typing import Optional
import statistics

@dataclass
class FairnessEvaluation:
    """
    Evaluates a RAG system for bias across demographic or categorical subgroups.
    Bias in RAG systems manifests as: different refusal rates by topic,
    different retrieval quality for different language registers,
    or inconsistent citation of sources by origin.
    """
    system_id: str
    eval_date: str
    subgroup_metric: str    # e.g. "query_language" | "topic_category" | "user_role"
    subgroups: dict[str, dict]  # subgroup_name → {metric: value}

    def compute_disparate_impact(
        self, metric_name: str, privileged_group: str
    ) -> dict:
        """
        Compute disparate impact ratio: min_group_score / privileged_group_score.
        Ratio < 0.8 indicates potential adverse impact (80% rule).
        """
        privileged_score = self.subgroups.get(privileged_group, {}).get(metric_name)
        if not privileged_score:
            return {"error": "Privileged group not found"}

        ratios = {}
        for group, metrics in self.subgroups.items():
            score = metrics.get(metric_name)
            if score and privileged_score > 0:
                ratio = score / privileged_score
                ratios[group] = {
                    "score": round(score, 4),
                    "ratio": round(ratio, 4),
                    "flag": "POTENTIAL_BIAS" if ratio < 0.80 else "OK"
                }
        return ratios

    def summary(self) -> dict:
        all_metrics = set()
        for metrics in self.subgroups.values():
            all_metrics.update(metrics.keys())

        result = {}
        for metric in all_metrics:
            scores = [v[metric] for v in self.subgroups.values()
                      if metric in v and v[metric] is not None]
            if len(scores) >= 2:
                result[metric] = {
                    "mean": round(statistics.mean(scores), 4),
                    "min":  round(min(scores), 4),
                    "max":  round(max(scores), 4),
                    "range": round(max(scores) - min(scores), 4),
                    "flag": "REVIEW" if (max(scores) - min(scores)) > 0.10 else "OK"
                }
        return result


class BiasMonitor:
    """
    Continuous bias monitoring: detects performance drift across
    query categories in production traffic.
    """
    def __init__(self, categories: list[str], window_size: int = 1000):
        self.categories = categories
        self.window_size = window_size
        self._buffers: dict[str, list] = {c: [] for c in categories}

    def record(self, category: str, metrics: dict):
        if category not in self._buffers:
            self._buffers[category] = []
        self._buffers[category].append(metrics)
        if len(self._buffers[category]) > self.window_size:
            self._buffers[category].pop(0)

    def detect_disparity(self, metric_name: str,
                         threshold: float = 0.10) -> list[dict]:
        """Flag categories with metric value more than threshold below the best."""
        scores = {}
        for cat, records in self._buffers.items():
            vals = [r[metric_name] for r in records
                    if metric_name in r and r[metric_name] is not None]
            if vals:
                scores[cat] = statistics.mean(vals)

        if not scores:
            return []

        best = max(scores.values())
        flagged = []
        for cat, score in scores.items():
            gap = best - score
            if gap > threshold:
                flagged.append({
                    "category": cat,
                    "score": round(score, 4),
                    "best_score": round(best, 4),
                    "gap": round(gap, 4),
                    "severity": "HIGH" if gap > 0.20 else "MEDIUM"
                })
        return sorted(flagged, key=lambda x: -x["gap"])
```

---

### 3.2 Transparency and Explainability

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExplainableResponse:
    """
    Wraps a RAG response with transparency metadata.
    Provides users and auditors with a full account of how
    the answer was constructed.
    """
    answer: str
    is_refusal: bool

    # Source transparency
    cited_sources: list[dict]   # [{id, title, excerpt, relevance_score}]
    source_count_retrieved: int
    source_count_used: int

    # Process transparency
    retrieval_model: str        # e.g. "all-MiniLM-L6-v2"
    generation_model: str       # e.g. "gpt-4o-mini"
    prompt_template_id: str
    prompt_version: str

    # Confidence indicators
    min_source_relevance: float
    max_source_relevance: float
    answer_grounded: bool       # All key claims traceable to cited sources

    # Uncertainty signals
    hedging_phrases: list[str]  # Phrases like "may", "typically", "I believe"
    refusal_triggered_by: Optional[str] = None  # Which rule triggered refusal

    def to_user_explanation(self) -> str:
        """Human-readable explanation of how the answer was generated."""
        if self.is_refusal:
            return (f"I was unable to answer this question because the relevant "
                    f"information was not found in my knowledge base "
                    f"({self.source_count_retrieved} sources retrieved, "
                    f"none sufficiently relevant).")

        sources_text = "\n".join(
            f"  [{i+1}] {s['title']} (relevance: {s['relevance_score']:.2f})"
            for i, s in enumerate(self.cited_sources)
        )
        return (
            f"This answer was generated using {self.generation_model} based on "
            f"{self.source_count_used} of {self.source_count_retrieved} retrieved sources.\n\n"
            f"Sources used:\n{sources_text}\n\n"
            f"Retrieval model: {self.retrieval_model}\n"
            f"Prompt template: {self.prompt_template_id} v{self.prompt_version}"
        )
```

---

### 3.3 Human-in-the-Loop Controls

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

class ReviewTrigger(str, Enum):
    LOW_CONFIDENCE     = "low_confidence"      # Source relevance below threshold
    HIGH_STAKES_TOPIC  = "high_stakes_topic"   # Medical, legal, financial advice
    NOVEL_QUERY        = "novel_query"          # No close match in recent queries
    USER_DISPUTE       = "user_dispute"         # User flagged answer as wrong
    REGULATORY_TOPIC   = "regulatory_topic"     # Compliance-sensitive question
    LOW_USER_RATING    = "low_user_rating"      # User rated 1–2/5

@dataclass
class HumanReviewRequest:
    request_id: str
    trigger: ReviewTrigger
    query: str
    generated_answer: str
    retrieved_sources: list[dict]
    confidence_score: float
    urgency: str            # "low" | "medium" | "high"
    assignee_role: str      # Role required to review
    created_at: str
    reviewed: bool = False
    reviewer_id: Optional[str] = None
    approved: Optional[bool] = None
    corrected_answer: Optional[str] = None
    review_notes: Optional[str] = None

class HumanInTheLoopGateway:
    """
    Routes RAG responses through human review when configured triggers fire.
    In async mode: returns a pending response and notifies reviewer.
    In sync mode: blocks until human reviews (not suitable for interactive use).
    """
    def __init__(
        self,
        review_queue,            # Queue (Redis, SQS) for review requests
        high_stakes_topics: list[str] = None,
        min_confidence_threshold: float = 0.60
    ):
        self.queue = review_queue
        self.high_stakes = set(high_stakes_topics or [
            "medical", "legal", "financial", "regulatory",
            "privacy", "discrimination", "employment"
        ])
        self.min_confidence = min_confidence_threshold

    def should_review(
        self,
        query: str,
        answer: str,
        confidence: float,
        topic: Optional[str] = None
    ) -> tuple[bool, ReviewTrigger]:
        """Determine if this response requires human review."""
        if confidence < self.min_confidence:
            return True, ReviewTrigger.LOW_CONFIDENCE
        if topic and topic.lower() in self.high_stakes:
            return True, ReviewTrigger.HIGH_STAKES_TOPIC
        # Check for disclaimer-laden answers (model expressing uncertainty)
        hedges = ["consult a professional", "seek legal advice",
                  "speak to a doctor", "i cannot provide financial advice"]
        if any(h in answer.lower() for h in hedges):
            return True, ReviewTrigger.HIGH_STAKES_TOPIC
        return False, None

    def submit_for_review(
        self,
        trigger: ReviewTrigger,
        query: str,
        answer: str,
        sources: list[dict],
        confidence: float
    ) -> str:
        import uuid
        from datetime import datetime
        request_id = uuid.uuid4().hex
        review = HumanReviewRequest(
            request_id=request_id,
            trigger=trigger,
            query=query,
            generated_answer=answer,
            retrieved_sources=sources,
            confidence_score=confidence,
            urgency="high" if trigger == ReviewTrigger.HIGH_STAKES_TOPIC else "medium",
            assignee_role="domain_expert",
            created_at=datetime.utcnow().isoformat()
        )
        self.queue.push(review)
        return request_id
```

---

### 3.4 Incident Management for AI Systems

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

class AIIncidentType(str, Enum):
    HALLUCINATION      = "hallucination"        # Factually wrong answer presented as fact
    HARMFUL_CONTENT    = "harmful_content"       # Output that could cause harm
    BIAS_INCIDENT      = "bias_incident"         # Discriminatory or unfair output
    PII_DISCLOSURE     = "pii_disclosure"        # PII in response
    PROMPT_INJECTION   = "prompt_injection"      # Successful injection attack
    DATA_POISONING     = "data_poisoning"        # Malicious content in corpus
    QUALITY_REGRESSION = "quality_regression"    # Systematic quality drop
    COST_OVERRUN       = "cost_overrun"          # Runaway token spend

@dataclass
class AIIncident:
    incident_id: str
    incident_type: AIIncidentType
    severity: str           # "critical" | "high" | "medium" | "low"
    title: str
    description: str
    detected_at: str
    reported_by: str
    affected_system_id: str
    affected_users_count: Optional[int] = None
    sample_request_id: Optional[str] = None   # Example request that triggered it
    root_cause: Optional[str] = None
    remediation_steps: list[str] = field(default_factory=list)
    resolved_at: Optional[str] = None
    status: str = "open"                       # open | investigating | resolved | closed
    # Post-incident
    lessons_learned: Optional[str] = None
    preventive_measures: list[str] = field(default_factory=list)
    governance_notified: bool = False
    regulatory_notification_required: bool = False

class AIIncidentManager:
    def __init__(self, store_path: str = "governance/incidents"):
        from pathlib import Path
        self.store = Path(store_path)
        self.store.mkdir(parents=True, exist_ok=True)
        self._incidents: dict[str, AIIncident] = {}

    def open_incident(self, incident: AIIncident) -> str:
        self._incidents[incident.incident_id] = incident
        self._persist(incident)
        # Critical and high incidents trigger immediate notification
        if incident.severity in ("critical", "high"):
            self._notify_governance(incident)
        return incident.incident_id

    def resolve(self, incident_id: str, root_cause: str,
                lessons_learned: str, preventive_measures: list[str]):
        incident = self._incidents.get(incident_id)
        if not incident:
            raise KeyError(f"Incident {incident_id} not found")
        incident.root_cause = root_cause
        incident.lessons_learned = lessons_learned
        incident.preventive_measures = preventive_measures
        incident.resolved_at = datetime.utcnow().isoformat()
        incident.status = "resolved"
        self._persist(incident)

    def _notify_governance(self, incident: AIIncident):
        print(f"[GOVERNANCE ALERT] {incident.severity.upper()} incident: "
              f"{incident.title} ({incident.incident_type.value})")

    def _persist(self, incident: AIIncident):
        import json, dataclasses
        path = self.store / f"{incident.incident_id}.json"
        path.write_text(json.dumps(dataclasses.asdict(incident), indent=2))
```

---

> ### 📋 Chapter Summary
>
> - **Fairness evaluation** measures performance disparities across subgroups. A disparate impact ratio < 0.80 flags potential bias requiring investigation.
> - **Explainability** wraps every response with provenance metadata: which sources were used, which models, and which prompt template version — satisfying transparency obligations.
> - **Human-in-the-loop controls** route low-confidence, high-stakes, or flagged responses to human review before (or after) they reach the user.
> - **AI incident management** classifies incidents by type (hallucination, PII disclosure, prompt injection, bias) and severity, with mandatory governance notification for critical/high incidents.

---

> ### ❓ Comprehension Questions
>
> 1. `BiasMonitor.detect_disparity` flags categories more than 0.10 below the best-performing category. At 1000 QPS with 8 categories, the buffer holds 125 records per category. Is this statistically sufficient to detect a 0.10 gap with 95% confidence? What sample size would you require?
> 2. `ExplainableResponse.answer_grounded` is a boolean. What process would you use to compute this field — how do you determine whether every key claim in the answer is traceable to a cited source?
> 3. `HumanInTheLoopGateway` detects hedge phrases in the answer text. A compliant, well-calibrated answer about medical topics appropriately says "consult a professional". This triggers a review. Is this a false positive, and should it be? Argue the trade-off.
> 4. An `AIIncident` of type `HALLUCINATION` is opened. What `root_cause` categories would you investigate, and what `preventive_measures` address each root cause?
> 5. `GovernanceGate.blocking=True` means work cannot proceed without approval. A critical security patch needs deploying immediately — the security engineer is on holiday and cannot approve GATE-03. Design an emergency bypass process that is both rapid and governance-compliant.

---

## References

### Documentation
- [AI Fairness 360 (IBM)](https://aif360.readthedocs.io/en/stable/) — Fairness metrics toolkit.
- [Responsible AI Practices (Google)](https://ai.google/responsibility/responsible-ai-practices/)
- [Microsoft Responsible AI Standards](https://blogs.microsoft.com/on-the-issues/2022/06/21/microsofts-framework-for-building-ai-systems-responsibly/)

### Papers
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Mitchell et al., 2019.
- [Fairness and Abstraction in Sociotechnical Systems](https://dl.acm.org/doi/10.1145/3287560.3287598) — Selbst et al., 2019.

---

---
[« Back to governance Index](index.md) | [🏠 Home](../../index.md)