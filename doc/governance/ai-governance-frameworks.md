## Chapter 1 — AI Governance Frameworks

### 1.1 Why AI Governance Matters

Governance is the set of policies, processes, and controls that ensure an AI system is developed, deployed, and operated responsibly. Without governance, engineering teams make locally rational decisions that collectively produce systems no one consciously chose to build — systems that are biased, opaque, unaccountable, or legally non-compliant.

Governance is not bureaucracy. It is the mechanism by which engineering work becomes auditable, reversible, and trustworthy. Three forces make governance non-negotiable for enterprise AI:

**Regulation.** The EU AI Act (effective 2024–2026), the US Executive Order on AI (2023), GDPR, and sector-specific regulations (HIPAA, FCA, MiFID II) impose concrete engineering requirements: impact assessments, logging, human oversight, accuracy documentation, and conformity declarations.

**Liability.** When an AI system causes harm — a wrong medical recommendation, a biased hiring decision, a hallucinated legal answer — someone is liable. Governance creates the paper trail that determines accountability and enables remediation.

**Trust.** Enterprise customers and regulators require evidence that AI systems behave as claimed. Governance produces that evidence: model cards, data lineage, evaluation reports, incident records.

---

### 1.2 The EU AI Act — Engineering Implications

The [EU AI Act](https://artificialintelligenceact.eu) is the most comprehensive AI regulation globally. It categorises AI systems by risk level and imposes proportionate obligations. For engineers building RAG and LLM systems, the key implications are:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class EUAIActRiskTier(str, Enum):
    UNACCEPTABLE = "unacceptable"   # Banned: social scoring, real-time biometric surveillance
    HIGH_RISK    = "high_risk"      # Strict obligations: hiring, credit, medical, law enforcement
    LIMITED_RISK = "limited_risk"   # Transparency obligations: chatbots must disclose AI nature
    MINIMAL_RISK = "minimal_risk"   # No specific obligations: spam filters, AI games

@dataclass
class EUAIActObligation:
    risk_tier: EUAIActRiskTier
    obligation: str
    engineering_requirement: str
    deadline: str

EU_AI_ACT_OBLIGATIONS = [
    EUAIActObligation(
        EUAIActRiskTier.HIGH_RISK,
        "Risk management system",
        "Implement and maintain a documented risk management process covering "
        "identification, analysis, estimation, and mitigation of AI risks",
        "August 2026"
    ),
    EUAIActObligation(
        EUAIActRiskTier.HIGH_RISK,
        "Data governance",
        "Document training/validation datasets: origin, collection method, "
        "preprocessing, known limitations, biases present",
        "August 2026"
    ),
    EUAIActObligation(
        EUAIActRiskTier.HIGH_RISK,
        "Technical documentation",
        "Maintain technical docs covering: system purpose, architecture, "
        "training data, performance metrics, limitations, intended use",
        "August 2026"
    ),
    EUAIActObligation(
        EUAIActRiskTier.HIGH_RISK,
        "Logging and audit trail",
        "Automatic logging of all inputs and outputs enabling post-hoc reconstruction "
        "of decisions for at least 6 months",
        "August 2026"
    ),
    EUAIActObligation(
        EUAIActRiskTier.HIGH_RISK,
        "Human oversight",
        "Implement technical controls enabling humans to monitor, override, "
        "and shut down the AI system",
        "August 2026"
    ),
    EUAIActObligation(
        EUAIActRiskTier.HIGH_RISK,
        "Accuracy and robustness",
        "Document accuracy metrics and demonstrate robustness against "
        "reasonably foreseeable adversarial inputs",
        "August 2026"
    ),
    EUAIActObligation(
        EUAIActRiskTier.LIMITED_RISK,
        "Transparency disclosure",
        "Inform users they are interacting with an AI system. "
        "Chatbots and synthetic content generators must disclose AI nature.",
        "August 2026"
    ),
    EUAIActObligation(
        EUAIActRiskTier.HIGH_RISK,
        "Conformity assessment",
        "Conduct and document conformity assessment before deployment. "
        "Register in the EU database for high-risk AI systems.",
        "August 2026"
    ),
]

def classify_rag_system(use_case: str, sector: str) -> EUAIActRiskTier:
    """
    Classify a RAG system by EU AI Act risk tier based on use case and sector.
    Conservative classification — when in doubt, classify higher.
    """
    high_risk_sectors = {
        "hiring", "recruitment", "employment",
        "credit_scoring", "lending", "insurance",
        "medical", "healthcare", "clinical",
        "law_enforcement", "border_control",
        "education_assessment", "vocational_training",
        "critical_infrastructure"
    }
    high_risk_use_cases = {
        "determine_eligibility", "rank_candidates",
        "assess_creditworthiness", "evaluate_students",
        "predict_recidivism", "triage_patients"
    }
    if sector.lower() in high_risk_sectors:
        return EUAIActRiskTier.HIGH_RISK
    if use_case.lower() in high_risk_use_cases:
        return EUAIActRiskTier.HIGH_RISK
    if "chatbot" in use_case.lower() or "assistant" in use_case.lower():
        return EUAIActRiskTier.LIMITED_RISK
    return EUAIActRiskTier.MINIMAL_RISK
```

---

### 1.3 Internal Governance Structures

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class GovernanceRole(str, Enum):
    AI_PRODUCT_OWNER    = "ai_product_owner"
    DATA_STEWARD        = "data_steward"
    ML_ENGINEER         = "ml_engineer"
    LEGAL_COMPLIANCE    = "legal_compliance"
    PRIVACY_OFFICER     = "privacy_officer"
    SECURITY_ENGINEER   = "security_engineer"
    ETHICS_REVIEWER     = "ethics_reviewer"

@dataclass
class GovernanceGate:
    """A mandatory review point in the AI system lifecycle."""
    gate_id: str
    name: str
    lifecycle_stage: str  # design | development | pre_launch | post_launch
    required_approvers: list[GovernanceRole]
    required_artefacts: list[str]
    blocking: bool        # If True, work cannot proceed without approval

AI_GOVERNANCE_GATES = [
    GovernanceGate(
        "GATE-01", "Use Case Approval",
        lifecycle_stage="design",
        required_approvers=[
            GovernanceRole.AI_PRODUCT_OWNER,
            GovernanceRole.LEGAL_COMPLIANCE,
            GovernanceRole.PRIVACY_OFFICER,
        ],
        required_artefacts=[
            "use_case_description.md",
            "eu_ai_act_classification.md",
            "data_processing_impact_assessment.md",
        ],
        blocking=True
    ),
    GovernanceGate(
        "GATE-02", "Data Governance Review",
        lifecycle_stage="development",
        required_approvers=[
            GovernanceRole.DATA_STEWARD,
            GovernanceRole.PRIVACY_OFFICER,
        ],
        required_artefacts=[
            "corpus_provenance_report.md",
            "pii_scan_report.json",
            "data_retention_policy.md",
            "consent_records.csv",
        ],
        blocking=True
    ),
    GovernanceGate(
        "GATE-03", "Security and Ethics Review",
        lifecycle_stage="pre_launch",
        required_approvers=[
            GovernanceRole.SECURITY_ENGINEER,
            GovernanceRole.ETHICS_REVIEWER,
        ],
        required_artefacts=[
            "threat_model.md",
            "adversarial_test_report.json",
            "bias_evaluation_report.md",
            "model_card.md",
        ],
        blocking=True
    ),
    GovernanceGate(
        "GATE-04", "Production Launch Approval",
        lifecycle_stage="pre_launch",
        required_approvers=[
            GovernanceRole.AI_PRODUCT_OWNER,
            GovernanceRole.LEGAL_COMPLIANCE,
            GovernanceRole.SECURITY_ENGINEER,
        ],
        required_artefacts=[
            "evaluation_report.json",
            "conformity_assessment.md",
            "incident_response_plan.md",
            "human_oversight_documentation.md",
        ],
        blocking=True
    ),
    GovernanceGate(
        "GATE-05", "Post-Launch Monitoring Review",
        lifecycle_stage="post_launch",
        required_approvers=[GovernanceRole.AI_PRODUCT_OWNER],
        required_artefacts=[
            "monthly_quality_report.md",
            "bias_drift_report.md",
            "incident_log.md",
        ],
        blocking=False   # Advisory — does not block operation
    ),
]
```

---

### 1.4 Risk Tiers and Classification

```python
@dataclass
class AISystemRiskProfile:
    """Internal risk classification for an AI system."""
    system_id: str
    name: str
    use_case: str
    sector: str
    eu_ai_act_tier: EUAIActRiskTier
    handles_pii: bool
    autonomous_decisions: bool    # Makes decisions without human review
    affects_individuals: bool     # Output affects individual people's rights/interests
    internal_risk_tier: str       # "critical" | "high" | "medium" | "low"
    review_frequency_days: int    # How often governance review is required

def compute_internal_risk_tier(profile: AISystemRiskProfile) -> str:
    score = 0
    if profile.eu_ai_act_tier == EUAIActRiskTier.HIGH_RISK:
        score += 4
    elif profile.eu_ai_act_tier == EUAIActRiskTier.LIMITED_RISK:
        score += 2
    if profile.handles_pii:
        score += 2
    if profile.autonomous_decisions:
        score += 3
    if profile.affects_individuals:
        score += 2

    if score >= 8:
        return "critical"
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"

REVIEW_FREQUENCY_BY_TIER = {
    "critical": 30,   # Monthly review
    "high":     90,   # Quarterly
    "medium":  180,   # Biannual
    "low":     365,   # Annual
}
```

---

> ### 📋 Chapter Summary
>
> - AI governance addresses regulatory compliance (EU AI Act, GDPR), liability management, and stakeholder trust through documented processes and controls.
> - The **EU AI Act** classifies AI systems into four risk tiers. High-risk systems (hiring, medical, credit) face strict obligations: risk management, data documentation, 6-month audit logs, human oversight, and conformity assessment.
> - **Governance gates** are mandatory review checkpoints at lifecycle stages (design → development → pre-launch → post-launch) with required approvers and artefacts.
> - Internal risk tiering combines EU Act classification, PII handling, autonomy, and individual impact into a scoring model that determines review frequency.

---

> ### ❓ Comprehension Questions
>
> 1. A legal team deploys a RAG system that summarises contracts and highlights risk clauses for human lawyers to review. The lawyers make all final decisions. Is this a high-risk system under the EU AI Act? Argue both classifications.
> 2. `GATE-03` requires an `ethics_reviewer` as an approver. In a 20-person startup with no dedicated ethics role, who should fill this function, and what minimum competence is required?
> 3. `compute_internal_risk_tier` assigns `autonomous_decisions` a weight of 3 — the highest single factor. Justify this weighting against the alternative of weighting `handles_pii` equally.
> 4. The EU AI Act requires 6-month audit log retention. The system processes 500 RPS. Estimate the daily log volume in GB at the `RAGRequestTrace` schema size (~2KB per record), and design a tiered storage strategy satisfying the retention requirement cost-effectively.
> 5. A system is classified as LIMITED_RISK because it is a customer support chatbot. The product team adds a feature that automatically routes escalated conversations to human agents based on LLM sentiment analysis. Does this change the risk classification, and what new obligations does it trigger?

---

## References

### Documentation
- [EU AI Act Official Text](https://artificialintelligenceact.eu)
- [NIST AI Risk Management Framework](https://www.nist.gov/system/files/documents/2023/01/26/NIST.AI.100-1.pdf)
- [UK ICO AI Guidance](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/)
- [EU AI Act Compliance Checker](https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/)

### Books
- *Atlas of AI* — Kate Crawford (Yale University Press). The societal stakes of AI governance.

---

---
[« Back to governance Index](index.md) | [🏠 Home](../index.md)