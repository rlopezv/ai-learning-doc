# Part XV — Governance

---

> **Navigation**
> [← Part XIV — Security](part_14_security.md) | [→ Part XVI — Operations](part_16_operations.md)

---

## Contents

- [Chapter 1 — AI Governance Frameworks](#chapter-1--ai-governance-frameworks)
  - [1.1 Why AI Governance Matters](#11-why-ai-governance-matters)
  - [1.2 The EU AI Act — Engineering Implications](#12-the-eu-ai-act--engineering-implications)
  - [1.3 Internal Governance Structures](#13-internal-governance-structures)
  - [1.4 Risk Tiers and Classification](#14-risk-tiers-and-classification)
- [Chapter 2 — Model and Data Lifecycle Governance](#chapter-2--model-and-data-lifecycle-governance)
  - [2.1 Model Cards and Documentation](#21-model-cards-and-documentation)
  - [2.2 Data Lineage Tracking](#22-data-lineage-tracking)
  - [2.3 Corpus Provenance and Consent](#23-corpus-provenance-and-consent)
  - [2.4 Versioned Governance Artefacts](#24-versioned-governance-artefacts)
- [Chapter 3 — Responsible AI Engineering](#chapter-3--responsible-ai-engineering)
  - [3.1 Fairness and Bias Detection](#31-fairness-and-bias-detection)
  - [3.2 Transparency and Explainability](#32-transparency-and-explainability)
  - [3.3 Human-in-the-Loop Controls](#33-human-in-the-loop-controls)
  - [3.4 Incident Management for AI Systems](#34-incident-management-for-ai-systems)
- [Chapter 4 — Compliance Automation 🧪](#chapter-4--compliance-automation-)
  - [4.1 Policy as Code](#41-policy-as-code)
  - [4.2 Automated Compliance Checks in CI/CD](#42-automated-compliance-checks-in-cicd)
  - [4.3 Evidence Collection and Audit Readiness](#43-evidence-collection-and-audit-readiness)
  - [🧪 Hands-on Lab: Governance Scorecard](#-hands-on-lab-governance-scorecard)

---

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

## Chapter 2 — Model and Data Lifecycle Governance

### 2.1 Model Cards and Documentation

A **model card** is a structured document that describes an AI system's purpose, capabilities, limitations, and evaluation results. Originally proposed by [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993), model cards have become the standard artefact for AI transparency.

```python
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from datetime import datetime

@dataclass
class ModelCard:
    """
    Structured model card following Mitchell et al. (2019) and
    EU AI Act Technical Documentation requirements.
    """
    # Identity
    system_id: str
    system_name: str
    version: str
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    authors: list[str] = field(default_factory=list)
    contact: str = ""

    # Intended use
    intended_use: str = ""
    intended_users: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)
    primary_use_case: str = ""
    sector: str = ""

    # Model details
    model_type: str = ""          # e.g. "RAG pipeline", "classification"
    llm_provider: str = ""        # e.g. "OpenAI GPT-4o"
    llm_version: str = ""
    embedding_model: str = ""
    framework_versions: dict = field(default_factory=dict)

    # Training and corpus data
    corpus_description: str = ""
    corpus_size_documents: int = 0
    corpus_update_frequency: str = ""
    corpus_languages: list[str] = field(default_factory=list)
    corpus_sources: list[str] = field(default_factory=list)
    data_collection_date_range: str = ""
    known_data_limitations: list[str] = field(default_factory=list)

    # Performance
    evaluation_datasets: list[str] = field(default_factory=list)
    evaluation_metrics: dict = field(default_factory=dict)
    # e.g. {"faithfulness": 0.91, "relevancy": 0.87, "recall_at_5": 0.84}
    performance_by_subgroup: dict = field(default_factory=dict)

    # Limitations and risks
    known_limitations: list[str] = field(default_factory=list)
    known_biases: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    mitigation_measures: list[str] = field(default_factory=list)

    # Governance
    eu_ai_act_tier: str = "minimal_risk"
    human_oversight_mechanisms: list[str] = field(default_factory=list)
    audit_log_retention_days: int = 180
    last_review_date: str = ""
    next_review_date: str = ""
    approvals: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render model card as human-readable Markdown."""
        lines = [
            f"# Model Card: {self.system_name} v{self.version}",
            f"*Generated: {self.created_at}*",
            "",
            "## Intended Use",
            f"**Primary use case:** {self.primary_use_case}",
            f"**Intended users:** {', '.join(self.intended_users)}",
            "",
            "**Out-of-scope uses:**",
        ] + [f"- {u}" for u in self.out_of_scope_uses] + [
            "",
            "## Model Details",
            f"- LLM: {self.llm_provider} ({self.llm_version})",
            f"- Embedding model: {self.embedding_model}",
            "",
            "## Corpus",
            f"- Size: {self.corpus_size_documents:,} documents",
            f"- Languages: {', '.join(self.corpus_languages)}",
            f"- Update frequency: {self.corpus_update_frequency}",
            "",
            "**Known data limitations:**",
        ] + [f"- {l}" for l in self.known_data_limitations] + [
            "",
            "## Performance",
        ] + [f"- {k}: {v}" for k, v in self.evaluation_metrics.items()] + [
            "",
            "## Limitations",
        ] + [f"- {l}" for l in self.known_limitations] + [
            "",
            "## Governance",
            f"- EU AI Act tier: {self.eu_ai_act_tier}",
            f"- Audit log retention: {self.audit_log_retention_days} days",
            f"- Next review: {self.next_review_date}",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str):
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown())
        (p.parent / (p.stem + ".json")).write_text(self.to_json())
```

---

### 2.2 Data Lineage Tracking

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid

@dataclass
class DataLineageNode:
    """A node in the data lineage graph."""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    node_type: str = ""   # "source" | "transform" | "index" | "output"
    name: str = ""
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class CorpusLineageRecord:
    """
    Complete lineage record for a document corpus.
    Tracks every transformation from source to vector index.
    """
    corpus_id: str
    corpus_version: str
    lineage_nodes: list[DataLineageNode] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_source(self, name: str, url: str, access_date: str,
                   license: str, record_count: int) -> str:
        node = DataLineageNode(
            node_type="source", name=name,
            description=f"Source: {url}",
            metadata={"url": url, "access_date": access_date,
                      "license": license, "record_count": record_count}
        )
        self.lineage_nodes.append(node)
        return node.node_id

    def add_transform(self, name: str, description: str,
                      parent_ids: list[str], params: dict) -> str:
        node = DataLineageNode(
            node_type="transform", name=name,
            description=description,
            parent_ids=parent_ids,
            metadata={"params": params}
        )
        self.lineage_nodes.append(node)
        return node.node_id

    def add_index_build(self, collection_name: str, vector_count: int,
                        embedding_model: str, parent_ids: list[str]) -> str:
        node = DataLineageNode(
            node_type="index", name="VectorIndexBuild",
            description=f"Built index: {collection_name}",
            parent_ids=parent_ids,
            metadata={"collection": collection_name,
                      "vector_count": vector_count,
                      "embedding_model": embedding_model}
        )
        self.lineage_nodes.append(node)
        return node.node_id

    def trace_document(self, document_id: str) -> list[DataLineageNode]:
        """Return all lineage nodes for a specific document."""
        return [n for n in self.lineage_nodes
                if document_id in n.metadata.get("document_ids", [])]


class LineageTracker:
    """Persists and queries lineage records."""
    def __init__(self, store_path: str = "governance/lineage"):
        from pathlib import Path
        self.store = Path(store_path)
        self.store.mkdir(parents=True, exist_ok=True)

    def save(self, record: CorpusLineageRecord):
        import json, dataclasses
        path = self.store / f"{record.corpus_id}_{record.corpus_version}.json"
        path.write_text(json.dumps(dataclasses.asdict(record), indent=2))

    def load(self, corpus_id: str, corpus_version: str) -> Optional[CorpusLineageRecord]:
        import json
        path = self.store / f"{corpus_id}_{corpus_version}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        record = CorpusLineageRecord(**{
            k: v for k, v in data.items() if k != "lineage_nodes"
        })
        record.lineage_nodes = [DataLineageNode(**n) for n in data["lineage_nodes"]]
        return record
```

---

### 2.3 Corpus Provenance and Consent

```python
from dataclasses import dataclass, field
from enum import Enum

class LicenseType(str, Enum):
    CC0          = "CC0"           # Public domain
    CC_BY        = "CC-BY"         # Attribution required
    CC_BY_SA     = "CC-BY-SA"      # Share-alike
    PROPRIETARY  = "proprietary"   # Requires explicit permission
    INTERNAL     = "internal"      # Company-owned content
    SCRAPE       = "scrape"        # Web scrape — verify ToS
    CONSENT      = "consent"       # User-provided with consent

class ConsentStatus(str, Enum):
    VERIFIED    = "verified"    # Explicit consent documented
    IMPLIED     = "implied"     # ToS permits use — document the ToS
    PENDING     = "pending"     # Consent collection in progress
    EXPIRED     = "expired"     # Consent period elapsed
    NOT_REQUIRED = "not_required"  # Public domain or company-owned

@dataclass
class CorpusSource:
    source_id: str
    name: str
    url: str
    license: LicenseType
    consent_status: ConsentStatus
    consent_evidence: str          # URL to ToS, consent record, or legal opinion
    data_subject_categories: list[str]  # e.g. ["employees", "customers"]
    contains_pii: bool
    retention_policy_days: int
    last_verified: str

@dataclass
class ConsentRecord:
    """Per-document consent record for user-provided content."""
    document_id: str
    subject_id: str           # Anonymised subject identifier
    consent_given_at: str
    consent_scope: str        # What the subject consented to
    consent_version: str      # Version of consent form
    withdrawal_requested: bool = False
    withdrawal_date: Optional[str] = None

class CorpusProvenanceManager:
    """Tracks provenance and consent for all corpus sources."""

    def __init__(self):
        self.sources: dict[str, CorpusSource] = {}
        self.consent_records: dict[str, ConsentRecord] = {}

    def register_source(self, source: CorpusSource):
        self.sources[source.source_id] = source

    def record_consent(self, record: ConsentRecord):
        self.consent_records[record.document_id] = record

    def withdraw_consent(self, document_id: str) -> list[str]:
        """
        Handle consent withdrawal (GDPR Art. 17 — Right to erasure).
        Returns list of actions required.
        """
        actions = []
        if document_id in self.consent_records:
            self.consent_records[document_id].withdrawal_requested = True
            self.consent_records[document_id].withdrawal_date = (
                datetime.utcnow().isoformat()
            )
        actions.append(f"DELETE_FROM_INDEX:{document_id}")
        actions.append(f"DELETE_FROM_OBJECT_STORE:{document_id}")
        actions.append(f"TRIGGER_INDEX_REBUILD:corpus_id=main")
        actions.append(f"AUDIT_LOG:consent_withdrawal:{document_id}")
        return actions

    def compliance_report(self) -> dict:
        total = len(self.sources)
        verified = sum(1 for s in self.sources.values()
                       if s.consent_status == ConsentStatus.VERIFIED)
        pii_sources = [s.name for s in self.sources.values() if s.contains_pii]
        problematic = [s.name for s in self.sources.values()
                       if s.consent_status in
                       (ConsentStatus.PENDING, ConsentStatus.EXPIRED)]
        return {
            "total_sources": total,
            "verified_sources": verified,
            "compliance_rate": verified / total if total else 0,
            "pii_sources": pii_sources,
            "requires_attention": problematic
        }
```

---

### 2.4 Versioned Governance Artefacts

```python
from pathlib import Path
import json
import hashlib
from datetime import datetime

class GovernanceArtefactStore:
    """
    Version-controlled store for governance artefacts:
    model cards, lineage records, evaluation reports, consent logs.
    Each write is immutable — artefacts are never overwritten.
    """
    def __init__(self, base_path: str = "governance/artefacts"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        system_id: str,
        artefact_type: str,   # "model_card" | "lineage" | "eval_report" | "consent"
        version: str,
        content: str,         # JSON or Markdown string
        author: str,
        approved_by: str = ""
    ) -> dict:
        artefact_dir = self.base / system_id / artefact_type
        artefact_dir.mkdir(parents=True, exist_ok=True)

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        filename = f"{version}_{content_hash}.json"
        record = {
            "system_id":    system_id,
            "artefact_type": artefact_type,
            "version":      version,
            "author":       author,
            "approved_by":  approved_by,
            "stored_at":    datetime.utcnow().isoformat(),
            "content_hash": content_hash,
            "content":      content,
        }
        (artefact_dir / filename).write_text(json.dumps(record, indent=2))
        return {"filename": filename, "content_hash": content_hash}

    def list_versions(self, system_id: str, artefact_type: str) -> list[dict]:
        artefact_dir = self.base / system_id / artefact_type
        if not artefact_dir.exists():
            return []
        versions = []
        for f in sorted(artefact_dir.glob("*.json")):
            record = json.loads(f.read_text())
            versions.append({
                "version":    record["version"],
                "author":     record["author"],
                "stored_at":  record["stored_at"],
                "hash":       record["content_hash"],
            })
        return versions

    def get_current(self, system_id: str, artefact_type: str) -> Optional[dict]:
        versions = self.list_versions(system_id, artefact_type)
        if not versions:
            return None
        latest = max(versions, key=lambda v: v["stored_at"])
        artefact_dir = self.base / system_id / artefact_type
        filename = f"{latest['version']}_{latest['hash']}.json"
        return json.loads((artefact_dir / filename).read_text())
```

---

> ### 📋 Chapter Summary
>
> - **Model cards** capture intended use, training data, performance metrics, limitations, and governance metadata — they satisfy EU AI Act technical documentation requirements.
> - **Data lineage** tracks every transformation (source → clean → chunk → embed → index) as a directed acyclic graph, enabling full provenance reconstruction.
> - **Corpus provenance** records license type and consent status per source; `withdraw_consent()` implements GDPR Art. 17 (Right to Erasure) as a concrete operational workflow.
> - **Immutable artefact storage** with content hashing ensures governance documents cannot be retroactively modified.

---

> ### ❓ Comprehension Questions
>
> 1. A model card records `evaluation_metrics` but only for the overall corpus. Regulators request performance metrics broken down by language (EN, FR, DE). What changes to the `ModelCard` dataclass and evaluation pipeline are required?
> 2. `CorpusLineageRecord.trace_document` returns lineage nodes that reference a document ID in their metadata. In practice, a transform node processes 50,000 documents. How would you scale this query to avoid scanning all nodes for every document lookup?
> 3. A user withdraws consent for their document. `withdraw_consent()` returns a list of actions but does not execute them. Design the execution workflow: what service executes each action, in what order, and how is completion verified?
> 4. `GovernanceArtefactStore` uses the file system. For a regulated environment requiring tamper-evident storage with multi-party access, what storage backend and access control model would you use?
> 5. `LicenseType.SCRAPE` marks content scraped from websites. A lawyer advises that scraping without explicit permission may violate ToS even if technically possible. How would you design a corpus intake process that gates `SCRAPE` sources on legal review before ingestion?

---

## References

### Documentation
- [EU AI Act Technical Documentation Requirements (Annex IV)](https://artificialintelligenceact.eu/annex/4/)
- [Google Model Card Toolkit](https://github.com/google/model-card-toolkit)
- [Hugging Face Model Cards](https://huggingface.co/docs/hub/model-cards)
- [GDPR Article 17 — Right to Erasure](https://gdpr-info.eu/art-17-gdpr/)

### Papers
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Mitchell et al., 2019.

---

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

## Chapter 4 — Compliance Automation 🧪

### 4.1 Policy as Code

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

class PolicyOutcome(str, Enum):
    PASS    = "pass"
    FAIL    = "fail"
    WARNING = "warning"
    SKIP    = "skip"   # Policy not applicable to this system

@dataclass
class PolicyCheck:
    policy_id: str
    name: str
    description: str
    category: str       # "data" | "model" | "operations" | "legal"
    severity: str       # "critical" | "high" | "medium" | "low"
    check_fn: Callable[[dict], tuple[PolicyOutcome, str]]
    remediation: str

def check_model_card_exists(system_context: dict) -> tuple[PolicyOutcome, str]:
    from pathlib import Path
    system_id = system_context.get("system_id", "")
    card_path = Path(f"governance/artefacts/{system_id}/model_card")
    if not card_path.exists() or not list(card_path.glob("*.json")):
        return PolicyOutcome.FAIL, "No model card found"
    return PolicyOutcome.PASS, "Model card present"

def check_pii_scan_on_corpus(system_context: dict) -> tuple[PolicyOutcome, str]:
    last_scan = system_context.get("last_pii_scan_date")
    if not last_scan:
        return PolicyOutcome.FAIL, "No PII scan recorded"
    from datetime import datetime, timedelta
    last = datetime.fromisoformat(last_scan)
    if (datetime.utcnow() - last).days > 90:
        return PolicyOutcome.WARNING, f"PII scan is {(datetime.utcnow()-last).days} days old"
    return PolicyOutcome.PASS, "PII scan recent"

def check_audit_logging_enabled(system_context: dict) -> tuple[PolicyOutcome, str]:
    if not system_context.get("audit_logging_enabled", False):
        return PolicyOutcome.FAIL, "Audit logging not enabled"
    retention = system_context.get("audit_log_retention_days", 0)
    if retention < 180:
        return PolicyOutcome.FAIL, f"Retention {retention}d < required 180d (EU AI Act)"
    return PolicyOutcome.PASS, f"Audit logging enabled, retention={retention}d"

def check_human_oversight_mechanism(system_context: dict) -> tuple[PolicyOutcome, str]:
    mechanisms = system_context.get("human_oversight_mechanisms", [])
    if not mechanisms:
        return PolicyOutcome.FAIL, "No human oversight mechanism documented"
    return PolicyOutcome.PASS, f"Oversight mechanisms: {mechanisms}"

def check_evaluation_report_current(system_context: dict) -> tuple[PolicyOutcome, str]:
    last_eval = system_context.get("last_evaluation_date")
    if not last_eval:
        return PolicyOutcome.FAIL, "No evaluation report found"
    from datetime import datetime
    last = datetime.fromisoformat(last_eval)
    age_days = (datetime.utcnow() - last).days
    if age_days > 90:
        return PolicyOutcome.WARNING, f"Evaluation report is {age_days} days old"
    return PolicyOutcome.PASS, "Evaluation report current"

COMPLIANCE_POLICIES = [
    PolicyCheck("POL-001", "Model Card Exists",
        "Every AI system must have a current model card",
        "model", "critical", check_model_card_exists,
        "Create model card using ModelCard dataclass and store via GovernanceArtefactStore"),

    PolicyCheck("POL-002", "PII Scan on Corpus",
        "Corpus must be PII-scanned within last 90 days",
        "data", "high", check_pii_scan_on_corpus,
        "Run PII detection pipeline on full corpus and record scan date"),

    PolicyCheck("POL-003", "Audit Logging Enabled",
        "Audit logging must be active with ≥180-day retention",
        "operations", "critical", check_audit_logging_enabled,
        "Enable AuditLogger with store retention configured to 180+ days"),

    PolicyCheck("POL-004", "Human Oversight Mechanism",
        "System must have documented human oversight controls",
        "legal", "high", check_human_oversight_mechanism,
        "Implement and document HumanInTheLoopGateway triggers"),

    PolicyCheck("POL-005", "Evaluation Report Current",
        "Evaluation report must be ≤90 days old",
        "model", "medium", check_evaluation_report_current,
        "Run evaluation pipeline and store report via GovernanceArtefactStore"),
]
```

---

### 4.2 Automated Compliance Checks in CI/CD

```python
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ComplianceRunResult:
    system_id: str
    run_at: str
    policies_checked: int
    passed: int
    failed: int
    warnings: int
    critical_failures: list[str]
    overall_status: str   # "compliant" | "non_compliant" | "warnings"
    details: list[dict]

class ComplianceRunner:
    def __init__(self, policies: list[PolicyCheck]):
        self.policies = policies

    def run(self, system_context: dict) -> ComplianceRunResult:
        from datetime import datetime
        results = []
        critical_failures = []

        for policy in self.policies:
            try:
                outcome, message = policy.check_fn(system_context)
            except Exception as e:
                outcome = PolicyOutcome.FAIL
                message = f"Check errored: {e}"

            results.append({
                "policy_id":  policy.policy_id,
                "name":       policy.name,
                "category":   policy.category,
                "severity":   policy.severity,
                "outcome":    outcome.value,
                "message":    message,
                "remediation": policy.remediation if outcome == PolicyOutcome.FAIL else None
            })
            if outcome == PolicyOutcome.FAIL and policy.severity == "critical":
                critical_failures.append(policy.policy_id)

        passed   = sum(1 for r in results if r["outcome"] == "pass")
        failed   = sum(1 for r in results if r["outcome"] == "fail")
        warnings = sum(1 for r in results if r["outcome"] == "warning")

        if critical_failures:
            status = "non_compliant"
        elif failed > 0:
            status = "non_compliant"
        elif warnings > 0:
            status = "warnings"
        else:
            status = "compliant"

        return ComplianceRunResult(
            system_id=system_context.get("system_id", "unknown"),
            run_at=datetime.utcnow().isoformat(),
            policies_checked=len(self.policies),
            passed=passed, failed=failed, warnings=warnings,
            critical_failures=critical_failures,
            overall_status=status,
            details=results
        )

    def run_and_exit(self, system_context: dict, fail_on: str = "non_compliant"):
        """CI/CD integration: exit non-zero on policy failure."""
        import sys
        result = self.run(system_context)
        print(f"\nCompliance Check: {result.overall_status.upper()}")
        print(f"  Passed:   {result.passed}/{result.policies_checked}")
        print(f"  Failed:   {result.failed}")
        print(f"  Warnings: {result.warnings}")
        if result.critical_failures:
            print(f"\nCRITICAL FAILURES: {result.critical_failures}")
            for detail in result.details:
                if detail["outcome"] == "fail":
                    print(f"  [{detail['policy_id']}] {detail['message']}")
                    print(f"    → {detail['remediation']}")
        if result.overall_status == fail_on or (
            fail_on == "non_compliant" and result.overall_status == "non_compliant"
        ):
            sys.exit(1)
        return result
```

```yaml
# .github/workflows/compliance.yml
name: AI Governance Compliance Check
on:
  push:
    branches: [main, release/*]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"   # Weekly Monday 06:00 UTC

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.lock

      - name: Run compliance checks
        run: python scripts/run_compliance.py --system-id rag-support-api --fail-on non_compliant
        env:
          GOVERNANCE_STORE_PATH: ${{ github.workspace }}/governance

      - name: Upload compliance report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: compliance-report
          path: governance/compliance_reports/
```

---

### 4.3 Evidence Collection and Audit Readiness

```python
from pathlib import Path
import json, zipfile
from datetime import datetime

class AuditEvidencePackage:
    """
    Assembles a complete evidence package for a compliance audit.
    Collects all governance artefacts, compliance reports, and
    incident logs into a structured, signed ZIP archive.
    """
    def __init__(self, system_id: str,
                 governance_store: "GovernanceArtefactStore",
                 audit_logger: "AuditLogger"):
        self.system_id = system_id
        self.gov_store = governance_store
        self.audit_logger = audit_logger

    def assemble(self, output_path: str, period_start: str, period_end: str) -> str:
        """
        Assemble audit package for a specified period.
        Returns path to the ZIP archive.
        """
        package_id = f"{self.system_id}_{period_start}_{period_end}"
        zip_path = Path(output_path) / f"{package_id}_audit_evidence.zip"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        manifest = {
            "system_id":     self.system_id,
            "period_start":  period_start,
            "period_end":    period_end,
            "assembled_at":  datetime.utcnow().isoformat(),
            "contents":      []
        }

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Model card (current version)
            model_card = self.gov_store.get_current(self.system_id, "model_card")
            if model_card:
                zf.writestr("model_card.json", json.dumps(model_card, indent=2))
                manifest["contents"].append("model_card.json")

            # 2. All compliance check reports in period
            reports_dir = Path(f"governance/compliance_reports/{self.system_id}")
            if reports_dir.exists():
                for report in sorted(reports_dir.glob("*.json")):
                    data = json.loads(report.read_text())
                    if period_start <= data.get("run_at", "") <= period_end:
                        zf.write(report, f"compliance_reports/{report.name}")
                        manifest["contents"].append(f"compliance_reports/{report.name}")

            # 3. Incident log
            incidents_dir = Path("governance/incidents")
            if incidents_dir.exists():
                for inc_file in sorted(incidents_dir.glob("*.json")):
                    data = json.loads(inc_file.read_text())
                    if (period_start <= data.get("detected_at", "") <= period_end
                            and data.get("affected_system_id") == self.system_id):
                        zf.write(inc_file, f"incidents/{inc_file.name}")
                        manifest["contents"].append(f"incidents/{inc_file.name}")

            # 4. Audit log chain verification report
            valid, count, error = self.audit_logger.verify_chain()
            chain_report = {
                "verified_at": datetime.utcnow().isoformat(),
                "records_verified": count,
                "chain_intact": valid,
                "error": error
            }
            zf.writestr("audit_chain_verification.json",
                        json.dumps(chain_report, indent=2))
            manifest["contents"].append("audit_chain_verification.json")

            # 5. Manifest
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

        print(f"  ✓ Audit package assembled: {zip_path}")
        print(f"    {len(manifest['contents'])} documents included")
        return str(zip_path)
```

---

### 🧪 Hands-on Lab: Governance Scorecard

**Objective:** Run a governance compliance check on a simulated AI system context and produce a human-readable scorecard.

```python
#!/usr/bin/env python3
"""
governance_scorecard.py — Governance compliance check demo.
No external dependencies required.
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

# ── Inline minimal policy framework ───────────────────────────────
class PolicyOutcome(str, Enum):
    PASS    = "pass"
    FAIL    = "fail"
    WARNING = "warning"

@dataclass
class Policy:
    id: str; name: str; severity: str; check: Callable; remediation: str

def _days_ago(d: str) -> int:
    return (datetime.utcnow() - datetime.fromisoformat(d)).days

POLICIES = [
    Policy("P01", "Model card exists",     "critical",
           lambda ctx: (PolicyOutcome.PASS, "present")
           if ctx.get("model_card_version") else
           (PolicyOutcome.FAIL, "No model card found"),
           "Create and store a ModelCard document"),

    Policy("P02", "Corpus PII scan",       "high",
           lambda ctx: (PolicyOutcome.PASS, "recent")
           if ctx.get("pii_scan_date") and _days_ago(ctx["pii_scan_date"]) <= 90
           else (PolicyOutcome.WARNING, f"Scan is {_days_ago(ctx.get('pii_scan_date','2000-01-01'))}d old")
           if ctx.get("pii_scan_date") else
           (PolicyOutcome.FAIL, "No scan recorded"),
           "Run PII scan on full corpus"),

    Policy("P03", "Audit log retention",   "critical",
           lambda ctx: (PolicyOutcome.PASS, f"{ctx.get('audit_retention_days',0)}d")
           if ctx.get("audit_logging_enabled") and ctx.get("audit_retention_days", 0) >= 180
           else (PolicyOutcome.FAIL, "Logging disabled or retention < 180d"),
           "Enable audit logging with 180+ day retention"),

    Policy("P04", "Evaluation current",    "medium",
           lambda ctx: (PolicyOutcome.PASS, "recent")
           if ctx.get("last_eval_date") and _days_ago(ctx["last_eval_date"]) <= 90
           else (PolicyOutcome.WARNING, "Evaluation > 90 days old")
           if ctx.get("last_eval_date") else
           (PolicyOutcome.FAIL, "No evaluation report"),
           "Run evaluation pipeline and store report"),

    Policy("P05", "Human oversight",       "high",
           lambda ctx: (PolicyOutcome.PASS, str(ctx.get("oversight_mechanisms", [])))
           if ctx.get("oversight_mechanisms") else
           (PolicyOutcome.FAIL, "No oversight mechanism documented"),
           "Implement HumanInTheLoopGateway"),

    Policy("P06", "EU AI Act classification","critical",
           lambda ctx: (PolicyOutcome.PASS, ctx["eu_ai_act_tier"])
           if ctx.get("eu_ai_act_tier") else
           (PolicyOutcome.FAIL, "No EU AI Act classification"),
           "Classify system under EU AI Act risk tiers"),

    Policy("P07", "Incident response plan","medium",
           lambda ctx: (PolicyOutcome.PASS, "documented")
           if ctx.get("incident_response_plan") else
           (PolicyOutcome.WARNING, "Incident response plan missing"),
           "Document incident response plan for AI-specific incident types"),
]

# ── System context: simulated AI system registry ───────────────────
SYSTEMS = [
    {
        "system_id": "rag-support-v2",
        "name": "Customer Support RAG",
        "eu_ai_act_tier": "limited_risk",
        "model_card_version": "2.1.0",
        "pii_scan_date": (datetime.utcnow() - timedelta(days=45)).isoformat(),
        "audit_logging_enabled": True,
        "audit_retention_days": 180,
        "last_eval_date": (datetime.utcnow() - timedelta(days=20)).isoformat(),
        "oversight_mechanisms": ["human_review_queue", "refusal_fallback"],
        "incident_response_plan": True,
    },
    {
        "system_id": "hiring-screener-v1",
        "name": "Resume Screening RAG",
        "eu_ai_act_tier": "high_risk",
        "model_card_version": None,    # Missing
        "pii_scan_date": None,          # Missing
        "audit_logging_enabled": True,
        "audit_retention_days": 90,     # Too short
        "last_eval_date": (datetime.utcnow() - timedelta(days=120)).isoformat(),
        "oversight_mechanisms": [],     # Missing
        "incident_response_plan": False,
    },
]

# ── Run and render scorecard ───────────────────────────────────────
ICONS = {PolicyOutcome.PASS: "✓", PolicyOutcome.FAIL: "✗", PolicyOutcome.WARNING: "⚠"}
COLORS = {PolicyOutcome.PASS: "", PolicyOutcome.FAIL: "FAIL ", PolicyOutcome.WARNING: "WARN "}

for system in SYSTEMS:
    print(f"\n{'='*58}")
    print(f"  Governance Scorecard: {system['name']}")
    print(f"  System ID: {system['system_id']}")
    print(f"  EU AI Act tier: {system.get('eu_ai_act_tier','unknown').upper()}")
    print(f"{'='*58}")

    results = []
    for policy in POLICIES:
        outcome, message = policy.check(system)
        results.append((policy, outcome, message))

    passed   = sum(1 for _,o,_ in results if o == PolicyOutcome.PASS)
    failed   = sum(1 for _,o,_ in results if o == PolicyOutcome.FAIL)
    warnings = sum(1 for _,o,_ in results if o == PolicyOutcome.WARNING)
    critical_fails = [p.id for p,o,_ in results if o == PolicyOutcome.FAIL and p.severity == "critical"]

    for policy, outcome, message in results:
        icon = ICONS[outcome]
        print(f"  {icon} [{policy.id}] {policy.name:<30} {message}")
        if outcome == PolicyOutcome.FAIL:
            print(f"       ↳ {policy.remediation}")

    print(f"\n  Score: {passed}/{len(POLICIES)} passed  "
          f"| {failed} failed | {warnings} warnings")

    if critical_fails:
        print(f"  CRITICAL FAILURES: {critical_fails}")
        print(f"  STATUS: NON-COMPLIANT ← blocks production deployment")
    elif failed:
        print(f"  STATUS: NON-COMPLIANT")
    elif warnings:
        print(f"  STATUS: COMPLIANT WITH WARNINGS")
    else:
        print(f"  STATUS: FULLY COMPLIANT ✓")
```

**Run the lab:**
```bash
python governance_scorecard.py
```

**Extensions:**
- Add a `--fix` flag that automatically generates skeleton artefacts for failing policies (empty model card, PII scan trigger)
- Export the scorecard to `governance/compliance_reports/{system_id}_{date}.json` for audit trail
- Add policy `P08`: check that `last_eval_date` faithfulness score ≥ 0.80 by reading the evaluation JSON report

---

> ### 📋 Chapter Summary
>
> - **Policy as code** encodes governance requirements as executable `check_fn` callables, making compliance checks runnable in CI/CD pipelines.
> - `ComplianceRunner.run_and_exit` provides a standard CI/CD integration that exits non-zero on critical policy failures — blocking deploys of non-compliant systems.
> - **Audit evidence packages** assemble all governance artefacts (model card, compliance reports, incident logs, audit chain verification) into a signed ZIP — ready for regulatory inspection.
> - The governance scorecard lab demonstrates a complete compliance gate: two systems, 7 policies, clear COMPLIANT vs NON-COMPLIANT verdict.

---

> ### ❓ Comprehension Questions
>
> 1. `check_pii_scan_on_corpus` returns `WARNING` if the scan is more than 90 days old. A corpus that updates daily could contain newly ingested PII within 24 hours of a scan. Should the threshold be time-based or event-based (scan triggered on each ingestion)? Argue both approaches.
> 2. `ComplianceRunner.run_and_exit` blocks a production deploy when `status == "non_compliant"`. A compliance failure is detected on a Friday afternoon. The fix requires a model card to be written — a 2-hour task. Is blocking the deploy the correct control, or should a time-bounded waiver process exist?
> 3. `AuditEvidencePackage.assemble` includes the audit chain verification result. An auditor asks: "How do you know the audit log was not tampered with before the evidence package was assembled?" What additional mechanism would strengthen this assurance?
> 4. The CI/CD compliance workflow runs on `push` to `main` and on a weekly schedule. A governance artefact becomes stale (evaluation > 90 days old) between pushes. How long can this go undetected, and how would you reduce the detection window?
> 5. `POLICIES` contains 7 checks. A senior engineer proposes adding `P08: LLM provider contract reviewed annually`. This requires checking an external document management system. What are the implications for the `check_fn` interface, CI/CD execution time, and failure handling?

---

## References

### Documentation
- [Open Policy Agent (OPA)](https://www.openpolicyagent.org/docs/latest/) — Policy as code engine.
- [EU AI Act Conformity Assessment](https://artificialintelligenceact.eu/assessment/)
- [NIST SP 800-53](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final) — Security and privacy controls.
- [SOC 2 Compliance Guide](https://www.aicpa.org/resources/landing/system-and-organization-controls-soc-suite-of-services)

### Papers
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Mitchell et al., 2019.
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) — Gebru et al., 2021.

---

> **Navigation**
> [← Part XIV — Security](part_14_security.md) | [→ Part XVI — Operations](part_16_operations.md)
