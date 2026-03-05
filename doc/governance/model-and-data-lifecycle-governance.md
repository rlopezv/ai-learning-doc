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

---
[« Back to governance Index](index.md) | [🏠 Home](../index.md)