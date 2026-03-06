## Chapter 1 — Dataset Lifecycle

### 1.1 Datasets as First-Class Engineering Artifacts

In traditional software engineering, the primary artifact is code. In AI systems engineering, **datasets are equally primary**. The behaviour of an LLM system depends on the data used to evaluate it, fine-tune it, and populate its knowledge base — not only on the code that orchestrates it.

This shift demands applying the same discipline to datasets that engineers already apply to code: versioning, testing, lineage tracking, access control, and lifecycle management.

The absence of this discipline manifests in predictable failure modes:

- Evaluation results cannot be reproduced because the evaluation dataset changed silently
- A model regression is traced to a training dataset change that was not recorded
- PII from customer conversations leaks into a fine-tuning dataset
- A knowledge base returns contradictory answers because older and newer document versions coexist without version control

Dataset engineering is the practice of preventing these failures systematically.

---

### 1.2 Lifecycle Stages

Every dataset in an LLM system passes through a defined lifecycle:

```
Collection → Cleaning → Validation → Versioning → Storage → Consumption
     ↓            ↓           ↓            ↓           ↓           ↓
  Ingestion   Dedup/PII   Schema/      DVC/LakeFS   Artifact   Training /
  pipelines   removal     quality      Git-based    registry   Evaluation /
                          checks                               RAG index
                                                                    ↓
                                                             Monitoring → Deprecation
```

The lifecycle is not linear — datasets are consumed iteratively, updated incrementally, and may fork into specialised variants. Managing this complexity requires explicit tooling and process.

---

### 1.3 Dataset Taxonomy for LLM Systems

LLM systems consume datasets across multiple distinct functions:

| Dataset type | Purpose | Typical size | Update frequency |
|---|---|---|---|
| **Knowledge base** | RAG document corpus | 1K–10M documents | Continuous / daily |
| **Evaluation set** | Measure system quality | 100–10K examples | Quarterly |
| **Fine-tuning set** | Adapt model behaviour | 1K–100K examples | Per model release |
| **Prompt test suite** | Regression testing | 50–500 prompts | Per prompt change |
| **Conversation logs** | Observability, analysis | Millions of records | Continuous |
| **Feedback dataset** | Human preference labels | 100–10K records | Continuous |

Each type has different engineering requirements. A knowledge base requires near-real-time freshness and fine-grained update tracking. An evaluation set requires strict version control and must never be modified after a benchmark run. A fine-tuning set requires PII scrubbing, deduplication, and format validation before use.

---

### 1.4 Ownership and Governance

Each dataset must have a defined owner responsible for its quality, access control, and lifecycle decisions.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

class DatasetStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class SensitivityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class DatasetMetadata:
    name: str
    version: str
    owner: str
    team: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: DatasetStatus = DatasetStatus.DRAFT
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    contains_pii: bool = False
    record_count: Optional[int] = None
    size_bytes: Optional[int] = None
    source_datasets: list[str] = field(default_factory=list)
    schema_version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    license: Optional[str] = None
    retention_days: Optional[int] = 365

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "owner": self.owner,
            "status": self.status.value,
            "sensitivity": self.sensitivity.value,
            "contains_pii": self.contains_pii,
            "record_count": self.record_count,
            "source_datasets": self.source_datasets,
        }
```

---

### 1.5 Dataset Schema Design

Consistent schema design across dataset types enables tooling to work generically.

**Evaluation dataset schema:**
```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class EvalRecord(BaseModel):
    id: str
    query: str
    ground_truth_answer: str
    relevant_document_ids: list[str]
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    category: Optional[str] = None
    requires_multi_hop: bool = False
    created_by: str = "synthetic"       # "synthetic" | "human" | "production_log"
    source_document_id: Optional[str] = None

class KnowledgeBaseDocument(BaseModel):
    id: str
    content: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    source_file: Optional[str] = None
    content_hash: str                   # SHA256 for deduplication
    created_at: str
    updated_at: str
    tags: list[str] = Field(default_factory=list)
    language: str = "en"
    token_count: Optional[int] = None

class FineTuningRecord(BaseModel):
    id: str
    messages: list[dict]                # OpenAI chat format
    source: str
    quality_score: Optional[float] = None
    reviewer: Optional[str] = None
    created_at: str
```

**Java — Dataset record with validation:**
```java
import jakarta.validation.constraints.*;
import java.time.Instant;
import java.util.List;

public record EvalRecord(
    @NotBlank String id,
    @NotBlank @Size(max = 1000) String query,
    @NotBlank String groundTruthAnswer,
    @NotEmpty List<String> relevantDocumentIds,
    @NotNull DifficultyLevel difficulty,
    String category,
    boolean requiresMultiHop,
    @NotBlank String createdBy,
    Instant createdAt
) {
    public enum DifficultyLevel { EASY, MEDIUM, HARD }
}
```

---

> ### 📋 Chapter Summary
>
> - Datasets are **first-class engineering artifacts** requiring versioning, ownership, schema design, and lifecycle management.
> - LLM systems consume six distinct dataset types with different freshness, quality, and access requirements.
> - Every dataset requires a defined **owner**, **sensitivity classification**, and **retention policy**.
> - Consistent schema design enables generic tooling across all dataset types in the system.

---

> ### ❓ Comprehension Questions
>
> 1. A team's evaluation benchmark has been improving for three months, but production quality has not improved. Investigation reveals the evaluation dataset has been silently expanded with easier examples. What process failure caused this?
> 2. Explain the difference between a knowledge base dataset and a fine-tuning dataset in terms of update frequency, size, and engineering requirements.
> 3. A dataset is classified as `CONFIDENTIAL` containing customer conversation logs. What access controls, storage requirements, and processing constraints does this classification imply?
> 4. Why is `content_hash` a critical field in a `KnowledgeBaseDocument` schema? What operations does it enable?
> 5. A data scientist creates an evaluation dataset directly from production logs without review. What are the data quality and privacy risks?

---

## References

### Papers
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) — Gebru et al., 2018. Framework for dataset documentation and governance.
- [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — Sculley et al., 2015.

### Documentation
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets) — Dataset loading, processing, and sharing.
- [Pydantic Documentation](https://docs.pydantic.dev) — Schema validation for dataset records.
- [Apache Parquet](https://parquet.apache.org/docs/) — Columnar storage format for large datasets.
- [GDPR Article 17 — Right to Erasure](https://gdpr-info.eu/art-17-gdpr/) — Data retention compliance reference.

---

---
[« Back to dataset-engineering Index](index.md) | [🏠 Home](../index.md)