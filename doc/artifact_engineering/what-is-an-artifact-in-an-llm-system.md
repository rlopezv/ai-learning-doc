## Chapter 1 — What Is an Artifact in an LLM System

### 1.1 Artifacts Beyond Models

In traditional ML engineering, "artifact" typically refers to a trained model — a serialised set of weights. In LLM systems engineering, the concept is significantly broader. An LLM system produces and consumes multiple distinct artifacts, each changing independently, each requiring version control, and each capable of causing system regressions if mismanaged.

The critical insight is that **a prompt change is as consequential as a model weight change**. A poorly managed prompt update can silently degrade system quality just as a bad model update would — but it is far easier to make and far less likely to be tracked with the same rigour.

Artifact engineering applies consistent versioning, testing, promotion, and rollback discipline to every artifact a system depends on — not only models.

---

### 1.2 Artifact Taxonomy

| Artifact type | Examples | Change frequency | Risk if unversioned |
|---|---|---|---|
| **Prompt templates** | System prompt, RAG prompt, tool descriptions | Daily / weekly | Silent quality degradation |
| **Model weights** | Fine-tuned adapters (LoRA), base model versions | Per release | Regression, compliance breach |
| **Vector indexes** | Embedded knowledge base, semantic search index | Daily / per ingestion | Stale retrieval, inconsistency |
| **Evaluation datasets** | QA pairs, adversarial sets | Quarterly | Unreliable benchmarks |
| **Configuration** | Chunk size, top-K, embedding model name | Per deployment | Silent behaviour change |
| **Tool schemas** | Function definitions for tool-augmented LLMs | Per API change | Broken tool calls |
| **Pipeline DAGs** | Ingestion pipeline, evaluation pipeline | Per code release | Non-reproducible processing |

---

### 1.3 Artifact Lifecycle

Every artifact follows a defined promotion path:

```
Development → Staging → Production
     ↓              ↓          ↓
  Unit test    Integration   Canary
  Eval score   Eval gate     Full rollout
  Lint/format  A/B test      Monitoring
```

The promotion path enforces that artifacts only reach production after passing defined quality gates — evaluation score thresholds, regression checks, and where required, explicit human approval.

---

### 1.4 Artifact Registry Patterns

An artifact registry stores versioned artifacts with metadata, enabling promotion, rollback, and lineage queries.

```python
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ArtifactType(str, Enum):
    PROMPT = "prompt"
    MODEL = "model"
    INDEX = "index"
    DATASET = "dataset"
    CONFIG = "config"
    TOOL_SCHEMA = "tool_schema"

class ArtifactStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"

@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_type: ArtifactType
    version: str
    stage: ArtifactStage
    content_hash: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    promoted_at: Optional[str] = None
    created_by: str = "pipeline"
    parent_version: Optional[str] = None
    evaluation_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

class ArtifactRegistry:
    """
    File-based artifact registry.
    In production: back with a database or MLflow artifact store.
    """
    def __init__(self, registry_path: str = "./artifact_registry"):
        self.base = Path(registry_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base / "index.jsonl"

    def register(self, record: ArtifactRecord, content: bytes) -> str:
        artifact_dir = self.base / record.artifact_type.value / record.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{record.version}.bin").write_bytes(content)
        with open(self.index_file, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")
        return record.artifact_id

    def promote(self, artifact_id: str, version: str, target_stage: ArtifactStage):
        records = self._load_index()
        for r in records:
            if r["artifact_id"] == artifact_id and r["version"] == version:
                r["stage"] = target_stage.value
                r["promoted_at"] = datetime.utcnow().isoformat()
        self._save_index(records)

    def get_production(self, artifact_id: str) -> Optional[dict]:
        records = self._load_index()
        production = [
            r for r in records
            if r["artifact_id"] == artifact_id
            and r["stage"] == ArtifactStage.PRODUCTION.value
        ]
        return max(production, key=lambda r: r["created_at"]) if production else None

    def get_versions(self, artifact_id: str) -> list[dict]:
        return [r for r in self._load_index() if r["artifact_id"] == artifact_id]

    def _load_index(self) -> list[dict]:
        if not self.index_file.exists():
            return []
        with open(self.index_file) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _save_index(self, records: list[dict]):
        with open(self.index_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
```

---

> ### 📋 Chapter Summary
>
> - LLM systems depend on multiple artifact types beyond models: prompts, indexes, configurations, and tool schemas all require versioning.
> - A prompt change is as consequential as a model change — both can silently degrade production quality.
> - Every artifact follows a **promotion path** (development → staging → production) with quality gates at each transition.
> - An **artifact registry** stores versioned content with metadata enabling rollback, lineage queries, and promotion tracking.

---

> ### ❓ Comprehension Questions
>
> 1. A production RAG system degrades silently for two weeks. Investigation reveals a prompt template was updated without version control. What engineering controls would have caught this?
> 2. Explain why vector indexes should be treated as versioned artifacts with blue-green deployment rather than in-place updates.
> 3. An artifact registry records `parent_version`. What lineage queries does this field enable?
> 4. Configuration artifacts (chunk size, top-K, embedding model) change less often than prompts. Does this mean they require less rigorous versioning?
> 5. Design the promotion gates for a prompt artifact moving from staging to production. What must be true before promotion is allowed?

---

## References

### Documentation
- [MLflow Models](https://mlflow.org/docs/latest/models.html) — Model artifact packaging and registry.
- [Hugging Face Model Hub](https://huggingface.co/docs/hub/models-the-hub) — Model versioning and cards.
- [DVC Artifacts](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files) — Artifact tracking with DVC.
- [LangSmith](https://docs.smith.langchain.com) — Prompt versioning and evaluation.

### Papers
- [Hidden Technical Debt in ML Systems](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — Sculley et al., 2015.
- [Towards Reproducible ML](https://arxiv.org/abs/2012.04261) — Artifact versioning for reproducibility.

---

---
[« Back to artifact_engineering Index](index.md) | [🏠 Home](../../index.md)