## Chapter 2 — Dataset Versioning

### 2.1 Why Version Datasets

Dataset versioning solves three critical problems:

**Reproducibility.** Evaluation results are only meaningful relative to a fixed dataset version. Without versioning, a team cannot determine whether a quality improvement is real or an artefact of a changed evaluation set.

**Auditability.** Regulated industries require traceability from model decisions back to the training and evaluation data. "What data was used to produce this result?" must be answerable with a precise version reference.

**Rollback.** When a dataset update introduces quality regressions, the team must be able to restore the previous version.

Git is unsuitable for large binary datasets. [DVC](https://dvc.org/doc) and [LakeFS](https://docs.lakefs.io) solve this by storing dataset content in object storage while tracking metadata and pointers in Git.

---

### 2.2 DVC: Data Version Control

[DVC](https://dvc.org/doc) extends Git with data tracking. It stores files in a configured remote and tracks them via lightweight `.dvc` pointer files committed to Git.

```bash
# Initialise DVC in an existing Git repository
git init && dvc init

# Configure remote storage
dvc remote add -d s3remote s3://my-ai-datasets/dvc-store
# 🔓 On-premise: MinIO or NFS
dvc remote add -d localremote /mnt/nfs/dvc-store

# Track a dataset file
dvc add data/eval_dataset_v1.jsonl
git add data/eval_dataset_v1.jsonl.dvc .gitignore
git commit -m "feat: add evaluation dataset v1.0"
dvc push

# Restore exact dataset from any git commit
git checkout v1.0
dvc pull
```

**Python — Programmatic DVC integration:**
```python
import subprocess
import json
from pathlib import Path

class DVCDatasetManager:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)

    def add_dataset(self, dataset_path: str, metadata: dict) -> str:
        """Track a dataset with DVC and return the git commit hash."""
        path = Path(dataset_path)

        # Write metadata sidecar
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2))

        # DVC add + git commit
        subprocess.run(["dvc", "add", str(path)], check=True, cwd=self.repo_path)
        subprocess.run(
            ["git", "add", str(path) + ".dvc", str(meta_path), ".gitignore"],
            check=True, cwd=self.repo_path
        )
        msg = f"dataset: {path.name} v{metadata.get('version', 'unknown')}"
        subprocess.run(["git", "commit", "-m", msg], check=True, cwd=self.repo_path)
        subprocess.run(["dvc", "push"], check=True, cwd=self.repo_path)

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=self.repo_path
        )
        return result.stdout.strip()

    def checkout_version(self, git_commit: str, dataset_path: str):
        """Restore a specific dataset version by git commit."""
        subprocess.run(
            ["git", "checkout", git_commit, "--", dataset_path + ".dvc"],
            check=True, cwd=self.repo_path
        )
        subprocess.run(["dvc", "pull", dataset_path], check=True, cwd=self.repo_path)
```

---

### 2.3 LakeFS: Git for Data Lakes

[LakeFS](https://docs.lakefs.io) applies Git branching semantics directly to object storage. It is suited to large-scale environments where datasets may be terabytes.

```python
import lakefs_client
from lakefs_client.models import CommitCreation, BranchCreation

configuration = lakefs_client.Configuration(
    host="http://localhost:8000",
    username="access_key_id",
    password="secret_access_key"
)

with lakefs_client.ApiClient(configuration) as api_client:
    branches_api = lakefs_client.BranchesApi(api_client)
    commits_api = lakefs_client.CommitsApi(api_client)
    objects_api = lakefs_client.ObjectsApi(api_client)

    REPO = "ai-datasets"

    # Create a branch for the new version
    branches_api.create_branch(
        repository=REPO,
        branch_creation=BranchCreation(name="eval-v2", source="main")
    )

    # Upload dataset to branch
    with open("eval_dataset_v2.jsonl", "rb") as f:
        objects_api.upload_object(
            repository=REPO, branch="eval-v2",
            path="evaluation/eval_dataset.jsonl", content=f
        )

    # Commit
    commits_api.commit(
        repository=REPO, branch="eval-v2",
        commit_creation=CommitCreation(
            message="Expand eval dataset to 500 examples",
            metadata={"record_count": "500", "added_by": "data-team"}
        )
    )

    # Merge to main after review
    branches_api.merge_into_branch(
        repository=REPO, source_ref="eval-v2",
        destination_branch="main"
    )
```

---

### 2.4 Versioning Strategies by Dataset Type

| Dataset type | Strategy | Rationale |
|---|---|---|
| Evaluation set | Immutable versions; never modify after benchmark | Reproducibility |
| Knowledge base | Incremental updates with daily snapshots | Freshness + rollback |
| Fine-tuning set | Explicit versioned releases | Model traceability |
| Conversation logs | Append-only with date partitioning | Audit trail |
| Prompt test suite | Git-native (small text files) | Direct Git tracking |

**Immutable evaluation sets — integrity sealing:**
```python
import hashlib
from pathlib import Path

def seal_evaluation_dataset(path: str) -> str:
    """Compute and record a SHA256 checksum. Use before every benchmark run."""
    data = Path(path).read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    Path(path).with_suffix(".sha256").write_text(checksum)
    return checksum

def verify_dataset_integrity(path: str) -> bool:
    """Raise if dataset has been modified since sealing."""
    checksum_path = Path(path).with_suffix(".sha256")
    if not checksum_path.exists():
        raise FileNotFoundError(f"No checksum for {path}. Was it sealed?")
    expected = checksum_path.read_text().strip()
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError(
            f"Dataset integrity FAILED for {path}. "
            f"Expected {expected[:12]}..., got {actual[:12]}..."
        )
    return True
```

---

### 2.5 Dataset Lineage Tracking

Lineage records the provenance chain: where a dataset came from, what transformations were applied, and what downstream artifacts depend on it.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class LineageNode:
    dataset_id: str
    version: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parent_ids: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    created_by: str = "pipeline"
    checksum: Optional[str] = None

class LineageTracker:
    def __init__(self, storage_path: str = "lineage.jsonl"):
        self.storage_path = storage_path

    def record(self, node: LineageNode):
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(node.__dict__) + "\n")

    def get_ancestors(self, dataset_id: str) -> list[LineageNode]:
        nodes = self._load_all()
        by_id = {n.dataset_id: n for n in nodes}
        ancestors, queue, visited = [], [dataset_id], set()
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            node = by_id.get(current)
            if node:
                ancestors.append(node)
                queue.extend(node.parent_ids)
        return ancestors

    def _load_all(self) -> list[LineageNode]:
        try:
            with open(self.storage_path) as f:
                return [LineageNode(**json.loads(line)) for line in f if line.strip()]
        except FileNotFoundError:
            return []

# Usage
tracker = LineageTracker()
tracker.record(LineageNode(
    dataset_id="raw_docs_2024_03", version="1.0", created_by="ingestion_pipeline"
))
tracker.record(LineageNode(
    dataset_id="eval_dataset_v1", version="1.0",
    parent_ids=["raw_docs_2024_03"],
    transformations=["synthetic_qa_generation", "quality_filter", "deduplication"]
))
```

---

> ### 📋 Chapter Summary
>
> - Dataset versioning is required for **reproducibility**, **auditability**, and **rollback**.
> - [DVC](https://dvc.org/doc) extends Git for datasets up to hundreds of GB via remote object storage.
> - [LakeFS](https://docs.lakefs.io) applies Git branching to object storage — ideal for petabyte-scale data lakes.
> - Evaluation datasets must be **immutable and checksummed** — modification after a benchmark run invalidates all results.
> - **Lineage tracking** records the full provenance chain from raw sources through transformations.

---

> ### ❓ Comprehension Questions
>
> 1. A team's RAG benchmark shows a 12% quality improvement after updating the knowledge base. Without versioning, why is this result potentially meaningless?
> 2. Compare [DVC](https://dvc.org/doc) and [LakeFS](https://docs.lakefs.io). Which would you choose for a 2TB knowledge base corpus updated daily by an automated ingestion pipeline?
> 3. An evaluation dataset checksum fails verification before a benchmark run. What steps do you take before proceeding?
> 4. Why should conversation logs use append-only partitioned storage rather than mutable tables?
> 5. A fine-tuning dataset is derived from internal documentation, public web scrape, and synthetic LLM-generated examples. What lineage records would you create?

---

## References

### Documentation
- [DVC Documentation](https://dvc.org/doc) — Data Version Control for ML.
- [LakeFS Documentation](https://docs.lakefs.io) — Git-like semantics for data lakes.
- [Git LFS](https://git-lfs.com) — Large file storage for Git.
- [Delta Lake](https://delta.io) — ACID transactions for data lake tables.
- [Apache Iceberg](https://iceberg.apache.org/docs/latest/) — Table format with time travel for large analytic datasets.

### Papers
- [MLflow: A Platform for the ML Lifecycle](https://mlflow.org/docs/latest/index.html) — Zaharia et al., 2018.
- [Towards Reproducible ML Pipelines](https://arxiv.org/abs/2012.04261) — Sculley et al., 2015.

---

---
[« Back to dataset_engineering Index](index.md) | [🏠 Home](../index.md)