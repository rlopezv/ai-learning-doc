# Part V — Dataset Engineering

---

> **Navigation**
> [← Part IV — Advanced RAG](part_04_advanced_rag.md) | [→ Part VI — Artifact Engineering](part_06_artifact_engineering.md)

---

## Contents

- [Chapter 1 — Dataset Lifecycle](#chapter-1--dataset-lifecycle)
  - [1.1 Datasets as First-Class Engineering Artifacts](#11-datasets-as-first-class-engineering-artifacts)
  - [1.2 Lifecycle Stages](#12-lifecycle-stages)
  - [1.3 Dataset Taxonomy for LLM Systems](#13-dataset-taxonomy-for-llm-systems)
  - [1.4 Ownership and Governance](#14-ownership-and-governance)
  - [1.5 Dataset Schema Design](#15-dataset-schema-design)
- [Chapter 2 — Dataset Versioning](#chapter-2--dataset-versioning)
  - [2.1 Why Version Datasets](#21-why-version-datasets)
  - [2.2 DVC: Data Version Control](#22-dvc-data-version-control)
  - [2.3 LakeFS: Git for Data Lakes](#23-lakefs-git-for-data-lakes)
  - [2.4 Versioning Strategies by Dataset Type](#24-versioning-strategies-by-dataset-type)
  - [2.5 Dataset Lineage Tracking](#25-dataset-lineage-tracking)
- [Chapter 3 — Synthetic Dataset Generation 🧪](#chapter-3--synthetic-dataset-generation-)
  - [3.1 Why Synthetic Data](#31-why-synthetic-data)
  - [3.2 Generating QA Pairs from Documents](#32-generating-qa-pairs-from-documents)
  - [3.3 Persona-Driven Generation](#33-persona-driven-generation)
  - [3.4 Adversarial and Edge-Case Generation](#34-adversarial-and-edge-case-generation)
  - [3.5 Multi-Turn Conversation Datasets](#35-multi-turn-conversation-datasets)
  - [3.6 Quality Filtering for Synthetic Data](#36-quality-filtering-for-synthetic-data)
  - [🧪 Hands-on Lab: Build a RAG Evaluation Dataset](#-hands-on-lab-build-a-rag-evaluation-dataset)
- [Chapter 4 — Data Quality](#chapter-4--data-quality)
  - [4.1 Quality Dimensions](#41-quality-dimensions)
  - [4.2 Automated Quality Checks](#42-automated-quality-checks)
  - [4.3 Deduplication](#43-deduplication)
  - [4.4 PII Detection and Redaction](#44-pii-detection-and-redaction)
  - [4.5 Quality Scoring Pipelines](#45-quality-scoring-pipelines)
- [Chapter 5 — Data Augmentation](#chapter-5--data-augmentation)
  - [5.1 When Augmentation Is Needed](#51-when-augmentation-is-needed)
  - [5.2 Paraphrase Augmentation](#52-paraphrase-augmentation)
  - [5.3 Back-Translation](#53-back-translation)
  - [5.4 Noise Injection for Robustness](#54-noise-injection-for-robustness)
  - [5.5 Augmentation Pipelines in Production](#55-augmentation-pipelines-in-production)

---

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

## Chapter 3 — Synthetic Dataset Generation 🧪

### 3.1 Why Synthetic Data

Building high-quality datasets from human annotation is expensive, slow, and difficult to scale. Synthetic generation — using LLMs to create training and evaluation examples — addresses this constraint but introduces its own engineering challenges.

Synthetic data is valuable in four scenarios:

**Bootstrapping evaluation.** Before any production data exists, a synthetic evaluation set enables measurement from day one.

**Domain coverage expansion.** Human annotators cluster around common cases. LLMs can be instructed to generate rare, edge-case, and adversarial examples that humans tend to miss.

**Privacy preservation.** Synthetic data can replicate statistical properties of sensitive production data without containing real customer information.

**Scaling fine-tuning datasets.** High-quality fine-tuning data is scarce. Synthetic generation from a small set of expert-labelled examples can expand the dataset 10–100x.

The central risk is **distributional bias**: synthetic data reflects the generating LLM's biases, knowledge gaps, and stylistic tendencies. All synthetic datasets must be quality-filtered before use.

---

### 3.2 Generating QA Pairs from Documents

The most common synthetic generation task for RAG: given a corpus, generate `(question, answer, source_document)` triples.

```python
import json
import hashlib
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

client = OpenAI()

class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    source_document_id: str
    source_excerpt: str
    difficulty: str            # easy | medium | hard
    question_type: str         # factual | inferential | comparative | procedural
    requires_multi_hop: bool = False

def generate_qa_pairs(
    document: dict,
    n_pairs: int = 5,
    model: str = "gpt-4o"
) -> list[QAPair]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Generate exactly {n_pairs} question-answer pairs from the document.

Requirements:
- Questions must be answerable SOLELY from the provided text
- Include a mix of: factual, inferential, procedural, and comparative questions
- Vary difficulty: easy (direct lookup), medium (requires combining info), hard (inference)
- Identify the exact excerpt that supports each answer
- For hard questions, set requires_multi_hop: true when two sections must be combined

Return JSON:
{{
  "pairs": [
    {{
      "question": "...",
      "answer": "...",
      "source_excerpt": "exact quote from text",
      "difficulty": "easy|medium|hard",
      "question_type": "factual|inferential|comparative|procedural",
      "requires_multi_hop": false
    }}
  ]
}}"""
            },
            {"role": "user", "content": f"Document ID: {document['id']}\n\n{document['content']}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    data = json.loads(response.choices[0].message.content)
    pairs = []
    for pair in data.get("pairs", []):
        pair_id = hashlib.md5(f"{document['id']}_{pair['question']}".encode()).hexdigest()[:12]
        pairs.append(QAPair(
            id=pair_id,
            question=pair["question"],
            answer=pair["answer"],
            source_document_id=document["id"],
            source_excerpt=pair.get("source_excerpt", ""),
            difficulty=pair.get("difficulty", "medium"),
            question_type=pair.get("question_type", "factual"),
            requires_multi_hop=pair.get("requires_multi_hop", False)
        ))
    return pairs

def generate_eval_dataset(
    documents: list[dict],
    pairs_per_doc: int = 5,
    max_workers: int = 5
) -> list[QAPair]:
    all_pairs = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(generate_qa_pairs, doc, pairs_per_doc) for doc in documents]
        for future in futures:
            try:
                all_pairs.extend(future.result())
            except Exception as e:
                print(f"Generation failed: {e}")
    return all_pairs
```

**Java — QA generation with [LangChain4j](https://docs.langchain4j.dev):**
```java
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

interface QAPairGenerator {
    @SystemMessage("""
        Generate 5 question-answer pairs from the document.
        Mix difficulty (easy/medium/hard) and type (factual/inferential/procedural).
        Return JSON: {"pairs": [{"question":"...","answer":"...","difficulty":"...","question_type":"..."}]}
        """)
    @UserMessage("Document:\n{{document}}")
    String generate(String document);
}

QAPairGenerator generator = AiServices.builder(QAPairGenerator.class)
    .chatLanguageModel(model)
    .build();

String json = generator.generate(documentContent);
// Parse with Jackson or Gson
```

---

### 3.3 Persona-Driven Generation

Generating from a single prompt produces questions that reflect one implicit user type. Real query distributions are far more diverse. Persona-driven generation models different user types explicitly.

```python
PERSONAS = [
    {
        "name": "technical_expert",
        "description": "Senior engineer, precise technical terminology, asks about edge cases",
        "example_queries": [
            "What is the thread safety model for concurrent API calls?",
            "What happens to in-flight requests during rolling deployment?"
        ]
    },
    {
        "name": "business_user",
        "description": "Non-technical manager, plain language, focused on costs and outcomes",
        "example_queries": [
            "How much does it cost to process 10,000 documents?",
            "What happens if the service goes down?"
        ]
    },
    {
        "name": "new_employee",
        "description": "Recently joined, unfamiliar with domain, asks basic questions",
        "example_queries": ["What is a webhook?", "Where is the documentation?"]
    },
    {
        "name": "power_user",
        "description": "Advanced customer, knows the product deeply, asks about edge cases",
        "example_queries": [
            "Can I use custom embedding models?",
            "How do I override the default chunking strategy?"
        ]
    }
]

def generate_persona_qa(
    document: dict,
    persona: dict,
    n_pairs: int = 3,
    model: str = "gpt-4o-mini"
) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are simulating a {persona['name']}: {persona['description']}.

Example questions this persona would ask:
{chr(10).join('- ' + q for q in persona['example_queries'])}

Generate {n_pairs} questions this persona would ask about the document.
Questions must be answerable from the document.
Return JSON: {{"pairs": [{{"question": "...", "answer": "...", "persona": "{persona['name']}"}}]}}"""
            },
            {"role": "user", "content": document["content"]}
        ],
        response_format={"type": "json_object"},
        temperature=0.9
    )
    return json.loads(response.choices[0].message.content).get("pairs", [])

def generate_diverse_eval_dataset(
    documents: list[dict],
    pairs_per_persona: int = 2
) -> list[dict]:
    all_pairs = []
    for doc in documents:
        for persona in PERSONAS:
            pairs = generate_persona_qa(doc, persona, n_pairs=pairs_per_persona)
            all_pairs.extend(pairs)
    return all_pairs
```

---

### 3.4 Adversarial and Edge-Case Generation

A system that performs well on clean queries may fail on adversarial inputs. Deliberately generating difficult examples builds a more robust evaluation set.

```python
ADVERSARIAL_TYPES = {
    "unanswerable": "Generate a question that CANNOT be answered from the document.",
    "ambiguous": "Generate a question whose answer is ambiguous given only the document.",
    "contradictory_premise": "Generate a question containing a false premise about the content.",
    "multi_hop": "Generate a question requiring information from at least two sections.",
    "negation": "Generate a question phrased with negation ('what is NOT included...').",
    "numerical": "Generate a question requiring numerical calculation from the document.",
}

def generate_adversarial_pairs(
    document: dict,
    adversarial_types: list[str] = None,
    model: str = "gpt-4o"
) -> list[dict]:
    types_to_use = adversarial_types or list(ADVERSARIAL_TYPES.keys())
    pairs = []
    for adv_type in types_to_use:
        instruction = ADVERSARIAL_TYPES[adv_type]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"""{instruction}
Return JSON: {{"question": "...", "expected_behaviour": "...", "adversarial_type": "{adv_type}"}}
For unanswerable: expected_behaviour = "system should say it cannot find this information"."""
                },
                {"role": "user", "content": document["content"]}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        pair = json.loads(response.choices[0].message.content)
        pair["source_document_id"] = document["id"]
        pairs.append(pair)
    return pairs
```

---

### 3.5 Multi-Turn Conversation Datasets

For conversational systems, evaluation requires datasets that test context retention and follow-up handling.

```python
def generate_conversation(
    document: dict,
    n_turns: int = 4,
    model: str = "gpt-4o"
) -> dict:
    """Generate a realistic multi-turn conversation grounded in a document."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Generate a realistic {n_turns}-turn conversation between
a user and a knowledge assistant about the provided document.

Requirements:
- Turn 1: User asks a general opening question
- Turns 2–{n_turns}: Follow-up questions referencing prior context
  (use pronouns, "what about the other option?", "how does that compare?")
- Answers grounded in the document
- Include at least one clarification request

Return JSON:
{{
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "document_id": "{document['id']}"
}}"""
            },
            {"role": "user", "content": document["content"]}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    return json.loads(response.choices[0].message.content)
```

---

### 3.6 Quality Filtering for Synthetic Data

All synthetic data must be filtered before use. Common issues: unanswerable questions, hallucinated answers, duplicates, and malformed output.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SyntheticDataQualityFilter:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(embed_model_name)

    def filter(self, pairs: list[dict], documents_by_id: dict[str, str]) -> list[dict]:
        filtered = self._remove_malformed(pairs)
        filtered = self._remove_ungrounded(filtered, documents_by_id)
        filtered = self._remove_near_duplicates(filtered, threshold=0.92)
        return filtered

    def _remove_malformed(self, pairs: list[dict]) -> list[dict]:
        return [
            p for p in pairs
            if p.get("question") and p.get("answer")
            and len(p["question"]) >= 20
            and len(p["answer"]) >= 10
        ]

    def _remove_ungrounded(
        self, pairs: list[dict], documents_by_id: dict[str, str]
    ) -> list[dict]:
        """
        Filter answers with low embedding similarity to their source document.
        Low similarity indicates the answer was not grounded in the document text.
        """
        passing = []
        for pair in pairs:
            doc_text = documents_by_id.get(pair.get("source_document_id", ""), "")
            if not doc_text:
                continue
            a_emb = self.embed_model.encode(pair["answer"])
            d_emb = self.embed_model.encode(doc_text[:1000])
            sim = float(cosine_similarity([a_emb], [d_emb])[0][0])
            if sim >= 0.4:
                pair["grounding_score"] = sim
                passing.append(pair)
        return passing

    def _remove_near_duplicates(
        self, pairs: list[dict], threshold: float = 0.92
    ) -> list[dict]:
        if len(pairs) < 2:
            return pairs
        questions = [p["question"] for p in pairs]
        embeddings = self.embed_model.encode(questions)
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        keep = np.ones(len(pairs), dtype=bool)
        for i in range(len(pairs)):
            if keep[i]:
                keep[np.where((sim_matrix[i] >= threshold) & (np.arange(len(pairs)) > i))] = False
        return [p for p, k in zip(pairs, keep) if k]
```

---

### 🧪 Hands-on Lab: Build a RAG Evaluation Dataset

**Objective:** Generate, filter, and validate a synthetic evaluation dataset from a small document corpus.

**Prerequisites:** `openai`, `sentence-transformers`, `scikit-learn`, `pandas`

```python
import json, hashlib, pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

DOCUMENTS = [
    {
        "id": "doc_001", "title": "Refund Policy",
        "content": """Our standard refund policy allows returns within 30 days of purchase.
Items must be in original, unopened condition. Digital products are non-refundable once downloaded.
Enterprise customers have an extended 90-day return window.
Refunds are processed in 5–7 business days for card payments, 10–14 days for bank transfers."""
    },
    {
        "id": "doc_002", "title": "Subscription Plans",
        "content": """We offer Free, Professional ($150/mo), and Enterprise ($500/mo) tiers.
Free: 5 users, 10GB, community support. Professional: 25 users, 100GB, 24h email SLA.
Enterprise: unlimited users, 1TB, dedicated account manager, 4-hour SLA for critical issues.
Annual billing provides a 20% discount on all paid tiers."""
    },
    {
        "id": "doc_003", "title": "API Reference",
        "content": """The REST API uses OAuth 2.0. Tokens expire after 3600 seconds.
Rate limits: Free 100 req/min, Professional 1000 req/min, Enterprise 10000 req/min.
Current stable API version is v2. Version v1 is deprecated and retires December 31, 2025."""
    }
]

docs_by_id = {d["id"]: d["content"] for d in DOCUMENTS}

# ── Step 1: Generate ──────────────────────────────────────
print("Generating QA pairs...")
all_pairs = []
for doc in DOCUMENTS:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Generate 6 diverse QA pairs from the document.
Include: 2 easy factual, 2 medium inferential, 1 hard multi-step, 1 unanswerable.
For unanswerable: set answer to "NOT_IN_DOCUMENT" and answerable: false.
Return JSON: {"pairs": [{"question":"...","answer":"...","difficulty":"...","answerable":true}]}"""
            },
            {"role": "user", "content": f"Document {doc['id']}:\n{doc['content']}"}
        ],
        response_format={"type": "json_object"}, temperature=0.8
    )
    pairs = json.loads(response.choices[0].message.content).get("pairs", [])
    for p in pairs:
        p["source_document_id"] = doc["id"]
        p["id"] = hashlib.md5(p["question"].encode()).hexdigest()[:10]
    all_pairs.extend(pairs)
    print(f"  {doc['id']}: {len(pairs)} pairs generated")

print(f"\nTotal generated: {len(all_pairs)}")

# ── Step 2: Quality filtering ─────────────────────────────
valid = [p for p in all_pairs if len(p.get("question","")) >= 20 and len(p.get("answer","")) >= 5]
print(f"After malformed filter: {len(valid)}")

grounded = []
for p in valid:
    if not p.get("answerable", True):
        p["grounding_score"] = 0.0
        grounded.append(p)
        continue
    doc_text = docs_by_id.get(p["source_document_id"], "")
    a_emb = embed_model.encode(p["answer"])
    d_emb = embed_model.encode(doc_text[:800])
    sim = float(cosine_similarity([a_emb], [d_emb])[0][0])
    p["grounding_score"] = sim
    if sim >= 0.35:
        grounded.append(p)
print(f"After grounding filter: {len(grounded)}")

import numpy as np
questions = [p["question"] for p in grounded]
q_embs = embed_model.encode(questions)
sim_matrix = cosine_similarity(q_embs)
np.fill_diagonal(sim_matrix, 0)
keep = np.ones(len(grounded), dtype=bool)
for i in range(len(grounded)):
    if keep[i]:
        keep[(sim_matrix[i] >= 0.90) & (np.arange(len(grounded)) > i)] = False
final = [p for p, k in zip(grounded, keep) if k]
print(f"After dedup filter: {len(final)}")

# ── Step 3: Statistics ────────────────────────────────────
df = pd.DataFrame(final)
print(f"\n── Dataset Statistics ───────────────────────")
print(f"Total: {len(final)}")
print(f"Difficulty:\n{df['difficulty'].value_counts()}")
print(f"Avg grounding score: {df['grounding_score'].mean():.3f}")

# ── Step 4: Save and seal ─────────────────────────────────
output = "/tmp/eval_dataset_v1.jsonl"
with open(output, "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")

checksum = hashlib.sha256(open(output, "rb").read()).hexdigest()
open(output + ".sha256", "w").write(checksum)
print(f"\nSaved: {output}")
print(f"SHA256: {checksum[:16]}...")
```

**Extensions:**
- Add persona-driven generation for 4 personas × 3 questions × 3 documents
- Add adversarial examples (unanswerable, contradictory premises)
- Compare grounding score threshold 0.3 vs 0.5 and measure impact on dataset size
- Generate multi-turn conversations and verify context retention

---

> ### 📋 Chapter Summary
>
> - Synthetic data solves the bootstrapping problem — you can measure quality before any production data exists.
> - **QA pair generation** from documents is the core technique; diversity requires explicit difficulty and question type variation.
> - **Persona-driven generation** captures the real diversity of user query styles.
> - **Adversarial generation** exposes system weaknesses: unanswerable questions, false premises, multi-hop reasoning.
> - All synthetic data requires **quality filtering**: malformed removal, grounding checks, and deduplication.

---

> ### ❓ Comprehension Questions
>
> 1. A synthetic evaluation dataset has 90% easy factual questions. How would you modify the generation prompt to produce a balanced difficulty distribution?
> 2. Explain the grounding score filter. What does a low grounding score indicate about a generated answer?
> 3. A team uses GPT-4o to generate a test set and GPT-4o as the system under test. What evaluation bias does this introduce?
> 4. Persona-driven generation produces diverse queries. What is the risk of over-representing one persona?
> 5. Your system must handle unanswerable queries gracefully. How do you include them in an evaluation set and what metric measures this capability?

---

## References

### Papers
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Es et al., 2023. Includes synthetic dataset generation methodology.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al., 2022.
- [Generating Training Data with LLMs for Information Extraction](https://arxiv.org/abs/2205.09712) — Møller et al., 2022.

### Documentation
- [OpenAI Fine-tuning Data Preparation](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [RAGAS TestsetGenerator](https://docs.ragas.io/en/latest/getstarted/testset_generation.html)
- [LangSmith Dataset Management](https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application)

---

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

## Chapter 5 — Data Augmentation

### 5.1 When Augmentation Is Needed

Data augmentation artificially expands a dataset by generating variants of existing examples. In LLM systems engineering, augmentation addresses three problems:

**Class imbalance in evaluation sets.** A generated evaluation set may have 90% factual questions and 10% inferential. Augmenting underrepresented categories creates a balanced distribution.

**Vocabulary coverage gaps.** A system trained on formal English may fail on informal queries. Augmenting the test set with informal paraphrases exposes this gap before production.

**Robustness testing.** A model may perform well on clean inputs but fail with typos, mixed languages, or noisy formatting. Augmenting with noisy variants tests this robustness.

---

### 5.2 Paraphrase Augmentation

Generate semantically equivalent variants of existing examples to expand coverage and test consistency.

```python
def paraphrase_question(
    question: str,
    n_variants: int = 3,
    model: str = "gpt-4o-mini"
) -> list[str]:
    """Generate paraphrase variants in different registers."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Generate {n_variants} paraphrase variants of the question.
Vary the register: formal, informal, and technical.
Each variant must have the same information need as the original.
Return JSON: {{"variants": ["variant1", "variant2", ...]}}"""
            },
            {"role": "user", "content": f"Original: {question}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    return json.loads(response.choices[0].message.content).get("variants", [])

def augment_with_paraphrases(
    records: list[dict],
    n_variants: int = 2
) -> list[dict]:
    """Variants inherit the original answer — valid only for factual answers."""
    augmented = []
    for record in records:
        variants = paraphrase_question(record["question"], n_variants=n_variants)
        for i, variant in enumerate(variants):
            augmented.append({
                **record,
                "id": f"{record['id']}_aug{i}",
                "question": variant,
                "is_augmented": True,
                "original_id": record["id"],
                "augmentation_type": "paraphrase"
            })
    return augmented
```

---

### 5.3 Back-Translation

Back-translation generates paraphrases by translating to an intermediate language and back. It produces naturally varied phrasing via translation divergence.

```python
def back_translate(
    text: str,
    intermediate_language: str = "Spanish",
    model: str = "gpt-4o-mini"
) -> str:
    """Translate to intermediate language, then back to English."""
    # Step 1: to intermediate
    intermediate = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Translate to {intermediate_language}. Return only the translation."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()

    # Step 2: back to English
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Translate to English. Return only the translation."},
            {"role": "user", "content": intermediate}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()

def augment_with_back_translation(
    records: list[dict],
    languages: list[str] = ("Spanish", "French", "German"),
    field: str = "question"
) -> list[dict]:
    augmented = []
    for record in records:
        for lang in languages:
            bt = back_translate(record[field], intermediate_language=lang)
            augmented.append({
                **record,
                "id": f"{record['id']}_bt_{lang[:2].lower()}",
                field: bt,
                "is_augmented": True,
                "original_id": record["id"],
                "augmentation_type": f"back_translation_{lang}"
            })
    return augmented
```

---

### 5.4 Noise Injection for Robustness

Deliberately introduce realistic noise to test system robustness.

```python
import random, string, re

class NoiseInjector:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def add_typos(self, text: str, error_rate: float = 0.05) -> str:
        result = list(text)
        for i in range(len(result)):
            if random.random() < error_rate and result[i].isalpha():
                op = random.choice(["substitute", "delete", "insert", "swap"])
                if op == "substitute":
                    result[i] = random.choice(string.ascii_lowercase)
                elif op == "delete":
                    result[i] = ""
                elif op == "insert":
                    result[i] = result[i] + random.choice(string.ascii_lowercase)
                elif op == "swap" and i < len(result) - 1:
                    result[i], result[i+1] = result[i+1], result[i]
        return "".join(result)

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

    def add_whitespace_noise(self, text: str) -> str:
        return re.sub(r' ', lambda m: '  ' if random.random() < 0.1 else ' ', text)

def create_robustness_test_set(
    records: list[dict],
    noise_types: list[str] = None
) -> list[dict]:
    noise_types = noise_types or ["typos", "lowercase", "no_punctuation"]
    injector = NoiseInjector()
    fns = {
        "typos":          lambda t: injector.add_typos(t, 0.08),
        "lowercase":      injector.lowercase,
        "no_punctuation": injector.remove_punctuation,
        "whitespace":     injector.add_whitespace_noise,
    }
    augmented = []
    for record in records:
        for noise_type in noise_types:
            fn = fns.get(noise_type)
            if fn:
                augmented.append({
                    **record,
                    "id": f"{record['id']}_noise_{noise_type}",
                    "question": fn(record["question"]),
                    "is_augmented": True,
                    "augmentation_type": f"noise_{noise_type}"
                })
    return augmented
```

---

### 5.5 Augmentation Pipelines in Production

Augmentation is a pipeline stage — reproducible, configurable, and bounded.

```python
from dataclasses import dataclass
from enum import Enum

class AugmentationType(str, Enum):
    PARAPHRASE = "paraphrase"
    BACK_TRANSLATION = "back_translation"
    NOISE = "noise"

@dataclass
class AugmentationConfig:
    augmentation_types: list[AugmentationType]
    paraphrase_n_variants: int = 2
    back_translation_languages: list[str] = None
    noise_types: list[str] = None
    max_augmentation_ratio: float = 3.0  # Never grow beyond 3x original

    def __post_init__(self):
        if self.back_translation_languages is None:
            self.back_translation_languages = ["Spanish", "French"]
        if self.noise_types is None:
            self.noise_types = ["typos", "lowercase"]

class AugmentationPipeline:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def run(self, records: list[dict]) -> dict:
        original_count = len(records)
        all_records = list(records)
        max_additions = int(original_count * (self.config.max_augmentation_ratio - 1))
        total_added = 0
        report = {"original": original_count, "counts": {}}

        if AugmentationType.PARAPHRASE in self.config.augmentation_types:
            added = augment_with_paraphrases(records, self.config.paraphrase_n_variants)
            added = added[:max_additions - total_added]
            all_records.extend(added)
            total_added += len(added)
            report["counts"]["paraphrase"] = len(added)

        if AugmentationType.NOISE in self.config.augmentation_types:
            added = create_robustness_test_set(records, self.config.noise_types)
            added = added[:max_additions - total_added]
            all_records.extend(added)
            total_added += len(added)
            report["counts"]["noise"] = len(added)

        report["final_count"] = len(all_records)
        report["augmentation_ratio"] = round(len(all_records) / original_count, 2)
        return {"records": all_records, "report": report}
```

---

> ### 📋 Chapter Summary
>
> - Augmentation expands datasets to cover vocabulary gaps, class imbalance, and robustness test cases.
> - **Paraphrase augmentation** generates stylistically varied versions of existing queries.
> - **Back-translation** produces naturally varied paraphrases via translation divergence.
> - **Noise injection** tests robustness to typos, casing, and punctuation.
> - Augmentation pipelines must be **configurable, reproducible, and bounded** via `max_augmentation_ratio`.

---

> ### ❓ Comprehension Questions
>
> 1. A RAG system performs well on clean test queries but poorly on mobile queries with typos. What augmentation strategy would you apply?
> 2. Back-translation through Spanish produces different paraphrases than through French. Why, and what does this imply about using multiple intermediate languages?
> 3. An augmentation pipeline produces a 10x expansion. What problems arise from excessive augmentation?
> 4. Paraphrase augmentation reuses the original answer. Under what conditions is this assumption incorrect?
> 5. Design an augmentation pipeline for a multilingual RAG system supporting English, Spanish, and French.

---

## References

### Papers
- [EDA: Easy Data Augmentation for Text Classification](https://arxiv.org/abs/1901.11196) — Wei & Zou, 2019.
- [Back-Translation as Data Augmentation for NMT](https://arxiv.org/abs/1511.06709) — Sennrich et al., 2016.
- [Data Augmentation Approaches in NLP: A Survey](https://arxiv.org/abs/2110.01852) — Feng et al., 2021.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al., 2022.

### Documentation
- [Hugging Face Datasets Processing](https://huggingface.co/docs/datasets/process) — Dataset transformation.
- [nlpaug](https://github.com/makcedward/nlpaug) — Text augmentation library.
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning) — Data preparation including augmentation.
- [LangChain4j Documentation](https://docs.langchain4j.dev) — Java dataset handling.

---

> **Navigation**
> [← Part IV — Advanced RAG](part_04_advanced_rag.md) | [→ Part VI — Artifact Engineering](part_06_artifact_engineering.md)
