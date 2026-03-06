## Chapter 4 — Index Artifacts

### 4.1 Vector Index as a Deployable Artifact

A vector index is not a mutable database table — it is a versioned artifact that is built, validated, and deployed atomically. This distinction has significant engineering implications.

**Why atomic deployment matters:** An in-place update during active serving creates a window where the index contains a mix of old and new embeddings. If the embedding model changed between builds, queries compare vectors from two different semantic spaces — a silent, hard-to-debug failure.

Treating the index as an artifact that is built, tested, and swapped atomically eliminates this class of failure.

---

### 4.2 Index Versioning and Blue-Green Deployment

```python
import time
from enum import Enum
from dataclasses import dataclass

class IndexStatus(str, Enum):
    BUILDING = "building"
    STAGING = "staging"
    ACTIVE = "active"
    DEPRECATED = "deprecated"

@dataclass
class IndexVersion:
    index_id: str
    version: str
    embedding_model: str
    chunk_strategy: str
    chunk_size: int
    document_count: int
    collection_name: str
    status: IndexStatus
    created_at: str
    content_hash: str
    evaluation_score: float = 0.0

class BlueGreenIndexManager:
    """
    Build new index in a separate collection (green),
    validate it, then switch traffic atomically.
    """
    def __init__(self, vector_client, embedding_service):
        self.client = vector_client
        self.embedding_service = embedding_service
        self.active_collection: Optional[str] = None

    def build_new_version(
        self,
        documents: list[dict],
        config: dict
    ) -> IndexVersion:
        """Build a new index in an isolated collection."""
        version = f"v{int(time.time())}"
        collection_name = f"index_{version}"

        # Ingest into new collection
        self._build_collection(collection_name, documents, config)

        return IndexVersion(
            index_id="main_kb",
            version=version,
            embedding_model=config["embedding_model"],
            chunk_strategy=config["chunk_strategy"],
            chunk_size=config["chunk_size"],
            document_count=len(documents),
            collection_name=collection_name,
            status=IndexStatus.STAGING,
            created_at=datetime.utcnow().isoformat(),
            content_hash=self._hash_documents(documents)
        )

    def validate(
        self,
        index_version: IndexVersion,
        eval_queries: list[dict],
        min_recall: float = 0.80
    ) -> bool:
        """Evaluate retrieval quality on staging index before promotion."""
        hits = sum(
            1 for item in eval_queries
            if any(
                rid in [r["id"] for r in self._search(index_version.collection_name, item["query"])]
                for rid in item["relevant_ids"]
            )
        )
        recall = hits / len(eval_queries)
        index_version.evaluation_score = recall
        print(f"Index validation: Recall@5 = {recall:.2%} (min: {min_recall:.2%})")
        return recall >= min_recall

    def promote(self, index_version: IndexVersion):
        """Atomically switch active collection to new version."""
        old = self.active_collection
        self.active_collection = index_version.collection_name
        index_version.status = IndexStatus.ACTIVE
        print(f"Promoted to active: {index_version.collection_name}")
        if old:
            print(f"Deprecated: {old}")

    def _build_collection(self, collection_name, documents, config):
        pass  # Vector DB specific implementation

    def _search(self, collection_name, query, top_k=5):
        return []  # Vector DB specific implementation

    def _hash_documents(self, documents: list[dict]) -> str:
        content = json.dumps(sorted([d.get("id", "") for d in documents]))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

---

### 4.3 Index Lineage

```python
from dataclasses import dataclass

@dataclass
class IndexLineage:
    index_version: str
    source_corpus_id: str
    source_corpus_version: str
    embedding_model: str
    chunking_pipeline_version: str
    document_count: int
    built_at: str
    quality_metrics: dict

def record_index_lineage(
    index_version: IndexVersion,
    corpus_metadata: dict,
    pipeline_versions: dict
) -> IndexLineage:
    return IndexLineage(
        index_version=index_version.version,
        source_corpus_id=corpus_metadata["corpus_id"],
        source_corpus_version=corpus_metadata["version"],
        embedding_model=index_version.embedding_model,
        chunking_pipeline_version=pipeline_versions.get("chunking", "unknown"),
        document_count=index_version.document_count,
        built_at=index_version.created_at,
        quality_metrics={"recall_at_5": index_version.evaluation_score}
    )
```

---

> ### 📋 Chapter Summary
>
> - Vector indexes are **deployable artifacts**, not in-place mutable databases.
> - **Blue-green deployment** builds in a separate collection, validates, then switches traffic atomically.
> - Index validation enforces a minimum Recall@K gate before promotion to production.
> - **Index lineage** records source documents, embedding model, and pipeline versions for each index build.

---

> ### ❓ Comprehension Questions
>
> 1. An in-place update adds documents embedded with a new model into an index built with an older model. What retrieval failure occurs and why?
> 2. Blue-green promotion switches `active_collection` atomically. What happens to in-flight queries at the switch moment?
> 3. Index Recall@5 = 0.75, below the 0.80 threshold. What diagnostics would you run before deciding whether to lower the threshold?
> 4. A corpus is updated with 10% new documents. Should you rebuild the entire index or incrementally update? Discuss the trade-offs.
> 5. Index lineage records the embedding model version. How does this support a legal audit of "what documents underpinned responses last month"?

---

## References

### Documentation
- [Qdrant Collection Management](https://qdrant.tech/documentation/concepts/collections/)
- [Weaviate Multi-tenancy](https://weaviate.io/developers/weaviate/manage-data/multi-tenancy)
- [Milvus Collection Alias](https://milvus.io/docs/manage-collections.md) — Atomic collection switching.
- [Chroma Documentation](https://docs.trychroma.com/getting-started)

### Papers
- [BEIR: Zero-shot Evaluation of Information Retrieval](https://arxiv.org/abs/2104.08663) — Thakur et al., 2021.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020.

---

---
[« Back to artifact-engineering Index](index.md) | [🏠 Home](../index.md)