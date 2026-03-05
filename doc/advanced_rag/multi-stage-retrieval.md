## Chapter 2 — Multi-Stage Retrieval

### 2.1 Retrieval Pipeline Stages

Single-stage retrieval — one similarity search, done — is adequate for small corpora and simple queries. As corpus size, query complexity, and quality requirements grow, a multi-stage architecture becomes necessary.

Multi-stage retrieval applies progressively more expensive and accurate ranking functions to progressively smaller candidate sets:

```
Stage 1 — Candidate Generation    Large corpus → Top 100–500 candidates
                                   Fast, approximate (ANN + BM25)

Stage 2 — Filtering & Pruning     Top 100–500 → Top 20–50
                                   Metadata filters, threshold pruning,
                                   lightweight scoring

Stage 3 — Final Ranking           Top 20–50 → Top 5–10
                                   Cross-encoder reranking, MMR,
                                   business rule application

Stage 4 — Context Assembly        Top 5–10 → Model input
                                   Budget management, ordering,
                                   attribution markup
```

Each stage reduces the candidate set, allowing the next stage to apply a more expensive model to a smaller, higher-quality candidate pool.

---

### 2.2 Candidate Generation

The first stage casts a wide net. The goal is **high recall** — capturing all potentially relevant documents, accepting some noise. Speed is critical; this stage runs on every query.

```python
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

@dataclass
class Candidate:
    content: str
    source: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: dict = None

class CandidateGenerator:
    def __init__(
        self,
        vector_collection,
        embedding_service,
        bm25_retriever: "BM25Retriever",
        candidate_k: int = 50
    ):
        self.vector_collection = vector_collection
        self.embedding_service = embedding_service
        self.bm25 = bm25_retriever
        self.candidate_k = candidate_k

    def generate(self, query: str) -> list[Candidate]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_f = pool.submit(self._dense, query)
            sparse_f = pool.submit(self._sparse, query)
            dense = dense_f.result()
            sparse = sparse_f.result()

        return self._rrf_merge(dense, sparse)

    def _dense(self, query: str) -> list[Candidate]:
        q_emb = self.embedding_service.embed_single(query)
        res = self.vector_collection.query(
            query_embeddings=[q_emb],
            n_results=self.candidate_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            Candidate(
                content=doc, source=meta.get("source", ""),
                dense_score=1 - dist, metadata=meta
            )
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def _sparse(self, query: str) -> list[Candidate]:
        results = self.bm25.retrieve(query, top_k=self.candidate_k)
        return [
            Candidate(content=r["content"], source=r.get("source",""), sparse_score=r["bm25_score"])
            for r in results
        ]

    def _rrf_merge(self, dense: list[Candidate], sparse: list[Candidate]) -> list[Candidate]:
        scores: dict[str, float] = {}
        by_content: dict[str, Candidate] = {}
        for rank, c in enumerate(dense, 1):
            key = c.content[:80]
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
            by_content[key] = c
        for rank, c in enumerate(sparse, 1):
            key = c.content[:80]
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
            if key not in by_content:
                by_content[key] = c
        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
        for key in ranked:
            by_content[key].rrf_score = scores[key]
        return [by_content[k] for k in ranked]
```

---

### 2.3 Filtering and Pruning

Stage 2 applies fast, deterministic filters to reduce the candidate set before expensive reranking.

```python
from datetime import datetime

class CandidateFilter:
    def __init__(
        self,
        min_rrf_score: float = 0.005,
        allowed_sources: Optional[list[str]] = None,
        max_age_days: Optional[int] = None,
        max_candidates: int = 20
    ):
        self.min_rrf_score = min_rrf_score
        self.allowed_sources = set(allowed_sources) if allowed_sources else None
        self.max_age_days = max_age_days
        self.max_candidates = max_candidates

    def filter(self, candidates: list[Candidate]) -> list[Candidate]:
        result = candidates

        # Score threshold filter
        result = [c for c in result if (c.rrf_score or 0) >= self.min_rrf_score]

        # Source allowlist filter (e.g., department-scoped access)
        if self.allowed_sources:
            result = [c for c in result if c.source in self.allowed_sources]

        # Recency filter
        if self.max_age_days and result:
            cutoff = datetime.utcnow().timestamp() - (self.max_age_days * 86400)
            result = [
                c for c in result
                if (c.metadata or {}).get("created_ts", float('inf')) >= cutoff
            ]

        return result[:self.max_candidates]
```

---

### 2.4 Final Ranking

Stage 3 applies the most accurate, most expensive model to the small filtered set.

```python
from sentence_transformers import CrossEncoder

class FinalRanker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        final_k: int = 5
    ):
        self.model = CrossEncoder(model_name)
        self.final_k = final_k

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        pairs = [(query, c.content) for c in candidates]
        scores = self.model.predict(pairs)
        for c, score in zip(candidates, scores):
            c.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:self.final_k]
```

**Java — Multi-stage pipeline with [LangChain4j](https://docs.langchain4j.dev):**
```java
@Service
public class MultiStageRetrievalService {
    private final EmbeddingStoreContentRetriever denseRetriever;
    private final BM25Retriever sparseRetriever;
    private final CrossEncoderScoringModel reranker;

    public List<String> retrieve(String query) {
        // Stage 1: Candidate generation (dense + sparse)
        List<String> denseCandidates = denseRetriever.retrieve(query, 30);
        List<String> sparseCandidates = sparseRetriever.retrieve(query, 30);
        List<String> merged = rrfMerge(denseCandidates, sparseCandidates);

        // Stage 2: Filter to top 15
        List<String> filtered = merged.stream().limit(15).collect(Collectors.toList());

        // Stage 3: Cross-encoder reranking
        Map<String, Double> scores = new HashMap<>();
        for (String candidate : filtered) {
            double score = reranker.score(query, candidate);
            scores.put(candidate, score);
        }
        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(5)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }
}
```

---

### 2.5 Latency Budget Across Stages

Multi-stage retrieval adds latency. Each stage must fit within its budget.

| Stage | Operation | Target latency |
|---|---|---|
| 1 — Candidate generation | ANN search + BM25 (parallel) | 20–50ms |
| 2 — Filtering | Metadata + score filter | <5ms |
| 3 — Reranking | Cross-encoder on 20 candidates | 50–150ms |
| 4 — Context assembly | Token counting + ordering | <10ms |
| **Total retrieval** | | **80–215ms** |

At P99, multi-stage retrieval with cross-encoder reranking typically adds 100–200ms versus single-stage retrieval. For most enterprise use cases, this is acceptable. For latency-critical applications (<200ms end-to-end), use a smaller cross-encoder model or skip reranking entirely.

---

> ### 📋 Chapter Summary
>
> - Multi-stage retrieval applies progressively more expensive models to progressively smaller candidate sets: **generate → filter → rank → assemble**.
> - Stage 1 optimises for **recall**; stage 3 optimises for **precision**. These goals are in tension and require separate optimisation.
> - The total retrieval latency budget for multi-stage pipelines typically falls between 80–215ms — acceptable for most enterprise use cases.
> - Each stage is independently replaceable: improving the cross-encoder model in stage 3 requires no changes to stages 1 or 2.

---

> ### ❓ Comprehension Questions
>
> 1. Explain why stage 1 should optimise for recall rather than precision. What is the cost of a false negative in stage 1 that would be correctly handled in stage 3?
> 2. A multi-stage pipeline has a P99 latency of 450ms. Profiling reveals the cross-encoder accounts for 380ms. List three strategies to reduce cross-encoder latency without removing it from the pipeline.
> 3. A content filter in stage 2 restricts retrieval to documents tagged with the user's department. What are the security implications if this filter is bypassed, and how would you enforce it at the infrastructure level?
> 4. Design a fallback strategy for when stage 3 reranking service is unavailable. How should the pipeline degrade gracefully?
> 5. Compare multi-stage retrieval to a database query execution plan (table scan → index scan → filter → sort). What does this analogy reveal about the engineering principles involved?

---

## References

### Papers
- [Multi-Stage Document Ranking with BERT](https://arxiv.org/abs/1910.14424) — Nogueira et al., 2019. Multi-stage ranking with neural models.
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) — Nogueira & Cho, 2019.
- [The MS MARCO Ranking Dataset](https://arxiv.org/abs/1611.09268) — Nguyen et al., 2016. Benchmark for ranking systems.

### Documentation
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
- [LangChain4j Retrieval Pipeline](https://docs.langchain4j.dev/tutorials/rag)
- [Qdrant Query API](https://qdrant.tech/documentation/concepts/search/)

---

---
[« Back to advanced_rag Index](index.md) | [🏠 Home](../../index.md)