## Chapter 5 — Reranking

### 5.1 Why First-Stage Retrieval Is Not Enough

Vector similarity search is a fast, approximate heuristic. It measures the geometric distance between dense embedding vectors — a proxy for semantic relevance that is effective but imperfect. Two failure modes are common in production:

**Semantic proximity ≠ answer relevance.** A chunk may be topically related to a query without actually containing the answer. A question about "annual subscription refund deadlines" may retrieve a chunk about "subscription benefits" (high cosine similarity) ahead of the specific refund policy (slightly lower similarity).

**Query-document asymmetry.** Bi-encoder embedding models encode queries and documents independently. They have no mechanism to reason about how well a specific document answers a specific question. Cross-encoder models, by contrast, attend jointly to query and document, producing significantly more accurate relevance scores.

Reranking addresses these failures by applying a more expensive, more accurate relevance model to the small candidate set returned by first-stage retrieval.

```
First stage:  Vector similarity → Top-20 candidates  (fast, approximate)
Second stage: Cross-encoder → Top-5 reranked         (slower, accurate)
```

This two-stage architecture — fast approximate retrieval followed by precise reranking — is the standard pattern in modern production RAG systems.

---

### 5.2 Cross-Encoder Reranking

A cross-encoder takes a (query, document) pair as joint input and produces a relevance score. Unlike bi-encoders (embedding models), cross-encoders attend to both texts simultaneously, capturing fine-grained relevance signals.

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder relevance model.
    🔓 Fully local — no API calls required.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[str],
        top_k: int = 5
    ) -> list[dict]:
        if not chunks:
            return []

        # Score all (query, chunk) pairs
        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)

        # Sort by score descending
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"content": text, "rerank_score": float(score)}
            for text, score in ranked[:top_k]
        ]

# Usage in RAG pipeline
retriever = VectorRetriever(collection, embedding_service, top_k=20)
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rag_with_reranking(query: str) -> str:
    # Stage 1: broad retrieval
    candidates = retriever.retrieve(query)
    candidate_texts = [c["content"] for c in candidates]

    # Stage 2: precise reranking
    reranked = reranker.rerank(query, candidate_texts, top_k=5)

    # Stage 3: generation with reranked context
    context = "\n\n".join([r["content"] for r in reranked])
    return generate_answer(query, context)
```

**Cross-encoder models ranked by quality/speed:**

| Model | Size | Latency | Quality | Use case |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | ~5ms/pair | Good | Production latency-sensitive |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | ~10ms/pair | Better | Production quality-focused |
| `BAAI/bge-reranker-large` | 560M | ~40ms/pair | Excellent | High-accuracy requirements |
| `Cohere rerank-english-v3.0` | API | ~200ms | Excellent | Managed service option |

---

### 5.3 LLM-Based Reranking

An LLM can act as a reranker by scoring or ordering retrieved chunks relative to a query. More flexible than cross-encoders but significantly more expensive.

```python
def llm_reranker(
    query: str,
    chunks: list[str],
    model: str = "gpt-4o-mini",
    top_k: int = 5
) -> list[str]:
    """
    Ask the LLM to rank chunks by relevance to the query.
    Returns top_k most relevant chunks.
    """
    numbered_chunks = "\n\n".join([
        f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)
    ])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Rank the provided text chunks by relevance to the question.
Return ONLY a JSON array of chunk numbers in order of relevance (most relevant first).
Example: [3, 1, 5, 2, 4]"""
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nChunks:\n{numbered_chunks}"
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    import json
    ranking = json.loads(response.choices[0].message.content)
    # Handle both {"ranking": [...]} and direct array
    if isinstance(ranking, dict):
        ranking = list(ranking.values())[0]

    reranked = [chunks[i - 1] for i in ranking if 1 <= i <= len(chunks)]
    return reranked[:top_k]
```

**📐 Architecture decision:** LLM reranking adds 1-2 seconds of latency and significant cost per query. Use cross-encoder reranking (local, milliseconds) for standard RAG pipelines. Reserve LLM reranking for offline evaluation pipelines or very low-volume, high-stakes retrieval where cost is not a constraint.

---

### 5.4 Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) is a score-free fusion algorithm that combines ranked lists from multiple retrieval strategies (e.g., vector search + BM25) into a single merged ranking.

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[str]:
    """
    Fuse multiple ranked lists using RRF.
    RRF score = sum over lists of 1 / (k + rank)
    k=60 is the standard value from the original paper.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank)

    # Sort by descending RRF score
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# Example: fuse vector search and BM25 results
vector_results = retriever.retrieve_texts(query)  # Top-20
bm25_results = bm25_retriever.retrieve_texts(query)  # Top-20

fused = reciprocal_rank_fusion([vector_results, bm25_results])
top_5 = fused[:5]
```

RRF is a key component of hybrid search, covered in depth in **[Part IV, Chapter 1](../advanced_rag/index.md)**.

---

### 5.5 Reranking in Production

A complete two-stage retrieval pipeline:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RetrievalResult:
    content: str
    source: str
    vector_score: float
    rerank_score: Optional[float] = None

class TwoStageRetriever:
    def __init__(
        self,
        vector_retriever: "VectorRetriever",
        reranker: "CrossEncoderReranker",
        first_stage_k: int = 20,
        final_k: int = 5
    ):
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self.first_stage_k = first_stage_k
        self.final_k = final_k

    def retrieve(self, query: str) -> list[RetrievalResult]:
        # Stage 1: vector similarity
        candidates = self.vector_retriever.retrieve_top_k(query, k=self.first_stage_k)

        # Stage 2: cross-encoder reranking
        reranked = self.reranker.rerank(
            query,
            [c["content"] for c in candidates],
            top_k=self.final_k
        )

        # Build final results with both scores
        content_to_vector_score = {c["content"]: c["similarity"] for c in candidates}
        return [
            RetrievalResult(
                content=r["content"],
                source=content_to_vector_score.get(r["content"], 0),
                vector_score=content_to_vector_score.get(r["content"], 0),
                rerank_score=r["rerank_score"]
            )
            for r in reranked
        ]
```

---

> ### 📋 Chapter Summary
>
> - First-stage vector retrieval is fast but approximate; **cross-encoder reranking** applies a more accurate relevance model to the candidate set.
> - The two-stage architecture — broad retrieval (top-20 to 50) followed by precise reranking (top-5) — is the production standard.
> - **Cross-encoder models** run fully locally ([sentence-transformers](https://www.sbert.net)) with millisecond latency; no API calls needed.
> - **LLM-based reranking** is more flexible but too expensive for real-time query pipelines; best suited to offline evaluation.
> - **Reciprocal Rank Fusion** merges multiple ranked lists without requiring score normalisation; essential for hybrid search.

---

> ### ❓ Comprehension Questions
>
> 1. A RAG system without reranking returns the ground-truth answer in position 4 or 5 of the top-5 results 40% of the time. After adding cross-encoder reranking, position 1 hit rate improves to 72%. Explain why this improvement occurs at an architectural level.
> 2. Compare the computational cost of `ms-marco-MiniLM-L-6-v2` reranking 20 candidates vs. one `gpt-4o-mini` API call for LLM reranking. At what query volume would the cost difference become significant?
> 3. Why does a bi-encoder embedding model have an inherent disadvantage in relevance scoring compared to a cross-encoder?
> 4. A system uses vector retrieval (top-20) followed by cross-encoder reranking (top-5). P99 latency is 800ms. The cross-encoder accounts for 650ms of this. What optimisations would you evaluate?
> 5. Explain Reciprocal Rank Fusion. Why is it preferable to simple score averaging when fusing results from vector search and BM25?

---

## References

### Papers
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) — Nogueira & Cho, 2019. Cross-encoder reranking with BERT.
- [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114) — Cormack et al., 2009. Original RRF paper.
- [BGE Reranker](https://arxiv.org/abs/2312.15503) — Xiao et al., 2023. State-of-the-art open-source reranker.
- [RankLLM: Reranking with Large Language Models](https://arxiv.org/abs/2309.15088) — Pradeep et al., 2023.

### Documentation
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/docs/cross_encoder/pretrained_models.html) — Available models and benchmarks.
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking) — Managed reranking service.
- [LangChain Reranking](https://python.langchain.com/docs/concepts/retrievers/#document-compressors-and-rerankers) — Integration patterns.
- [LlamaIndex Reranking](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/) — Reranker node postprocessors.

---

---
[« Back to rag_engineering Index](index.md) | [🏠 Home](../index.md)