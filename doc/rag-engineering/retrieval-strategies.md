## Chapter 4 — Retrieval Strategies 🧪

### 4.1 Beyond Simple Similarity Search

The baseline retrieval strategy — embed the query, find the top-K closest vectors — is the correct starting point. It is simple, fast, and effective for well-formed queries against well-chunked corpora. But production systems encounter conditions that strain this baseline:

**Ambiguous queries.** "Tell me about the plan" — plan could refer to a subscription tier, a project roadmap, or a compliance plan.

**Multi-part questions.** "What is the refund policy for enterprise customers and how does it compare to the standard tier?" — requires two distinct retrievals.

**Vocabulary mismatch.** A user asks about "termination of contract"; the knowledge base uses "contract cancellation". Embedding similarity may not bridge this gap sufficiently.

**Diversity requirements.** Top-K retrieval by similarity tends to return semantically redundant chunks. The model receives five versions of the same information rather than five distinct perspectives.

Each of the strategies in this chapter addresses one or more of these conditions. As with chunking, the right strategy depends on your query distribution and corpus structure — and should be validated with explicit metrics.

---

### 4.2 Top-K Retrieval

The standard retrieval strategy. Returns the K chunks with highest cosine similarity to the query embedding.

```python
class VectorRetriever:
    def __init__(self, collection, embedding_service: "EmbeddingService", top_k: int = 5):
        self.collection = collection
        self.embedding_service = embedding_service
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_embedding = self.embedding_service.embed_single(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "content": doc,
                "metadata": meta,
                "similarity": 1 - dist  # Convert distance to similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
```

**Choosing K:** K is a balance between recall and context budget. A reasonable default is K=5 for 512-token chunks with a 16K context window. Measure [Recall@K](https://arxiv.org/abs/2309.15217) on your evaluation set to find the minimum K that captures the ground truth answer.

---

### 4.3 Threshold-Based Retrieval

Instead of always returning exactly K results, threshold-based retrieval returns only chunks above a minimum similarity score. This prevents the model from receiving low-quality, weakly relevant context.

```python
def threshold_retrieval(
    query: str,
    collection,
    embedding_service: "EmbeddingService",
    min_similarity: float = 0.75,
    max_results: int = 8
) -> list[dict]:
    query_embedding = embedding_service.embed_single(query)
    # Retrieve more than needed, then filter
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results,
        include=["documents", "metadatas", "distances"]
    )
    filtered = [
        {"content": doc, "metadata": meta, "similarity": 1 - dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
        if (1 - dist) >= min_similarity
    ]
    return filtered
```

**📐 Architecture decision:** Threshold retrieval can return zero results for out-of-domain queries, which is the correct behaviour — the model should not attempt to answer from irrelevant context. Implement a fallback response ("I cannot find relevant information in the knowledge base") rather than generating an answer from low-quality context.

---

### 4.4 Maximum Marginal Relevance

Maximum Marginal Relevance (MMR) balances relevance with diversity. It iteratively selects chunks that are relevant to the query but dissimilar to already-selected chunks, reducing redundancy.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_retrieval(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidate_texts: list[str],
    top_k: int = 5,
    lambda_param: float = 0.5  # 0=max diversity, 1=max relevance
) -> list[str]:
    """
    MMR iteratively selects the next chunk that maximises:
    lambda * similarity(chunk, query) - (1-lambda) * max_similarity(chunk, selected)
    """
    if not candidate_embeddings:
        return []

    query_arr = np.array(query_embedding).reshape(1, -1)
    cand_arr = np.array(candidate_embeddings)

    # Relevance scores: similarity to query
    relevance = cosine_similarity(query_arr, cand_arr)[0]

    selected_indices = []
    remaining = list(range(len(candidate_texts)))

    for _ in range(min(top_k, len(candidate_texts))):
        if not selected_indices:
            # First: select most relevant
            best = remaining[np.argmax([relevance[i] for i in remaining])]
        else:
            # MMR score for each remaining candidate
            selected_arr = cand_arr[selected_indices]
            scores = []
            for idx in remaining:
                rel = relevance[idx]
                redundancy = cosine_similarity(
                    cand_arr[idx].reshape(1, -1), selected_arr
                ).max()
                scores.append(lambda_param * rel - (1 - lambda_param) * redundancy)
            best = remaining[np.argmax(scores)]

        selected_indices.append(best)
        remaining.remove(best)

    return [candidate_texts[i] for i in selected_indices]
```

**When to use MMR:** When evaluation shows top-K retrieval returns multiple near-duplicate chunks (common with overlapping chunking or highly repetitive corpora). Lambda=0.7 is a reasonable starting point — prioritises relevance while penalising redundancy.

---

### 4.5 Multi-Query Retrieval

Multi-query retrieval generates multiple paraphrases of the original query, retrieves independently for each, and merges the results. This addresses vocabulary mismatch and query ambiguity.

```python
from openai import OpenAI
import json

client = OpenAI()

def generate_query_variants(query: str, n_variants: int = 3) -> list[str]:
    """
    Generate semantically equivalent query paraphrases
    to improve retrieval recall.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Generate {n_variants} different phrasings of the given question.
Each phrasing should retrieve different but relevant documents.
Respond with a JSON array of strings only."""
            },
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    data = json.loads(response.choices[0].message.content)
    # Handle both {"queries": [...]} and direct array
    if isinstance(data, dict):
        return list(data.values())[0]
    return data

def multi_query_retrieval(
    query: str,
    retriever: "VectorRetriever",
    n_variants: int = 3,
    top_k_per_query: int = 3
) -> list[dict]:
    """
    Retrieve for original query + N variants, deduplicate by content.
    """
    all_queries = [query] + generate_query_variants(query, n_variants)
    seen_contents = set()
    results = []

    for q in all_queries:
        for chunk in retriever.retrieve_top_k(q, k=top_k_per_query):
            # Deduplicate by content fingerprint
            fingerprint = chunk["content"][:100]
            if fingerprint not in seen_contents:
                seen_contents.add(fingerprint)
                results.append(chunk)

    # Re-rank merged results by similarity to original query
    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return results[:top_k_per_query * 2]
```

**Java — Multi-query retrieval with [LangChain4j](https://docs.langchain4j.dev):**
```java
interface QueryExpander {
    @dev.langchain4j.service.SystemMessage("""
        Generate 3 different phrasings of the given question.
        Return as JSON array. Example: ["phrasing 1", "phrasing 2", "phrasing 3"]
        """)
    String expand(String originalQuery);
}

List<String> allResults = new ArrayList<>();
String variantsJson = queryExpander.expand(userQuery);
List<String> variants = objectMapper.readValue(variantsJson, List.class);

// Retrieve for each variant and deduplicate
Set<String> seen = new HashSet<>();
for (String variant : variants) {
    List<Content> contents = contentRetriever.retrieve(
        new Query(variant, Metadata.from(userMessage, ChatMemory.SYSTEM_TOKEN))
    );
    contents.stream()
        .filter(c -> seen.add(c.textSegment().text().substring(0, 80)))
        .forEach(c -> allResults.add(c.textSegment().text()));
}
```

---

### 4.6 Contextual Compression Retrieval

Full retrieved chunks frequently contain content that is irrelevant to the specific question. Contextual compression passes each retrieved chunk through an LLM to extract only the relevant portion, reducing context noise.

```python
def contextual_compression(
    query: str,
    chunks: list[str],
    model: str = "gpt-4o-mini"
) -> list[str]:
    """
    Extract question-relevant content from each retrieved chunk.
    Returns empty string if chunk is irrelevant.
    """
    compressed = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """Extract only the content relevant to the question.
If the chunk contains no relevant information, respond with exactly: IRRELEVANT
Do not add explanation. Return only the relevant excerpt."""
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nChunk:\n{chunk}"
                }
            ],
            temperature=0,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        if result != "IRRELEVANT" and len(result) > 20:
            compressed.append(result)

    return compressed
```

**Cost consideration:** Contextual compression adds one LLM call per retrieved chunk. For top-K=5, this adds 5 additional calls per query. Use `gpt-4o-mini` or a local model to minimise cost. Apply only when evaluation shows that context noise is a measured quality problem.

---

### 4.7 Self-Querying Retrieval

Self-querying retrieval allows the LLM to translate a natural language question into a structured query combining semantic search with precise metadata filters.

```python
def self_querying_retrieval(
    natural_language_query: str,
    collection,
    embedding_service: "EmbeddingService"
) -> list[dict]:
    """
    LLM generates both the semantic query and the metadata filters
    from the natural language question.
    """
    # Step 1: LLM extracts query intent and metadata filters
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Analyse the question and extract:
1. semantic_query: the core question for vector search
2. filters: metadata conditions (department, year, doc_type, etc.)

Respond with JSON:
{
  "semantic_query": "...",
  "filters": {"department": "HR", "year": 2024}
}
If no filters apply, set filters to {}."""
            },
            {"role": "user", "content": natural_language_query}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    parsed = json.loads(response.choices[0].message.content)
    semantic_query = parsed["semantic_query"]
    filters = parsed.get("filters", {})

    # Step 2: Embed semantic query
    query_embedding = embedding_service.embed_single(semantic_query)

    # Step 3: Apply metadata filters if present
    where_clause = {f"${k}": {"$eq": v} for k, v in filters.items()} if filters else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
```

---

### 🧪 Hands-on Lab: Retrieval Strategy Benchmark

**Objective:** Compare top-K, MMR, and multi-query retrieval on a shared evaluation dataset. Measure hit rate and response diversity.

**Step 1 — Build shared index:**
```python
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client()
col = db.create_collection("retrieval_lab")

CORPUS = [
    "Enterprise customers receive priority support with a 4-hour response SLA.",
    "Standard support tier offers email support with 24-hour response time.",
    "The premium plan includes dedicated account management.",
    "Refunds for enterprise customers are processed within 14 business days.",
    "Standard refunds are processed within 5-7 business days.",
    "Annual subscriptions cancelled in the first 30 days receive a full refund.",
    "Monthly subscriptions have no minimum contract period.",
    "All plans include 99.9% uptime SLA for production environments.",
    "Downtime compensation is calculated as service credits for affected hours.",
    "Data export is available in CSV and JSON format for all paid plans.",
]

embeddings = embed_model.encode(CORPUS).tolist()
col.add(documents=CORPUS, embeddings=embeddings, ids=[str(i) for i in range(len(CORPUS))])

EVAL_QUERIES = [
    {"query": "What is the support SLA for enterprise?", "relevant_ids": ["0"]},
    {"query": "How long do standard refunds take?", "relevant_ids": ["4"]},
    {"query": "Can I cancel a monthly plan anytime?", "relevant_ids": ["6"]},
    {"query": "What formats can I export my data in?", "relevant_ids": ["9"]},
]
```

**Step 2 — Implement benchmark:**
```python
def top_k_retrieve(query: str, k: int = 3) -> list[str]:
    q_emb = embed_model.encode(query).tolist()
    res = col.query(query_embeddings=[q_emb], n_results=k)
    return res["ids"][0]

def mmr_retrieve(query: str, k: int = 3, candidate_k: int = 8, lam: float = 0.7) -> list[str]:
    q_emb = embed_model.encode(query).tolist()
    res = col.query(query_embeddings=[q_emb], n_results=candidate_k,
                    include=["embeddings", "documents"])
    ids = res["ids"][0]
    embs = res["embeddings"][0]
    docs = res["documents"][0]
    selected_docs = mmr_retrieval(q_emb, embs, docs, top_k=k, lambda_param=lam)
    # Map back to ids
    return [ids[docs.index(d)] for d in selected_docs if d in docs]

def evaluate(strategy_fn, name: str):
    hits = 0
    for item in EVAL_QUERIES:
        retrieved_ids = strategy_fn(item["query"])
        if any(rid in retrieved_ids for rid in item["relevant_ids"]):
            hits += 1
    rate = hits / len(EVAL_QUERIES)
    print(f"{name:<20} hit_rate={rate:.0%}  ({hits}/{len(EVAL_QUERIES)})")

evaluate(lambda q: top_k_retrieve(q, k=3), "Top-K (k=3)")
evaluate(lambda q: top_k_retrieve(q, k=5), "Top-K (k=5)")
evaluate(lambda q: mmr_retrieve(q, k=3, lam=0.7), "MMR (λ=0.7, k=3)")
evaluate(lambda q: mmr_retrieve(q, k=3, lam=0.3), "MMR (λ=0.3, k=3)")
```

**Step 3 — Extend the lab (optional):**

- Implement multi-query retrieval and add it to the benchmark (requires OpenAI API key)
- Add a diversity metric: measure average pairwise similarity among retrieved chunks (lower = more diverse)
- Vary the minimum similarity threshold in threshold-based retrieval and plot hit rate vs. threshold

---

> ### 📋 Chapter Summary
>
> - **Top-K retrieval** is the baseline; always start here and measure before adding complexity.
> - **Threshold-based retrieval** prevents low-confidence context from reaching the model; critical for out-of-domain query handling.
> - **MMR** reduces retrieval redundancy; most valuable when corpus contains repetitive or overlapping content.
> - **Multi-query retrieval** improves recall for ambiguous or vocabulary-mismatched queries at the cost of additional LLM calls.
> - **Contextual compression** reduces context noise; justified only when measured analysis shows it improves generation quality.
> - **Self-querying retrieval** enables natural language queries that combine semantic search with structured metadata filtering.

---

> ### ❓ Comprehension Questions
>
> 1. A RAG system returns 5 chunks for every query but 3 of them are near-identical. Which retrieval strategy addresses this, and how do you tune its primary parameter?
> 2. Explain the trade-off between `lambda=0.3` and `lambda=0.9` in MMR retrieval. For which query type would each extreme be appropriate?
> 3. A user asks "What is the refund policy for enterprise customers in Germany after 2023?". Which retrieval strategy is best suited to this query and why?
> 4. Contextual compression adds one LLM call per retrieved chunk. For a system processing 10,000 queries per day with K=5, calculate the additional cost using `gpt-4o-mini` (approximately $0.15 per million input tokens, 200 tokens per chunk).
> 5. Multi-query retrieval generates 3 paraphrases per query. If the original retrieval takes 50ms, estimate the additional latency and propose an architecture that parallelises the paraphrase queries.

---

## References

### Papers
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — Karpukhin et al., 2020. DPR and top-K retrieval fundamentals.
- [The Diversity of Retrieval](https://arxiv.org/abs/2210.10816) — Carbonell & Goldstein, 1998. Original MMR paper.
- [Query2Doc: Query Expansion with Large Language Models](https://arxiv.org/abs/2303.07678) — Wang et al., 2023. LLM-based query expansion for retrieval.
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — Es et al., 2023. Retrieval evaluation metrics.

### Documentation
- [LangChain Retrieval Strategies](https://python.langchain.com/docs/concepts/retrievers/) — Multi-query, contextual compression, and self-querying retrievers.
- [LangChain4j Content Retriever](https://docs.langchain4j.dev/tutorials/rag#content-retriever) — Java retrieval implementation.
- [LlamaIndex Retrieval Modes](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/) — Node post-processing and reranking.
- [RAGAS Documentation](https://docs.ragas.io) — Retrieval evaluation framework.

---

---
[« Back to rag-engineering Index](index.md) | [🏠 Home](../index.md)