## Chapter 2 — Embeddings

### 2.1 What Embeddings Represent

An embedding is a dense vector representation of text that encodes semantic meaning. Two texts with similar meaning will have embeddings close together in vector space, regardless of exact wording. This property is what enables semantic search.

```
"What is the refund policy?"   → [0.023, -0.187, 0.441, ...]
"How do I return a product?"   → [0.031, -0.201, 0.433, ...]
"Company financial results Q3" → [-0.312, 0.089, -0.123, ...]
```

The first two queries are semantically similar (high cosine similarity). The third is unrelated (low cosine similarity). A vector search will correctly retrieve chunks about refunds for both of the first queries, even though the words differ.

Understanding embeddings is critical not just for using them, but for diagnosing retrieval failures. Poor retrieval often traces to embedding model mismatch — using a general-purpose model for a specialised domain, or using different models for ingestion and query time.

---

### 2.2 Embedding Model Selection

The choice of embedding model is the second most important decision in a RAG system (after chunking). The [MTEB (Massive Text Embedding Benchmark)](https://arxiv.org/abs/2210.07316) leaderboard at [Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) provides standardised retrieval quality scores across models.

| Model | Dimensions | Max tokens | Speed | Cost | Best for |
|---|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | 8191 | Fast | $0.02/1M tokens | General English, cost-optimised |
| `text-embedding-3-large` | 3072 | 8191 | Medium | $0.13/1M tokens | High-accuracy English |
| `BAAI/bge-large-en-v1.5` 🔓 | 1024 | 512 | Medium | Free | On-premise English |
| `BAAI/bge-m3` 🔓 | 1024 | 8192 | Slow | Free | Multilingual, long context |
| `nomic-embed-text-v1.5` 🔓 | 768 | 8192 | Fast | Free | Long-context on-premise |
| `all-MiniLM-L6-v2` 🔓 | 384 | 256 | Very fast | Free | Development and prototyping |
| `Cohere embed-v3` | 1024 | 512 | Fast | Pay-per-use | Multilingual, Cohere stack |

**📐 Architecture decision:** Never mix embedding models between ingestion and query time. All documents in an index must be embedded with the same model. If you upgrade the model, the entire index must be rebuilt.

---

### 2.3 Generating Embeddings in Production

**Python — Production embedding service:**
```python
import time
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Literal

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Unified embedding service supporting both cloud and local models.
    Includes batching, retry logic, and rate limit handling.
    """
    def __init__(
        self,
        provider: Literal["openai", "local"] = "openai",
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        if provider == "openai":
            self.client = OpenAI()
        else:
            logger.info(f"Loading local model: {model}")
            self.local_model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with automatic batching."""
        all_embeddings = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start:batch_start + self.batch_size]
            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    return [item.embedding for item in response.data]
                else:
                    return self.local_model.encode(
                        batch,
                        batch_size=32,
                        normalize_embeddings=True
                    ).tolist()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt  # Exponential backoff
                logger.warning(f"Embedding attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                time.sleep(wait)
        return []

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

**Java — Embedding generation with [LangChain4j](https://docs.langchain4j.dev):**
```java
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.data.embedding.Embedding;

// Cloud: OpenAI
EmbeddingModel openAiModel = OpenAiEmbeddingModel.builder()
    .apiKey(System.getenv("OPENAI_API_KEY"))
    .modelName("text-embedding-3-small")
    .build();

// 🔓 On-premise: Ollama (nomic-embed-text running locally)
EmbeddingModel localModel = OllamaEmbeddingModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("nomic-embed-text")
    .build();

// Generate embedding
Response<Embedding> response = openAiModel.embed("What is the refund policy?");
float[] vector = response.content().vector();
System.out.println("Embedding dimensions: " + vector.length);
```

---

### 2.4 Embedding Dimensions and Trade-offs

Embedding dimensionality affects three system properties:

**Storage cost.** A 1536-dimension float32 embedding occupies 6.1KB. For 1 million chunks, that is 6.1GB of embedding storage — before the index overhead.

**Query latency.** Higher-dimensional vectors require more compute for similarity calculations. At scale, this becomes a latency concern.

**Quality.** Higher dimensions generally capture more semantic nuance, but with diminishing returns. `text-embedding-3-small` (1536d) achieves 90%+ of the quality of `text-embedding-3-large` (3072d) at 10% of the cost.

**Dimensionality reduction:** OpenAI's third-generation embedding models support Matryoshka Representation Learning (MRL), allowing truncation to lower dimensions with controlled quality loss:

```python
# Truncate to 256 dimensions — 6x storage reduction
response = client.embeddings.create(
    input=["text to embed"],
    model="text-embedding-3-small",
    dimensions=256  # Matryoshka truncation
)
```

---

### 2.5 Batch Embedding Pipelines

Production ingestion pipelines must process large corpora efficiently. Key optimisations:

```python
import asyncio
from openai import AsyncOpenAI

async def embed_corpus_async(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_concurrent: int = 5
) -> list[list[float]]:
    """
    Async batch embedding with concurrency control.
    Respects OpenAI rate limits while maximising throughput.
    """
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    all_embeddings = [None] * len(texts)

    async def embed_batch(batch: list[str], indices: list[int]):
        async with semaphore:
            response = await client.embeddings.create(
                input=batch,
                model=model
            )
            for i, item in zip(indices, response.data):
                all_embeddings[i] = item.embedding

    tasks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        indices = list(range(start, start + len(batch)))
        tasks.append(embed_batch(batch, indices))

    await asyncio.gather(*tasks)
    return all_embeddings
```

---

### 2.6 Embedding Evaluation

Embeddings should be evaluated on the specific domain and query patterns of your application, not just on general benchmarks.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_embedding_model(
    model_name: str,
    query_doc_pairs: list[tuple[str, str, float]],  # (query, doc, relevance_score)
    threshold: float = 0.7
) -> dict:
    """
    Evaluate an embedding model on domain-specific query-document pairs.
    relevance_score: 1.0 = highly relevant, 0.0 = irrelevant.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    predicted_similarities = []
    true_relevances = []

    for query, doc, relevance in query_doc_pairs:
        q_emb = model.encode(query)
        d_emb = model.encode(doc)
        sim = cosine_similarity([q_emb], [d_emb])[0][0]
        predicted_similarities.append(sim)
        true_relevances.append(relevance)

    # Spearman correlation between predicted similarity and human relevance
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(predicted_similarities, true_relevances)

    return {
        "model": model_name,
        "spearman_correlation": round(correlation, 4),
        "p_value": round(p_value, 4),
        "mean_similarity_relevant": np.mean([
            s for s, r in zip(predicted_similarities, true_relevances) if r > 0.7
        ]),
        "mean_similarity_irrelevant": np.mean([
            s for s, r in zip(predicted_similarities, true_relevances) if r < 0.3
        ])
    }
```

---

> ### 📋 Chapter Summary
>
> - Embeddings encode semantic meaning as dense vectors; similarity in vector space corresponds to semantic relatedness.
> - The embedding model must be **identical** at ingestion time and query time.
> - Model selection involves explicit trade-offs between quality, dimensionality, cost, and on-premise viability.
> - The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the standard reference for comparing embedding models on retrieval tasks.
> - Production embedding pipelines require batching, retry logic, and async concurrency to handle large corpora efficiently.
> - Embedding quality should be validated on **domain-specific** query-document pairs, not only on general benchmarks.

---

> ### ❓ Comprehension Questions
>
> 1. A team rebuilds their knowledge base with a higher-quality embedding model. They update the query pipeline to use the new model but keep the existing index. What will happen and why?
> 2. Your corpus contains 2 million chunks. You are comparing `text-embedding-3-small` (1536d) and `all-MiniLM-L6-v2` (384d). Calculate the storage difference and explain when the smaller model would be the better architectural choice.
> 3. Explain Matryoshka Representation Learning. What is the engineering benefit of supporting truncatable embeddings?
> 4. An embedding model achieves state-of-the-art MTEB scores but performs poorly on your enterprise support knowledge base. What is the likely cause and how would you address it?
> 5. Design an A/B test to compare two embedding models in a production RAG system without taking the system offline.

---

## References

### Papers
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — Muennighoff et al., 2022. Standard benchmark for embedding model evaluation.
- [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings](https://arxiv.org/abs/2402.03216) — Chen et al., 2024.
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — Kusupati et al., 2022. Flexible-dimension embeddings.
- [E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) — Wang et al., 2022.

### Documentation
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — Official embeddings API documentation and best practices.
- [Sentence Transformers Documentation](https://www.sbert.net) — Open-source embedding models and evaluation.
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Up-to-date embedding model comparison.
- [LangChain4j Embedding Models](https://docs.langchain4j.dev/integrations/embedding-models/) — Java embedding model integrations.
- [Cohere Embeddings Documentation](https://docs.cohere.com/docs/embeddings) — Multilingual embedding API.

---

---
[« Back to rag-engineering Index](index.md) | [🏠 Home](../index.md)