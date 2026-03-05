## Chapter 4 — Context Compression

### 4.1 Context Noise and Its Effects

Every retrieved chunk contains some content that is directly relevant to the query and some that is not. This irrelevant content — context noise — has measurable negative effects:

**Distraction.** LLMs can fixate on irrelevant passages and generate responses that reference them instead of the relevant content.

**Token waste.** Noisy chunks consume context window budget that could hold additional relevant sources.

**Hallucination amplification.** When the signal-to-noise ratio in context is low, models are more likely to "fill in" missing information from their parametric knowledge.

Context compression reduces retrieved chunks to their relevant essence before they enter the context window.

---

### 4.2 Extractive Compression

Extractive compression identifies and returns the specific sentences or spans within a chunk that are relevant to the query, discarding the rest.

```python
def extractive_compress(
    query: str,
    chunk: str,
    model: str = "gpt-4o-mini",
    min_length: int = 30
) -> Optional[str]:
    """
    Extract only the relevant portion of a chunk for the given query.
    Returns None if chunk contains no relevant content.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Extract the sentences from the text that are directly relevant to the question.
Return only the extracted sentences, no paraphrasing.
If no sentences are relevant, respond with exactly: IRRELEVANT"""
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nText:\n{chunk}"
            }
        ],
        temperature=0,
        max_tokens=300
    )
    result = response.choices[0].message.content.strip()
    if result == "IRRELEVANT" or len(result) < min_length:
        return None
    return result

def compress_candidates(
    query: str,
    chunks: list[str],
    max_workers: int = 5
) -> list[str]:
    """Compress all chunks in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(
            lambda chunk: extractive_compress(query, chunk),
            chunks
        ))
    return [r for r in results if r is not None]
```

---

### 4.3 Abstractive Compression

Abstractive compression generates a new, concise summary of the chunk tailored to the query — rather than extracting existing sentences. It can synthesise information across multiple chunks.

```python
def abstractive_compress(
    query: str,
    chunks: list[str],
    max_words: int = 150,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Summarise retrieved chunks in relation to the query.
    Produces a compressed, query-focused synthesis.
    """
    combined = "\n\n---\n\n".join(chunks)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Summarise the provided context focusing on information relevant to the question.
Maximum {max_words} words. Be precise and factual. Do not add information not present in the context."""
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{combined}"
            }
        ],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()
```

**📐 Architecture decision:** Abstractive compression is a double-edged tool. It can improve focus, but it also introduces an intermediary that may lose nuance, misinterpret the original source, or silently drop critical details. For regulated or audit-sensitive applications, extractive compression (which preserves original wording) is safer than abstractive.

---

### 4.4 LLMLingua: Token-Level Compression

[LLMLingua](https://arxiv.org/abs/2310.05736) (Microsoft Research, 2023) takes a different approach: instead of summarising or extracting, it removes individual tokens from the prompt that a small LM rates as low-importance, preserving the compressed prompt in near-original form.

```python
# 🔓 LLMLingua — fully local, no API calls
# pip install llmlingua

from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True
)

def compress_with_llmlingua(
    context: str,
    question: str,
    target_ratio: float = 0.5  # Compress to 50% of original tokens
) -> str:
    result = compressor.compress_prompt(
        context,
        question=question,
        target_token=int(len(context.split()) * target_ratio),
        condition_compare=True,
        condition_in_question="after"
    )
    return result["compressed_prompt"]
```

LLMLingua achieves 2–5x compression ratios while preserving 90%+ of answer quality in benchmarks. It runs locally and requires no LLM API calls for the compression step.

---

### 4.5 Compression in the Pipeline

Compression adds latency and cost — it should be applied only where it demonstrably improves quality. The recommended integration point is after reranking, before context assembly:

```python
class CompressedRAGPipeline:
    def __init__(
        self,
        retriever,
        reranker,
        budget_manager,
        compression_strategy: str = "none"  # none | extractive | llmlingua
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.budget_manager = budget_manager
        self.compression_strategy = compression_strategy

    def run(self, query: str, system_prompt: str) -> dict:
        # 1. Retrieve
        candidates = self.retriever.retrieve(query)

        # 2. Rerank
        reranked = self.reranker.rank(query, candidates)
        texts = [c.content for c in reranked]

        # 3. Compress (optional)
        if self.compression_strategy == "extractive":
            texts = compress_candidates(query, texts)
        elif self.compression_strategy == "llmlingua":
            combined = "\n\n".join(texts)
            texts = [compress_with_llmlingua(combined, query)]

        # 4. Fit to budget
        fitted = self.budget_manager.fit_chunks_to_budget(texts, system_prompt, query)

        # 5. Generate
        context = "\n\n".join(fitted)
        return {"context": context, "chunks_used": len(fitted)}
```

---

> ### 📋 Chapter Summary
>
> - Context noise reduces generation quality by distracting the model and wasting token budget.
> - **Extractive compression** returns relevant sentences verbatim — safer for regulated applications.
> - **Abstractive compression** generates a query-focused summary — higher information density but lossy.
> - **[LLMLingua](https://arxiv.org/abs/2310.05736)** removes low-importance tokens at the token level — fully local, no API cost, 2–5x compression.
> - Apply compression only where evaluation confirms quality improvement; it always adds latency.

---

> ### ❓ Comprehension Questions
>
> 1. A legal discovery RAG system retrieves contract clauses. A paralegal reports that the AI summaries sometimes miss critical sub-clauses. Which compression strategy would you remove from the pipeline and why?
> 2. LLMLingua compresses prompts at token level. How does this differ from extractive compression at sentence level? When would LLMLingua be preferable?
> 3. Compression adds one LLM call per chunk. For a system with K=5 reranked chunks and 10,000 daily queries, calculate daily API cost using `gpt-4o-mini` at $0.15/M input tokens with 400 tokens per chunk.
> 4. Abstractive compression across multiple chunks merges information from different sources. What attribution problem does this create and how would you address it?
> 5. Design an evaluation to determine whether adding extractive compression improves or degrades answer quality for your specific use case.

---

## References

### Papers
- [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736) — Jiang et al. (Microsoft), 2023.
- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) — Pan et al., 2024.
- [Compressing Context to Enhance Inference Efficiency of Large Language Models](https://arxiv.org/abs/2310.06201) — Li et al., 2023.

### Documentation
- [LLMLingua GitHub](https://github.com/microsoft/LLMLingua) — Microsoft's token-level compression library.
- [LangChain Contextual Compression](https://python.langchain.com/docs/how_to/contextual_compression/) — Extractive compression integration.
- [LlamaIndex Node Postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/) — Compression as a postprocessing step.

---

---
[« Back to advanced_rag Index](index.md) | [🏠 Home](../../index.md)