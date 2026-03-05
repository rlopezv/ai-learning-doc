## Chapter 6 — Context Construction

### 6.1 From Retrieved Chunks to Model Input

Retrieval produces a set of ranked, relevant chunks. Context construction is the process of assembling these chunks into the prompt that the LLM will reason over. It is the bridge between the retrieval and generation stages.

Context construction decisions affect three things:

**Answer quality.** How the model receives and interprets the context — ordering, formatting, attribution — directly influences generation accuracy.

**Token budget.** Each chunk consumes tokens. Context construction must fit the available budget without truncating critical information.

**Traceability.** Well-constructed context enables the model to cite sources, enabling verification and trust.

---

### 6.2 Context Window Budget Management

A typical production context window budget allocation for a RAG system:

```
Total context window: 16,000 tokens
─────────────────────────────────────
System prompt:            ~800 tokens  (5%)
Conversation history:   ~2,000 tokens  (12.5%)
Retrieved context:      ~8,000 tokens  (50%)
Query + instructions:     ~500 tokens  (3%)
Reserved for output:    ~4,700 tokens  (29.5%)
```

```python
import tiktoken

class ContextBudgetManager:
    def __init__(
        self,
        model: str = "gpt-4o",
        total_budget: int = 16_000,
        output_reserve: int = 4_000
    ):
        self.enc = tiktoken.encoding_for_model(model)
        self.available = total_budget - output_reserve

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def fit_chunks_to_budget(
        self,
        chunks: list[str],
        system_prompt: str,
        query: str,
        history: str = ""
    ) -> list[str]:
        """
        Select as many top-ranked chunks as fit within the context budget.
        Chunks are assumed pre-ranked (best first).
        """
        fixed_tokens = (
            self.count_tokens(system_prompt)
            + self.count_tokens(query)
            + self.count_tokens(history)
            + 200  # Formatting overhead
        )
        remaining_budget = self.available - fixed_tokens
        selected, used = [], 0

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)
            if used + chunk_tokens <= remaining_budget:
                selected.append(chunk)
                used += chunk_tokens
            else:
                break  # No more budget

        return selected
```

---

### 6.3 Context Ordering Strategies

The order in which chunks are presented to the model affects generation quality. Research on the ["lost-in-the-middle"](https://arxiv.org/abs/2307.03172) phenomenon (Liu et al., 2023) shows that LLMs tend to use information at the beginning and end of the context window more effectively than information in the middle.

```python
from enum import Enum
from typing import Callable

class ContextOrder(Enum):
    RELEVANCE_DESC = "relevance_desc"   # Most relevant first (baseline)
    RELEVANCE_ASC = "relevance_asc"     # Most relevant last
    LOST_IN_MIDDLE = "lost_in_middle"   # Highest relevance at edges
    CHRONOLOGICAL = "chronological"     # By document date

def order_context_chunks(
    chunks: list[dict],  # Each: {"content": str, "rerank_score": float, "metadata": dict}
    strategy: ContextOrder = ContextOrder.LOST_IN_MIDDLE
) -> list[str]:
    if strategy == ContextOrder.RELEVANCE_DESC:
        ordered = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

    elif strategy == ContextOrder.RELEVANCE_ASC:
        ordered = sorted(chunks, key=lambda x: x.get("rerank_score", 0))

    elif strategy == ContextOrder.LOST_IN_MIDDLE:
        # Place highest-scoring chunks at start and end, lower in middle
        sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
        if len(sorted_chunks) <= 2:
            ordered = sorted_chunks
        else:
            result = []
            left, right = [], []
            for i, chunk in enumerate(sorted_chunks):
                if i % 2 == 0:
                    left.append(chunk)
                else:
                    right.append(chunk)
            ordered = left + list(reversed(right))

    elif strategy == ContextOrder.CHRONOLOGICAL:
        ordered = sorted(chunks, key=lambda x: x.get("metadata", {}).get("created_at", ""))

    return [c["content"] for c in ordered]
```

---

### 6.4 Source Attribution and Citations

Production RAG systems in regulated or high-stakes domains must trace generated content back to specific source documents. Attribution enables verification, audit trails, and user trust.

```python
def build_attributed_context(
    chunks: list[dict],
    include_source_markers: bool = True
) -> tuple[str, list[dict]]:
    """
    Build context with inline source markers.
    Returns (context_text, sources_list).
    """
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks, start=1):
        source_id = f"[{i}]"
        source_info = {
            "id": i,
            "title": chunk.get("metadata", {}).get("document_title", "Unknown"),
            "source": chunk.get("metadata", {}).get("source", ""),
            "heading": chunk.get("metadata", {}).get("heading", "")
        }
        sources.append(source_info)

        if include_source_markers:
            context_parts.append(f"{source_id} {chunk['content']}")
        else:
            context_parts.append(chunk['content'])

    context_text = "\n\n".join(context_parts)
    return context_text, sources

ATTRIBUTED_SYSTEM_PROMPT = """
You are a corporate knowledge assistant.
Answer the question using only the provided context.
When citing information, reference the source number in brackets, e.g. [1], [2].
If the answer is not in the context, state: "I cannot find this information in the available documents."
Do not invent information.
"""

def generate_attributed_answer(query: str, chunks: list[dict]) -> dict:
    context, sources = build_attributed_context(chunks)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ATTRIBUTED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }
```

---

### 6.5 Dynamic Context Assembly

Advanced systems dynamically adapt context construction based on query type, available budget, and retrieval quality.

```python
from enum import Enum

class QueryType(Enum):
    FACTUAL = "factual"           # Single-fact lookup
    COMPARATIVE = "comparative"   # Comparing two or more items
    PROCEDURAL = "procedural"     # Step-by-step instructions
    ANALYTICAL = "analytical"     # Synthesis across sources

def classify_query_type(query: str) -> QueryType:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Classify query as: factual, comparative, procedural, or analytical.
Respond with one word only."""
            },
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    raw = response.choices[0].message.content.strip().lower()
    try:
        return QueryType(raw)
    except ValueError:
        return QueryType.FACTUAL

def dynamic_context_assembly(
    query: str,
    retriever: "TwoStageRetriever",
    budget_manager: "ContextBudgetManager",
    system_prompt: str
) -> str:
    query_type = classify_query_type(query)

    # Adapt retrieval depth by query type
    k_map = {
        QueryType.FACTUAL: 3,
        QueryType.COMPARATIVE: 6,
        QueryType.PROCEDURAL: 5,
        QueryType.ANALYTICAL: 8
    }
    k = k_map[query_type]
    results = retriever.retrieve(query)[:k]

    # Apply lost-in-middle ordering for analytical queries
    ordering = (
        ContextOrder.LOST_IN_MIDDLE
        if query_type == QueryType.ANALYTICAL
        else ContextOrder.RELEVANCE_DESC
    )
    ordered_chunks = order_context_chunks(
        [{"content": r.content, "rerank_score": r.rerank_score, "metadata": {}}
         for r in results],
        strategy=ordering
    )

    # Fit to budget
    fitted = budget_manager.fit_chunks_to_budget(
        ordered_chunks, system_prompt, query
    )
    return "\n\n".join(fitted)
```

---

> ### 📋 Chapter Summary
>
> - Context construction assembles retrieved chunks into the model prompt: it determines which chunks are included, their order, and how they are formatted.
> - **Token budget management** is mandatory: system prompt, history, retrieved context, and output reserve must all be explicitly allocated.
> - **Context ordering matters**: the ["lost-in-the-middle"](https://arxiv.org/abs/2307.03172) effect shows LLMs favour information at the edges of the context window.
> - **Source attribution** — inline markers and a sources list — is essential for regulated or high-stakes RAG systems.
> - **Dynamic context assembly** adapts retrieval depth and ordering strategy based on query type classification.

---

> ### ❓ Comprehension Questions
>
> 1. A RAG system with a 128K context window is ingesting retrieved chunks without a budget manager. Describe the failure modes that emerge at scale and the engineering controls that prevent them.
> 2. Explain the "lost-in-the-middle" phenomenon. Design a context ordering strategy that mitigates it for a 10-chunk context window.
> 3. A compliance team requires that every AI-generated answer in a financial services application be traceable to a specific source document with page number. Design the metadata schema and context construction pipeline that supports this requirement.
> 4. A query is classified as "analytical". Why does this justify retrieving more chunks (K=8) than a factual query (K=3)?
> 5. A context budget manager receives 8 reranked chunks totalling 6,400 tokens but the available budget is 4,800 tokens. Describe two strategies for deciding which chunks to drop, and their respective trade-offs.

---

## References

### Papers
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu et al., 2023. Empirical evidence for context position effects on LLM performance.
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — Es et al., 2023. Context precision and recall metrics.
- [FActScoring: Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2305.14251) — Min et al., 2023. Source attribution and factual grounding evaluation.

### Documentation
- [OpenAI Chat Completions — Context Management](https://platform.openai.com/docs/guides/conversations) — Managing conversation history and context.
- [tiktoken](https://github.com/openai/tiktoken) — Accurate token counting for budget management.
- [LangChain Context Construction](https://python.langchain.com/docs/concepts/rag/) — RAG pipeline and context assembly patterns.
- [LangChain4j RAG — Augmentation](https://docs.langchain4j.dev/tutorials/rag#augmentor) — Java context augmentation reference.
- [Anthropic Context Window Documentation](https://docs.anthropic.com/en/docs/about-claude/models/overview) — Claude model context specifications.

---

> **Navigation**
> [← Part II — LLM Architectures](../architectures/index.md) | [→ Part IV — Advanced RAG](../advanced_rag/index.md)

---
[« Back to rag_engineering Index](index.md) | [🏠 Home](../index.md)