## Annual Subscriptions

Annual subscribers who cancel within the first 30 days receive a full refund.
Cancellations after 30 days receive a prorated refund for remaining full months.
"""

# Evaluation dataset: question → relevant section keywords
EVAL_DATASET = [
    {
        "question": "How long do I have to return a product?",
        "relevant_keywords": ["30 days", "return", "refund policy"]
    },
    {
        "question": "What is the refund policy for enterprise customers?",
        "relevant_keywords": ["enterprise", "90 days", "support contracts"]
    },
    {
        "question": "How long does a refund take to process?",
        "relevant_keywords": ["5-7 business days", "processing"]
    },
    {
        "question": "Can I get a refund on an annual subscription?",
        "relevant_keywords": ["annual", "30 days", "prorated"]
    },
    {
        "question": "Are digital products refundable?",
        "relevant_keywords": ["digital products", "non-refundable"]
    },
]
```

**Step 2 — Define chunking strategies:**

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
token_len = lambda t: len(enc.encode(t))

def strategy_fixed(text: str) -> list[str]:
    """Fixed 256-character chunks, no overlap."""
    size = 256
    return [text[i:i+size].strip() for i in range(0, len(text), size) if text[i:i+size].strip()]

def strategy_recursive(text: str) -> list[str]:
    """Recursive character splitter, 256 tokens, 32 overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=32, length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def strategy_structure(text: str) -> list[str]:
    """Split on Markdown headings."""
    import re
    chunks = []
    sections = re.split(r'\n(?=#{1,3} )', text)
    for section in sections:
        if section.strip():
            chunks.append(section.strip())
    return chunks

def strategy_overlap(text: str) -> list[str]:
    """Sliding window: 256 tokens, 128 token stride (50% overlap)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=128, length_function=token_len
    )
    return splitter.split_text(text)

STRATEGIES = {
    "fixed":      strategy_fixed,
    "recursive":  strategy_recursive,
    "structure":  strategy_structure,
    "overlap":    strategy_overlap,
}
```

**Step 3 — Build a collection per strategy and evaluate:**

```python
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
db_client = chromadb.Client()

def build_collection(name: str, chunks: list[str]):
    try:
        db_client.delete_collection(name)
    except Exception:
        pass
    col = db_client.create_collection(name)
    embeddings = embed_model.encode(chunks).tolist()
    col.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{name}_{i}" for i in range(len(chunks))]
    )
    return col

def evaluate_strategy(collection, eval_dataset: list[dict], top_k: int = 3) -> dict:
    hits = 0
    for item in eval_dataset:
        q_emb = embed_model.encode(item["question"]).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        retrieved = " ".join(results["documents"][0]).lower()
        # Hit: at least one keyword found in retrieved chunks
        if any(kw.lower() in retrieved for kw in item["relevant_keywords"]):
            hits += 1
    return {
        "hit_rate": hits / len(eval_dataset),
        "hits": hits,
        "total": len(eval_dataset)
    }

# Run comparison
print(f"{'Strategy':<12} {'Chunks':>7} {'Hit Rate':>10}  {'Hits':>6}")
print("-" * 42)
results = {}
for name, strategy_fn in STRATEGIES.items():
    chunks = strategy_fn(SAMPLE_DOCUMENT)
    col = build_collection(f"lab_{name}", chunks)
    metrics = evaluate_strategy(col, EVAL_DATASET)
    results[name] = {"chunks": len(chunks), **metrics}
    print(f"{name:<12} {len(chunks):>7} {metrics['hit_rate']:>10.0%}  {metrics['hits']:>4}/{metrics['total']}")
```

**Expected output (indicative):**
```
Strategy      Chunks   Hit Rate    Hits
──────────────────────────────────────────
fixed              9       60%    3/5
recursive          6       80%    4/5
structure          5      100%    5/5
overlap            9       80%    4/5
```

**Step 4 — Inspect what each strategy produced:**

```python
for name, strategy_fn in STRATEGIES.items():
    chunks = strategy_fn(SAMPLE_DOCUMENT)
    print(f"\n── {name.upper()} ({len(chunks)} chunks) ──")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {len(c):>4} chars | {c[:80].strip()}...")
```

**Step 5 — Extend the lab (optional):**

- Add your own document (a company policy, technical spec, or product manual)
- Add the hierarchical strategy and compare
- Replace `all-MiniLM-L6-v2` with `BAAI/bge-large-en-v1.5` — does hit rate change?
- Vary chunk size (128, 256, 512, 1024) and plot hit rate vs. chunk size

---

> ### 📋 Chapter Summary
>
> - **Chunking is the most impactful RAG design decision**: chunk size and strategy directly determine retrieval precision, generation quality, and token cost.
> - **Fixed-size** chunking is the simplest baseline; **recursive character** splitting is the general-purpose default.
> - **Semantic chunking** respects topic boundaries but is computationally expensive; best for long-form documents.
> - **Structure-aware chunking** is the highest-quality strategy for well-formatted documents (Markdown, HTML, manuals).
> - **Hierarchical (parent-child)** chunking decouples retrieval precision from generation context richness.
> - Every chunk should carry enriched metadata (source, heading, token count, creation date) to enable filtering and attribution.

---

> ### ❓ Comprehension Questions
>
> 1. A RAG system built on a 10,000-article knowledge base shows high retrieval recall but poor generation quality — the LLM receives the right sections but produces vague answers. What chunking-related factors might explain this, and what strategy changes would you evaluate?
> 2. An enterprise deploys a RAG system over legal contracts. Lawyers report that retrieved chunks frequently cut in the middle of a clause. Which chunking strategies would best address this, and what are the trade-offs of each?
> 3. Explain the parent-child chunking pattern. What problem does it solve that standard recursive chunking cannot?
> 4. Your ingestion pipeline processes 50,000 PDF documents nightly. Semantic chunking takes 8 hours; recursive character splitting takes 45 minutes. At what quality difference would you justify the computational cost of semantic chunking?
> 5. A chunk contains the heading text from its parent section prepended to the body content. Why does this improve retrieval quality, and what metadata field would you use to store the heading separately?

---

## References

### Papers
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — Karpukhin et al., 2020. Foundational paper on dense retrieval; chunk quality directly impacts DPR performance.
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — Es et al., 2023. Framework for measuring chunking and retrieval quality.
- [Improving Document Retrieval via Contextual Chunk Headers](https://arxiv.org/abs/2312.11702) — Anthropic, 2023. Evidence for heading-enriched chunks.

### Documentation
- [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/) — Comprehensive guide to chunking strategies in Python.
- [LangChain4j Document Splitters](https://docs.langchain4j.dev/tutorials/rag#document-splitter) — Java chunking reference.
- [LlamaIndex Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) — Alternative chunking implementations.
- [tiktoken](https://github.com/openai/tiktoken) — Token counting library for accurate chunk sizing.
- [Sentence Transformers](https://www.sbert.net) — Open-source embedding models for semantic chunking.

### Articles
- [Evaluating RAG: How to Measure Chunk Quality](https://www.pinecone.io/learn/chunking-strategies/) — Pinecone, 2024. Practical chunking strategy guide with benchmarks.
- [Five Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials) — Greg Kamradt, 2023. In-depth chunking comparison with evaluation.

---

---
[« Back to rag_engineering Index](index.md) | [🏠 Home](../../index.md)