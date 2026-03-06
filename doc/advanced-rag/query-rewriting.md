## Chapter 3 — Query Rewriting 🧪

### 3.1 Why Queries Fail

Users rarely formulate queries that are optimal for vector search. Common failure patterns:

**Vocabulary mismatch.** User asks "how do I cancel my account" but knowledge base uses "account termination procedure". Cosine similarity between these embeddings may be insufficient to retrieve the correct document.

**Incomplete context.** In a multi-turn conversation, "what about the enterprise tier?" refers to a topic discussed earlier. Without that context, the query fails.

**Overly broad queries.** "Tell me about pricing" produces low-recall, high-noise retrieval across many pricing-related documents.

**Multi-part questions.** "Compare the standard and enterprise plans, including SLAs and pricing" requires retrieving from at least four distinct sections.

Query rewriting transforms the user's input into one or more queries that are better suited for retrieval — before the retrieval stage runs.

---

### 3.2 Query Expansion

Query expansion generates additional search terms or paraphrases to broaden retrieval coverage.

```python
from openai import OpenAI
import json

client = OpenAI()

def expand_query(
    query: str,
    n_expansions: int = 3,
    model: str = "gpt-4o-mini"
) -> list[str]:
    """
    Generate semantically equivalent query variants to improve recall.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Generate {n_expansions} alternative phrasings of the user's question.
Each variant should approach the same information need from a different angle.
Return JSON: {{"variants": ["variant1", "variant2", ...]}}"""
            },
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    data = json.loads(response.choices[0].message.content)
    variants = data.get("variants", [])
    return [query] + variants  # Include original

# Example
expansions = expand_query("how do I cancel my account?")
# → ["how do I cancel my account?",
#    "account termination procedure",
#    "how to close or deactivate my subscription",
#    "steps to end my account access"]
```

**Contextual query expansion** — for multi-turn conversations, inject prior context:

```python
def contextual_rewrite(
    query: str,
    conversation_history: list[dict],
    model: str = "gpt-4o-mini"
) -> str:
    """
    Rewrite a query to be self-contained, incorporating conversation context.
    """
    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_history[-4:]  # Last 2 turns
    ])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Rewrite the follow-up question as a standalone, self-contained query.
Incorporate any necessary context from the conversation history.
Return only the rewritten query, no explanation."""
            },
            {
                "role": "user",
                "content": f"Conversation:\n{history_text}\n\nFollow-up: {query}"
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Example
history = [
    {"role": "user", "content": "Tell me about your pricing plans."},
    {"role": "assistant", "content": "We offer Standard ($50/mo) and Enterprise ($200/mo) plans."}
]
rewritten = contextual_rewrite("what about SLAs?", history)
# → "What are the SLA guarantees for the Standard and Enterprise pricing plans?"
```

---

### 3.3 Query Decomposition

Complex, multi-part questions are decomposed into a set of focused sub-queries, each retrievable independently.

```python
from pydantic import BaseModel

class DecomposedQuery(BaseModel):
    sub_queries: list[str]
    reasoning: str

def decompose_query(query: str) -> DecomposedQuery:
    """
    Break a complex query into focused sub-queries for independent retrieval.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Analyse the question and decompose it into simple, focused sub-questions.
Each sub-question should be retrievable independently from a knowledge base.
Return JSON: {"sub_queries": ["q1", "q2", ...], "reasoning": "..."}
If the question is already simple, return a single sub-query."""
            },
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    return DecomposedQuery(**json.loads(response.choices[0].message.content))

def retrieve_decomposed(
    query: str,
    retriever,
    top_k_per_subquery: int = 3
) -> list[dict]:
    decomposed = decompose_query(query)
    seen = set()
    results = []

    for sub_query in decomposed.sub_queries:
        chunks = retriever.retrieve(sub_query)[:top_k_per_subquery]
        for chunk in chunks:
            fp = chunk["content"][:80]
            if fp not in seen:
                seen.add(fp)
                chunk["sub_query"] = sub_query
                results.append(chunk)

    return results

# Example
decomposed = decompose_query(
    "Compare the Standard and Enterprise plans — what are the price, SLA, and user limits?"
)
# sub_queries:
#   "What is the price of the Standard plan?"
#   "What is the price of the Enterprise plan?"
#   "What is the SLA for the Standard plan?"
#   "What is the SLA for the Enterprise plan?"
#   "What are the user limits for each plan?"
```

---

### 3.4 HyDE: Hypothetical Document Embeddings

HyDE ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)) is an elegant technique: instead of embedding the query directly, ask the LLM to generate a hypothetical document that would answer the query, then embed that document for retrieval.

The intuition is that a hypothetical answer is semantically closer to real answer documents than the query itself, especially for short or ambiguous queries.

```
Traditional:  embed("refund policy enterprise customers") → search
HyDE:         LLM generates hypothetical answer → embed(answer) → search
```

```python
def hyde_retrieve(
    query: str,
    retriever,
    model: str = "gpt-4o-mini",
    top_k: int = 5
) -> list[dict]:
    """
    HyDE: generate hypothetical document, use its embedding for retrieval.
    """
    # Step 1: Generate hypothetical answer
    hypo_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Write a short, factual paragraph (3-5 sentences) that would 
directly answer the following question. Write as if you are a policy document.
Do not hedge or say you don't know — write a plausible answer."""
            },
            {"role": "user", "content": query}
        ],
        temperature=0.5,
        max_tokens=200
    )
    hypothetical_doc = hypo_response.choices[0].message.content

    # Step 2: Use hypothetical document embedding for retrieval
    results = retriever.retrieve_by_text(hypothetical_doc, top_k=top_k)
    for r in results:
        r["hyde_source"] = hypothetical_doc[:100]
    return results
```

**When HyDE helps most:** Queries that are short, ambiguous, or phrased very differently from the documents in the corpus. Measured improvement over standard retrieval typically ranges from 5–20% in recall on technical or specialised corpora.

**📐 Architecture decision:** HyDE adds one LLM call per query. Use it selectively — in an A/B test or for query types that have demonstrated retrieval quality gaps — rather than as a default pipeline stage.

---

### 3.5 Step-Back Prompting

Step-back prompting ([Zheng et al., 2023](https://arxiv.org/abs/2310.06117)) generates a more general, abstract version of the query before retrieval. This helps when the specific query requires broader contextual knowledge to answer correctly.

```python
def step_back_retrieve(
    query: str,
    retriever,
    model: str = "gpt-4o-mini"
) -> list[dict]:
    """
    Generate an abstract step-back query, retrieve for both,
    merge and return combined context.
    """
    # Generate step-back query
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Given a specific question, generate a more general, abstract question 
that would provide useful background context for answering the specific question.
Return only the abstract question."""
            },
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    stepback_query = response.choices[0].message.content.strip()

    # Retrieve for both queries
    specific_results = retriever.retrieve(query, top_k=3)
    general_results = retriever.retrieve(stepback_query, top_k=3)

    # Merge, deduplicate
    seen = set()
    merged = []
    for r in specific_results + general_results:
        fp = r["content"][:80]
        if fp not in seen:
            seen.add(fp)
            merged.append(r)

    return merged

# Example
# Original:  "What is the penalty for late payment on Invoice INV-2024-0847?"
# Step-back: "What are the standard terms and penalties in payment agreements?"
```

---

### 🧪 Hands-on Lab: Query Rewriting Pipeline

**Objective:** Build a modular query rewriting pipeline. Compare standard retrieval against HyDE and query expansion on vocabulary-mismatched queries.

**Prerequisites:** `openai`, `sentence-transformers`, `chromadb`

```python
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json

client = OpenAI()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client()
col = db.create_collection("qr_lab")

# Corpus uses formal/technical language
CORPUS = [
    "Account termination requires written notice submitted via the customer portal.",
    "Subscription cancellation takes effect at the end of the current billing cycle.",
    "Service discontinuation requests must include the account identifier and reason.",
    "Pricing schedule: Standard tier $50/month, Professional tier $150/month, Enterprise $500/month.",
    "Monthly recurring charges are invoiced on the first business day of each month.",
    "SLA uptime commitment is 99.9% for production environments, measured monthly.",
    "Critical incident response time is 1 hour for Enterprise, 4 hours for Professional.",
    "Data export functionality supports CSV, JSON, and XML formats.",
    "Personal data deletion requests are processed within 30 days per GDPR Article 17.",
    "API authentication uses OAuth 2.0 with JWT bearer tokens, expiring after 3600 seconds.",
]

embeddings = embed_model.encode(CORPUS).tolist()
col.add(documents=CORPUS, embeddings=embeddings, ids=[str(i) for i in range(len(CORPUS))])

# Evaluation: informal queries that should match formal corpus entries
EVAL = [
    {"query": "how do I cancel my subscription?",    "relevant_idx": [0, 1, 2]},
    {"query": "how much does it cost?",               "relevant_idx": [3]},
    {"query": "when do I get charged?",               "relevant_idx": [4]},
    {"query": "is the service reliable?",             "relevant_idx": [5]},
    {"query": "how fast do you fix critical issues?", "relevant_idx": [6]},
]

# ── Retrieval functions ────────────────────────────────────
def baseline_retrieve(query: str, k: int = 3) -> list[int]:
    q_emb = embed_model.encode(query).tolist()
    res = col.query(query_embeddings=[q_emb], n_results=k)
    return [int(i) for i in res["ids"][0]]

def expansion_retrieve(query: str, k: int = 3) -> list[int]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 'Generate 2 formal/technical variants of this query. Return JSON: {"variants": [...]}'},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"}, temperature=0.5
    )
    variants = json.loads(response.choices[0].message.content).get("variants", [])
    all_queries = [query] + variants

    scores: dict[int, float] = {}
    for q in all_queries:
        q_emb = embed_model.encode(q).tolist()
        res = col.query(query_embeddings=[q_emb], n_results=k)
        for rank, idx in enumerate(res["ids"][0], 1):
            scores[int(idx)] = scores.get(int(idx), 0) + 1/(60 + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:k]

def hyde_retrieve_lab(query: str, k: int = 3) -> list[int]:
    hypo = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a formal 2-sentence policy document excerpt that answers this question."},
            {"role": "user", "content": query}
        ],
        temperature=0.3, max_tokens=100
    ).choices[0].message.content
    q_emb = embed_model.encode(hypo).tolist()
    res = col.query(query_embeddings=[q_emb], n_results=k)
    return [int(i) for i in res["ids"][0]]

# ── Benchmark ─────────────────────────────────────────────
def evaluate(fn, name: str):
    hits = sum(
        1 for item in EVAL
        if any(idx in fn(item["query"]) for idx in item["relevant_idx"])
    )
    print(f"{name:<20} {hits}/{len(EVAL)} = {hits/len(EVAL):.0%}")

print("Strategy             Hits  HitRate")
print("─" * 38)
evaluate(baseline_retrieve, "Baseline")
evaluate(expansion_retrieve, "Query Expansion")
evaluate(hyde_retrieve_lab, "HyDE")
```

**Extensions:** Combine expansion + HyDE (generate variants, then HyDE each). Add step-back prompting. Test with a corpus in Spanish (change to `paraphrase-multilingual-MiniLM-L12-v2`).

---

> ### 📋 Chapter Summary
>
> - **Query expansion** generates paraphrases to improve recall for vocabulary-mismatched queries.
> - **Contextual rewriting** makes follow-up queries self-contained by incorporating conversation history.
> - **Query decomposition** splits multi-part questions into independent sub-queries for focused retrieval.
> - **HyDE** generates a hypothetical answer and uses its embedding for retrieval — particularly effective for short or ambiguous queries.
> - **Step-back prompting** generates an abstract version of the query to provide broader contextual retrieval.

---

> ### ❓ Comprehension Questions
>
> 1. A user in a multi-turn conversation asks "what about the cancellation fee?". Without contextual rewriting, why does this query fail in a stateless RAG system?
> 2. HyDE generates a hypothetical answer before retrieval. What are the risks of this approach, and under what conditions does the hypothetical answer degrade retrieval quality?
> 3. Query decomposition adds multiple LLM calls and multiple retrieval operations. Design an architecture that minimises latency while still benefiting from decomposition.
> 4. Your evaluation dataset shows that query expansion improves hit rate from 60% to 85% but increases P99 latency from 120ms to 340ms. How would you decide whether the quality gain justifies the latency cost?
> 5. Describe a query type for which step-back prompting would provide no benefit. Justify your answer.

---

## References

### Papers
- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496) — Gao et al., 2022. Hypothetical Document Embeddings.
- [Take a Step Back: Evoking Reasoning via Abstraction](https://arxiv.org/abs/2310.06117) — Zheng et al., 2023. Step-back prompting.
- [Query2Doc: Query Expansion with Large Language Models](https://arxiv.org/abs/2303.07678) — Wang et al., 2023.
- [Interleaving Retrieval with Chain-of-Thought Reasoning](https://arxiv.org/abs/2304.01083) — Trivedi et al., 2023. IRCoT — multi-step retrieval with reasoning.

### Documentation
- [LangChain Multi-Query Retriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) — Query expansion implementation.
- [LlamaIndex Query Transformations](https://docs.llamaindex.ai/en/stable/module_guides/querying/query_transformations/) — HyDE and step-back in LlamaIndex.
- [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat) — API reference for query rewriting.

---

---
[« Back to advanced-rag Index](index.md) | [🏠 Home](../index.md)