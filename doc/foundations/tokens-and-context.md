# Chapter 5 — Tokens and Context

[⬅ Back to Foundations](index.md)

## Context

Large language models process text as sequences of **tokens**, not as raw characters or words. Every request sent to an LLM—whether a prompt, retrieved context, or generated output—is internally represented as tokens. This representation directly affects how models interpret input, how much information can be processed in a single request, and how much the interaction costs.

Understanding tokens and context windows is therefore essential for AI systems engineering. Token limits influence prompt design, retrieval pipelines, system latency, and operational cost. In many real-world systems, especially Retrieval-Augmented Generation (RAG) architectures and conversational agents, **context management becomes a central engineering problem**.

This chapter explains how tokenization works, why token counts matter, how context windows constrain system design, and how engineers manage context efficiently in production AI systems.

---

## Concept Overview

Large language models operate on tokens rather than natural language strings.

A token is a **numerical representation of a piece of text** produced by a tokenizer. Tokens may correspond to:

- individual words
- parts of words
- punctuation
- whitespace

Example tokenization:

```text id="tok01"
Input text:
"Artificial intelligence systems are powerful."

Possible tokens:
["Artificial", " intelligence", " systems", " are", " powerful", "."]
```

Each token is mapped to an integer identifier used internally by the model.

```text id="tok02"
Text
↓
Tokenizer
↓
Token IDs
↓
Model Processing
```

Because models operate on tokens, **all system limits and costs are measured in tokens**, not characters or words.

**Key Concept — Tokens Define the Unit of Computation**

In LLM systems, tokens represent the fundamental unit of computation. Context limits, inference cost, and generation length are all measured in tokens.

---

## 5.1 The Token as the Fundamental Unit

Tokenization converts text into numerical sequences that can be processed by neural networks.

Common tokenization algorithms include:

- Byte Pair Encoding (BPE)
- WordPiece
- SentencePiece

These algorithms break text into units that balance vocabulary size with expressive power.

Example tokenization:

```text id="tok03"
"unbelievable"
↓
["un", "believ", "able"]
```

This approach allows models to represent large vocabularies while keeping the token dictionary manageable.

From a systems perspective, tokenization introduces a key engineering constraint: **all model inputs and outputs must fit within the model’s context window**.

---

## 5.2 What is a Token

Tokens do not necessarily correspond to words.

Typical patterns include:

| Text         | Possible Tokens    |
| ------------ | ------------------ |
| "AI systems" | ["AI", " systems"] |
| "running"    | ["run", "ning"]    |
| "2024"       | ["20", "24"]       |

As a rough rule of thumb:

```text id="tok04"
1 token ≈ 3–4 characters of English text
```

However, the exact mapping depends on the tokenizer and language.

Because token boundaries vary across models and languages, engineers must **measure token counts directly rather than estimating them from text length**.

---

## 5.3 Why Token Count Matters

Token counts influence several core aspects of system design.

### Model Constraints

Every model has a **maximum context window**, which limits the number of tokens that can be processed in a single request.

Example context windows:

| Model        | Context Window                |
| ------------ | ----------------------------- |
| GPT-3.5      | ~4k–16k tokens                |
| GPT-4        | up to ~128k tokens            |
| Newer models | hundreds of thousands or more |

If the combined token count of input and output exceeds this limit, the request must be truncated or rejected.

---

### Cost

Most LLM APIs charge based on token usage.

```text id="tok05"
Cost ≈ (input tokens + output tokens)
```

Large prompts, retrieved documents, and long responses increase operational cost.

---

### Latency

Processing more tokens increases inference time.

```text id="tok06"
More tokens
↓
More computation
↓
Higher latency
```

Systems handling large contexts must balance **accuracy, responsiveness, and cost**.

---

## 5.4 The Context Window

The **context window** is the maximum number of tokens a model can process in a single request.

The window includes multiple components:

```text id="tok07"
System prompt
+
User prompt
+
Retrieved context
+
Conversation history
+
Generated output
```

All of these must fit within the model’s context limit.

Example allocation:

```text id="tok08"
Context window: 32k tokens

System prompt:        500
User input:           300
Retrieved documents: 6000
Conversation history: 5000
Available generation: ~20k
```

Effective systems carefully allocate tokens across these components to maintain performance and reliability.

---

## 5.5 Attention Scaling and Long Contexts

Long contexts introduce computational challenges due to the **attention mechanism** used by transformer models.

Self-attention complexity grows approximately with the square of the sequence length:

```text id="tok09"
Attention complexity ≈ O(n²)
```

As token counts increase:

```text id="tok10"
Longer context
↓
More attention computations
↓
Higher latency and memory usage
```

This is why extremely large prompts may slow down inference significantly even when the model technically supports large context windows.

For system architects, this means that **maximizing context size is not always optimal**. Efficient context selection is often more effective than simply increasing prompt length.

---

## 5.6 Measuring Token Usage in Code

In production systems, token usage is typically measured programmatically.

Example using a tokenizer library:

```python id="tok11"
from tiktoken import get_encoding

enc = get_encoding("cl100k_base")
tokens = enc.encode("AI systems engineering is evolving rapidly.")
print(len(tokens))
```

Token measurement allows engineers to:

- estimate API cost
- enforce prompt limits
- control retrieval size
- optimize system performance

Monitoring token usage is a standard practice in production LLM systems.

---

## 5.7 Context Window Management in RAG Systems

Context management becomes especially important in **Retrieval-Augmented Generation systems**.

Typical pipeline:

```text id="tok12"
User Query
↓
Retriever
↓
Top-K Documents
↓
Context Builder
↓
LLM
```

The system must decide:

- how many documents to retrieve
- how large each chunk should be
- how to rank and filter results

If too much context is inserted, the model may:

- exceed context limits
- become slower
- dilute important information

This phenomenon is sometimes referred to as the **lost-in-the-middle problem**, where relevant information placed in the middle of long contexts receives less attention from the model.

Effective RAG systems therefore implement **context selection and ranking strategies**.

---

## 5.8 Chunking Strategies for Retrieval

Documents must typically be divided into smaller pieces before being embedded and indexed.

Common chunking strategies include:

| Strategy            | Description                                        |
| ------------------- | -------------------------------------------------- |
| Fixed-size chunking | Documents split into equal token segments          |
| Overlapping chunks  | Adjacent segments share tokens to preserve context |
| Sliding window      | Sequential overlapping segments                    |
| Semantic chunking   | Boundaries aligned with paragraphs or sections     |

Choosing the right chunk size is a trade-off:

- small chunks improve retrieval precision
- large chunks preserve semantic context

Chunking decisions significantly affect RAG system performance.

---

## 5.9 Token Budgeting in Conversational Systems

Conversational applications accumulate tokens over time as dialogue history grows.

Example conversation context:

```text id="tok13"
System prompt
+
User messages
+
Assistant responses
+
Retrieved documents
```

Without careful management, prompts may exceed context limits or become inefficient.

Common mitigation strategies include:

- conversation summarization
- selective message retention
- memory compression
- retrieval-based history reconstruction

Managing conversation memory is therefore an important aspect of production AI systems.

---

## 5.10 Generation Parameters and Their Effect

LLM responses can be influenced by generation parameters.

Common parameters include:

| Parameter         | Purpose                    |
| ----------------- | -------------------------- |
| temperature       | Controls randomness        |
| top_p             | Nucleus sampling threshold |
| max_tokens        | Maximum output length      |
| frequency_penalty | Reduces repetition         |
| presence_penalty  | Encourages topic diversity |

Example request structure:

```text id="tok14"
Prompt
↓
LLM
(parameters: temperature, max_tokens)
↓
Generated Output
```

These parameters allow engineers to balance:

- creativity
- determinism
- response length
- operational cost

---

## 5.11 Token Optimization Strategies

Efficient token usage is essential for production systems.

Common optimization techniques include:

### Prompt Compression

Removing redundant instructions or repeated context.

### Context Filtering

Selecting only the most relevant retrieved documents.

### Retrieval Ranking

Prioritizing documents with the highest semantic relevance.

### Conversation Summarization

Condensing long dialogue histories into shorter summaries.

Example optimization pipeline:

```text id="tok15"
User Query
↓
Retrieve Documents
↓
Filter + Rank
↓
Context Compression
↓
LLM
```

These strategies help maintain accuracy while controlling latency and cost.

---

## 📋 Chapter Summary

- Large language models process text as **tokens**, which represent the fundamental unit of computation.
- Token counts determine **context limits, inference cost, and system latency**.
- The **context window** defines the maximum number of tokens that can be processed in a single request.
- Long contexts increase computational cost due to **attention scaling** and may introduce issues such as the **lost-in-the-middle problem**.
- In systems such as Retrieval-Augmented Generation pipelines, engineers must carefully manage context size and token allocation.
- Efficient token management and context optimization are essential for building scalable AI systems.

---

## ❓ Comprehension Questions

1. Why do large language models operate on tokens rather than raw text or words?

2. How does token count influence both the cost and latency of LLM inference?

3. What components typically consume tokens within an LLM context window?

4. Why does context management become a critical design problem in Retrieval-Augmented Generation systems?

5. What architectural strategies can engineers use to control token growth in long-running conversational systems?

---

## References

### Documentation

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Interactive tokenizer tool.
- [tiktoken GitHub](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokeniser library.
- [OpenAI Models & Context Windows](https://platform.openai.com/docs/models) — Official model specs and token limits.
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)
- [Anthropic Claude Context Window](https://docs.anthropic.com/en/docs/about-claude/models/overview) — Claude model specifications.
- [LangChain4j Tokenizer](https://docs.langchain4j.dev/integrations/language-models/open-ai#tokenization) — Java tokenisation utilities.

### Articles

- [OpenAI Cookbook: How to Count Tokens](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) — Practical token counting guide.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu et al., 2023. Why large context windows do not guarantee performance with long inputs.

### Papers

- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508) — Bai et al., 2023. Evaluation of long-context model performance.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. , 2017

### Papers

- Vaswani et al. — _Attention Is All You Need_ (2017)
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### Documentation

- OpenAI Tokenizer Documentation
  [https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer)

---

## See Also

Related chapters:

- Chapter 4 — AI System Types
- Chapter 6 — Prompt Engineering
- Chapter 7 — Limitations of LLMs

---

## Key Takeaways

- Tokens are the **fundamental processing unit** for LLMs.
- Context windows limit how much information can be processed in a single request.
- Token usage directly impacts **system cost, latency, and scalability**.
- Retrieval pipelines must carefully manage context size and document chunking.
- Efficient token management is a core responsibility in production AI systems engineering.
