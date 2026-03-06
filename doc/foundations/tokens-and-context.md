# Chapter 5 — Tokens and Context

[⬅ Back to Foundations](index.md)

## Context

Large language models (LLMs) process and generate text using **tokens**, not raw text. Tokens represent smaller units of text such as words, subwords, or characters. Understanding how tokens work is essential for designing AI systems because tokenization directly affects **model input limits, system latency, cost, and prompt design**.

In addition to tokens, LLMs operate within a **context window**, which defines the maximum number of tokens that the model can process in a single request. All inputs to the model—including prompts, user queries, retrieved documents, and system instructions—must fit within this window.

For AI systems engineers, tokens and context windows influence architectural decisions such as:

- prompt construction
- retrieval chunking strategies
- system latency and cost
- memory management
- model selection

This chapter introduces the concepts of tokens and context windows and explains how they influence the design and behavior of AI systems.

---

## Concept Overview

Large language models operate on token sequences rather than natural language strings.

```

Text
↓
Tokenization
↓
Token IDs
↓
Model Processing
↓
Generated Tokens
↓
Text Output

```

When a user sends text to a model, the text is first **tokenized**, meaning it is converted into a sequence of tokens. Each token corresponds to a numerical identifier in the model's vocabulary.

For example:

```

Sentence:
"Large language models are powerful tools."

Tokens:
["Large", " language", " models", " are", " powerful", " tools", "."]

```

Tokenization is necessary because neural networks operate on **numerical representations**, not raw text.

**Key Concept — Tokens Are the Fundamental Unit of Model Computation**

All model operations—including inference, attention mechanisms, and generation—are performed on tokens.

It is important to note that **tokens do not correspond directly to words**. A single word may be split into multiple tokens, while some short words may be combined into a single token.

Example:

```

Word: "unbelievable"

Tokens:
["un", "believ", "able"]

```

This means token counts often differ from word counts, which is important when estimating cost or context usage.

---

## 5.1 Tokenization

Tokenization is the process of converting text into tokens that the model can process.

Different models use different tokenization strategies, including:

- **Byte Pair Encoding (BPE)**
- **WordPiece**
- **Unigram Language Models**
- **Byte-level tokenization**

These algorithms break text into pieces that balance vocabulary size with representation efficiency.

Typical properties of tokenization:

- common words often map to single tokens
- rare words may split into multiple tokens
- punctuation is often separate tokens
- whitespace can influence token boundaries

Tokenization efficiency also varies across languages. Languages such as English often require fewer tokens than languages with more complex morphology or character-based writing systems.

Because tokenization varies across models, the same text may produce different token counts depending on the model used.

---

## 5.2 Token Costs and Latency

Tokens are the primary unit used to measure **model usage and computational cost**.

Most LLM APIs charge based on the number of tokens processed during inference.

Token usage includes:

- input tokens
- generated output tokens

Example request:

```

Prompt tokens: 120
Generated tokens: 80
Total tokens processed: 200

```

Higher token counts increase:

- inference cost
- processing latency
- memory usage

LLMs generate text **token by token**, which means output length directly affects latency.

Because transformer attention mechanisms scale approximately with the **square of the token count (O(n²))**, larger contexts require significantly more computation.

For this reason, AI systems engineers often design prompts and pipelines to **minimize unnecessary token usage**.

Typical optimization strategies include:

- shorter prompts
- structured prompts instead of verbose instructions
- summarizing context before model calls
- limiting output length

Managing token usage is essential for maintaining **cost-efficient production systems**.

---

## 5.3 Token Budgeting

In production systems, engineers often allocate a **token budget** for different components of the prompt.

Example token budget:

| Component           | Token Budget |
| ------------------- | ------------ |
| System prompt       | 500          |
| User query          | 200          |
| Retrieved documents | 3,000        |
| Output allowance    | 1,000        |

Token budgeting helps ensure that prompts remain within the model's context window while preserving space for generated responses.

Token budgeting is especially important in **retrieval-augmented systems**, where retrieved documents may consume a large portion of the available context.

---

## 5.4 Context Windows

A **context window** defines the maximum number of tokens that a model can process in a single inference request.

All inputs to the model share the same context window, including:

- system prompts
- user messages
- retrieved documents
- tool responses
- conversation history

Example:

```

Context Window: 8,000 tokens

System prompt: 500
User query: 50
Retrieved documents: 2,000
Conversation history: 1,000
Remaining capacity: 4,450 tokens

```

If the total token count exceeds the context window, the model cannot process the request.

This constraint affects many aspects of system design, including:

- retrieval pipelines
- memory strategies
- prompt engineering
- conversation management

Modern models support increasingly large context windows, but the constraint remains an important engineering consideration.

---

## 5.5 Context Construction

In many AI systems, the context provided to the model is **constructed dynamically**.

Typical context components include:

```

System Instructions
+
User Input
+
Retrieved Knowledge
+
Conversation History

```

These elements must be combined into a prompt that fits within the model's context window.

This process is often called **context construction** or **prompt assembly**.

A typical context-building pipeline might look like this:

```

User Query
↓
Retriever
↓
Top-K Chunks
↓
Prompt Template
↓
Context Assembly
↓
Model

```

Effective context construction requires balancing:

- relevance
- token limits
- cost
- model performance

Poorly constructed context may lead to hallucinations, irrelevant responses, or wasted tokens.

---

## 5.6 Context Management Strategies

Because context windows are limited, AI systems often use strategies to manage token usage.

Common strategies include:

### Context Truncation

Older messages or less relevant content are removed to maintain token limits.

### Context Summarization

Conversation history or documents may be summarized to reduce token usage.

Example pipeline:

```

Conversation History
↓
Summarization Model
↓
Compressed Context

```

### Retrieval-Based Context

Instead of sending entire documents, systems retrieve only the most relevant text segments.

This strategy is commonly used in **Retrieval-Augmented Generation (RAG)** systems.

### Chunking

Large documents are split into smaller segments called **chunks**, which can be retrieved individually.

Chunk size strongly influences retrieval quality and token efficiency.

---

## 5.7 Long Context Models

Recent models support much larger context windows than earlier LLMs.

Examples of context window sizes:

| Model Type          | Typical Context Window |
| ------------------- | ---------------------- |
| Early LLMs          | 2k–4k tokens           |
| Modern LLMs         | 32k–128k tokens        |
| Long-context models | 200k+ tokens           |

Large context windows enable new capabilities such as:

- long document analysis
- multi-document reasoning
- extended conversations
- codebase understanding

However, large contexts introduce challenges.

One well-known issue is the **lost-in-the-middle problem**, where information placed in the middle of a long context is more likely to be ignored by the model.

Long contexts also increase:

- computational cost
- latency
- attention complexity

Because attention mechanisms scale with token count, extremely large contexts may degrade performance.

---

## 5.8 Context Window vs Memory

Context windows should not be confused with **system memory**.

The context window represents the tokens currently processed by the model.

Memory systems, by contrast, store information across interactions.

Examples of memory mechanisms include:

- conversation history
- vector database storage
- knowledge bases
- external state stores

Memory systems allow AI applications to maintain information beyond the limits of a single context window.

Example architecture:

```

User Query
↓
Memory Store
↓
Retriever
↓
Context Builder
↓
Model

```

Memory management becomes increasingly important in:

- conversational assistants
- agent systems
- long-running workflows

---

## Chapter Summary

- Large language models process text using **tokens**, not raw strings.
- Tokenization converts text into numerical token sequences used by the model.
- Tokens determine **model cost, latency, and context limits**.
- The **context window** defines the maximum number of tokens a model can process at once.
- Context construction combines prompts, retrieved knowledge, and conversation history.
- Token budgeting helps manage context usage in production systems.
- Long-context models enable new capabilities but introduce computational trade-offs.
- Memory systems extend AI applications beyond the limits of a single context window.

---

## Comprehension Questions

1. Why do large language models operate on tokens instead of raw text?
2. Why is token count different from word count?
3. How does token usage affect system cost and latency?
4. What is token budgeting and why is it important?
5. What components typically contribute to a model's context window?
6. Why is context management important for RAG systems?
7. What is the difference between a context window and system memory?

---

## References

- Vaswani et al. _Attention Is All You Need_.  
  https://arxiv.org/abs/1706.03762

- OpenAI Tokenization Guide  
  https://platform.openai.com/docs/tokenizer

- Hugging Face Tokenizers Documentation  
  https://huggingface.co/docs/tokenizers

- Anthropic Context Window Documentation  
  https://docs.anthropic.com
