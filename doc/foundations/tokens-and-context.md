# Tokens and Context

[⬅ Back to Foundations](index.md)

---

## Context

Large language models process text differently from traditional software systems. Instead of operating directly on words or sentences, these models work with **tokens**, which are numerical representations of pieces of text.

Because modern AI systems rely heavily on **prompts and contextual information**, understanding how text is represented and processed inside the model becomes essential. Every prompt, retrieved document, and generated response must be converted into tokens before the model can process it.

Another critical constraint is the **context window**, which defines how many tokens the model can process at one time. The size of the context window determines how much information the model can consider when generating a response.

Understanding tokens and context windows is essential for AI Systems Engineering because they influence:

- system architecture
- prompt design
- retrieval pipelines
- latency and cost
- response quality

These constraints shape how engineers design prompts, retrieval systems, and workflows for production AI systems.

---

## Concept Overview

Large language models do not operate directly on raw text. Instead, they transform text into tokens that can be processed by neural networks.

The processing pipeline typically looks like this:

```
Text Input
↓
Tokenization
↓
Token IDs
↓
Model Processing
↓
Generated Tokens
↓
Decoded Text
```

Two concepts are fundamental to understanding how language models operate:

- **tokens** — the units used to represent text
- **context window** — the maximum number of tokens the model can process at once

These two factors strongly influence how AI systems are designed and how information flows through them.

**Key Concept — Tokens Are the Basic Unit of LLM Computation**

Large language models do not operate on words or sentences directly. Instead, they process tokens, which represent pieces of text converted into numerical identifiers.

Every prompt, retrieved document, and generated response must be represented as a sequence of tokens before it can be processed by the model.

---

## 5.1 Tokenization

Tokenization is the process of converting raw text into tokens.

Depending on the tokenizer used, tokens may represent:

- individual characters
- parts of words
- entire words
- punctuation symbols

For example, the sentence:

```
AI systems are transforming software engineering.
```

might be tokenized as:

```
["AI", " systems", " are", " transforming", " software", " engineering", "."]
```

Each token is then mapped to a numerical identifier known as a **token ID**, which allows the model to process the sequence mathematically.

```
["AI", " systems", " are"]
↓
[1543, 9281, 472]
```

The number of tokens does not always correspond directly to the number of words. Some words may be split into multiple tokens, while others may be combined.

Example:

| Text                   | Possible Tokenization                  |
| ---------------------- | -------------------------------------- |
| AI                     | ["AI"]                                 |
| systems                | [" systems"]                           |
| engineering            | [" engineer", "ing"]                   |
| AI systems engineering | ["AI", " systems", " engineer", "ing"] |

Different models may use different tokenization strategies, such as:

- Byte Pair Encoding (BPE)
- WordPiece
- SentencePiece

Although the details vary, the core idea remains the same: converting text into tokens that the model can process.

**Key Concept — Tokenization Converts Text into Numerical Sequences**

Language models cannot directly process text. Tokenization converts human-readable text into numerical token identifiers that can be processed by neural networks.

---

## 5.2 Context Windows

The **context window** defines how many tokens a model can process at one time.

When a model receives an input prompt, the entire sequence must fit within the model's context window.

```
Prompt Tokens
+
Retrieved Context
+
Model Response
≤ Context Window
```

If the total number of tokens exceeds this limit, the model cannot process the entire sequence.

For example:

| Model                    | Typical Context Window |
| ------------------------ | ---------------------- |
| Early transformer models | 512–1024 tokens        |
| GPT-3                    | ~4K tokens             |
| Modern LLMs              | 32K–200K+ tokens       |

The context window determines:

- how much information can be included in a prompt
- how many documents can be retrieved for context
- how long responses can be

Because the response tokens are also counted within the context window, engineers must carefully manage how tokens are allocated between input context and output generation.

---

## 5.3 Token Budgeting

Because the context window is limited, AI systems must carefully allocate tokens across different components of the prompt.

A typical LLM request might include:

```
System Instructions
+
User Query
+
Retrieved Documents
+
Tool Outputs
+
Generated Response
```

All of these elements must fit within the model's context window.

This leads to the concept of **token budgeting**, where engineers allocate tokens across different parts of the prompt.

Example token allocation:

| Component           | Example Token Budget |
| ------------------- | -------------------- |
| System prompt       | 200                  |
| User query          | 50                   |
| Retrieved documents | 2500                 |
| Model response      | 500                  |

Effective token budgeting is important for:

- maximizing useful context
- avoiding truncation
- controlling response length
- managing system costs

---

## 5.4 Cost Implications

Token usage also affects the **cost of operating AI systems**.

Most LLM APIs charge based on the number of tokens processed during inference.

```
Total Cost ∝ Input Tokens + Output Tokens
```

As a result, token usage directly impacts:

- operational cost
- latency
- system scalability

For example, large retrieval contexts may improve response quality but also increase token usage and cost.

Engineers must therefore balance:

- response quality
- latency
- operational cost

This trade-off is a central concern when designing production AI systems.

**Key Concept — Tokens Drive Cost and Latency**

Token usage directly influences both the cost and performance of AI systems.

Efficient prompt design and retrieval strategies help minimize unnecessary token usage while preserving useful context.

---

## 5.5 Context Management in AI Systems

Because context windows are limited, AI systems must carefully manage how information is provided to the model.

Common context management strategies include:

- document chunking
- retrieval ranking
- context compression
- summarization
- conversation memory management

For example, in a Retrieval-Augmented Generation system:

```
User Query
↓
Retriever
↓
Vector Database
↓
Top-K Documents
↓
Prompt Construction
↓
LLM
↓
Response
```

The retriever must select a **small number of highly relevant documents** so that the prompt remains within the context window.

Context management is therefore closely tied to:

- retrieval system design
- prompt construction
- conversation memory strategies

These techniques help ensure that the model receives the most relevant information without exceeding context limits.

---

## 5.6 Context Window Limitations

Despite recent improvements, context windows remain a practical constraint in AI systems.

Large prompts can lead to several issues:

- increased latency
- higher operational cost
- reduced model attention to earlier tokens
- prompt truncation

Even when models support very large context windows, performance may degrade when the context becomes excessively long.

For this reason, production systems often rely on **retrieval pipelines and summarization techniques** rather than sending entire documents directly to the model.

**Key Concept — Context Is a Scarce Resource**

Although modern models support large context windows, context capacity is still limited.

Effective AI systems treat context as a scarce resource and prioritize delivering the most relevant information to the model.

---

## 5.7 Tokens in System Architecture

Tokens and context windows influence the architecture of many AI systems.

Several architectural components exist primarily to manage token constraints:

- vector databases for retrieving relevant documents
- chunking strategies for splitting large documents
- summarization pipelines for compressing information
- conversation memory systems for managing dialogue history

These components form part of the **retrieval and orchestration layers** of AI systems.

By carefully managing tokens and context, engineers can design systems that deliver high-quality responses while remaining efficient and scalable.

---

## Chapter Summary

- Language models process text as **tokens**, which are numerical representations of text fragments.
- **Tokenization** converts human-readable text into token sequences used by the model.
- The **context window** defines how many tokens the model can process in a single request.
- Token limits require engineers to manage how prompts, retrieved documents, and responses fit within the context window.
- Token usage directly influences system cost, latency, and scalability.
- Production AI systems rely on retrieval pipelines, summarization, and context management techniques to work within these constraints.

---

## Comprehension Questions

1. Why do large language models operate on tokens rather than raw text?
2. What role does tokenization play in the processing pipeline of a language model?
3. What is a context window, and why does it constrain prompt design?
4. Why must token budgets be managed in production AI systems?
5. How do retrieval systems help manage context window limitations?
6. Why does token usage affect both cost and latency in AI applications?

---

## References

### Papers

Attention Is All You Need — Vaswani et al., 2017
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

Language Models are Few-Shot Learners — Brown et al., 2020
[https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

### Documentation

OpenAI Tokenizer Documentation
[https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer)

Hugging Face Tokenizers Documentation
[https://huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers)

---

## Key Takeaways

- Tokens are the fundamental unit used by language models to represent text.
- Tokenization converts text into numerical sequences that neural networks can process.
- The context window limits how many tokens a model can process in a single request.
- Token budgeting is essential for balancing context quality, response length, and cost.
- Efficient context management is a critical component of production AI system architecture.
