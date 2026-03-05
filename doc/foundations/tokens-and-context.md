## Chapter 4 — Tokens and Context

### 4.1 The Token as the Fundamental Unit

Every interaction with a Large Language Model — whether sending a prompt or receiving a response — is expressed in units called **tokens**. Tokens are the atomic unit of LLM computation: they are what the model reads, processes, and produces. Understanding tokens is not optional background knowledge — it directly shapes every design decision involving context management, cost control, and system performance.

---

### 4.2 What is a Token

A token is a fragment of text as represented by the model's vocabulary. Tokenization is the process of splitting input text into these fragments using a tokenization algorithm — most modern models use **Byte Pair Encoding (BPE)** or similar subword algorithms.

Tokenization does not correspond to words or characters in a simple way:

```
"authentication"         → ["auth", "ent", "ication"]         (3 tokens)
"login"                  → ["login"]                           (1 token)
"I cannot access my account" → ["I", " cannot", " access",
                                " my", " account"]             (5 tokens)
"gpt-4o"                 → ["g", "pt", "-", "4", "o"]         (5 tokens)
```

**Rules of thumb for token estimation:**
- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words in English
- 100 tokens ≈ 75 words ≈ half a paragraph
- 1,000 tokens ≈ 750 words ≈ a short article

These approximations vary by language. Code, technical terminology, and non-Latin scripts tokenize differently.

---

### 4.3 Why Token Count Matters

Token count is not merely a performance consideration — it is a core engineering constraint that affects three dimensions simultaneously:

**Cost.** Most LLM APIs price per token. A system handling 100,000 requests per day with an average of 2,000 tokens per request is consuming 200 million tokens daily. At GPT-4o pricing, this translates to hundreds of dollars per day — a meaningful cost that must be managed through architecture decisions.

**Latency.** LLM inference time scales with token count, particularly for output generation. Long prompts and large context windows increase time-to-first-token and total generation time.

**Quality.** The model's ability to reason over context degrades for very long inputs. While modern models support large context windows, retrieval precision — finding the right information within a long context — is an active research challenge. Smaller, more targeted contexts often outperform large ones.

---

### 4.4 The Context Window

The **context window** is the maximum number of tokens an LLM can process in a single inference call. It represents the model's "working memory" — everything that can influence the generation of a response must fit within this window.

The context window includes:
- System prompt
- Conversation history (in multi-turn interactions)
- Retrieved documents (in RAG systems)
- User query
- Generated response (counted against the window in some implementations)

**Context window comparison across major models:**

| Model | Context Window | On-premise Available |
|---|---|---|
| GPT-3.5-turbo | 16,384 tokens | No |
| GPT-4o | 128,000 tokens | No |
| GPT-4o-mini | 128,000 tokens | No |
| Claude 3.5 Sonnet | 200,000 tokens | No |
| [Llama 3](https://ai.meta.com/llama/).1 8B | 128,000 tokens | ✅ Yes |
| Llama 3.1 70B | 128,000 tokens | ✅ Yes |
| [Mistral](https://docs.mistral.ai) 7B | 32,768 tokens | ✅ Yes |
| [Qwen](https://huggingface.co/Qwen) 2.5 72B | 128,000 tokens | ✅ Yes |

> **Architecture implication:** A 128k token context window does not mean you should fill it. Token costs scale with context size, and retrieval precision degrades in very large contexts. Context window management — deciding what to include, what to compress, and what to exclude — is a critical RAG system design skill.

---

### 4.5 Measuring Token Usage in Code

Token counting must be implemented at the application layer, not treated as an afterthought. Before sending a prompt to an LLM, you should know its token cost.

**Python — Counting tokens with [tiktoken](https://github.com/openai/tiktoken):**
```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  model: str = "gpt-4o") -> float:
    # Prices as of 2024 — always verify current pricing
    pricing = {
        "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
        "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    }
    p = pricing.get(model, pricing["gpt-4o"])
    return (prompt_tokens * p["input"]) + (completion_tokens * p["output"])

# Usage
prompt = "Classify this ticket: I cannot log in to my account."
tokens = count_tokens(prompt)
print(f"Prompt tokens: {tokens}")             # → 14
print(f"Estimated cost: ${estimate_cost(tokens, 20):.6f}")
```

**Java — Token counting with [LangChain4j](https://docs.langchain4j.dev):**
```java
import dev.langchain4j.model.Tokenizer;
import dev.langchain4j.model.openai.OpenAiTokenizer;

public class TokenCounter {

    private final Tokenizer tokenizer;

    public TokenCounter() {
        this.tokenizer = new OpenAiTokenizer("gpt-4o");
    }

    public int countTokens(String text) {
        return tokenizer.estimateTokenCountInText(text);
    }

    public int countMessagesTokens(List<ChatMessage> messages) {
        return tokenizer.estimateTokenCountInMessages(messages);
    }
}
```

---

### 4.6 Context Window Management in RAG Systems

In RAG systems, the context window budget must be explicitly allocated across all components of the prompt. Failing to plan this allocation leads to context overflow — a situation where the assembled prompt exceeds the model's context window and must be truncated, often silently discarding critical information.

**Budget allocation example for a 128k token model:**

```
┌─────────────────────────────────────────────────────┐
│  Total context window: 128,000 tokens               │
├─────────────────────────────────────────────────────┤
│  System prompt:          ~500 tokens                │
│  Conversation history:  ~2,000 tokens               │
│  Retrieved context:    ~10,000 tokens  (top-K docs) │
│  User query:              ~200 tokens               │
│  Response (reserved):   ~2,000 tokens               │
├─────────────────────────────────────────────────────┤
│  Total used:           ~14,700 tokens               │
│  Buffer:              ~113,300 tokens               │
└─────────────────────────────────────────────────────┘
```

For cost-optimized systems, minimizing the used context while preserving answer quality is the primary design goal.

**Python — Context window manager:**
```python
import tiktoken
from dataclasses import dataclass
from typing import List

@dataclass
class ContextBudget:
    max_tokens: int = 128_000
    system_prompt_budget: int = 500
    conversation_budget: int = 2_000
    response_budget: int = 2_000

    @property
    def retrieval_budget(self) -> int:
        return (self.max_tokens
                - self.system_prompt_budget
                - self.conversation_budget
                - self.response_budget
                - 500)  # Safety margin

class ContextWindowManager:
    def __init__(self, model: str = "gpt-4o", budget: ContextBudget = None):
        self.encoder = tiktoken.encoding_for_model(model)
        self.budget = budget or ContextBudget()

    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def fit_chunks_to_budget(self, chunks: List[str]) -> List[str]:
        """Select chunks that fit within the retrieval budget."""
        selected = []
        used = 0
        for chunk in chunks:
            chunk_tokens = self.count(chunk)
            if used + chunk_tokens > self.budget.retrieval_budget:
                break
            selected.append(chunk)
            used += chunk_tokens
        return selected
```

**Java — Context budget enforcer:**
```java
public class ContextWindowManager {

    private static final int MAX_CONTEXT_TOKENS = 128_000;
    private static final int SYSTEM_BUDGET = 500;
    private static final int CONVERSATION_BUDGET = 2_000;
    private static final int RESPONSE_BUDGET = 2_000;
    private static final int SAFETY_MARGIN = 500;

    private final Tokenizer tokenizer;

    public int getRetrievalBudget() {
        return MAX_CONTEXT_TOKENS
            - SYSTEM_BUDGET
            - CONVERSATION_BUDGET
            - RESPONSE_BUDGET
            - SAFETY_MARGIN;
    }

    public List<String> fitChunksToBudget(List<String> chunks) {
        List<String> selected = new ArrayList<>();
        int usedTokens = 0;
        int budget = getRetrievalBudget();

        for (String chunk : chunks) {
            int chunkTokens = tokenizer.estimateTokenCountInText(chunk);
            if (usedTokens + chunkTokens > budget) {
                break;
            }
            selected.add(chunk);
            usedTokens += chunkTokens;
        }
        return selected;
    }
}
```

---

### 4.7 Generation Parameters and Their Effect

The generation process can be controlled through several parameters that affect output quality, creativity, and consistency. Understanding these parameters is essential for prompt engineering and system configuration.

**Temperature** controls the randomness of token selection during generation. At `temperature=0`, the model always selects the most probable next token — deterministic behavior. At higher values, lower-probability tokens are selected more frequently, producing more varied but potentially less accurate outputs.

```
temperature = 0.0  → deterministic, consistent, factual tasks
temperature = 0.3  → low variation, suitable for extraction and classification
temperature = 0.7  → moderate creativity, suitable for generation tasks
temperature = 1.0  → high variation, suitable for creative writing
```

**Top-p (nucleus sampling)** limits token selection to the smallest set whose cumulative probability exceeds p. Combined with temperature, it prevents the model from sampling very low-probability tokens.

**Max tokens** limits the length of the generated response. Setting this appropriately prevents unexpectedly long (and expensive) responses.

**Python — Parameter configuration:**
```python
from openai import OpenAI

client = OpenAI()

# Extraction task — deterministic, concise
extraction_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract the invoice number from: INV-2024-00847"}],
    temperature=0,
    max_tokens=50
)

# Summary task — some variation acceptable
summary_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": f"Summarize: {document_text}"}],
    temperature=0.3,
    max_tokens=500
)
```

---

### 4.8 Token Optimization Strategies

Managing token consumption is a recurring architectural concern. The following strategies reduce cost and latency while preserving answer quality:

**Strategy 1 — Model routing.** Use smaller, cheaper models for simple tasks and larger models only when complexity demands it.

```
simple classification → gpt-4o-mini or llama3:8b   (~10x cheaper)
complex reasoning     → gpt-4o or llama3:70b
```

**Strategy 2 — Prompt compression.** Remove redundant instructions, verbose examples, and unnecessary context from prompts. Every saved token reduces cost.

**Strategy 3 — Response format constraints.** Request structured, concise outputs rather than narrative prose when the downstream system processes the response programmatically.

```python
# Instead of: "Explain whether the document is relevant..."
# Use: "Reply with exactly one word: relevant or irrelevant"
```

**Strategy 4 — Semantic caching.** Cache responses for semantically similar queries rather than sending each request to the LLM. Covered in depth in **Part XI**.

**Strategy 5 — Context compression.** Summarize retrieved documents before including them in the prompt, reducing the token cost of context while preserving key information.

---

> ### 📋 Chapter Summary
>
> - A **token** is the atomic unit of LLM computation — roughly 4 characters or 0.75 words in English.
> - The **context window** defines the maximum token capacity per inference call; it must be explicitly budgeted across system prompt, history, retrieved context, and response.
> - Token count directly drives **cost**, **latency**, and **quality** — all three must be considered together in architectural decisions.
> - **Generation parameters** (temperature, top-p, max tokens) control output behavior and must be configured per task type.
> - **Token optimization** — model routing, prompt compression, semantic caching — is a first-class engineering concern in production systems.

---

> ### ❓ Comprehension Questions
>
> 1. A RAG system assembles prompts by concatenating the system prompt, the top-10 retrieved chunks, and the user query. The system is deployed against a model with a 32k token context window. What is the risk of this approach, and how would you implement a context budget to mitigate it?
> 2. A classification endpoint is called 500,000 times per day. The current implementation uses GPT-4o with `temperature=0`. What architectural changes would you evaluate to reduce cost without sacrificing classification accuracy?
> 3. Why is `max_tokens` an important production configuration, not just a quality parameter?
> 4. An engineer sets `temperature=1.0` for a data extraction task that produces JSON output. What failure mode is likely to occur, and why?
> 5. Your system's context window budget calculation shows that the top-5 retrieved chunks consume 80% of the available retrieval budget. What strategies would you consider to either reduce context size or improve the information density of retrieved content?

---

## References

### Documentation
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Interactive tokenizer tool.
- [tiktoken GitHub](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokeniser library.
- [OpenAI Models & Context Windows](https://platform.openai.com/docs/models) — Official model specs and token limits.
- [Anthropic Claude Context Window](https://docs.anthropic.com/en/docs/about-claude/models/overview) — Claude model specifications.
- [LangChain4j Tokenizer](https://docs.langchain4j.dev/integrations/language-models/open-ai#tokenization) — Java tokenisation utilities.

### Articles
- [OpenAI Cookbook: How to Count Tokens](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) — Practical token counting guide.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu et al., 2023. Why large context windows do not guarantee performance with long inputs.

### Papers
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508) — Bai et al., 2023. Evaluation of long-context model performance.

---
[« Back to foundations Index](index.md) | [🏠 Home](../index.md)