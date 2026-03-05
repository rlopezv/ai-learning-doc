## Chapter 5 — Prompt Engineering

### 5.1 Prompts as System Configuration

In traditional software, system behavior is defined by code. In LLM-based systems, a significant portion of system behavior is defined by **prompts** — natural language instructions that configure model behavior at runtime. This shift has a profound implication: prompt engineering is not a soft skill or a creative activity. It is a **core engineering discipline** with the same rigor requirements as any other system configuration.

Prompts in production systems are:
- **Versioned** artifacts under source control
- **Tested** against evaluation datasets before deployment
- **Monitored** in production for quality regressions
- **Decoupled** from application code to enable independent lifecycle management

```
Prompt + Context + Model configuration → System behavior
```

---

### 5.2 The Anatomy of a Production Prompt

A well-structured production prompt has four distinct components, each with a specific engineering function:

```
┌─────────────────────────────────────────────────────┐
│  1. SYSTEM INSTRUCTIONS                             │
│  Defines model role, constraints, output format,   │
│  and behavioral guardrails.                         │
├─────────────────────────────────────────────────────┤
│  2. CONTEXT                                         │
│  Dynamically injected at runtime: retrieved         │
│  documents, conversation history, user profile.    │
├─────────────────────────────────────────────────────┤
│  3. USER QUERY / TASK                               │
│  The specific input or instruction for this         │
│  inference call.                                    │
├─────────────────────────────────────────────────────┤
│  4. OUTPUT FORMAT SPECIFICATION                     │
│  Explicit instruction for response structure:       │
│  JSON schema, enumeration, structured fields.       │
└─────────────────────────────────────────────────────┘
```

**Example production prompt — support ticket classifier:**

```
SYSTEM:
You are a support ticket classification engine for an enterprise software platform.
Your task is to classify incoming support tickets into predefined categories.

Rules:
- Respond only with valid JSON matching the specified schema
- If the ticket does not clearly belong to any category, use "other"
- Do not add explanations or commentary outside the JSON

OUTPUT SCHEMA:
{
  "category": "<authentication|billing|technical|other>",
  "confidence": "<high|medium|low>",
  "reasoning": "<one sentence>"
}

CONTEXT:
{retrieved_examples}

USER TICKET:
{ticket_text}
```

---

### 5.3 System Prompts: The Behavioral Contract

The system prompt defines the **behavioral contract** between the application and the model. It is the closest LLM equivalent to a service interface specification or a class invariant. Every production LLM system should have an explicit, versioned system prompt.

**Python — System prompt as a managed constant:**
```python
SUPPORT_CLASSIFIER_SYSTEM_PROMPT = """
You are a support ticket classification engine for an enterprise software platform.

Classification categories:
- authentication: login failures, password resets, access issues, MFA problems
- billing: payment failures, invoice disputes, subscription changes, refund requests
- technical: application errors, performance issues, feature malfunctions, data corruption
- other: requests that do not fit the above categories

Rules:
- Classify based on the core issue, not surface-level keywords
- Respond with valid JSON only: {"category": "...", "confidence": "high|medium|low"}
- Do not include explanations outside the JSON response
"""

def classify_ticket(ticket_text: str, client: OpenAI) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SUPPORT_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": ticket_text}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)
```

**Java — System prompt with [LangChain4j](https://docs.langchain4j.dev):**
```java
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

interface SupportClassifier {
    @SystemMessage("""
        You are a support ticket classification engine.

        Categories:
        - authentication: login, password, access, MFA issues
        - billing: payment, invoice, subscription issues
        - technical: errors, crashes, performance, data issues
        - other: anything that does not fit above

        Respond with valid JSON only:
        {"category": "...", "confidence": "high|medium|low"}
        """)
    @UserMessage("{{ticketText}}")
    String classify(String ticketText);
}
```

---

### 5.4 Prompting Strategies

Different task types benefit from different prompting strategies. Understanding the trade-offs allows you to choose the appropriate strategy per use case.

#### Zero-Shot Prompting

The model performs a task based solely on the instruction, without examples. Appropriate when the task is well-defined and the model has strong prior knowledge.

```python
ZERO_SHOT_PROMPT = """
Classify the following support ticket into one of:
authentication, billing, technical, other.
Respond with only the category name.

Ticket: {ticket_text}
"""
```

**When to use:** Simple, well-defined tasks. Low token cost per call.
**When to avoid:** Tasks requiring specific output format or edge-case handling.

#### Few-Shot Prompting

The prompt includes labeled examples that demonstrate the expected input-output behavior. This is the most reliable technique for tasks with specific format requirements or non-obvious classification criteria.

```python
FEW_SHOT_PROMPT = """
Classify support tickets. Examples:

Ticket: "I forgot my password and cannot log in"
Category: authentication

Ticket: "My payment was declined yesterday"
Category: billing

Ticket: "The export feature generates an empty file"
Category: technical

Ticket: "I would like to know your office hours"
Category: other

Now classify:
Ticket: "{ticket_text}"
Category:
"""
```

**Python — Dynamic few-shot example selection:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class DynamicFewShotSelector:
    """Selects the most relevant examples for a given query."""

    def __init__(self, examples: list[dict], model_name: str = "all-MiniLM-L6-v2"):
        self.examples = examples
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [ex["text"] for ex in examples]
        )

    def select(self, query: str, k: int = 3) -> list[dict]:
        query_embedding = self.model.encode(query)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_k_indices]
```

#### [Chain-of-Thought](https://arxiv.org/abs/2201.11903) Prompting

Instructs the model to reason step-by-step before producing the final answer. Significantly improves accuracy on complex classification, reasoning, or analysis tasks.

```python
COT_PROMPT = """
Analyze the following support ticket and classify it.

Ticket: "{ticket_text}"

Think through this step by step:
1. What is the core problem the user is experiencing?
2. Which system or feature is involved?
3. Which category best fits this problem?
4. How confident are you in this classification?

Final answer (JSON only):
{"category": "...", "confidence": "...", "reasoning": "..."}
"""
```

**When to use:** Ambiguous inputs, multi-criteria decisions, cases where reasoning transparency matters for auditing.
**Trade-off:** Increases token count significantly. Use `gpt-4o-mini` or a local model to control cost when this pattern is applied at scale.

#### Role Prompting

Assigns a specific expert persona to the model. Effective for domain-specific analysis where the framing of the task benefits from an expert perspective.

```python
ARCHITECTURE_REVIEW_PROMPT = """
You are a senior solution architect with 15 years of experience
designing distributed systems for financial services enterprises.
You specialize in identifying scalability bottlenecks, single points
of failure, and security vulnerabilities in complex architectures.

Review the following system design and provide a structured assessment:

{architecture_description}

Assess:
1. Scalability risks (identify specific bottlenecks)
2. Reliability concerns (single points of failure, failure modes)
3. Security considerations (attack surfaces, data exposure risks)
4. Recommended architectural changes (prioritized by impact)
"""
```

---

### 5.5 Prompt Templates and Dynamic Construction

Production prompts are not static strings — they are **templates** with dynamic sections populated at runtime from retrieved context, user input, and system state.

**Python — Template management with Jinja2:**
```python
from jinja2 import Template
from dataclasses import dataclass
from typing import List

@dataclass
class PromptContext:
    retrieved_chunks: List[str]
    user_query: str
    conversation_history: List[dict] = None

RAG_PROMPT_TEMPLATE = Template("""
You are a corporate knowledge assistant.
Answer the user's question using only the provided context.
If the answer is not in the context, say so explicitly.
Do not speculate or use knowledge outside the provided context.

CONTEXT:
{% for chunk in retrieved_chunks %}
---
{{ chunk }}
---
{% endfor %}

QUESTION:
{{ user_query }}

ANSWER:
""")

def build_prompt(context: PromptContext) -> str:
    return RAG_PROMPT_TEMPLATE.render(
        retrieved_chunks=context.retrieved_chunks,
        user_query=context.user_query
    )
```

**Java — Template with Spring:**
```java
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;

@Component
public class PromptBuilder {

    @Value("classpath:prompts/rag_prompt_v3.txt")
    private Resource promptTemplate;

    public String buildRagPrompt(List<String> chunks, String query) throws IOException {
        String template = Files.readString(promptTemplate.getFile().toPath());
        String context = String.join("\n---\n", chunks);
        return template
            .replace("{{context}}", context)
            .replace("{{query}}", query);
    }
}
```

> **Engineering recommendation:** Store prompt templates as files in your source repository (`prompts/v3/rag_prompt.txt`), not as inline strings in code. This enables independent versioning, review, and diff tracking of prompt changes — exactly as you would manage SQL migration scripts or configuration files.

---

### 5.6 Prompt Versioning

Prompts must be versioned with the same rigor as code. A prompt change can alter system behavior as significantly as a code change — and without versioning, it is impossible to attribute quality changes to specific prompt modifications.

**Recommended repository structure:**
```
prompts/
├── support-classifier/
│   ├── v1/
│   │   ├── system.txt
│   │   └── metadata.yaml      # model, parameters, eval results
│   ├── v2/
│   │   ├── system.txt
│   │   └── metadata.yaml
│   └── current -> v2/         # symlink to deployed version
├── rag-assistant/
│   ├── v1/
│   └── v2/
└── CHANGELOG.md
```

**metadata.yaml:**
```yaml
version: v2
model: gpt-4o
temperature: 0
created: 2024-11-15
author: platform-team
eval_dataset: support-classifier-eval-v3
eval_results:
  accuracy: 0.94
  f1_weighted: 0.93
changes_from_v1: |
  Added explicit confidence scoring.
  Improved handling of ambiguous authentication/technical boundary.
```

---

### 5.7 Prompt Testing

Before deploying a new prompt version, it must be tested against a representative evaluation dataset. This is the LLM equivalent of unit testing — but operating on probabilistic outputs.

**Python — Prompt evaluation:**
```python
from dataclasses import dataclass
from typing import List, Callable
import json

@dataclass
class TestCase:
    input: str
    expected_category: str

@dataclass
class EvalResult:
    total: int
    passed: int
    failed: List[dict]

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

def evaluate_prompt(
    test_cases: List[TestCase],
    classify_fn: Callable[[str], dict]
) -> EvalResult:
    failed = []
    passed = 0

    for case in test_cases:
        result = classify_fn(case.input)
        predicted = result.get("category", "unknown")
        if predicted == case.expected_category:
            passed += 1
        else:
            failed.append({
                "input": case.input,
                "expected": case.expected_category,
                "predicted": predicted,
                "confidence": result.get("confidence")
            })

    return EvalResult(total=len(test_cases), passed=passed, failed=failed)
```

**Java — Prompt test runner:**
```java
public class PromptEvaluator {

    private final SupportClassifier classifier;

    public EvalReport evaluate(List<TestCase> testCases) {
        int passed = 0;
        List<FailedCase> failures = new ArrayList<>();

        for (TestCase tc : testCases) {
            String result = classifier.classify(tc.getInput());
            ClassificationResult parsed = parseResult(result);

            if (tc.getExpectedCategory().equals(parsed.getCategory())) {
                passed++;
            } else {
                failures.add(new FailedCase(
                    tc.getInput(),
                    tc.getExpectedCategory(),
                    parsed.getCategory()
                ));
            }
        }

        return new EvalReport(testCases.size(), passed, failures);
    }
}
```

---

### 5.8 Common Prompt Engineering Pitfalls

| Pitfall | Description | Mitigation |
|---|---|---|
| **Ambiguous instructions** | Model interprets instructions differently across runs | Use explicit, specific language; test with diverse inputs |
| **Missing output format** | Responses vary in structure, breaking downstream parsing | Always specify exact output format; use `response_format: json_object` |
| **Context overflow** | Retrieved context exceeds budget, truncating critical content | Implement explicit token budget management |
| **Prompt overfitting** | Prompt optimized for test cases, fails on production distribution | Maintain separate evaluation and test datasets |
| **Hardcoded prompts** | Prompts embedded in code, no independent versioning | Store prompts as files under version control |
| **Missing negative constraints** | Model speculates beyond available context | Add explicit constraints: "Do not use knowledge outside the provided context" |

---

> ### 📋 Chapter Summary
>
> - **Prompts are system configuration** — they must be versioned, tested, and managed with the same rigor as code.
> - A production prompt has four components: system instructions, context, user query, and output format specification.
> - **Prompting strategies** — zero-shot, few-shot, chain-of-thought, role prompting — each address different task types and complexity levels.
> - **Prompt templates** separate the static structure from the dynamic runtime content, enabling independent lifecycle management.
> - **Prompt versioning** requires a repository structure that tracks changes, evaluation results, and deployment metadata alongside the prompt text.

---

> ### ❓ Comprehension Questions
>
> 1. A team stores prompt templates as string constants in their Java service classes. What operational risks does this create, and what alternative structure would you propose?
> 2. Explain why few-shot examples should ideally be selected dynamically based on similarity to the input query, rather than using a fixed set of examples for all inputs.
> 3. A chain-of-thought prompt improves classification accuracy from 87% to 94% on the evaluation dataset. The prompt adds 400 tokens per request. The system handles 200,000 requests per day. Model cost is $0.0025/1k input tokens. Calculate the daily cost increase and discuss whether the accuracy improvement justifies it.
> 4. What is the difference between a prompt template and a prompt version? How do they relate to each other in a prompt management system?
> 5. You deploy a new prompt version and observe that a specific category's precision drops from 91% to 78% in production. What debugging process would you follow to identify the root cause?

---

## References

### Papers
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Wei et al., 2022. Foundational paper on CoT prompting.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020. Introduces few-shot prompting.
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Wang et al., 2022. Improving CoT with multiple reasoning paths.
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) — Kojima et al., 2022. "Let's think step by step" zero-shot CoT.

### Guides & Documentation
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — Official best practices.
- [Anthropic Prompt Engineering Overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — Claude-specific prompt engineering.
- [OpenAI Cookbook](https://cookbook.openai.com) — Practical examples and patterns.
- [LangChain Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/) — Template management in Python.
- [LangChain4j Prompt Templates](https://docs.langchain4j.dev/tutorials/ai-services) — Template management in Java.

### Books
- [Prompt Engineering for LLMs](https://www.oreilly.com/library/view/prompt-engineering-for/9781098153427/) — John Berryman & Albert Ziegler, O'Reilly, 2024.

---
[« Back to foundations Index](index.md) | [🏠 Home](../../index.md)