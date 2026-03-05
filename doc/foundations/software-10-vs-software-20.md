## Chapter 2 — Software 1.0 vs [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35)

### 2.1 The Programming Paradigm That Shaped Enterprise Software

Enterprise software development has been dominated for decades by a single paradigm: engineers write explicit instructions that determine system behavior. If a classification rule changes, an engineer modifies the corresponding `if` statement. If a new business rule is added, it is encoded as a new code branch. The system's behavior is fully auditable by reading the source code.

This paradigm — **Software 1.0** — has enabled the construction of extraordinarily complex systems. Java EE application servers, Spring-based microservices, Oracle-backed business platforms: all of these are built on the same foundational principle that system behavior equals code.

```
input → explicit rules → deterministic output
```

The strengths of this model are well understood by any senior Java developer:

- **Full debuggability** — a breakpoint at any line reveals exact system state
- **Deterministic testing** — the same input always produces the same output
- **Auditable behavior** — system logic is expressed in readable code
- **Incremental evolution** — changing behavior means changing identifiable code

The limitations became apparent when engineers attempted to apply this paradigm to problems involving natural language, unstructured data, or complex perceptual tasks.

---

### 2.2 The Limits of Explicit Rules

Consider a customer support ticket classification system. A Software 1.0 approach might look like this:

**Python**
```python
def classify_ticket(text: str) -> str:
    text_lower = text.lower()
    if "password" in text_lower or "login" in text_lower:
        return "authentication"
    elif "payment" in text_lower or "billing" in text_lower:
        return "billing"
    elif "error" in text_lower or "crash" in text_lower or "bug" in text_lower:
        return "technical"
    else:
        return "unknown"
```

**Java**
```java
public String classifyTicket(String text) {
    String lower = text.toLowerCase();
    if (lower.contains("password") || lower.contains("login")) {
        return "authentication";
    } else if (lower.contains("payment") || lower.contains("billing")) {
        return "billing";
    } else if (lower.contains("error") || lower.contains("crash")) {
        return "technical";
    }
    return "unknown";
}
```

This approach works for a narrow set of inputs. It fails immediately when a user writes:

- *"I cannot get into my account"* → authentication, but no keyword matches
- *"My card was charged twice"* → billing, but "card" and "charged" are not in the rule set
- *"The app keeps stopping"* → technical, but neither "error" nor "crash" appears

The rule set grows rapidly to compensate, creating a maintenance burden that scales poorly with linguistic variation. More rules introduce more conflicts. The system becomes fragile.

---

### 2.3 Software 2.0 — Learned Behavior

Software 2.0 addresses this limitation by replacing hand-written rules with models trained on data. Instead of encoding `if "password" in text`, the engineer curates a labeled dataset and trains a classifier.

```
input → trained model (learned from data) → output
```

The classification logic is not written — it is *learned*. The model generalizes from examples to handle linguistic variation that no rule set can fully anticipate.

**Training dataset example:**

```
"I forgot my password"         → authentication
"Cannot log in to my account"  → authentication
"My payment did not go through"→ billing
"I was charged the wrong amount" → billing
"The app crashed after update" → technical
"Getting a 500 error on login" → technical
```

**Python — Training and inference with scikit-learn:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])
pipeline.fit(train_texts, train_labels)

# Inference
category = pipeline.predict(["I cannot get into my account"])[0]
# → "authentication"
```

**Java — Inference via REST API to a deployed model:**
```java
@Service
public class TicketClassifier {

    private final RestTemplate restTemplate;

    public String classify(String ticketText) {
        ClassificationRequest request = new ClassificationRequest(ticketText);
        ClassificationResponse response = restTemplate.postForObject(
            "http://ml-service/classify",
            request,
            ClassificationResponse.class
        );
        return response.getCategory();
    }
}
```

The code no longer encodes the classification logic — it orchestrates a call to a model that has internalized that logic from data. This is a fundamental architectural shift: **the model is a dependency, not an implementation detail**.

---

### 2.4 The Comparative Anatomy of Both Paradigms

| Dimension | Software 1.0 | Software 2.0 |
|---|---|---|
| **Behavior defined by** | Explicit code | Trained model weights |
| **Determinism** | High — same input, same output | Probabilistic — outputs may vary |
| **Debugging approach** | Code reading, breakpoints | Data analysis, error distribution |
| **System evolution** | Modify code, redeploy | Retrain or fine-tune with new data |
| **Testing strategy** | Exact output assertions | Statistical metrics, threshold-based |
| **Failure mode** | Logic errors, edge cases | Distribution shift, hallucination |
| **Auditability** | Full — code is the behavior | Partial — weights are opaque |
| **Domain expertise required** | Software engineering | Software + ML + data engineering |

---

### 2.5 Large Language Models — A Step Further

Machine learning classifiers still require task-specific training. A model trained to classify support tickets cannot be directly used to summarize documents or translate code. Each task requires a separate training pipeline.

Large Language Models change this. A single LLM, pre-trained on vast corpora, can perform diverse tasks via **natural language instructions** — no task-specific fine-tuning required.

```
input → prompt instruction → LLM → output
```

The same model can classify tickets, summarize documents, extract structured data, and answer questions — the task is specified in the prompt, not baked into model weights.

**Python — LLM-based classification:**
```python
from openai import OpenAI

client = OpenAI()

def classify_with_llm(ticket_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the support ticket into exactly one of these categories: "
                    "authentication, billing, technical, other. "
                    "Respond with only the category name."
                )
            },
            {"role": "user", "content": ticket_text}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
```

**Java — LLM classification with [LangChain4j](https://docs.langchain4j.dev):**
```java
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;

interface TicketClassifier {
    @SystemMessage("""
        Classify the support ticket into exactly one of these categories:
        authentication, billing, technical, other.
        Respond with only the category name.
        """)
    String classify(String ticketText);
}

// Usage
OpenAiChatModel model = OpenAiChatModel.builder()
    .apiKey(System.getenv("OPENAI_API_KEY"))
    .modelName("gpt-4o")
    .temperature(0.0)
    .build();

TicketClassifier classifier = AiServices.create(TicketClassifier.class, model);
String category = classifier.classify("I cannot get into my account");
// → "authentication"
```

**On-premise / open-source alternative with [Ollama](https://ollama.com):**
```python
import requests

def classify_with_local_llm(ticket_text: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": (
                f"Classify this support ticket into: authentication, billing, "
                f"technical, or other. Reply with only the category.\n\n"
                f"Ticket: {ticket_text}"
            ),
            "stream": False
        }
    )
    return response.json()["response"].strip()
```

> **On-premise note:** [Ollama](https://ollama.com) provides a straightforward runtime for running open-weight models ([Llama 3](https://ai.meta.com/llama/), [Mistral](https://docs.mistral.ai), [Qwen](https://huggingface.co/Qwen)) locally or on private infrastructure. For Java-based enterprise systems, [LangChain4j](https://github.com/langchain4j/langchain4j) provides native support for Ollama as a model backend.

---

### 2.6 Real LLM Systems: Beyond Simple Prompting

In production, LLM-based systems rarely consist of a single prompt call. The complete pipeline for a knowledge-intensive application looks like this:

```
user query
     │
     ▼
query preprocessing (intent detection, query rewriting)
     │
     ▼
retrieval system (vector DB + keyword search)
     │
     ▼
context assembly (top-K chunks + metadata)
     │
     ▼
prompt construction (template + context + query)
     │
     ▼
LLM inference (model selection, parameter configuration)
     │
     ▼
output processing (parsing, validation, formatting)
     │
     ▼
response
```

This pattern — **Retrieval-Augmented Generation (RAG)** — is the dominant architectural pattern for enterprise LLM systems. It is covered in depth in **Parts III and IV**.

---

### 2.7 Hybrid Systems: Combining Both Paradigms

The most robust production systems combine LLM capabilities with traditional deterministic logic. The LLM handles the parts of the problem that resist rule-based encoding — natural language understanding, knowledge synthesis — while code handles validation, business rule enforcement, and structured data processing.

**Architecture pattern:**
```
user input
     │
     ▼
LLM analysis → structured output (JSON)
     │
     ▼
deterministic validation rules (Java/Python)
     │
     ▼
business logic execution
     │
     ▼
final decision
```

**Python — LLM extraction + validation:**
```python
import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError

class ExtractedTicket(BaseModel):
    customer_name: str
    issue_category: str
    priority: str
    product: str

def extract_and_validate(ticket_text: str) -> ExtractedTicket:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract ticket information as JSON with fields: "
                    "customer_name, issue_category, priority (low/medium/high), product."
                )
            },
            {"role": "user", "content": ticket_text}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    raw = json.loads(response.choices[0].message.content)
    return ExtractedTicket(**raw)  # Pydantic validates structure and types
```

**Java — Validation layer:**
```java
@Component
public class TicketValidator {

    public void validate(ExtractedTicket ticket) {
        if (ticket.getCustomerName() == null || ticket.getCustomerName().isBlank()) {
            throw new ValidationException("customer_name is required");
        }
        if (!VALID_CATEGORIES.contains(ticket.getIssueCategory())) {
            throw new ValidationException(
                "Invalid category: " + ticket.getIssueCategory()
            );
        }
        if (!VALID_PRIORITIES.contains(ticket.getPriority())) {
            throw new ValidationException(
                "Invalid priority: " + ticket.getPriority()
            );
        }
    }

    private static final Set<String> VALID_CATEGORIES = Set.of(
        "authentication", "billing", "technical", "other"
    );
    private static final Set<String> VALID_PRIORITIES = Set.of(
        "low", "medium", "high"
    );
}
```

This hybrid pattern is the recommended approach for enterprise systems: use LLMs for what they do well (language understanding, knowledge synthesis), and enforce correctness guarantees with traditional code.

---

### 2.8 Implications for Solution Architects

The shift from Software 1.0 to LLM-based systems has direct implications for architecture and engineering practices:

**Versioning extends beyond code.** Prompts, datasets, and model versions must all be tracked and managed. A deployed system is not fully described by its Git repository.

**Testing must be probabilistic.** Automated test suites need to evaluate outputs against semantic criteria, not exact string matches. Evaluation pipelines become first-class engineering deliverables.

**Operational observability changes.** Beyond latency and error rates, you must monitor token usage, prompt effectiveness, retrieval quality, and answer faithfulness.

**Cost is a design constraint.** Every LLM inference call has a direct monetary cost. Architecture decisions — model selection, caching strategies, context management — have direct cost implications.

**The development loop changes.** Improving system behavior may require updating a prompt, curating a dataset, or adjusting a retrieval pipeline — not writing a new code branch.

---

> ### 📋 Chapter Summary
>
> - **Software 1.0** defines behavior through explicit code; deterministic, debuggable, but brittle for natural language problems.
> - **Software 2.0** encodes behavior in trained model weights; robust to linguistic variation but requires data-centric development practices.
> - **LLM systems** generalize further: a single model handles multiple tasks via prompt instructions, eliminating per-task training.
> - **Hybrid systems** are the production norm: LLMs handle language understanding, traditional code enforces validation and business rules.
> - The paradigm shift has direct implications for architecture, testing, observability, and cost management.

---

> ### ❓ Comprehension Questions
>
> 1. A product manager proposes extending the rule-based ticket classifier to handle 50 new customer issue types. What engineering argument would you make against this approach, and what alternative would you propose?
> 2. Why is `temperature=0` a sensible default for classification and extraction tasks, but potentially inappropriate for generative tasks?
> 3. In the hybrid architecture pattern, what is the rationale for performing validation in Java/Python code rather than trusting the LLM to produce valid output?
> 4. An LLM-based extraction system occasionally produces JSON with missing required fields. What engineering mechanisms can catch this at the boundary between the LLM response and the downstream business logic?
> 5. How does the concept of "system evolution" differ between Software 1.0 and LLM-based systems? What new skills and practices does this require from a development team?

---

## References

### Articles & Essays
- [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35) — Andrej Karpathy, Medium, 2017. The original essay introducing the Software 2.0 paradigm.
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Rich Sutton, 2019. Why general methods that leverage compute beat hand-coded approaches.

### Papers
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020. Demonstrates LLMs as general-purpose task solvers via prompting.

### Books
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen, O'Reilly, 2022.

### Documentation
- [LangChain4j Getting Started](https://docs.langchain4j.dev/get-started) — Java LLM integration.
- [Ollama Model Library](https://ollama.com/library) — Available open-weight models for local deployment.

---
[« Back to foundations Index](index.md) | [🏠 Home](../index.md)