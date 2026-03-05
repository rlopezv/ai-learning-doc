## Chapter 3 — Machine Learning vs LLM

### 3.1 Two Approaches to Intelligent Behavior

The distinction between classical machine learning and large language models is not merely technical — it reflects a fundamentally different approach to building intelligent system components. Understanding this distinction is essential for making sound architecture decisions: when is a traditional ML model the right choice, and when do LLMs offer clear advantages?

---

### 3.2 Classical Machine Learning: Task-Specific Models

Classical supervised ML follows a well-defined workflow: collect labeled data, engineer features, train a model on a specific task, evaluate, and deploy. The result is a model that performs one task well — but only that task.

```
labeled_dataset
      │
      ▼
feature engineering
      │
      ▼
model training (task-specific)
      │
      ▼
evaluation
      │
      ▼
deployed model (single task)
```

**Strengths:**
- High precision on well-defined tasks with sufficient training data
- Computationally efficient at inference time
- Deterministic (for non-probabilistic models)
- Easier to explain and audit (for linear models, decision trees)
- No prompt engineering required

**Weaknesses:**
- Requires substantial labeled data per task
- Does not generalize across tasks
- Retraining required when task definition changes
- Limited handling of natural language nuance

---

### 3.3 Large Language Models: Generalist Reasoning Engines

LLMs are pre-trained on enormous text corpora to learn general language understanding and reasoning. They are not designed for a specific task — they are designed to follow instructions expressed in natural language.

```
general_text_corpus (pre-training)
      │
      ▼
LLM (general purpose)
      │
      ▼
prompt instruction (task specification at runtime)
      │
      ▼
response (task-specific output)
```

The same model can classify text, summarize documents, extract structured data, translate languages, generate code, and answer questions — the task is specified in the prompt, not in the model architecture.

**Strengths:**
- Zero-shot and few-shot generalization to new tasks
- Handles linguistic variation and ambiguity robustly
- No task-specific training required for most use cases
- Continuously improving base models
- Capable of complex multi-step reasoning

**Weaknesses:**
- Non-deterministic outputs
- Prone to hallucination
- Token-based inference cost
- Context window limits the amount of information per call
- Requires prompt engineering discipline
- Knowledge cutoff — no awareness of events after training

---

### 3.4 Side-by-Side Comparison

| Dimension | Classical ML | LLM |
|---|---|---|
| **Task scope** | Single task | Multiple tasks via prompting |
| **Training requirement** | Labeled dataset per task | Pre-trained; minimal or no fine-tuning |
| **Inference cost** | Very low | Higher (token-based) |
| **Determinism** | High (for most models) | Low (probabilistic sampling) |
| **Explainability** | Moderate to high | Low (opaque weights) |
| **Natural language handling** | Limited | Excellent |
| **Structured output** | Native | Requires prompt engineering |
| **Hallucination risk** | None | Present |
| **Knowledge cutoff** | N/A | Yes — training data cutoff |

---

### 3.5 When to Use Each Approach

The choice between classical ML and LLMs is an architectural decision with performance, cost, and maintainability implications.

**Use classical ML when:**
- The task is well-defined and training data is available
- Inference cost at scale is a primary concern
- Determinism and explainability are required (regulated environments)
- Real-time latency requirements are strict (< 10ms)
- The feature space is structured (tabular data, time series)

**Use LLMs when:**
- The task involves natural language understanding or generation
- Training data is unavailable or insufficient for a task-specific model
- The task definition may evolve (prompts are easier to update than retraining)
- Multi-step reasoning or synthesis is required
- You need a single model to handle multiple tasks

**Use both (hybrid approach) when:**
- An LLM handles language understanding, a classifier handles structured routing
- An LLM generates candidate outputs, a traditional model scores or ranks them
- Real-time filtering (ML) feeds context to an LLM for deeper analysis

---

### 3.6 Practical Example: Document Routing System

Consider a system that receives customer documents and must route them to the correct processing pipeline. Two approaches:

**Approach A — Classical ML classifier:**

**Python**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Training phase
clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("classifier", RandomForestClassifier(n_estimators=100))
])
clf.fit(train_texts, train_labels)
joblib.dump(clf, "document_router.pkl")

# Inference — very fast, no API call
def route_document_ml(document_text: str) -> str:
    model = joblib.load("document_router.pkl")
    return model.predict([document_text])[0]
```

**Approach B — LLM classifier:**

**Python**
```python
from openai import OpenAI

ROUTING_PROMPT = """
Classify the following document into one of these categories:
- invoice
- contract
- support_request
- technical_specification
- other

Respond with only the category name.

Document:
{document_text}
"""

def route_document_llm(document_text: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Cost-optimized for classification
        messages=[{"role": "user", "content": ROUTING_PROMPT.format(
            document_text=document_text[:2000]  # Truncate for cost control
        )}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
```

**Java — Approach B with [LangChain4j](https://docs.langchain4j.dev) and local model:**
```java
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;

interface DocumentRouter {
    @dev.langchain4j.service.SystemMessage("""
        Classify the document into one of: invoice, contract,
        support_request, technical_specification, other.
        Respond with only the category name.
        """)
    String route(@UserMessage String documentText);
}

// On-premise deployment with Ollama
OllamaChatModel model = OllamaChatModel.builder()
    .baseUrl("http://ollama-server:11434")
    .modelName("mistral")
    .temperature(0.0)
    .build();

DocumentRouter router = AiServices.create(DocumentRouter.class, model);
String category = router.route(documentText);
```

The ML approach is faster and cheaper at scale for well-defined categories. The LLM approach handles ambiguous documents better and requires no labeled training data — but costs more per inference and introduces non-determinism.

---

### 3.7 Fine-tuning: Bridging ML and LLM

Fine-tuning takes a pre-trained LLM and adapts it to a specific domain or task using a targeted dataset. It combines the general capabilities of LLMs with the precision of task-specific training.

```
pre-trained LLM (general)
      │
      ▼
fine-tuning dataset (domain-specific)
      │
      ▼
fine-tuned LLM (domain-adapted)
```

**When fine-tuning makes sense:**
- A specific output format must be enforced consistently
- Domain vocabulary is specialized enough to confuse the base model
- Prompt engineering alone cannot achieve required quality
- Inference cost at scale justifies the upfront training investment

**When to avoid fine-tuning:**
- The problem can be solved with prompt engineering (cheaper and faster to iterate)
- The task or knowledge base changes frequently (fine-tuning is expensive to repeat)
- The required data volume for fine-tuning is unavailable

> **Architecture recommendation:** Always attempt to solve the problem with prompt engineering and RAG first. Fine-tuning is a last resort when other approaches have been exhausted and quality requirements are not met.

---

### 3.8 Embedding Models: A Special Category

Embedding models occupy a unique position — they are ML models in the traditional sense (deterministic, fast, cheap) but are purpose-built to support LLM systems by converting text into dense vector representations.

```
text → embedding_model → vector (e.g., 1536 dimensions)

"authentication issue"    → [0.23, -0.17,  0.91, ...]
"cannot log in"           → [0.21, -0.14,  0.88, ...]  ← similar
"payment processing error"→ [-0.12, 0.44, -0.31, ...]  ← dissimilar
```

Vectors that are close in the embedding space represent semantically similar texts. This property is the foundation of vector similarity search in RAG systems.

**Key embedding models:**

| Model | Provider | Dimensions | On-premise |
|---|---|---|---|
| `text-embedding-3-large` | OpenAI | 3072 | No |
| `text-embedding-3-small` | OpenAI | 1536 | No |
| `embed-english-v3.0` | [Cohere](https://docs.cohere.com) | 1024 | No |
| `all-MiniLM-L6-v2` | HuggingFace | 384 | ✅ Yes |
| `[nomic-embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1)-text` | Nomic / Ollama | 768 | ✅ Yes |
| `bge-large-en-v1.5` | BAAI / HuggingFace | 1024 | ✅ Yes |

> **On-premise note:** For environments where data cannot leave the organization, `[sentence-transformers](https://www.sbert.net)` (Python) and `djl` (Java) provide production-grade embedding generation using open-weight models.

---

> ### 📋 Chapter Summary
>
> - **Classical ML** produces task-specific models from labeled data: fast, deterministic, and cost-effective at scale for well-defined tasks.
> - **LLMs** are general-purpose reasoning engines that perform tasks via natural language instructions — flexible, but probabilistic and token-cost-constrained.
> - The architectural choice depends on task definition stability, data availability, latency requirements, and cost tolerance.
> - **Fine-tuning** bridges the two approaches: it adapts a pre-trained LLM to a specific domain but requires data, cost, and careful trade-off analysis.
> - **Embedding models** are deterministic ML models that generate semantic vector representations — the backbone of retrieval systems.

---

> ### ❓ Comprehension Questions
>
> 1. A fraud detection system must classify 50,000 transactions per second with < 5ms latency. Which approach — classical ML or LLM — is appropriate? Justify your answer.
> 2. Your team is building a system to extract structured fields from unstructured medical reports. Training data is scarce. Compare the trade-offs of fine-tuning a small model versus using a general LLM with a structured extraction prompt.
> 3. Why is it important that the embedding model used during document ingestion is the same as the one used during query processing?
> 4. Describe a hybrid architecture where a classical ML model and an LLM collaborate on the same task. What does each component contribute?
> 5. A stakeholder asks why the system sometimes produces different outputs for identical inputs. How would you explain LLM non-determinism to a team accustomed to deterministic systems?

---

## References

### Papers
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020. Establishes LLMs as general-purpose task solvers.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018. Foundational pre-trained language model.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021. Efficient fine-tuning technique.
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — Muennighoff et al., 2022. Standard benchmark for embedding model evaluation.

### Documentation
- [Sentence Transformers Documentation](https://www.sbert.net) — Open-source embedding models.
- [Hugging Face Model Hub](https://huggingface.co/models) — Repository of open-weight models and embedding models.
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — Official embeddings API documentation.
- [scikit-learn Documentation](https://scikit-learn.org/stable/) — Classical ML library reference.

### Books
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen, O'Reilly, 2022.

---
[« Back to foundations Index](index.md) | [🏠 Home](../../index.md)