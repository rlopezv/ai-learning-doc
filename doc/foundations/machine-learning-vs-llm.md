# Chapter 3 — Machine Learning vs LLM

[⬅ Back to Foundations](index.md)

## Context

Artificial intelligence systems can be built using different types of models. Traditional **machine learning models** are typically trained to solve a specific task using structured input features. In contrast, **large language models (LLMs)** are general-purpose foundation models capable of performing many tasks through prompting and contextual input.

For AI systems engineers, understanding the differences between these approaches is essential. The choice between classical machine learning and LLM-based solutions affects system architecture, infrastructure requirements, operational cost, evaluation strategies, and system reliability.

This chapter explains how classical machine learning systems differ from LLM-based systems and when each approach is most appropriate in production environments.

Within the **AI Systems Reference Stack**, these approaches primarily affect:

- **Model Layer** — type of model used for inference
- **Prompt Layer** — how model behavior is controlled
- **Retrieval Layer** — how external knowledge is incorporated
- **Application Layer** — how models are integrated into business workflows

Understanding these distinctions allows engineers to design **hybrid AI systems** that combine specialized models with general-purpose reasoning systems.

---

## Concept Overview

Machine learning systems and LLM systems represent two different approaches to intelligent behavior.

```

AI Systems
│
├ Classical Machine Learning
│ ├ Task-specific models
│ ├ Structured feature inputs
│ └ Deterministic prediction pipelines
│
└ Large Language Models
├ Foundation models
├ Prompt-driven behavior
└ Context-based reasoning

```

Classical machine learning models are optimized for **specific prediction tasks**, while LLMs are designed to **generalize across many tasks using prompts and contextual information**.

**Key Concept — Specialized vs General Models**

Traditional machine learning systems are highly specialized and efficient for a single task. Large language models trade efficiency for flexibility by providing a single model capable of performing many tasks through prompt configuration.

This distinction leads to fundamentally different engineering practices.

| Aspect             | Classical ML              | LLM Systems              |
| ------------------ | ------------------------- | ------------------------ |
| Model purpose      | Single task               | General-purpose          |
| Input format       | Structured features       | Natural language         |
| Adaptation         | Retraining required       | Prompt engineering       |
| Inference behavior | Deterministic predictions | Probabilistic generation |
| System control     | Feature engineering       | Prompt + context         |

A deeper comparison highlights additional engineering differences.

| Property    | Classical ML           | LLM                     |
| ----------- | ---------------------- | ----------------------- |
| Training    | Task-specific training | Large-scale pretraining |
| Interface   | Feature input / API    | Prompt interface        |
| Adaptation  | Retraining required    | Prompt engineering      |
| Scope       | Single task            | Many tasks              |
| Output type | Prediction             | Generated output        |

Another important difference lies in how systems represent input data.

| Property            | Classical ML                  | LLM Systems             |
| ------------------- | ----------------------------- | ----------------------- |
| Feature design      | Manual feature engineering    | Learned representations |
| Data representation | Structured numerical features | Token embeddings        |

Modern AI systems often combine both approaches.

---

## 3.1 Two Approaches to Intelligent Behavior

AI systems can broadly be divided into two categories:

1. **Task-specific models** trained for a single purpose.
2. **General-purpose models** capable of performing many tasks.

Classical machine learning falls into the first category.

Examples include:

- fraud detection
- credit risk prediction
- recommendation systems
- anomaly detection
- image classification

Large language models belong to the second category. They are pretrained on large corpora and can perform multiple tasks through prompting.

Examples include:

- summarization
- translation
- conversational assistants
- document analysis
- code generation

This distinction has significant implications for system architecture, data pipelines, and infrastructure design.

---

## 3.2 Classical Machine Learning: Task-Specific Models

Traditional machine learning systems follow a structured development pipeline.

```

Data Collection
↓
Feature Engineering
↓
Model Training
↓
Evaluation
↓
Deployment

```

Most classical ML systems rely on **supervised learning**, where models are trained using labeled datasets that map inputs to expected outputs.

The model is trained using labeled datasets designed for a specific prediction task.

Examples:

| Use Case               | Typical Model                           |
| ---------------------- | --------------------------------------- |
| Spam detection         | Logistic regression / gradient boosting |
| Fraud detection        | Gradient boosting                       |
| Recommendation systems | Collaborative filtering                 |
| Image recognition      | Convolutional neural networks           |

These systems rely heavily on **feature engineering**, where domain experts design the input features used by the model.

Once deployed, the model performs inference using a structured pipeline:

```

Input Features
↓
Model
↓
Prediction

```

Predictions are typically:

- fast
- inexpensive
- deterministic

This makes classical ML particularly well suited for **high-throughput decision systems**.

---

## 3.3 Large Language Models: Generalist Reasoning Engines

Large language models follow a different paradigm.

Instead of training separate models for each task, a **single pretrained model** is trained on massive datasets containing text and code.

These models are commonly referred to as **foundation models** because they provide a base capability that can be adapted to many downstream tasks.

The model learns:

- language structure
- semantic relationships
- reasoning patterns
- general world knowledge

LLMs operate on **tokenized representations of text**, meaning that input text is converted into sequences of tokens before being processed by the model.

Tasks are performed using prompts.

```

Prompt

- Context
- Model
  ↓
  Generated Output

```

Unlike classical ML models that produce discrete predictions, LLMs **generate output tokens sequentially**, producing text, code, or structured responses.

LLMs operate within a limited **context window**, which restricts how much information can be processed in a single inference request.

For example, the same LLM can perform:

- translation
- summarization
- classification
- code generation

without retraining.

However, LLMs are **probabilistic systems**. Their outputs may vary depending on decoding parameters such as:

- temperature
- top-p sampling
- token limits

This flexibility makes LLMs powerful but also introduces new engineering challenges such as prompt sensitivity, hallucinations, and higher inference cost.

LLM inference also tends to have **higher latency** than traditional ML models due to autoregressive token generation.

---

## 3.4 Side-by-Side Comparison

The differences between classical machine learning and LLM systems affect multiple dimensions of system design.

| Dimension      | Classical ML            | LLM Systems                         |
| -------------- | ----------------------- | ----------------------------------- |
| Training       | Task-specific datasets  | Large-scale pretraining             |
| Adaptation     | Retraining required     | Prompt-based configuration          |
| Input format   | Structured features     | Natural language                    |
| Output         | Predictions             | Generated text or structured output |
| Inference cost | Low                     | Higher                              |
| Latency        | Milliseconds            | Often seconds                       |
| Flexibility    | Limited to trained task | Multi-purpose                       |

Another useful comparison highlights the engineering perspective:

| Property   | ML Model               | LLM                     |
| ---------- | ---------------------- | ----------------------- |
| Training   | Task-specific training | Large-scale pretraining |
| Interface  | Feature input / API    | Prompt interface        |
| Adaptation | Retraining required    | Prompt engineering      |
| Scope      | Single task            | Many tasks              |

This comparison highlights an important trade-off:

**specialization vs flexibility**.

In classical ML systems, evaluation focuses on **model-level metrics** such as:

- accuracy
- precision
- recall
- F1 score

In contrast, LLM-based systems often require **system-level evaluation**, measuring end-to-end task performance across prompts, retrieval, and model outputs.

---

## 3.5 When to Use Each Approach

Selecting the appropriate model type depends on system requirements.

### When Classical Machine Learning is Preferred

Classical ML is often better when:

- the problem is well-defined
- large labeled datasets exist
- latency requirements are strict
- predictions must be highly consistent
- inference must scale to millions of requests

Examples include:

- fraud detection
- recommendation ranking
- demand forecasting
- anomaly detection
- real-time bidding systems

### When LLMs Are Preferred

LLMs are more appropriate when:

- tasks involve natural language
- reasoning or explanation is required
- workflows combine multiple tasks
- inputs are unstructured documents
- flexibility is more important than efficiency

Examples include:

- conversational assistants
- document analysis
- knowledge assistants
- coding assistants
- research copilots

---

## 3.6 Practical Example: Document Routing

Consider a system that routes documents to the correct department.

A classical machine learning approach might use:

```

Document
↓
Feature Extraction
↓
Text Classification Model
↓
Department Prediction

```

An LLM-based approach might look like:

```

Document
↓
Prompt + Context
↓
LLM
↓
Structured Output

```

The ML solution is usually:

- faster
- cheaper
- more predictable

The LLM solution is:

- more flexible
- easier to adapt
- capable of reasoning over complex documents

---

## 3.7 Fine-Tuning: Bridging ML and LLM

Fine-tuning allows engineers to adapt a pretrained model to a specific domain.

Instead of training from scratch, the model is updated using a smaller domain-specific dataset.

Fine-tuning can improve:

- accuracy
- domain knowledge
- response consistency

However, it also introduces additional operational complexity:

- training infrastructure
- dataset management
- model versioning

Because of this complexity, many production systems instead rely on **retrieval-augmented generation (RAG)** to provide domain knowledge without retraining the model.

Retrieval systems supply external knowledge to LLMs dynamically during inference.

---

## 3.8 Embedding Models: A Special Category

Embedding models convert text into numerical vectors that capture semantic meaning.

Typical pipeline:

```

Text
↓
Embedding Model
↓
Vector Representation
↓
Vector Database Search

```

Embedding models enable:

- semantic search
- document retrieval
- clustering
- similarity matching

They are a fundamental component of **RAG architectures**, where embeddings allow relevant documents to be retrieved and injected into the model's context.

Embedding models often bridge traditional machine learning and LLM-based architectures.

---

## 📋 Chapter Summary

- Classical machine learning models are designed for **specific prediction tasks** using structured input features.
- Large language models are **general-purpose foundation models** capable of performing many tasks through prompts.
- ML models are typically more efficient and predictable for narrow tasks.
- LLMs provide flexibility and reasoning capabilities but introduce higher inference cost and architectural complexity.
- Fine-tuning allows pretrained models to be adapted to specific domains.
- Embedding models enable semantic retrieval and are a core component of modern **RAG architectures**.
- Many production AI systems combine classical ML models, embedding models, and LLM reasoning within a single architecture.

---

## ❓ Comprehension Questions

1. What are the main differences between classical machine learning models and large language models in terms of training and system integration?
2. Why are classical machine learning models typically more efficient for narrow prediction tasks?
3. How do prompts enable LLMs to perform tasks without retraining?
4. In which situations would a classical ML solution be preferable to an LLM-based system?
5. What role do embedding models play in modern AI architectures?

---

## References

### Papers

- Language Models are Few-Shot Learners (GPT-3) — Brown et al., 2020
  https://arxiv.org/abs/2005.14165

- BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al., 2018
  https://arxiv.org/abs/1810.04805

- LoRA: Low-Rank Adaptation of Large Language Models — Hu et al., 2021
  https://arxiv.org/abs/2106.09685

- MTEB: Massive Text Embedding Benchmark — Muennighoff et al., 2022
  https://arxiv.org/abs/2210.07316

### Documentation

- Sentence Transformers Documentation — https://www.sbert.net
- Hugging Face Model Hub — https://huggingface.co/models
- OpenAI Embeddings Guide — https://platform.openai.com/docs/guides/embeddings
- scikit-learn Documentation — https://scikit-learn.org/stable/

### Books

- Designing Machine Learning Systems — Chip Huyen, O'Reilly, 2022

---

## See Also

Related chapters:

- Chapter 2 — Software 1.0 vs Software 2.0
- Chapter 4 — AI System Types
- Chapter 5 — Tokens and Context

---

## Key Takeaways

- Classical machine learning and LLM systems represent different paradigms for building intelligent software.
- ML models are specialized and efficient for narrow prediction tasks.
- LLMs are general-purpose reasoning systems configured through prompts.
- Embedding models bridge traditional ML techniques and modern LLM architectures.
- Many production AI systems combine **ML models, retrieval systems, and LLM reasoning** in hybrid architectures.
