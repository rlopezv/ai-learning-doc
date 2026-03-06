# Chapter 2 — Software 1.0 vs Software 2.0

[⬅ Back to Foundations](index.md)

## Context

This chapter explains the conceptual shift from traditional software development to machine learning–driven systems. Understanding this shift is essential for **AI Systems Engineering** because it fundamentally changes how system behavior is defined, implemented, and maintained.

In classical software systems, developers explicitly encode behavior using deterministic algorithms. In contrast, modern AI systems increasingly rely on models trained from data, where behavior emerges statistically rather than being directly programmed.

The distinction between **Software 1.0** and **Software 2.0** was popularized by Andrej Karpathy in his essay _Software 2.0_. The framework describes how modern systems increasingly move from handwritten logic toward learned behavior derived from data.

This chapter introduces the concepts of **Software 1.0**, **Software 2.0**, and **LLM-based systems**, explaining how each paradigm defines and controls system behavior.

Understanding these paradigms provides a mental model for reasoning about the architecture of modern AI systems. In the broader **AI Systems Reference Stack**, these paradigms influence several layers:

- **Application Layer** — deterministic application logic
- **Model Layer** — trained machine learning models
- **Prompt Layer** — structured instructions guiding model behavior
- **Retrieval Layer** — contextual knowledge used during inference

Recognizing where each paradigm applies helps engineers design systems that combine deterministic logic with probabilistic AI capabilities.

---

## Concept Overview

The evolution of software systems can be understood through three stages.

Software 1.0

Code + Input → Output

Software 2.0

Training Data + Learning Algorithm → Model  
Model + Input → Output

LLM Systems

Model + Prompt + Context → Output

These paradigms represent different ways of defining system behavior.

| Paradigm     | Behavior Defined By      | Key Artifact                  |
| ------------ | ------------------------ | ----------------------------- |
| Software 1.0 | Explicit code            | Source code                   |
| Software 2.0 | Learned parameters       | Model weights                 |
| LLM Systems  | Model + prompt + context | Prompts and retrieval context |

A useful way to understand the transition is through the **control surface** of the system.

The **control surface** refers to the primary artifact engineers modify in order to influence system behavior.

| Paradigm     | Control Surface     |
| ------------ | ------------------- |
| Software 1.0 | Source code         |
| Software 2.0 | Training data       |
| LLM Systems  | Prompts and context |

A broader comparison highlights the engineering implications of these paradigms.

| Aspect              | Software 1.0  | Software 2.0        | LLM Systems               |
| ------------------- | ------------- | ------------------- | ------------------------- |
| Behavior defined by | Code          | Training data       | Prompt + context          |
| Primary artifact    | Source code   | Model weights       | Prompts                   |
| Engineering focus   | Programming   | Model training      | Prompt + retrieval design |
| Testing method      | Unit tests    | Evaluation datasets | System evaluation         |
| Determinism         | Deterministic | Probabilistic       | Probabilistic             |

Each paradigm shifts where engineering effort is applied when building and improving systems.

A simplified comparison:

Software 1.0

Input  
↓  
Code  
↓  
Output

Software 2.0

Input  
↓  
Model  
↓  
Prediction

LLM Systems

Input  
↓  
Prompt + Context  
↓  
Model  
↓  
Generated Output

In modern AI systems, these paradigms often coexist within a single architecture. Deterministic application logic may coordinate model inference, while prompts and retrieval pipelines shape how models behave.

Understanding these differences is essential for designing modern AI systems.

---

## 1.1 Software 1.0: Deterministic Programs

Traditional software systems are built from deterministic algorithms written by developers.

Example structure:

Input  
↓  
Program Logic  
↓  
Output

Developers explicitly specify:

- control flow
- decision rules
- data transformations
- algorithms

Because the logic is deterministic, system behavior is predictable and reproducible. Given identical inputs, the program always produces the same output.

Typical characteristics of Software 1.0 systems include:

- clearly defined logic
- explicit control structures
- strong reliance on unit testing
- deterministic correctness guarantees
- well-defined failure modes

These systems are well suited for domains where rules can be clearly specified, such as:

- accounting systems
- transaction processing
- network protocols
- compilers
- database engines

However, many real-world problems do not have easily definable rules.

---

## 1.2 Limitations of Rule-Based Systems

Tasks involving perception, language, or complex reasoning are difficult to encode using explicit rules.

Examples include:

- natural language understanding
- speech recognition
- image recognition
- semantic search
- conversational interaction

Attempts to implement these capabilities using rule-based systems often lead to:

- brittle logic
- extremely large rule sets
- high maintenance cost
- limited scalability
- poor generalization

Rule-based systems typically struggle with **generalization**, the ability to handle new inputs not explicitly covered by predefined rules.

For example, a rule-based system designed to parse language would require an enormous number of handcrafted rules to handle grammar variations, idioms, and ambiguity. Maintaining such systems becomes impractical as complexity grows.

This limitation motivated the adoption of **machine learning techniques**, where systems learn patterns directly from data.

---

## 1.3 Software 2.0: Programs Learned from Data

Machine learning introduces a new paradigm where system behavior is learned from data rather than directly written by developers.

Instead of implementing explicit rules, engineers define:

- a model architecture
- a training procedure
- a dataset

Training then produces a model whose parameters encode learned behavior.

Training Pipeline:

Dataset  
↓  
Training Pipeline  
↓  
Model  
↓  
Evaluation

Inference Pipeline:

Input  
↓  
Model  
↓  
Prediction

In this paradigm:

- the **dataset** becomes a critical artifact
- the **training process** replaces manual rule writing
- the resulting **model parameters** encode system behavior

Rather than editing code to change system behavior, engineers improve performance by:

- collecting better data
- improving training procedures
- tuning model architectures
- refining evaluation metrics

This shift transforms how software systems are built and maintained.

Unlike deterministic programs, machine learning systems are **probabilistic**. Even when the same input is provided, small variations in model inference may produce different outputs.

---

## 1.4 Characteristics of Software 2.0 Systems

Software 2.0 systems differ from traditional software in several important ways.

| Aspect              | Software 1.0     | Software 2.0        |
| ------------------- | ---------------- | ------------------- |
| Behavior source     | Handwritten code | Learned model       |
| Development process | Programming      | Training            |
| Primary artifact    | Source code      | Model weights       |
| Testing             | Unit tests       | Evaluation datasets |
| Determinism         | Deterministic    | Probabilistic       |

Because behavior is learned rather than explicitly written, engineers must focus on:

- dataset quality
- training procedures
- evaluation metrics
- model monitoring
- experiment tracking

Software 2.0 development therefore introduces new engineering practices such as:

- dataset versioning
- model evaluation pipelines
- experiment management
- automated retraining

These challenges led to the emergence of **MLOps (Machine Learning Operations)** and specialized machine learning engineering practices.

---

## 1.5 From Software 2.0 to LLM Systems

Large language models introduce another shift in how AI capabilities are integrated into software systems.

Instead of training a new model for each task, engineers can reuse **pretrained foundation models** and control their behavior through prompts and contextual data.

Foundation models are large pretrained models trained on massive datasets that can be adapted to many tasks.

This results in a new inference structure:

Model

- Prompt
- Context  
  ↓  
  Generated Output

In this paradigm:

- the **model** provides general reasoning capability
- the **prompt** defines task instructions
- the **context** provides domain knowledge

In some cases, engineers further adapt foundation models using techniques such as **fine-tuning**, **instruction tuning**, or **parameter-efficient adaptation** methods.

Inference behavior can also be influenced by **decoding parameters**, such as:

- temperature
- top-p sampling
- maximum token limits

Unlike many Software 2.0 systems, LLM applications are often **inference-first systems**. Instead of training models from scratch, engineers focus primarily on how models are used during inference.

This approach significantly reduces the cost and complexity of building intelligent systems, allowing a single model to support many tasks.

Examples include:

- summarization
- question answering
- code generation
- document analysis
- conversational assistants

---

## 1.6 Prompts as System Artifacts

In LLM-based systems, prompts become **first-class engineering artifacts**.

Prompts define:

- task instructions
- reasoning structure
- output format
- constraints and behavioral guidelines

Example prompt structure:

System Prompt  
↓  
User Query  
↓  
Retrieved Context  
↓  
Model Response

Because prompts strongly influence model behavior, they must be treated as managed artifacts within the system.

This means prompts should be:

- versioned
- evaluated
- iterated
- integrated into system architecture

Many modern systems also rely on **embeddings**, which represent text as numerical vectors. These embeddings enable **semantic search** and allow retrieval systems to find relevant information based on meaning rather than exact keyword matching.

Because models operate within a limited **context window**, retrieval pipelines are often used to dynamically supply relevant information during inference.

This architectural pattern is commonly known as **Retrieval-Augmented Generation (RAG)**.

Prompt engineering therefore becomes a core part of AI system design.

Because LLM systems depend on prompts, retrieval, and orchestration, **evaluation must occur at the system level**, rather than only at the model level.

---

## 1.7 Architectural Implications

The shift from Software 1.0 to AI-driven systems changes the architecture of modern applications.

Traditional architecture:

Application  
↓  
Deterministic Logic  
↓  
Database

AI-driven architecture:

User Query  
↓  
Application Layer  
↓  
Orchestration Layer  
↓  
Retriever  
↓  
Prompt + Context Builder  
↓  
Model Inference  
↓  
Tool Calls (optional)  
↓  
Response

Some systems allow models to call **external tools**, such as:

- APIs
- databases
- search engines
- computation services

These interactions enable models to retrieve data, perform calculations, or execute actions beyond pure text generation.

In these systems, behavior emerges from interactions between multiple artifacts:

- prompts
- models
- retrieval pipelines
- embeddings
- external tools
- evaluation datasets

Modern AI systems therefore behave as **artifact-driven systems**, where these artifacts collectively determine system behavior.

Because LLM inference is computationally expensive, engineers must also consider **latency and cost per query**. Token usage, model size, and the number of model calls directly influence operational cost.

These components correspond to architectural layers introduced in the previous chapter, including:

- **Application Layer**
- **Orchestration Layer**
- **Prompt Layer**
- **Retrieval Layer**
- **Model Layer**

Understanding how these layers interact is essential for designing scalable AI systems.

---

## 1.8 A Hybrid Paradigm

Modern AI applications rarely rely exclusively on a single paradigm.

Instead, they combine elements of all three.

Software 1.0  
Application logic, APIs, orchestration

Software 2.0  
Machine learning models and classifiers

LLM Systems  
Prompt-driven reasoning and generation

Example hybrid architecture:

User Query  
↓  
Application Logic (Software 1.0)  
↓  
Retriever (Software 2.0 components)  
↓  
Prompt Construction  
↓  
LLM Inference  
↓  
Generated Response

In practice:

- **Software 1.0 components** manage system control and integration
- **Software 2.0 components** provide specialized prediction capabilities
- **LLM systems** provide flexible reasoning and language generation

This hybrid architecture is the foundation of most modern AI applications.

Understanding how these paradigms interact is essential for designing **robust, scalable AI systems**.

---

## 📋 Chapter Summary

- **Software 1.0** refers to traditional deterministic programs written explicitly by developers.
- **Software 2.0** describes systems where behavior is learned from data using machine learning models.
- The concept of Software 2.0 was popularized by Andrej Karpathy.
- Large language model systems extend this paradigm by allowing engineers to reuse **foundation models** and guide their behavior through prompts and context.
- Prompts and retrieval pipelines become first-class artifacts that influence model reasoning and output generation.
- Many modern AI applications are **inference-first systems** that focus on prompt design and contextual information rather than model training.
- Production AI systems typically combine **Software 1.0, Software 2.0, and LLM-based components** within a hybrid architecture.

---

## ❓ Comprehension Questions

1. What distinguishes Software 1.0 systems from Software 2.0 systems?
2. Why do rule-based systems struggle with tasks such as language understanding or image recognition?
3. What artifacts replace handwritten rules in Software 2.0 systems?
4. How do LLM-based systems extend the Software 2.0 paradigm?
5. What does it mean for an AI system to be **inference-first**?
6. Why are prompts considered first-class artifacts in LLM-based systems?
7. Why do modern AI systems often combine deterministic application logic with LLM-based reasoning?

---

## References

### Articles

Karpathy, Andrej. _Software 2.0_  
https://karpathy.medium.com/software-2-0-a64152b37c35

### Foundational Papers

Vaswani et al. _Attention Is All You Need_  
https://arxiv.org/abs/1706.03762

### Books

Chip Huyen. _Designing Machine Learning Systems_. O'Reilly, 2022.

Martin Kleppmann. _Designing Data-Intensive Applications_. O'Reilly, 2017.

### Additional Resources

OpenAI Documentation  
https://platform.openai.com/docs

Hugging Face Documentation  
https://huggingface.co/docs
