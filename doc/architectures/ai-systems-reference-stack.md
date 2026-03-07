# AI Systems Reference Stack

[⬅ Back to Architectures](index.md)

---

## Context

Modern AI applications are not built as single model calls. Instead, they are **composed systems** made of multiple layers of functionality that interact to process user requests, retrieve knowledge, execute tasks, and generate responses.

As AI systems grow in complexity, engineers need a structured way to reason about how different components interact. Without a clear architectural model, it becomes difficult to design systems, debug failures, or reason about operational responsibilities.

The **AI Systems Reference Stack** provides a conceptual architecture for organizing the components of modern AI systems. It separates responsibilities into layers that represent different concerns of the system.

This layered model helps engineers understand:

- where different components belong in the system
- how data flows between components
- how responsibilities are separated across the architecture

The stack introduced in this chapter serves as a **reference architecture used throughout the book** to explain how AI systems are designed and operated.

The following chapters in the **Architectures** section describe architectural patterns built on top of this stack, including prompt-based systems, retrieval-augmented systems, workflow architectures, and agent systems.

---

## Concept Overview

The AI Systems Reference Stack organizes an AI system into a set of conceptual layers, each responsible for a specific part of system behavior.

```id="stack-overview"
Interaction Layer
↓
Application Layer
↓
Orchestration Layer
↓
Prompt Layer
↓
Retrieval Layer
↓
Model Layer
↓
Data Layer
↓
Infrastructure Layer
```

Each layer represents a different level of abstraction in the system.

Higher layers focus on **user interaction and application logic**, while lower layers provide **models, data, and infrastructure** that support system execution.

This layered architecture is similar to reference models used in other areas of software engineering, such as the **OSI networking model** or layered cloud architectures.

Not every AI system includes every layer. Simpler systems may omit components such as retrieval or orchestration, while more advanced architectures incorporate additional layers and services.

---

## 1. Interaction Layer

The **Interaction Layer** represents how users or external systems communicate with the AI application.

Examples include:

- chat interfaces
- web applications
- mobile applications
- API endpoints
- messaging platforms
- voice interfaces

Example flow:

```id="interaction-flow"
User
↓
Chat Interface
↓
AI Application
```

This layer focuses on **user experience and request handling** rather than AI-specific processing.

Responsibilities typically include:

- user input collection
- request formatting
- authentication
- session management

---

## 2. Application Layer

The **Application Layer** contains the main business logic of the system.

This layer determines how user requests are interpreted and how the system should respond.

Example:

```id="application-flow"
User Request
↓
Application Service
↓
Task Definition
```

Typical responsibilities include:

- request interpretation
- task routing
- business logic execution
- integration with application services

For example, an enterprise assistant may determine whether a request requires:

- document retrieval
- data analysis
- report generation
- question answering

---

## 3. Orchestration Layer

The **Orchestration Layer** coordinates the sequence of operations required to complete a task.

AI systems often require multiple steps such as:

- retrieving documents
- calling tools
- performing reasoning steps
- validating responses

Example orchestration flow:

```id="orchestration-flow"
User Query
↓
Orchestrator
↓
Retrieve Context
↓
Call Tool
↓
Generate Response
```

The orchestrator determines **how different system components interact** during execution.

Typical orchestration technologies include:

- workflow engines
- agent frameworks
- orchestration services

---

## 4. Prompt Layer

The **Prompt Layer** is responsible for constructing the prompts sent to language models.

Prompts define:

- system instructions
- task context
- user input
- formatting constraints

Example prompt composition:

```id="prompt-composition"
System Prompt
+
Task Instructions
+
Retrieved Context
+
User Query
```

Prompt templates are often treated as **versioned artifacts** that can evolve over time.

Effective prompt construction is essential for controlling model behavior and producing reliable outputs.

---

## 5. Retrieval Layer

The **Retrieval Layer** provides access to external knowledge sources.

Because language models cannot store all relevant information, retrieval systems allow AI applications to incorporate **domain-specific or up-to-date knowledge**.

Example retrieval pipeline:

```id="retrieval-pipeline"
User Query
↓
Embedding Model
↓
Vector Search
↓
Relevant Documents
↓
Context Builder
```

Typical retrieval components include:

- embedding models
- vector databases
- search indexes
- document stores

Retrieval pipelines are a core component of **Retrieval-Augmented Generation (RAG)** systems.

---

## 6. Model Layer

The **Model Layer** contains the machine learning models responsible for reasoning and generation.

Examples include:

- large language models
- embedding models
- reranking models
- classification models

Example model interaction:

```id="model-interaction"
Prompt
↓
LLM
↓
Generated Output
```

This layer performs the core tasks of:

- language understanding
- reasoning
- text generation
- semantic representation

Model selection and configuration strongly influence system behavior and performance.

---

## 7. Data Layer

The **Data Layer** stores the information used by AI systems.

This includes:

- documents
- knowledge bases
- embeddings
- evaluation datasets
- training datasets

Example structure:

```id="data-layer"
Documents
Embeddings
Evaluation Sets
Knowledge Sources
```

The quality of the data layer has a major impact on system performance and reliability.

Data pipelines are often responsible for preparing and updating these datasets.

---

## 8. Infrastructure Layer

The **Infrastructure Layer** provides the computational resources required to run the system.

This includes:

- cloud compute
- GPUs
- networking
- storage systems
- model inference services

Example infrastructure stack:

```id="infrastructure-stack"
Cloud Platform
↓
GPU Infrastructure
↓
Model Serving
↓
AI Application
```

This layer ensures the system can operate at scale while maintaining acceptable latency and reliability.

---

## 9. End-to-End Flow

Although these layers are conceptually separate, real AI systems involve interactions across multiple layers.

Example system flow:

```id="system-flow"
User
↓
Interaction Layer
↓
Application Layer
↓
Orchestration Layer
↓
Prompt Layer
↓
Retrieval Layer
↓
Model Layer
↓
Response
```

This layered execution model helps engineers understand:

- where failures occur
- how information flows through the system
- which components influence system behavior

---

## Chapter Summary

- Modern AI applications consist of multiple interacting components organized into architectural layers.
- The AI Systems Reference Stack provides a conceptual framework for understanding these layers.
- The stack separates responsibilities across interaction, application logic, orchestration, prompts, retrieval systems, models, data, and infrastructure.
- Not all systems require every layer, but the stack provides a useful reference model for reasoning about system design.
- This layered architecture helps engineers design scalable, maintainable, and observable AI systems.

---

## Comprehension Questions

1. What is the purpose of the AI Systems Reference Stack?
2. Which responsibilities belong to the Interaction Layer?
3. What role does the Orchestration Layer play in an AI system?
4. Why is the Prompt Layer important in LLM-based architectures?
5. How does the Retrieval Layer extend the capabilities of language models?
6. Why is the Data Layer critical for system reliability?

---

## References

### Papers

Attention Is All You Need — Vaswani et al., 2017
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al., 2020
[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

### Books

Designing Machine Learning Systems — Chip Huyen
[https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

Designing Data-Intensive Applications — Martin Kleppmann
[https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/)

---

## Key Takeaways

- AI systems are composed of multiple architectural layers.
- The AI Systems Reference Stack provides a framework for understanding these layers.
- Each layer has a specific responsibility within the system architecture.
- Separating responsibilities across layers improves scalability and maintainability.
- The layered architecture helps engineers reason about complex AI systems.
