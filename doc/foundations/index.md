# Foundations

---

## Context

Modern AI applications are fundamentally different from traditional software systems.
Instead of relying solely on deterministic algorithms, AI systems integrate probabilistic models, data pipelines, prompts, retrieval infrastructure, and orchestration logic.

Because of this shift, building reliable AI-powered applications requires a new engineering discipline: **AI Systems Engineering**.

The **Foundations** section introduces the core concepts required to understand how modern AI systems are designed, built, and operated. Rather than focusing on machine learning theory, these chapters establish the **system-level mental models** that engineers need when working with large language models and AI architectures.

The goal of this section is to help readers understand:

- how AI systems differ from traditional software
- how large language models behave
- how engineers control model behavior
- why modern AI architectures are necessary

By the end of this section, readers will understand the conceptual building blocks that underpin the rest of the book.

---

## Pedagogical Map of Foundations

The seven chapters in this section follow a deliberate progression designed to introduce AI Systems Engineering in a structured way.

Each chapter answers a specific question about how modern AI systems work.

```id="i2drts"
AI Systems Engineering
        ↓
AI System Types
        ↓
Software Paradigm Shift
        ↓
ML vs LLM Systems
        ↓
Tokens and Context
        ↓
Prompt Engineering
        ↓
Limitations of LLMs
```

This progression mirrors how engineers typically learn to reason about AI systems.

### 1. Understanding AI Systems

The section begins with **[Introduction to AI Systems Engineering](introduction-to-ai-systems-engineering.md)**, which explains how modern AI applications differ from traditional software and why AI systems must be engineered as composed systems rather than simple model calls.

### 2. Understanding AI Architectures

Next, **[AI System Types](ai-system-types.md)** introduces the main architectural categories used in modern AI applications, including prompt-based systems, retrieval-augmented systems, workflow systems, and agent systems.

This chapter provides the architectural vocabulary used throughout the book.

### 3. Understanding the Paradigm Shift

**[Software 1.0 vs Software 2.0](software-10-vs-software-20.md)** explains the historical shift from deterministic programming to machine learning systems where behavior emerges from data.

This chapter establishes the conceptual foundation for understanding why modern AI systems behave differently from traditional applications.

### 4. Understanding Model Types

**[Machine Learning vs LLM Systems](machine-learning-vs-llm.md)** explains how traditional machine learning models differ from modern large language models and how LLM-based systems shift engineering focus from training pipelines to inference architectures.

### 5. Understanding Model Constraints

**[Tokens and Context](tokens-and-context.md)** introduces the computational constraints of large language models, including tokenization and context windows.

These constraints shape how prompts, retrieval pipelines, and system architectures must be designed.

### 6. Understanding Model Control

**[Prompt Engineering](prompt-engineering.md)** explains how prompts act as the primary interface for controlling model behavior and how prompt templates, patterns, and system prompts are used in production systems.

### 7. Understanding Model Limitations

Finally, **[Limitations of LLMs](limitations-of-llms.md)** explains the inherent weaknesses of language models, including hallucinations, reasoning limitations, knowledge boundaries, and non-deterministic outputs.

These limitations explain why production AI systems rely on architectures such as retrieval pipelines, workflows, evaluation systems, and guardrails.

---

## How This Fits Together

Although each chapter introduces a specific concept, modern AI applications combine these ideas into a single system architecture.

A simplified AI system can be understood as a pipeline that transforms **user intent into model responses**.

```id="b2k7hc"
User Query
↓
Application Layer
↓
Prompt Builder
↓
Retriever
↓
Vector Database
↓
LLM Inference
↓
Response
```

Each stage of this pipeline corresponds to concepts introduced in the Foundations chapters.

- **AI System Types** introduces the architectural patterns used to structure these pipelines.
- **Tokens and Context** explains the computational limits that constrain how much information can be passed to the model.
- **Prompt Engineering** describes how prompts structure the instructions and context used during model inference.
- **Limitations of LLMs** explains why additional system components are required to improve reliability.

In real-world AI systems, additional architectural layers are typically present. The **AI Systems Reference Stack** introduced earlier in the Foundations section helps organize these responsibilities.

```id="4k3jpt"
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

This layered model illustrates how modern AI systems separate responsibilities across the stack.

Understanding how these components interact is the central goal of **AI Systems Engineering**.

The Foundations section therefore provides the conceptual framework needed to understand the more advanced architectures introduced later in the book.

---

## Why This Progression Matters

This sequence is intentional.

Engineers must first understand **how AI systems behave** before learning how to design complex architectures around them.

The Foundations section therefore moves from:

```id="y9tnmj"
system concepts
↓
architectures
↓
models
↓
model constraints
↓
prompt control
↓
model limitations
```

By the end of this section, readers will understand why modern AI systems are built as **layered architectures composed of multiple interacting components**.

This understanding prepares the reader for the rest of the book, which explores the engineering disciplines required to build production AI systems.

---

## Content

The **Foundations** section includes the following chapters:

1. **[Introduction to AI Systems Engineering](introduction-to-ai-systems-engineering.md)**
   Introduces the discipline of AI Systems Engineering and explains how modern AI systems differ from traditional software architectures.

2. **[AI System Types](ai-system-types.md)**
   Describes the major architectural categories used in AI systems, including prompt-based systems, retrieval-augmented systems, workflow systems, and agent systems.

3. **[Software 1.0 vs Software 2.0](software-10-vs-software-20.md)**
   Explains the paradigm shift from deterministic software to machine learning systems where behavior is learned from data.

4. **[Machine Learning vs LLM Systems](machine-learning-vs-llm.md)**
   Compares traditional machine learning systems with modern large language model architectures.

5. **[Tokens and Context](tokens-and-context.md)**
   Explains how language models process text using tokens and how context windows constrain system design.

6. **[Prompt Engineering](prompt-engineering.md)**
   Introduces prompt design techniques used to control model behavior and structure AI system inputs.

7. **[Limitations of LLMs](limitations-of-llms.md)**
   Explains the inherent limitations of large language models and why AI systems require additional architectural components to ensure reliability.

---

## What Comes Next

After understanding the conceptual foundations of AI systems, the rest of the book explores the engineering disciplines required to build production systems:

- **RAG Engineering** — designing retrieval pipelines and knowledge systems
- **Workflow and Agent Architectures** — orchestrating complex AI behaviors
- **Evaluation Engineering** — measuring system quality and reliability
- **Platform and Infrastructure Engineering** — operating AI systems at scale

These topics build directly on the mental models introduced in the Foundations section.
