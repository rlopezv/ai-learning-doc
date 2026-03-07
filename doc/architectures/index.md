# Architectures

[⬅ Back to Book Index](../index.md)

---

## Context

Artificial intelligence models are powerful capabilities, but they do not form complete applications on their own. Real-world AI systems are built by combining models with prompts, retrieval pipelines, tools, orchestration workflows, and external services.

These components must be organized into **coherent system architectures** that allow AI applications to operate reliably in production environments.

The purpose of this section is to introduce the **architectural foundations of modern AI systems**. It explains how models interact with prompts, knowledge systems, external tools, and orchestration frameworks to form complete applications.

Understanding these architectural patterns is essential for AI Systems Engineering because most of the complexity in AI systems arises not from the models themselves, but from how models are integrated into larger systems.

---

## How This Section Fits in the Book

The **Foundations** section introduced the conceptual building blocks of AI systems:

- the shift from deterministic software to probabilistic systems
- the role of prompts, tokens, and context
- the limitations of language models
- the emergence of AI systems as composed architectures

This **Architectures** section builds on those ideas by explaining **how AI systems are structured in practice**.

While the Foundations section focused on _concepts_, this section focuses on _system design_.

The architectural patterns presented here provide the conceptual basis for later sections of the book, which explore:

- retrieval engineering
- workflow and agent orchestration
- evaluation and testing
- platform and infrastructure design

---

## Architectural Progression

AI system architectures can be understood as a progression of increasing capability and complexity.

```id="architecture-progression"
Prompt-Based Systems
↓
Retrieval-Augmented Systems
↓
Tool-Augmented Systems
↓
Workflow-Orchestrated Systems
↓
Agent-Based Systems
```

Each stage introduces additional architectural components:

- **Prompt-based systems** rely primarily on prompt design and model inference.
- **Retrieval-augmented systems** introduce external knowledge through retrieval pipelines.
- **Tool-augmented systems** enable models to interact with external systems and APIs.
- **Workflow systems** coordinate structured multi-step processing pipelines.
- **Agent systems** introduce dynamic reasoning and action loops.

These architectures are not mutually exclusive. Most production AI systems combine several of these approaches into **hybrid architectures**, which are explored later in this section.

---

## The Role of the AI Systems Reference Stack

A key concept introduced in this section is the **AI Systems Reference Stack**, which organizes AI system responsibilities into architectural layers.

```id="ai-stack"
Interaction Layer
Application Layer
Orchestration Layer
Prompt Layer
Retrieval Layer
Model Layer
Data Layer
Infrastructure Layer
```

This layered architecture helps engineers reason about system design, responsibilities, and operational complexity.

Throughout this section, architectural patterns are mapped to different layers of this stack to clarify how AI systems are constructed.

---

## Contents

### Architectural Foundations

- **[AI Systems Reference Stack](ai-systems-reference-stack.md)**
  Introduces the layered architecture used to reason about AI system design.

- **[Prompt-Based Architectures](prompt-based-architectures.md)**
  Explains the simplest form of AI applications built around prompt-driven model interactions.

---

### Knowledge-Augmented Architectures

- **[Retrieval-Augmented Generation (RAG)](retrieval-augmented-generation.md)**
  Introduces architectures that combine language models with external knowledge sources.

---

### Tool-Integrated Architectures

- **[Tool-Augmented LLM Systems](tool-augmented-llm.md)**
  Explores architectures that allow models to interact with APIs, databases, and external services.

---

### Orchestration Architectures

- **[Workflow Systems](workflow-systems.md)**
  Explains how structured pipelines coordinate multi-step AI system execution.

- **[Agent Architectures](agent-architectures.md)**
  Introduces dynamic reasoning systems capable of planning actions and interacting with tools.

---

### Architectural Synthesis

- **[Architectural Patterns & Landscape](architectural-patterns.md)**
  Provides a synthesis of architectural patterns and an overview of the AI systems ecosystem, including model platforms, data infrastructure, orchestration frameworks, and agent systems.

---

## Key Ideas in This Section

Several key ideas guide the architectural perspective used throughout this section.

### AI Systems Are Composed Architectures

AI applications are rarely single model calls. Instead, they combine multiple interacting components including prompts, retrieval pipelines, orchestration systems, and tools.

### Architecture Determines System Capabilities

The architecture of a system determines what it can do. Introducing retrieval systems, tools, workflows, or agents expands the capabilities of the application but also increases system complexity.

### Engineering Trade-offs Are Central

Architectural decisions involve trade-offs between:

- capability
- reliability
- latency
- cost
- operational complexity

Understanding these trade-offs is essential for designing production AI systems.

---

## What You Will Learn

By the end of this section, you will understand:

- how modern AI systems are structured
- how prompts, retrieval systems, tools, and agents interact
- how different architectures map to the AI Systems Reference Stack
- how architectural choices influence system capabilities and operational complexity

These concepts form the architectural foundation required for building and operating production AI systems.

---

## See Also

- [Foundations](../foundations/index.md)
- [RAG Engineering](../rag-engineering/index.md)
