# Foundations

[⬅ Back to Main Table of Contents](../index.md)

## Context

Modern AI applications are not simply machine learning models exposed through APIs. They are **complex systems** that integrate models, prompts, retrieval pipelines, orchestration logic, and external tools.

Building these systems requires a different engineering mindset than traditional software development. Engineers must reason about probabilistic models, data-driven behavior, prompt design, retrieval architectures, and evaluation pipelines.

The goal of the **Foundations** section is to introduce the core concepts that underpin modern **AI Systems Engineering**. These chapters establish the conceptual vocabulary and mental models needed to understand how AI systems are designed, built, and operated in production.

Rather than focusing on model training or machine learning theory, this section focuses on **system-level thinking**. It explains how modern AI applications combine software engineering practices with machine learning components and large language models.

By the end of this section, readers will understand:

- how AI systems differ from traditional software systems
- the architectural role of machine learning models and LLMs
- the major types of AI system architectures
- how tokens, prompts, and context influence system behavior
- the limitations of large language models and their architectural implications

These concepts form the foundation for the more advanced topics explored in later parts of the book, including **retrieval engineering, evaluation pipelines, platform infrastructure, and production operations**.

---

## Content

The **Foundations** section introduces the fundamental concepts required to understand modern AI system architectures.

- **[Chapter 1 — Introduction to AI Systems Engineering](introduction-to-ai-systems-engineering.md)**  
  Introduces the discipline of AI Systems Engineering and explains how AI applications differ from traditional software systems.

- **[Chapter 2 — Software 1.0 vs Software 2.0](software-10-vs-software-20.md)**  
  Explains the shift from rule-based software to data-driven machine learning systems and how LLM-based systems extend this paradigm.

- **[Chapter 3 — Machine Learning vs LLM](machine-learning-vs-llm.md)**  
  Compares traditional machine learning models with large language models and explains when each approach is appropriate in system design.

- **[Chapter 4 — AI System Types](ai-system-types.md)**  
  Introduces common architectural patterns used in AI systems, including prompt-based systems, RAG architectures, workflow systems, and agent-based systems.

- **[Chapter 5 — Tokens and Context](tokens-and-context.md)**  
  Explains how language models process information using tokens and context windows, and how these constraints influence system design.

- **[Chapter 6 — Prompt Engineering](prompt-engineering.md)**  
  Introduces techniques for designing prompts that reliably control the behavior of language models in production systems.

- **[Chapter 7 — Limitations of Large Language Models](limitations-of-llms.md)**  
  Examines the inherent limitations of LLMs, including hallucinations, reasoning errors, and security vulnerabilities, and explains how system architectures mitigate these issues.

---

## How These Chapters Fit Together

The chapters in this section build progressively toward a system-level understanding of AI applications.

```

AI Systems Engineering
│
├ Software Paradigms
│   ├ Software 1.0
│   └ Software 2.0
│
├ Model Types
│   ├ Machine Learning Models
│   └ Large Language Models
│
├ System Architectures
│   ├ Prompt Systems
│   ├ Retrieval-Augmented Systems
│   ├ Workflow Systems
│   └ Agent Systems
│
└ System Constraints
├ Tokens and Context
├ Prompt Design
└ LLM Limitations

```

Together, these topics provide the conceptual foundation necessary to understand the architecture and engineering practices used in modern AI systems.
