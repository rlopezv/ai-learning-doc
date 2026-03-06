# Chapter 4 — AI System Types

[⬅ Back to Foundations](index.md)

## Context

This chapter introduces a practical classification of modern AI systems based on how they **access knowledge** and how much **autonomy they exercise during execution**. While previous chapters explored the paradigm shift from deterministic software to machine learning and large language models, this chapter focuses on how these technologies are assembled into real-world systems.

Understanding system types helps architects select the appropriate architecture for a given problem. Different designs offer different trade-offs in terms of reliability, flexibility, complexity, and operational cost.

This classification also prepares the reader for the next part of the book, where these patterns are explored in greater architectural depth.

---

## Concept Overview

AI applications built on large language models can be categorized by two primary dimensions:

- **Knowledge access** — whether the system relies only on the model's internal knowledge or accesses external data sources.
- **Execution autonomy** — whether the system follows a fixed workflow or dynamically decides which actions to perform.

These two dimensions produce several recognizable system patterns.

```text
AI System Types
│
├ Prompt-Based Systems
├ Retrieval-Augmented Systems (RAG)
├ Tool-Augmented Systems
├ Workflow Systems
├ Agent Systems
└ Hybrid Systems
```

Each type represents a different level of architectural sophistication and operational complexity.

**Key Concept — Knowledge Access vs Autonomy**

AI systems differ primarily in **how they obtain knowledge** and **how decisions about actions are made**. Simpler systems rely solely on model knowledge and fixed prompts. More advanced systems incorporate retrieval pipelines, external tools, or dynamic planning.

---

## 4.1 Prompt-Based Systems

Prompt-based systems represent the simplest form of LLM-powered applications.

In these systems, the model receives instructions directly through prompts and produces a response without interacting with external systems.

Typical architecture:

```text
User Input
↓
Prompt Template
↓
LLM
↓
Generated Response
```

These systems are easy to build and deploy but have several limitations:

- knowledge is limited to the model's training data
- responses may become outdated
- hallucination risk is higher

Common use cases include:

- content generation
- text summarization
- brainstorming tools
- basic chat assistants

Prompt-based systems are useful for narrow tasks but rarely sufficient for knowledge-intensive applications.

---

## 4.2 Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation systems enhance LLM responses by retrieving relevant information from external knowledge sources.

Typical architecture:

```text
User Query
↓
Retriever
↓
Vector Database
↓
Context Builder
↓
LLM
↓
Response
```

Instead of relying solely on model knowledge, the system dynamically injects relevant context into the prompt.

Benefits include:

- access to up-to-date knowledge
- reduced hallucination
- improved factual accuracy

Typical use cases:

- enterprise knowledge assistants
- document search systems
- technical support agents
- internal company copilots

RAG systems are among the most common production architectures for LLM applications.

---

## 4.3 Tool-Augmented Systems

Tool-augmented systems allow LLMs to interact with external tools such as APIs, databases, or computational services.

Example architecture:

```text
User Request
↓
LLM Reasoning
↓
Tool Invocation
↓
External System
↓
LLM Response
```

Tools may include:

- search APIs
- calculators
- database queries
- internal microservices

This approach allows models to perform tasks that require deterministic computation or interaction with external systems.

Common examples:

- travel assistants that call booking APIs
- financial assistants performing calculations
- coding assistants executing code tools

Tool usage increases system capability but also introduces additional complexity in orchestration and error handling.

---

## 4.4 Workflow Systems

Workflow systems introduce explicit orchestration logic that controls how tasks are executed.

Instead of allowing the model to decide every step dynamically, workflows define structured execution paths.

Example workflow:

```text
User Request
↓
Intent Classification
↓
Retrieve Documents
↓
Generate Answer
↓
Post-processing
```

Advantages of workflow-based architectures:

- improved reliability
- deterministic execution
- easier debugging
- clearer system observability

These systems are commonly used in production environments where predictability and control are critical.

---

## 4.5 Agent Systems

Agent systems allow models to dynamically plan and execute multi-step tasks.

Instead of following predefined workflows, the system allows the model to determine which actions to perform.

Typical loop:

```text
Goal
↓
Reason
↓
Select Tool
↓
Execute Action
↓
Observe Result
↓
Repeat
```

This enables systems capable of:

- complex reasoning
- multi-step planning
- iterative problem solving

However, agent systems introduce new engineering challenges:

- unpredictable execution paths
- higher latency
- more complex monitoring
- safety considerations

As a result, agents are often used in controlled environments or specialized tasks.

---

## 4.6 Hybrid Systems

In practice, most real-world AI applications combine multiple architectural patterns.

For example:

- a **RAG system** may also use **tool invocation**
- a **workflow system** may integrate **retrieval pipelines**
- an **agent system** may include structured workflow steps

Hybrid architectures combine the strengths of multiple approaches.

Example hybrid architecture:

```text
User Query
↓
Workflow Orchestrator
↓
Retrieval System
↓
LLM Reasoning
↓
Tool Execution
↓
Response
```

Hybrid systems are the norm in production environments. Pure architectural patterns are primarily conceptual models used to explain system design.

---

## 4.7 Decision Heuristics

Selecting the appropriate architecture depends on system requirements.

A useful rule of thumb is to **start with the simplest possible architecture and increase complexity only when necessary**.

Typical progression:

```text
Prompt
↓
RAG
↓
Tool-Augmented
↓
Workflow
↓
Agent
```

As system complexity increases, so do the challenges in:

- system reliability
- latency
- observability
- operational cost

Careful architectural decisions are therefore critical when building production AI systems.

---

## 4.8 Trade-offs Between System Types

Different architectures offer different trade-offs.

| System Type    | Strength                      | Limitation                        |
| -------------- | ----------------------------- | --------------------------------- |
| Prompt-Based   | Simple and fast to build      | Limited knowledge                 |
| RAG            | Access to external knowledge  | Requires retrieval infrastructure |
| Tool-Augmented | Interaction with real systems | Orchestration complexity          |
| Workflow       | Reliable and controllable     | Reduced flexibility               |
| Agents         | Highly flexible reasoning     | Harder to control and monitor     |

Understanding these trade-offs helps architects choose the appropriate design for a given problem.

---

## 📋 Chapter Summary

- AI systems can be classified by how they **access knowledge** and how much **autonomy** they exercise during execution.
- Prompt-based systems rely entirely on model knowledge and are suitable for narrow tasks.
- Retrieval-Augmented Generation (RAG) systems incorporate external knowledge through retrieval pipelines.
- Tool-augmented systems enable models to interact with external APIs and deterministic services.
- Workflow systems provide structured orchestration that improves reliability and observability.
- Agent systems allow dynamic reasoning and planning but introduce operational complexity.
- Most real-world AI applications are **hybrid systems** that combine multiple architectural patterns.

---

## ❓ Comprehension Questions

1. Why is classifying AI systems by knowledge access and execution autonomy useful for system architecture design?

2. What are the main differences between prompt-based systems and retrieval-augmented generation systems?

3. In which situations would a workflow-based architecture be preferable to an agent-based system?

4. What engineering challenges arise when introducing tool-augmented systems?

5. Why are hybrid architectures common in production AI systems?

---

## References

### Papers

- Lewis et al. — _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_ (2020)
  [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

### Books

- Chip Huyen — _Designing Machine Learning Systems_ (O'Reilly, 2022)

### Documentation

- OpenAI Platform Documentation
  [https://platform.openai.com/docs](https://platform.openai.com/docs)

- LangChain Documentation
  [https://python.langchain.com](https://python.langchain.com)

---

## See Also

Related chapters:

- Chapter 3 — Machine Learning vs LLM
- Chapter 5 — Tokens and Context
- Chapter 6 — Prompt Engineering

---

## Key Takeaways

- AI systems differ primarily in **how they access knowledge and how much autonomy they exercise**.
- Architectural complexity increases from prompt-based systems to agent systems.
- Retrieval and tool integration enable LLMs to interact with external knowledge and services.
- Workflow architectures provide reliability and operational control in production systems.
- Most real-world deployments use **hybrid architectures combining multiple system patterns**.
