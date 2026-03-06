# Chapter 4 — AI System Types

[⬅ Back to Foundations](index.md)

## Context

AI systems can be organized into different architectural types depending on how models are used, how information flows through the system, and how decisions are orchestrated.

Understanding these system types is essential for **AI Systems Engineering** because most real-world applications are not simply model calls. Instead, they are **composed systems** built from multiple components such as prompts, retrieval pipelines, orchestration workflows, and external tools.

This chapter introduces the primary architectural categories used in modern AI systems and explains how they differ in complexity, capability, and engineering requirements.

These system types primarily occupy the **Application**, **Orchestration**, **Prompt**, and **Retrieval** layers of the **AI Systems Reference Stack** introduced in the previous chapter.

The goal of this chapter is to provide a mental model for understanding how AI systems evolve from simple prompt-driven applications to more complex architectures involving retrieval, workflows, and autonomous agents.

---

## Concept Overview

An **AI system architecture** defines how models, data sources, tools, and orchestration logic interact to produce system behavior.

Modern AI applications can be grouped into several architectural system types based on how the model interacts with context, data sources, and tools.

```

AI System Types
│
├ Prompt-Based Systems
│
├ Retrieval-Augmented Systems (RAG)
│
├ Workflow-Based Systems
│
└ Agent Systems

```

These categories represent **increasing architectural complexity and capability**.

| System Type         | Key Capability           | Typical Components          |
| ------------------- | ------------------------ | --------------------------- |
| Prompt-Based        | Direct model interaction | Prompt + model              |
| Retrieval-Augmented | External knowledge       | Retriever + vector database |
| Workflow Systems    | Structured orchestration | Pipelines + tool execution  |
| Agent Systems       | Autonomous reasoning     | Planning + tools + memory   |

Each system type builds upon the capabilities of the previous one.

---

## 4.1 Prompt-Based Systems

Prompt-based systems are the simplest type of AI application.

In these systems, the application constructs a prompt and sends it directly to a language model. The model generates a response based solely on the provided prompt and its internal knowledge.

Example architecture:

```

User Input
↓
Prompt Template
↓
Model
↓
Response

```

Typical characteristics:

- minimal system complexity
- no external knowledge sources
- fast implementation
- limited controllability

Examples include:

- simple chatbots
- text summarization tools
- code generation assistants
- basic content generation applications

Although prompt-based systems are easy to build, they have important limitations:

- knowledge limited to model training data
- hallucination risk
- difficulty enforcing consistent structure
- lack of access to private or up-to-date information

Because of these limitations, most production systems extend this architecture using **retrieval pipelines**.

---

## 4.2 Retrieval-Augmented Systems (RAG)

Retrieval-Augmented Generation (RAG) systems combine language models with external knowledge sources.

Instead of relying solely on the model's training data, the system retrieves relevant documents and injects them into the prompt as context.

Example architecture:

```

User Query
↓
Retriever
↓
Vector Database
↓
Relevant Documents
↓
Prompt + Context
↓
Model
↓
Response

```

Key components:

- **embedding model** for vector representation
- **vector database** for similarity search
- **retriever** for document selection
- **context builder** for prompt construction

Advantages of RAG systems:

- access to domain-specific knowledge
- ability to use private data
- improved factual accuracy
- reduced hallucination risk

RAG has become one of the most common architectural patterns for production AI systems.

However, RAG systems also introduce new engineering challenges:

- retrieval quality
- context window management
- latency from additional retrieval steps
- complexity in evaluation

These systems are discussed in depth in the **RAG Engineering** section of the book.

---

## 4.3 Workflow-Based Systems

Workflow systems extend RAG architectures by introducing **structured orchestration pipelines**.

Instead of performing a single model call, the system coordinates multiple steps that may include:

- multiple model calls
- retrieval operations
- tool execution
- data transformations
- validation steps

Example architecture:

```

User Query
↓
Orchestrator
↓
Task Decomposition
↓
Multiple Processing Steps
↓
Tools / APIs
↓
Model
↓
Final Response

```

Many workflow systems allow models to interact with **external tools**, such as:

- APIs
- databases
- search systems
- calculation engines

Advantages:

- greater control over system behavior
- ability to combine multiple tools
- improved reliability
- structured reasoning pipelines

Typical use cases include:

- document processing pipelines
- research assistants
- multi-step analysis systems
- automated report generation

Workflow systems allow engineers to design **predictable pipelines around probabilistic models**.

Because these systems involve multiple components, **observability and tracing** become important engineering capabilities. Monitoring prompt inputs, tool calls, and intermediate outputs helps engineers debug and evaluate system behavior.

---

## 4.4 Agent Systems

Agent systems represent the most advanced category of AI architectures.

Agents are systems capable of:

- planning tasks
- selecting tools
- iteratively reasoning
- adapting actions based on intermediate results

Unlike workflow systems, where execution steps are predetermined, agents dynamically decide what actions to take.

Example architecture:

```

User Goal
↓
Agent
↓
Reasoning
↓
Tool Selection
↓
Action
↓
Observation
↓
Iteration
↓
Final Result

```

Typical agent components include:

- reasoning loop
- tool interfaces
- memory systems
- planning logic
- execution control

Memory systems may include:

- short-term conversational memory
- vector-based long-term memory
- external knowledge storage

Agent systems enable more autonomous behavior, making them useful for tasks such as:

- research automation
- complex task planning
- coding assistants
- multi-step data analysis

However, they also introduce additional challenges:

- unpredictability
- evaluation difficulty
- safety risks
- cost and latency

Because of these challenges, many production systems combine **agent capabilities with structured workflows**.

---

## 4.5 Increasing System Complexity

These system types can be viewed as a progression in architectural sophistication.

```

Prompt Systems
↓
RAG Systems
↓
Workflow Systems
↓
Agent Systems

```

Each stage introduces additional capabilities:

| Property           | Prompt | RAG     | Workflow | Agent     |
| ------------------ | ------ | ------- | -------- | --------- |
| External knowledge | ❌     | ✅      | ✅       | ✅        |
| Tool usage         | ❌     | Limited | ✅       | ✅        |
| Planning           | ❌     | ❌      | Limited  | ✅        |
| Autonomy           | ❌     | ❌      | ❌       | High      |
| System complexity  | Low    | Medium  | High     | Very High |

However, increased capability also introduces trade-offs:

- higher system complexity
- increased latency
- greater operational cost
- more challenging evaluation

Each additional component—retrieval pipelines, orchestration logic, or tool execution—adds processing overhead and increases system latency.

Selecting the appropriate system type depends on the **requirements and constraints of the application**.

---

## 4.6 Hybrid Architectures

Most real-world AI systems combine multiple architectural patterns.

Example hybrid architecture:

```

User Query
↓
Application Layer
↓
Workflow Orchestrator
↓
Retriever
↓
Vector Database
↓
Context Builder
↓
Model
↓
Tool Calls
↓
Response

```

This architecture may include elements of:

- retrieval systems
- structured workflows
- tool usage
- agent-style reasoning

Hybrid architectures allow engineers to balance:

- system reliability
- reasoning flexibility
- cost efficiency
- operational complexity

Designing these systems requires careful consideration of **latency, cost, reliability, and evaluation strategies**.

---

## 4.7 Choosing the Right System Type

Selecting the appropriate system architecture depends on the complexity of the task and the operational requirements of the application.

Prompt-based systems are often sufficient when:

- tasks are simple
- knowledge requirements are limited
- latency must be minimal

RAG systems are appropriate when:

- applications require domain knowledge
- systems must access private or frequently updated data
- factual accuracy is important

Workflow systems are useful when:

- tasks involve multiple steps
- systems must interact with external tools
- greater control over execution is required

Agent systems are most appropriate when:

- problems require dynamic planning
- tasks involve complex decision-making
- systems must adapt based on intermediate results

In practice, engineers often begin with simpler architectures and progressively introduce additional capabilities as system requirements grow.

---

## Chapter Summary

- AI applications can be categorized into several system types based on how models are integrated into system architecture.
- **Prompt-based systems** are the simplest architectures, relying on direct interaction with language models.
- **Retrieval-Augmented Generation (RAG)** systems incorporate external knowledge sources to improve accuracy.
- **Workflow systems** orchestrate multiple steps and tools to perform complex tasks.
- **Agent systems** introduce autonomous reasoning and dynamic task planning.
- As systems grow in capability, complexity, latency, and operational cost also increase.
- Many production AI systems combine multiple architectural patterns into **hybrid architectures**.

---

## Comprehension Questions

1. What distinguishes prompt-based systems from retrieval-augmented systems?
2. Why do RAG systems improve factual accuracy compared to prompt-only systems?
3. How do workflow systems improve reliability in AI applications?
4. What capabilities differentiate agent systems from workflow-based architectures?
5. What trade-offs arise as AI system architectures become more complex?
6. Why are hybrid architectures common in production AI systems?

---

## References

- Lewis et al. _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_.
  https://arxiv.org/abs/2005.11401

- Yao et al. _ReAct: Synergizing Reasoning and Acting in Language Models_.
  https://arxiv.org/abs/2210.03629

- Shinn et al. _Reflexion: Language Agents with Verbal Reinforcement Learning_.
  https://arxiv.org/abs/2303.11366

- OpenAI Documentation
  https://platform.openai.com/docs
