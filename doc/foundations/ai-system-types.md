# AI System Types

[⬅ Back to Foundations](index.md)

---

## Context

AI systems can be organized into different architectural types depending on how models are used, how information flows through the system, and how decisions are orchestrated.

Understanding these system types is essential for **AI Systems Engineering** because most real-world applications are not simply model calls. Instead, they are **composed systems** built from multiple components such as prompts, retrieval pipelines, orchestration workflows, and external tools.

This chapter introduces the primary architectural categories used in modern AI systems and explains how they differ in complexity, capability, and engineering requirements.

These system types primarily occupy the **Application**, **Orchestration**, **Prompt**, and **Retrieval** layers of the **AI Systems Reference Stack** introduced in the previous chapter.

The goal of this chapter is to provide a mental model for understanding how AI systems evolve from simple prompt-driven applications to more complex architectures involving retrieval, workflows, and autonomous agents.

---

AI applications may appear similar from the user interface, but they can differ significantly in internal architecture. Some systems rely on a single prompt and model call, while others combine retrieval, orchestration, tools, and iterative reasoning loops.

From an engineering perspective, these differences matter because system type affects:

- architecture complexity
- reliability and controllability
- latency and cost
- evaluation strategy
- operational risk

Choosing the right system type is therefore a core architectural decision in AI Systems Engineering.

---

## Concept Overview

An **AI system architecture** defines how models, data sources, tools, and orchestration logic interact to produce system behavior.

While many AI applications appear simple from the outside, their internal structure can vary significantly depending on how the model interacts with external information, tools, and system logic.

Modern AI applications can be grouped into several architectural system types based on how the model interacts with context, data sources, and tools.

Before examining each system type in detail, it is useful to understand the general progression of AI system architectures.

```id="5f2f8a"
Model Only
↓
Model + External Knowledge
↓
Model + Orchestration Pipelines
↓
Model + Autonomous Decision Making
```

This progression reflects how AI systems evolve as additional components are introduced around the model.

```id="dmo7q2"
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

These categories represent **increasing architectural complexity, capability, and system autonomy**.

| System Type         | Key Capability           | Typical Components          |
| ------------------- | ------------------------ | --------------------------- |
| Prompt-Based        | Direct model interaction | Prompt + model              |
| Retrieval-Augmented | External knowledge       | Retriever + vector database |
| Workflow Systems    | Structured orchestration | Pipelines + tool execution  |
| Agent Systems       | Autonomous reasoning     | Planning + tools + memory   |

Each system type builds upon the capabilities of the previous one. As additional components are introduced—such as retrieval pipelines, orchestrators, and tools—the system becomes more capable but also more complex to design and operate.

**Key Concept — AI Systems Are Architectural Patterns**

AI applications are not defined solely by the model they use, but by the **architecture that surrounds the model**.

Prompts, retrieval systems, orchestration pipelines, tools, and memory components collectively determine how the system behaves.

For this reason, AI applications should be understood primarily as **system architectures rather than isolated model calls**.

---

## 4.1 Prompt-Based Systems

Prompt-based systems are the simplest type of AI application.

In these systems, the application constructs a prompt and sends it directly to a language model. The model generates a response based solely on the provided prompt and its internal knowledge.

Example architecture:

```id="1lp9i2"
User Input
↓
Prompt Template
↓
Model
↓
Response
```

Typical characteristics include:

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

**Key Concept — Prompt Systems Rely on Model Knowledge**

Prompt-based systems depend almost entirely on the knowledge embedded in the model's training data.

Without external information sources, these systems cannot reliably access private data, recent knowledge, or domain-specific information.

---

## 4.2 Retrieval-Augmented Systems (RAG)

Because prompt-based systems are limited by model-internal knowledge, many production systems introduce retrieval as the next architectural step.

Retrieval-Augmented Generation (RAG) systems combine language models with external knowledge sources.

Instead of relying solely on the model's training data, the system retrieves relevant documents and injects them into the prompt as context.

This allows the model to reason over information that is not stored inside its parameters.

Example architecture:

```id="wg0fl1"
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

Key components include:

- **embedding model** for vector representation
- **vector database** for similarity search
- **retriever** for document selection
- **context builder** for prompt construction

Advantages of RAG systems include:

- access to domain-specific knowledge
- ability to use private data
- improved factual accuracy
- reduced hallucination risk

RAG has become one of the most common architectural patterns for production AI systems.

However, introducing retrieval also creates new engineering challenges:

- retrieval quality
- context window management
- latency from additional retrieval steps
- complexity in evaluation

These systems are discussed in depth in the **RAG Engineering** section of the book.

**Key Concept — Retrieval Extends Model Knowledge**

Retrieval-Augmented Generation allows models to reason over external information rather than relying solely on training data.

By retrieving relevant documents and injecting them into prompt context, the system can produce more accurate, current, and domain-specific responses.

---

## 4.3 Workflow-Based Systems

As systems take on more complex tasks, retrieval alone is often not sufficient. Many applications require multiple steps, intermediate decisions, and interactions with external tools.

Workflow systems extend simpler AI architectures by introducing **structured orchestration pipelines**.

Instead of performing a single model call, the system coordinates multiple steps that may include:

- multiple model calls
- retrieval operations
- tool execution
- data transformations
- validation steps

Example architecture:

```id="qcaqm2"
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

Advantages include:

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

**Key Concept — Workflows Add Deterministic Control**

Workflow architectures introduce deterministic control structures around probabilistic models.

By defining explicit execution steps, engineers can constrain system behavior, improve reliability, and make multi-step systems easier to evaluate and operate.

---

## 4.4 Agent Systems

When tasks require more flexible decision-making, systems may move beyond predefined workflows toward agent architectures.

Agent systems represent the most advanced category of AI architectures.

Agents are systems capable of:

- planning tasks
- selecting tools
- iteratively reasoning
- adapting actions based on intermediate results

Unlike workflow systems, where execution steps are predetermined, agents dynamically decide what actions to take during execution.

Example architecture:

```id="b4euhr"
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

However, this flexibility introduces additional challenges:

- unpredictability of system behavior
- evaluation difficulty
- safety risks
- higher cost and latency

Because of these challenges, many production systems combine **agent capabilities with structured workflows** rather than relying on fully autonomous agents.

Workflow systems follow **predefined execution pipelines designed by engineers**, while agent systems dynamically decide which actions to perform during execution.

This distinction has important implications for reliability, predictability, and system evaluation.

**Key Concept — Agents Introduce Autonomous Decision-Making**

Agent architectures allow models to dynamically decide which actions to perform during execution.

Instead of following a fixed pipeline, the system iteratively reasons, selects tools, performs actions, and evaluates results based on intermediate observations.

---

## 4.5 Increasing System Complexity

These system types can be viewed as a progression in architectural sophistication.

```id="pl4g78"
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

Selecting the appropriate system type therefore depends on the **requirements and constraints of the application**.

---

## 4.6 Hybrid Architectures

In practice, real production systems rarely fit cleanly into a single category.

Most real-world AI systems combine multiple architectural patterns.

Production systems often integrate elements from several system types to balance capability and operational reliability.

Example hybrid architecture:

```id="mdiul6"
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

**Key Concept — Production Systems Are Hybrid**

Most production AI systems combine multiple architectural patterns.

Retrieval, workflows, tools, and agent-style reasoning are often integrated into hybrid architectures designed to balance flexibility, reliability, and operational cost.

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

### Papers

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al., 2020
[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

ReAct: Synergizing Reasoning and Acting in Language Models — Yao et al., 2022
[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

Reflexion: Language Agents with Verbal Reinforcement Learning — Shinn et al., 2023
[https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)

### Community References

OpenAI Platform Documentation
[https://platform.openai.com/docs](https://platform.openai.com/docs)

---

## Key Takeaways

- AI applications can be organized into several architectural system types based on how models interact with context, tools, and orchestration logic.
- **Prompt-based systems** represent the simplest architecture and rely entirely on the model’s internal knowledge.
- **Retrieval-Augmented Generation (RAG)** systems extend model capabilities by incorporating external knowledge sources.
- **Workflow systems** introduce structured orchestration pipelines that coordinate multiple model calls, retrieval operations, and tool executions.
- **Agent systems** enable dynamic reasoning loops where models decide which actions to perform during execution.
- As AI systems evolve from prompts to agents, they gain capabilities but also increase in **architectural complexity, latency, and operational cost**.
- Most real-world AI applications use **hybrid architectures** that combine retrieval, workflows, and tool usage.
- In practice, AI applications should be understood as **system architectures built around models**, rather than as isolated model invocations.
