# Architectural Patterns & Landscape

[⬅ Back to Architectures](index.md)

---

## Context

The previous chapters introduced the primary architectural approaches used to build modern AI systems:

- prompt-based systems
- retrieval-augmented systems
- tool-augmented systems
- workflow systems
- agent systems

Each architecture introduces different capabilities and engineering trade-offs.

However, production AI platforms rarely implement a single architecture in isolation. Instead, real-world systems combine multiple components such as prompts, retrieval pipelines, tools, orchestrators, models, and data platforms.

As a result, modern AI applications are best understood as **composed architectures** built from reusable system patterns.

At the same time, a rapidly growing ecosystem of frameworks, platforms, and infrastructure tools has emerged to support the development and operation of these systems.

This chapter provides three complementary perspectives:

1. **Architectural patterns** used to compose AI systems
2. **The AI systems landscape**, including major categories of platforms and frameworks
3. **Architectural decision frameworks** that help engineers design production systems

Together these perspectives help engineers understand how architectural concepts map to real-world platforms and system implementations.

---

## Concept Overview

Modern AI applications combine multiple architectural elements into layered systems.

A simplified conceptual pipeline illustrates how these components interact:

```id="architecture-composition"
User Request
↓
Application Layer
↓
Orchestration Layer
↓
Prompt Construction
↓
Retrieval
↓
Model Reasoning
↓
Tool Execution
↓
Response
```

Different architectural patterns emphasize different stages of this pipeline depending on system requirements.

For example:

- prompt-based systems emphasize **prompt construction**
- retrieval systems emphasize **knowledge retrieval**
- tool systems emphasize **external system interaction**
- workflow systems emphasize **structured orchestration**
- agent systems emphasize **dynamic reasoning loops**

Understanding these patterns helps engineers design systems that balance capability, complexity, latency, and operational cost.

---

# AI Systems Landscape

The ecosystem of tools supporting AI systems can be organized according to the **AI Systems Reference Stack**.

Each category of tools supports one or more layers of the stack.

The landscape therefore includes platforms for:

- models
- data and knowledge systems
- prompt engineering
- retrieval pipelines
- tool integration
- workflow orchestration
- agent frameworks

This layered ecosystem reflects how modern AI systems are engineered in practice.

---

## 1. Model Platforms

Model platforms provide the foundational models used in AI systems.

These platforms offer APIs or infrastructure for running **foundation models**, including language models, embedding models, and multimodal models.

Typical capabilities include:

- hosted model inference
- model deployment and scaling
- model versioning
- embedding generation
- multimodal inference

Examples include:

- OpenAI
- Anthropic
- Google Gemini
- Cohere
- Mistral
- Hugging Face
- vLLM
- Ollama

These platforms primarily operate in the **Model Layer** of the AI Systems Reference Stack.

The choice of model platform strongly influences system performance, latency, cost, and capability.

---

## 2. Data and Knowledge Platforms

AI systems rely heavily on data infrastructure that stores and manages knowledge used during system execution.

These platforms support:

- document storage
- knowledge repositories
- vector databases
- knowledge graphs
- structured data warehouses

Typical technologies include:

- Pinecone
- Weaviate
- Vespa
- Elasticsearch
- Neo4j
- Snowflake
- Databricks
- BigQuery

These platforms operate primarily within the **Data Layer** and **Retrieval Layer** of the architecture.

They provide the foundation for knowledge-intensive AI systems.

---

## 3. Prompt Engineering Platforms

Prompt engineering platforms provide tools for constructing, managing, and evaluating prompts used in language model interactions.

Typical capabilities include:

- prompt templates
- structured prompt composition
- parameterized prompts
- prompt versioning
- prompt testing and evaluation

Example tools and frameworks include:

- LangChain prompt templates
- DSPy prompt programming
- Guidance
- OpenAI Assistants prompt management

These tools primarily operate within the **Prompt Layer** and **Application Layer** of the AI Systems Reference Stack.

Prompt platforms are commonly used in systems where model behavior is largely controlled through prompt design.

---

## 4. Retrieval Engineering Systems

Retrieval engineering systems support architectures that incorporate external knowledge sources.

Typical components include:

- embedding models
- vector databases
- retrieval pipelines
- document chunking systems
- context assembly pipelines

Common technologies include:

- LlamaIndex
- Haystack
- LangChain retrieval modules

These systems operate primarily within the **Retrieval Layer** of the AI architecture.

Retrieval engineering platforms enable systems to access domain-specific knowledge while maintaining scalability and performance.

---

## 5. Tool Integration Frameworks

Tool integration frameworks allow language models to interact with external systems.

Typical tools include:

- APIs
- databases
- search systems
- computation engines
- code execution environments

Example frameworks include:

- OpenAI function calling
- LangChain tools
- Semantic Kernel skills

These frameworks enable models to perform operations beyond text generation by delegating tasks to specialized systems.

Tool frameworks primarily operate within the **Orchestration Layer** and the **Application Layer**.

---

## 6. Workflow Orchestration Platforms

Workflow orchestration platforms coordinate multi-step AI pipelines.

These platforms manage:

- execution order
- intermediate state
- retries and error handling
- task dependencies
- distributed execution

Examples include:

- LangGraph
- Temporal
- Prefect
- Airflow
- Dagster

Workflow orchestration systems play a central role in the **Orchestration Layer** of AI system architectures.

They enable engineers to design reliable pipelines that combine retrieval systems, model calls, and tool interactions.

---

## 7. Agent Frameworks

Agent frameworks enable systems that perform dynamic reasoning and action selection.

These frameworks implement components such as:

- reasoning loops
- tool invocation systems
- memory management
- planning mechanisms

Examples include:

- LangChain agents
- AutoGen
- CrewAI
- MetaGPT

Agent frameworks typically operate across several layers of the architecture, including:

- Orchestration Layer
- Prompt Layer
- Model Layer

These frameworks support systems that require flexible decision-making and exploration of complex task spaces.

---

## 8. Mapping Architectures to the AI Systems Reference Stack

The architectural patterns introduced in previous chapters map to different layers of the **AI Systems Reference Stack**.

| Architecture Type           | Interaction | Application | Orchestration | Prompt | Retrieval | Model | Data | Infrastructure |
| --------------------------- | ----------- | ----------- | ------------- | ------ | --------- | ----- | ---- | -------------- |
| Prompt-Based Systems        | ✓           | ✓           | –             | ✓      | –         | ✓     | –    | ✓              |
| Retrieval-Augmented Systems | ✓           | ✓           | –             | ✓      | ✓         | ✓     | ✓    | ✓              |
| Tool-Augmented Systems      | ✓           | ✓           | ✓             | ✓      | –         | ✓     | ✓    | ✓              |
| Workflow Systems            | ✓           | ✓           | ✓             | ✓      | ✓         | ✓     | ✓    | ✓              |
| Agent Systems               | ✓           | ✓           | ✓             | ✓      | ✓         | ✓     | ✓    | ✓              |

This mapping illustrates how architectural complexity increases as additional system layers become involved.

For example:

- prompt systems rely primarily on **prompt and model layers**
- retrieval systems introduce the **retrieval and data layers**
- workflow and agent systems require **orchestration and tool integration**

Understanding this relationship helps engineers reason about system design and operational complexity.

---

## 9. Hybrid Architectures in Production

Most real-world AI systems combine several architectural patterns.

A typical production architecture may look like this:

```id="hybrid-architecture"
User Query
↓
Application Service
↓
Workflow Orchestrator
↓
Retriever
↓
Vector Database
↓
Context Builder
↓
LLM
↓
Tool Execution
↓
Response
```

In this architecture:

- retrieval pipelines provide contextual knowledge
- workflow orchestration coordinates system behavior
- tools provide access to external systems

Hybrid architectures allow engineers to balance:

- system reliability
- reasoning flexibility
- latency
- operational cost

---

## 10. Evolution of AI System Architectures

AI systems often evolve through several architectural stages.

A typical progression looks like this:

```id="architecture-evolution"
Prompt-Based System
↓
Retrieval-Augmented System
↓
Tool-Augmented System
↓
Workflow-Orchestrated System
↓
Agent-Based System
```

Organizations frequently begin with simple prompt-based prototypes and progressively introduce additional architectural components as system requirements increase.

Understanding this evolution helps teams design systems that can scale over time.

---

## 11. Choosing the Right Architecture

Selecting the appropriate architecture depends on the nature of the task and the operational requirements of the system.

Prompt-based systems are often sufficient when:

- tasks are simple
- knowledge requirements are limited
- low latency is required

Retrieval-augmented systems are appropriate when:

- systems must access domain knowledge
- responses must reference external documents
- hallucination risk must be reduced

Tool-augmented systems are useful when:

- tasks require interaction with external systems
- the model must perform calculations or data queries
- real-time information is required

Workflow architectures are appropriate when:

- tasks involve multi-step processing
- reliability and predictability are important
- execution must be structured and auditable

Agent architectures are most useful when:

- tasks require dynamic reasoning
- execution paths cannot be predetermined
- systems must adapt to intermediate results

In practice, engineers typically start with the simplest architecture that satisfies system requirements and introduce additional components only when necessary.

---

## 12. Architecture Decision Drivers

Architectural decisions in AI systems are influenced by several key factors.

| Decision Driver          | Architectural Impact                                        |
| ------------------------ | ----------------------------------------------------------- |
| Task complexity          | Determines whether prompts, workflows, or agents are needed |
| Knowledge requirements   | Determines whether retrieval systems are required           |
| Latency constraints      | Limits multi-step pipelines or agent loops                  |
| Cost constraints         | Influences number of model calls and system components      |
| Reliability requirements | Favors structured workflows over autonomous agents          |
| Data availability        | Enables retrieval-based architectures                       |

Understanding these drivers helps engineers evaluate trade-offs between system capability and operational complexity.

---

## 13. The C4 Model Applied to LLM Systems

Architectural documentation for AI systems can benefit from established software architecture modeling techniques.

One useful framework is the **C4 Model**, which describes systems across four levels of abstraction:

- **Context** — the system and its external interactions
- **Container** — the major system services and infrastructure components
- **Component** — internal modules within services
- **Code** — implementation details

Applied to an AI system, these levels may look like the following.

### Context Level

```id="c4-context"
User
↓
AI Application
↓
External APIs
↓
Enterprise Data Sources
```

### Container Level

```id="c4-container"
Web Application
API Service
Workflow Engine
Vector Database
LLM Service
Monitoring Platform
```

### Component Level

```id="c4-component"
Retriever
Prompt Builder
Context Assembler
Tool Interface
Response Formatter
```

Using the C4 model helps engineering teams document AI systems in a structured way and improves communication between architects, developers, and platform engineers.

---

## 14. Reference Production Architecture

A typical production AI system combines multiple architectural components across several layers of the AI Systems Reference Stack.

An example production architecture may look like this:

```id="reference-production-architecture"
User
↓
API Gateway
↓
Application Service
↓
Workflow Engine
↓
Retriever
↓
Vector Database
↓
Context Builder
↓
LLM
↓
Tool Execution
↓
Observability and Monitoring
↓
Response
```

In this architecture:

- the **API gateway** manages request routing and authentication
- the **application service** handles user interaction and request management
- the **workflow engine** orchestrates multi-step system execution
- the **retriever and vector database** provide domain knowledge
- the **LLM** performs reasoning and response generation
- **tool execution systems** integrate external APIs and services
- **observability platforms** monitor latency, cost, and system reliability

This architecture illustrates how modern AI systems combine multiple architectural patterns into a single integrated platform.

---

## Chapter Summary

- Modern AI applications are composed from multiple architectural patterns.
- These patterns include prompt systems, retrieval systems, tool systems, workflows, and agents.
- The AI ecosystem includes platforms for models, data infrastructure, prompts, retrieval systems, tools, workflows, and agents.
- Architectural decisions depend on factors such as task complexity, latency, cost, and reliability.
- The C4 model can be used to document AI system architectures at different levels of abstraction.
- Most production AI systems use hybrid architectures combining several patterns.

---

## Comprehension Questions

1. Why do production AI systems rarely rely on a single architectural pattern?
2. What categories of platforms appear in the AI systems landscape?
3. How do model platforms and data platforms support AI architectures?
4. What factors influence architectural decisions in AI systems?
5. How can the C4 model be applied to documenting LLM-based systems?

---

## References

### Papers

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al., 2020
[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

ReAct: Synergizing Reasoning and Acting in Language Models — Yao et al., 2022
[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

Toolformer: Language Models Can Teach Themselves to Use Tools — Schick et al., 2023
[https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)

### Books

Designing Machine Learning Systems — Chip Huyen, 2022
[https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

---

## Key Takeaways

- Architectural patterns provide reusable structures for organizing AI systems.
- The AI ecosystem includes platforms for models, data systems, prompt engineering, retrieval pipelines, tools, workflows, and agents.
- The AI Systems Reference Stack helps map these technologies to architectural layers.
- Architectural decisions should consider latency, cost, reliability, and task complexity.
- The C4 model provides a useful framework for documenting AI system architectures.
