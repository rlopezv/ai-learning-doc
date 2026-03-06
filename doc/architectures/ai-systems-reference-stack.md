# AI Systems Reference Stack

## Context

This chapter introduces a reference stack for reasoning about AI systems as layered engineering systems instead of isolated model calls. Many chapters in the book describe individual capabilities such as prompts, retrieval, model access, evaluation, or infrastructure. This chapter provides the unifying architectural frame that connects them.

## Concept

An AI system can be decomposed into layers, each with a distinct responsibility. This layered view helps engineers reason about:

- separation of concerns
- ownership boundaries
- scaling strategies
- failure isolation
- where a design decision belongs

The purpose of a reference stack is not to prescribe a single implementation, but to make architectural conversations more precise.

## The Stack

### 1. Interaction Layer

This is where users or upstream systems interact with the AI system.

Typical components:

- chat interfaces
- APIs
- CLI tools
- embedded copilots
- workflow triggers

This layer should remain thin. It handles presentation and request entry, not core AI logic.

### 2. Application Layer

The application layer contains product logic and business behavior. It defines what the system is supposed to do from a product perspective.

Typical responsibilities:

- session handling
- authorization decisions
- routing based on business rules
- user-specific behavior
- escalation and fallback logic

### 3. Orchestration Layer

The orchestration layer coordinates model calls, retrieval, tools, memory, and multi-step flows.

Typical responsibilities:

- prompt assembly
- retrieval coordination
- workflow state transitions
- tool selection and execution
- response composition

In many production systems, this is the most architecture-sensitive layer.

### 4. Prompt Layer

The prompt layer contains the instructions and templates that shape model behavior.

Typical responsibilities:

- system prompts
- task prompts
- response schemas
- few-shot examples
- prompt versioning

This layer is often underestimated. In practice, prompt artifacts behave like configuration and must be managed with the same rigor.

### 5. Retrieval Layer

The retrieval layer provides external knowledge to the model.

Typical responsibilities:

- query transformation
- embedding search
- metadata filtering
- reranking
- context assembly

This layer is central to RAG systems, but absent in pure prompt-only systems.

### 6. Model Layer

The model layer provides inference capabilities.

Typical components:

- LLM APIs
- local model runtimes
- embedding models
- rerankers
- moderation or classifier models

This layer should usually be accessed through abstraction points rather than embedded directly into application logic.

### 7. Data Layer

The data layer contains the artifacts and stores that the AI system depends on.

Typical components:

- document stores
- vector databases
- relational data
- prompt registries
- evaluation datasets
- model metadata

This layer is broader than a traditional database layer because it includes all persistent AI artifacts.

### 8. Infrastructure Layer

The infrastructure layer supports the runtime environment.

Typical components:

- containers
- compute nodes
- GPUs
- queues
- object storage
- networking
- observability backends

This layer determines the operational limits of the system, but should not leak directly into product logic.

## Why the Stack Matters

The reference stack is useful because it helps answer questions such as:

- Is this problem about prompts, orchestration, or retrieval?
- Which team should own this capability?
- Which layers change when the embedding model changes?
- Where should observability be instrumented?
- Which layers are affected when moving from prototype to production?

Without a layered model, AI systems often become a collection of tightly coupled scripts around a model API.

## Design Considerations

The stack also reveals important trade-offs:

| Layer | Typical design tension |
|---|---|
| Interaction | UX simplicity vs capability exposure |
| Application | business control vs orchestration complexity |
| Orchestration | flexibility vs debuggability |
| Prompt | expressiveness vs maintainability |
| Retrieval | grounding quality vs latency |
| Model | quality vs cost |
| Data | freshness vs governance |
| Infrastructure | scale vs operational complexity |

Not every system needs every layer equally. A prompt-only classifier may have little or no retrieval layer. A large enterprise RAG platform may use all layers extensively.

## Related Sections

- [Introduction to AI Systems Engineering](../foundations/introduction-to-ai-systems-engineering.md)
- [Architectural Patterns](architectural-patterns.md)
- [Retrieval-Augmented Generation](retrieval-augmented-generation.md)
- [The Internal AI Platform](../platform-engineering/the-internal-ai-platform.md)
- [Observability Strategy for LLM Systems](../observability/index.md)

## Key Takeaways

- The reference stack gives a shared architectural language for AI systems.
- It helps separate concerns across interaction, application, orchestration, prompts, retrieval, models, data, and infrastructure.
- It is a reasoning tool, not a rigid implementation prescription.
- The stack makes later chapters in the book easier to connect and compare.
