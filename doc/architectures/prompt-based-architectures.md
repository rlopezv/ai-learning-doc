# Prompt-Based Architectures

[⬅ Back to Architectures](index.md)

---

## Context

The simplest way to build an AI application is to interact directly with a language model using prompts. In these systems, the application constructs a prompt and sends it to the model, which generates a response.

This design pattern forms the basis of **prompt-based architectures**, the most basic category of AI system architectures.

Prompt-based systems rely primarily on **prompt design and model capabilities** rather than complex orchestration or external knowledge systems. Many early AI applications and prototypes are built using this architecture.

Although simple, prompt-based architectures remain useful for a wide range of tasks where the model’s internal knowledge is sufficient and external system integration is not required.

Understanding prompt-based architectures is important because they represent the **starting point for most AI applications**. More advanced architectures—such as Retrieval-Augmented Generation (RAG), workflow systems, and agent systems—extend this basic structure with additional capabilities.

---

## Concept Overview

A prompt-based architecture is an AI system in which the application interacts directly with a language model using prompts constructed at runtime.

A typical request follows a simple pipeline:

```id="y3h7de"
User Input
↓
Application Logic
↓
Prompt Construction
↓
LLM
↓
Response
```

The application builds a prompt that contains instructions and user input, sends it to the model, and returns the generated response to the user.

These systems are often used for tasks such as:

- text summarization
- content generation
- translation
- question answering
- coding assistance

**Key Concept — Prompt-Based Systems Rely on Model Capabilities**

In prompt-based architectures, most of the system’s intelligence comes from the language model itself. The surrounding application primarily constructs prompts and manages user interaction.

Because the architecture is simple, development is fast. However, this simplicity also introduces important limitations.

---

## 1. Architecture Structure

A minimal prompt-based architecture includes only a small number of components.

```id="tr3m9a"
User
↓
Application
↓
Prompt Template
↓
LLM API
↓
Response
```

The system typically contains:

- a user interface
- an application service
- a prompt template
- a language model API

The application constructs prompts dynamically by combining user input with predefined instructions.

Example prompt structure:

```id="j5uxye"
System Instruction:
Summarize the following document.

User Input:
{document_text}
```

In the **AI Systems Reference Stack**, this architecture primarily involves the following layers:

```id="jrbn6y"
Interaction Layer
↓
Application Layer
↓
Prompt Layer
↓
Model Layer
```

This layered view helps illustrate how responsibilities are separated in even the simplest AI systems.

---

## 2. Prompt Construction

Prompt construction is a central component of prompt-based architectures.

Prompts are often composed of multiple elements:

```id="8wty02"
System Prompt
+
Task Instructions
+
User Input
```

These elements are typically combined using **prompt templates** defined within the application.

Example template:

```id="7ypskg"
You are a helpful assistant.

Task:
Explain the following concept clearly.

Concept:
{user_input}
```

At runtime, the system replaces placeholders such as `{user_input}` with actual values.

In production systems, prompt templates often become **versioned system artifacts** that are stored, tested, and iterated over time.

Effective prompt design is therefore essential for achieving reliable outputs in prompt-based architectures.

---

## 3. Advantages of Prompt-Based Architectures

Prompt-based architectures have several advantages that make them attractive for many applications.

### Simplicity

These systems are straightforward to implement because they require few components.

### Rapid Development

Developers can build working prototypes quickly without designing complex infrastructure.

### Low Operational Complexity

Prompt-based systems do not require retrieval pipelines, vector databases, or orchestration engines.

### Flexible Task Adaptation

A single model can perform many tasks simply by changing the prompt instructions.

These characteristics make prompt-based architectures ideal for early-stage experimentation and prototyping.

---

## 4. Limitations of Prompt-Based Architectures

Despite their simplicity, prompt-based architectures have several important limitations.

### Limited Knowledge Access

The model can only use information contained in its training data or provided in the prompt.

### Hallucination Risk

Because the model generates responses probabilistically, incorrect or fabricated information may appear.

### Context Window Constraints

Only a limited amount of information can be included in the prompt due to token limits.

### Limited Control

Developers have limited control over reasoning steps or intermediate outputs.

Because of these limitations, prompt-based architectures are often extended with additional components in production systems.

---

## 5. Example Applications

Prompt-based architectures are commonly used in applications where external knowledge retrieval is not necessary.

Examples include:

- AI writing assistants
- brainstorming and ideation tools
- document summarization tools
- translation services
- grammar correction systems
- simple coding assistants

For example, a summarization application might follow this flow:

```id="o9v63g"
User uploads document
↓
Application extracts text
↓
Prompt template generates summarization prompt
↓
LLM produces summary
↓
Application returns result
```

In this scenario, the application simply transforms user input into a prompt and forwards it to the model.

---

## 6. Evolution Toward More Complex Architectures

Many production AI systems begin as prompt-based prototypes.

As system requirements grow, engineers often introduce additional components to address limitations.

Typical evolution:

```id="a0z0kt"
Prompt-Based System
↓
RAG System
↓
Workflow System
↓
Agent System
```

Each step introduces additional capabilities such as:

- external knowledge retrieval
- tool execution
- multi-step orchestration
- autonomous decision making

For example, when applications require access to **private data or domain-specific knowledge**, prompt-based architectures are typically extended with **retrieval systems**, leading to Retrieval-Augmented Generation (RAG) architectures.

**Key Concept — Prompt-Based Architectures Are the Starting Point**

Prompt-based architectures represent the simplest form of AI system architecture.

More advanced systems extend this design with additional components that provide external knowledge, structured workflows, and autonomous behavior.

---

## Chapter Summary

- Prompt-based architectures are the simplest form of AI system architecture.
- These systems interact directly with language models using prompts constructed by the application.
- Prompt templates combine system instructions and user input to create model requests.
- Prompt-based systems are simple to implement and enable rapid prototyping.
- However, they have limitations related to knowledge access, hallucination risk, and context constraints.
- Many production AI systems evolve from prompt-based architectures toward more complex architectures such as RAG and workflow systems.

---

## Comprehension Questions

1. What defines a prompt-based AI system architecture?
2. What components typically exist in a prompt-based architecture?
3. Why are prompt-based systems easy to prototype?
4. What limitations arise when relying solely on prompt-based interaction with language models?
5. Why do many AI systems evolve from prompt-based architectures to more complex architectures?

---

## References

### Papers

Language Models are Few-Shot Learners — Brown et al., 2020
[https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

### Documentation

OpenAI API Documentation
[https://platform.openai.com/docs](https://platform.openai.com/docs)

Anthropic API Documentation
[https://docs.anthropic.com](https://docs.anthropic.com)

---

## Key Takeaways

- Prompt-based architectures rely on direct interaction between an application and a language model.
- The application constructs prompts dynamically using templates and user input.
- These architectures are simple to implement and useful for prototyping.
- However, they rely heavily on model capabilities and have limited access to external knowledge.
- Many production AI systems extend prompt-based architectures with retrieval systems and orchestration workflows.
