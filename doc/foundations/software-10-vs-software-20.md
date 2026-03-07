# Software 1.0 vs Software 2.0

[⬅ Back to Foundations](index.md)

---

## Context

To understand why modern AI systems are engineered differently from traditional software systems, it is useful to examine how software itself has evolved over time.

Historically, software systems were built by explicitly programming rules and algorithms that transform inputs into outputs. However, many real-world problems—such as language understanding, image recognition, and knowledge extraction—are difficult to solve using deterministic rules alone.

Machine learning introduced a different paradigm: instead of writing explicit rules, engineers train models on large datasets so that systems **learn patterns from data**.

Large language models (LLMs) represent a further evolution of this idea. Instead of training models for a single task, LLMs provide **general-purpose reasoning and language capabilities** that can be adapted to many tasks using prompts and contextual information.

Understanding this progression—from rule-based software to model-driven systems—provides essential context for AI Systems Engineering.

---

Modern AI systems are often described using a conceptual framework popularized by Andrej Karpathy: **Software 1.0 and Software 2.0**.

This framework highlights a fundamental shift in how system behavior is defined and implemented.

In traditional software, developers explicitly write the logic that determines system behavior. In machine learning systems, behavior emerges from patterns learned from data during training.

As large language models become central components of applications, a further stage has effectively emerged: **systems where prompts and contextual information guide the behavior of large pretrained models**.

Understanding these distinctions helps engineers reason about the architecture, development practices, and operational challenges of modern AI systems.

---

## Concept Overview

The evolution of software can be understood through three conceptual stages.

```
Software 1.0
↓
Software 2.0
↓
LLM Systems
```

Each stage represents a different way of defining system behavior.

| Paradigm     | Behavior Defined By           | Primary Artifacts           |
| ------------ | ----------------------------- | --------------------------- |
| Software 1.0 | Hand-written code             | Source code                 |
| Software 2.0 | Learned model parameters      | Training datasets + models  |
| LLM Systems  | Prompts and contextual inputs | Prompts + knowledge sources |

In practice, modern AI systems often combine all three paradigms. Deterministic software components orchestrate probabilistic models, while prompts and contextual information guide model reasoning.

**Key Concept — Software Behavior Can Be Programmed or Learned**

Traditional software defines behavior through deterministic algorithms written by developers.

Machine learning systems define behavior through model parameters learned from data during training.

Modern AI systems frequently combine both approaches, using deterministic software infrastructure to control probabilistic model behavior.

---

## 3.1 Software 1.0 — Rule-Based Systems

**Software 1.0** refers to traditional software systems where behavior is explicitly defined by human-written code.

In this paradigm, developers implement algorithms that transform inputs into outputs through deterministic logic.

```
Input
↓
Algorithm (Code)
↓
Output
```

Typical characteristics of Software 1.0 systems include:

- deterministic behavior
- explicitly defined rules
- predictable execution paths
- correctness verified through testing

Examples of Software 1.0 systems include:

- web applications
- database systems
- operating systems
- financial transaction systems
- distributed infrastructure platforms

Because system behavior is determined by code, developers can reason precisely about how the system will behave for any given input.

Testing strategies such as **unit testing, integration testing, and static analysis** allow engineers to verify system correctness with high confidence.

However, Software 1.0 approaches struggle with problems where rules are difficult to specify explicitly, such as:

- natural language understanding
- speech recognition
- image classification
- recommendation systems
- anomaly detection

These domains require systems capable of recognizing complex patterns in data.

---

## 3.2 Software 2.0 — Learned Systems

Machine learning introduced a new paradigm often referred to as **Software 2.0**.

In Software 2.0 systems, developers do not explicitly program the decision logic. Instead, they design models that **learn patterns from data during training**.

```
Training Data
+
Learning Algorithm
↓
Trained Model
```

The trained model is then used during inference:

```
Input
↓
Trained Model
↓
Output
```

In this paradigm, system behavior is encoded in **model parameters** rather than source code.

Typical characteristics of Software 2.0 systems include:

- behavior learned from data
- probabilistic outputs
- performance dependent on dataset quality
- evaluation through statistical metrics

Examples of Software 2.0 applications include:

- image recognition systems
- recommendation engines
- fraud detection systems
- speech recognition systems
- predictive analytics platforms

Developers no longer write rules that directly solve the problem. Instead, they define:

- model architectures
- training datasets
- training objectives
- evaluation metrics

The resulting system behavior emerges from the training process.

**Key Concept — Model Parameters Encode System Behavior**

In Software 2.0 systems, behavior is not defined directly in source code. Instead, it is encoded in the numerical parameters learned by the model during training.

Improving system behavior therefore involves improving training data, model architecture, or training processes rather than modifying program logic.

---

## 3.3 LLM Systems — Prompt-Guided Behavior

Large language models extend the Software 2.0 paradigm by introducing **general-purpose models that can perform many tasks without retraining**.

Unlike traditional machine learning systems where a model is trained for a specific task, LLMs are pretrained on massive datasets and can be adapted to new tasks using prompts and contextual information.

Instead of retraining models for each new task, engineers guide model behavior using prompts.

```
Model
+
Prompt
+
Context
↓
Output
```

In LLM-based systems:

- the **model** provides general reasoning capabilities
- the **prompt** defines instructions and task structure
- the **context** provides task-specific or domain-specific information

This architecture allows a single model to perform many different tasks.

Examples include:

- chat assistants
- coding copilots
- document analysis systems
- enterprise knowledge assistants
- research assistants

LLM systems therefore introduce a new class of engineering artifacts:

- prompts
- system instructions
- contextual knowledge sources
- retrieval pipelines

Modern AI systems frequently extend this architecture with **external tools**, allowing models to interact with APIs, databases, calculators, and other computation systems. These tools allow models to perform actions that go beyond text generation.

Engineers design systems that **shape model behavior through prompts, context, and tool usage rather than retraining models**.

**Key Concept — Prompts Become a Form of Programming**

In LLM systems, prompts act as a form of lightweight programming that defines tasks, constraints, and reasoning strategies for the model.

While prompts do not provide deterministic control, they significantly influence how models interpret inputs and generate outputs.

---

## 3.4 Combining Software Paradigms

Modern AI systems rarely rely on a single paradigm. Instead, they combine Software 1.0 and Software 2.0 approaches within the same architecture.

A typical AI system might include:

- deterministic application services
- machine learning models
- prompts and contextual inputs
- retrieval systems
- workflow orchestration

```
User Request
↓
Application Logic (Software 1.0)
↓
Prompt + Context
↓
LLM Reasoning
↓
Response
```

Deterministic software components manage:

- application logic
- system orchestration
- tool execution
- monitoring and observability

Machine learning components provide capabilities such as:

- language understanding
- pattern recognition
- reasoning and generation

This hybrid architecture allows engineers to combine the strengths of both paradigms.

**Key Concept — AI Systems Combine Deterministic and Probabilistic Components**

Modern AI systems integrate deterministic software infrastructure with probabilistic model behavior.

Reliable AI systems therefore require careful engineering of both components: traditional software infrastructure and machine learning systems.

---

## 3.5 Engineering Implications

The shift from Software 1.0 to Software 2.0 has important implications for how systems are designed, tested, and operated.

Traditional software engineering emphasizes:

- algorithm correctness
- deterministic behavior
- strict test coverage

AI systems require additional practices, including:

- dataset management
- evaluation frameworks
- prompt engineering
- monitoring of probabilistic outputs
- continuous experimentation

Because behavior emerges from data and model interactions, AI systems must be evaluated statistically rather than verified deterministically.

| Traditional Engineering | AI Systems Engineering       |
| ----------------------- | ---------------------------- |
| Deterministic tests     | Statistical evaluation       |
| Code versioning         | Artifact versioning          |
| Debugging algorithms    | Debugging data and prompts   |
| Static system behavior  | Iterative system improvement |

Engineering teams must therefore adopt workflows that integrate **software engineering, data engineering, model evaluation, and system monitoring**.

---

## Chapter Summary

- Software systems have evolved from rule-based programs to data-driven learning systems.
- **Software 1.0** refers to traditional deterministic programs where behavior is defined by human-written code.
- **Software 2.0** refers to machine learning systems where behavior emerges from model parameters learned from data.
- **LLM systems** extend this paradigm by allowing model behavior to be shaped using prompts, context, and tool usage.
- Modern AI applications combine deterministic software infrastructure with probabilistic models.
- Building reliable AI systems requires new engineering practices such as dataset management, evaluation pipelines, and prompt engineering.

---

## Comprehension Questions

1. What distinguishes Software 1.0 systems from Software 2.0 systems?
2. Why are some problems difficult to solve using deterministic algorithms alone?
3. How do model parameters encode behavior in machine learning systems?
4. Why do large language models allow many tasks to be performed without retraining?
5. In what sense can prompts be considered a form of programming?
6. Why do modern AI systems combine deterministic and probabilistic components?

---

## References

### Papers

Attention Is All You Need — Vaswani et al., 2017
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### Talks and Articles

Software 2.0 — Andrej Karpathy
[https://karpathy.medium.com/software-2-0-a64152b37c35](https://karpathy.medium.com/software-2-0-a64152b37c35)

### Books

Designing Machine Learning Systems — Chip Huyen, 2022
[https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

---

## Key Takeaways

- Software systems have evolved from rule-based programming to data-driven learning systems.
- **Software 1.0** describes traditional software where behavior is defined by code.
- **Software 2.0** describes machine learning systems where behavior is encoded in model parameters learned from data.
- **LLM systems** extend this paradigm by allowing model behavior to be shaped using prompts and contextual information.
- Modern AI systems combine deterministic software infrastructure with probabilistic model reasoning.
- Effective AI Systems Engineering requires managing code, data, models, prompts, and evaluation artifacts together.
