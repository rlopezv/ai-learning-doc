
# Chapter 2 — Software 1.0 vs Software 2.0

[⬅ Back to Foundations](index.md)

## Context

The evolution of software systems has undergone a profound transformation over the last two decades. Traditional enterprise systems were built using deterministic logic written explicitly by developers. This paradigm, sometimes referred to as **Software 1.0**, dominated software engineering for decades.

Machine learning introduced a different paradigm in which system behavior is **learned from data rather than explicitly programmed**. This concept was famously described by Andrej Karpathy as **Software 2.0**, where neural network weights effectively become the program.

Large Language Models (LLMs) extend this idea even further. Instead of training specialized models for individual tasks, modern AI systems often rely on large pretrained foundation models whose behavior can be adapted dynamically through **prompts and contextual information**.

Understanding these paradigms is essential for AI systems engineers, because modern production AI platforms frequently combine elements from all three.

---

## Concept Overview

Software systems can be understood as evolving across three paradigms:

Software Evolution
│
├ Software 1.0
│   └ Behavior explicitly programmed
│
├ Software 2.0
│   └ Behavior learned from data
│
└ LLM Systems
    └ Behavior guided by prompts and context

Each paradigm defines **how system behavior is created and modified**.

---

## 2.1 The Programming Paradigm That Shaped Enterprise Software

Software 1.0 refers to the traditional programming model in which developers write explicit instructions that define system behavior.

Example:

Input Data  
↓  
Algorithm  
↓  
Output

A simple rule-based program might look like:

```python
if temperature > 30:
    recommendation = "Turn on air conditioning"
else:
    recommendation = "Open the windows"
```

In this paradigm:

- behavior is deterministic
- logic is encoded directly in code
- modifications require changing source code

Software 1.0 systems are highly predictable and easy to test using traditional unit testing approaches.

---

## 2.2 The Limits of Explicit Rules

While Software 1.0 works well for many problems, it struggles with tasks that require:

- perception
- natural language understanding
- pattern recognition
- reasoning over ambiguous data

Examples include:

- speech recognition
- image classification
- document understanding
- translation

Writing explicit rules for these problems is extremely difficult because the possible variations in real-world data are too large.

Machine learning addressed this limitation by allowing systems to **learn patterns directly from data**.

---

## 2.3 Software 2.0 — Learned Behavior

Software 2.0 describes systems in which program behavior is learned through model training rather than written directly in code.

Typical development pipeline:

Dataset  
↓  
Training Algorithm  
↓  
Neural Network  
↓  
Model Weights

Here, the **model weights effectively become the program**.

Instead of writing explicit rules, engineers:

- collect and label datasets
- select model architectures
- train models using optimization algorithms

The trained model is then used for inference:

Input Data  
↓  
Model  
↓  
Prediction

Examples of Software 2.0 systems include:

- image recognition systems
- recommendation engines
- fraud detection models
- autonomous driving perception systems

---

## 2.4 The Comparative Anatomy of Both Paradigms

The differences between Software 1.0 and Software 2.0 affect nearly every aspect of system development.

| Aspect | Software 1.0 | Software 2.0 |
|---|---|---|
| Program definition | Explicit code | Learned model parameters |
| Development method | Programming | Training |
| Main artifact | Source code | Dataset + model |
| Debugging | Code inspection | Data analysis |
| Testing | Unit tests | Evaluation datasets |
| Behavior modification | Code changes | Retraining |

These differences introduce new engineering challenges, particularly around **data management, evaluation, and reproducibility**.

---

## 2.5 Large Language Models — A Step Further

Large language models extend the Software 2.0 paradigm.

Instead of training models for individual tasks, LLMs are **large pretrained foundation models** trained on massive text corpora.

Once trained, these models can perform many tasks without retraining.

Model  
+  
Prompt  
+  
Context  
↓  
Output

This introduces a new form of programming sometimes called **prompt programming**.

Developers influence system behavior by modifying:

- prompt instructions
- contextual data
- examples included in prompts

LLMs can therefore be understood as **a layer built on top of Software 2.0 models**.

---

## 2.6 Real LLM Systems: Beyond Simple Prompting

In real production environments, LLM systems rarely consist of a single prompt and model.

Instead, they include multiple components:

User Input  
↓  
Application Logic  
↓  
Prompt Construction  
↓  
Context Retrieval  
↓  
LLM  
↓  
Post-processing

These systems may incorporate:

- vector databases
- retrieval pipelines
- workflow orchestration
- tool integration
- evaluation systems

As a result, modern AI platforms combine elements from:

- Software 1.0
- Software 2.0
- prompt-driven LLM systems

---

## 2.7 Hybrid Systems: Combining Both Paradigms

Most real-world AI applications combine deterministic software with machine learning components.

Example architecture:

User Request  
↓  
Application Logic (Software 1.0)  
↓  
ML Model (Software 2.0)  
↓  
LLM Reasoning  
↓  
Final Output

This hybrid architecture allows systems to combine:

- deterministic reliability
- statistical pattern recognition
- flexible reasoning capabilities

Software engineers must therefore design systems that integrate **deterministic and probabilistic components** effectively.

---

## 2.8 Implications for Solution Architects

The emergence of Software 2.0 and LLM systems changes how engineers design software systems.

Architects must now consider:

- how models are trained and updated
- how prompts influence behavior
- how retrieval systems supply knowledge
- how evaluation pipelines measure performance
- how system reliability is maintained despite probabilistic outputs

This requires expanding traditional software engineering practices to include:

- data lifecycle management
- model lifecycle management
- prompt versioning
- evaluation frameworks

AI systems engineering therefore represents a **convergence of software engineering, machine learning engineering, and data engineering practices**.

---

## 📋 Chapter Summary

- **Software 1.0** refers to traditional systems where behavior is explicitly programmed through deterministic logic.
- **Software 2.0** describes systems where behavior is learned from data through machine learning models.
- Large language models extend this paradigm by enabling systems to adapt behavior dynamically through **prompts and contextual information**.
- Modern AI applications typically combine deterministic software components with learned models.
- Understanding the interaction between these paradigms is essential for designing scalable AI systems.

---

## ❓ Comprehension Questions

1. What distinguishes Software 1.0 from Software 2.0 in terms of how system behavior is defined?
2. Why are rule-based systems insufficient for many perception and language tasks?
3. In what sense do neural network weights function as a “program” in Software 2.0?
4. How do large language models extend the Software 2.0 paradigm?
5. Why are hybrid architectures combining deterministic software and AI models common in modern systems?

---

## References

### Foundational Concepts

- Andrej Karpathy — *Software 2.0* (2017)  
https://karpathy.medium.com/software-2-0-a64152b37c35

### Papers

- Vaswani et al. — *Attention Is All You Need* (2017)  
https://arxiv.org/abs/1706.03762

- LeCun, Bengio, Hinton — *Deep Learning* (Nature, 2015)  
https://www.nature.com/articles/nature14539

### Books

- Chip Huyen — *Designing Machine Learning Systems* (O’Reilly, 2022)

- Martin Kleppmann — *Designing Data-Intensive Applications* (O’Reilly, 2017)

---

## Key Takeaways

- Software development has evolved from deterministic programming to systems driven by learned models.
- Software 2.0 introduces a new programming paradigm where model parameters encode system behavior.
- Large language models enable **prompt-driven behavior adaptation** without retraining.
- Production AI systems typically combine deterministic software with machine learning and LLM components.
- Architects must design systems that integrate both deterministic and probabilistic elements reliably.
