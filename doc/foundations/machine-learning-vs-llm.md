# Machine Learning vs LLM Systems

[⬅ Back to Foundations](index.md)

---

## Context

Machine learning systems have been widely deployed for more than a decade in applications such as recommendation systems, fraud detection, image classification, and predictive analytics. These systems typically rely on models trained for **specific tasks using labeled datasets**.

Large language models (LLMs) represent a different class of machine learning systems. Instead of being trained for a single task, they are **general-purpose models** capable of performing a wide variety of language and reasoning tasks.

This shift has important implications for how systems are designed, deployed, and operated. Traditional machine learning systems often require separate models for each task, while LLM-based systems can adapt to new tasks using prompts and contextual information.

Understanding the differences between traditional machine learning systems and LLM systems helps engineers design architectures that take advantage of the strengths of each approach.

---

Traditional machine learning systems remain essential for many applications. Structured prediction problems, numerical forecasting, and large-scale recommendation systems continue to rely heavily on specialized models optimized for specific tasks.

However, LLM systems introduce new capabilities in domains involving language, reasoning, and unstructured data. As a result, modern AI architectures often combine traditional machine learning models with large language models.

Understanding when to use each approach—and how they interact within a system architecture—is a key skill in AI Systems Engineering.

---

## Concept Overview

Traditional machine learning systems and LLM-based systems differ in several fundamental ways.

```
Traditional Machine Learning
↓
Task-Specific Models
↓
LLM Systems
```

The key distinction lies in how models are trained and how they are used within applications.

| Property            | Traditional ML Systems | LLM Systems                     |
| ------------------- | ---------------------- | ------------------------------- |
| Training objective  | Task-specific          | General language modeling       |
| Model scope         | Single task            | Multi-task                      |
| Training data       | Labeled datasets       | Large-scale pretraining corpora |
| Adaptation to tasks | Retraining required    | Prompting and context           |
| System architecture | Model-centric          | System-centric                  |

In traditional machine learning systems, the model itself contains most of the task-specific intelligence. In LLM-based systems, intelligence emerges from **the interaction between the model, prompts, and contextual information**.

This distinction reflects a broader architectural shift. Traditional ML systems are typically **model-centric**, while LLM systems are often **system-centric**, where prompts, retrieval pipelines, and orchestration components shape how the model is used.

**Key Concept — Task-Specific vs General-Purpose Models**

Traditional machine learning models are typically trained to solve a single well-defined task.

Large language models are trained as **general-purpose language models** capable of performing many tasks without retraining.

This distinction fundamentally changes how AI systems are designed and maintained.

---

## 4.1 Traditional Machine Learning Systems

Traditional machine learning systems are built around models trained to solve specific tasks.

Examples include:

- image classification models
- recommendation systems
- fraud detection models
- demand forecasting models
- ranking models for search systems

A typical machine learning pipeline looks like this:

```
Dataset
↓
Feature Engineering
↓
Model Training
↓
Evaluation
↓
Deployment
```

Once deployed, the trained model is used for inference:

```
Input Data
↓
Feature Extraction
↓
Model Inference
↓
Prediction
```

In these systems, most of the intelligence resides inside the trained model.

Improving performance typically requires:

- collecting additional training data
- improving feature engineering
- retraining models
- tuning model architectures

These workflows form the basis of **Machine Learning Engineering** and **MLOps** practices.

Traditional ML systems therefore tend to be **training-centered systems**, where improving performance requires improvements to training data and model development.

---

## 4.2 LLM Systems

Large language models are trained using a different paradigm.

Instead of training models to perform a specific task, LLMs are trained on massive corpora of text to **predict the next token in a sequence**. This training objective produces models capable of performing many language-related tasks.

Because of this general training objective, LLMs can be adapted to different tasks without retraining.

```
Pretrained Model
+
Prompt
+
Context
↓
Output
```

Examples of LLM-based applications include:

- chat assistants
- document summarization systems
- coding assistants
- knowledge assistants
- research automation systems

Unlike traditional ML systems, LLM systems rely heavily on **system architecture** rather than model specialization.

Typical components include:

- prompt templates
- retrieval pipelines
- external tool integrations
- orchestration workflows
- evaluation pipelines

These components allow engineers to shape model behavior without modifying the underlying model.

**Key Concept — LLM Systems Are Architecture-Driven**

In traditional ML systems, improving performance usually requires retraining the model.

In LLM systems, improvements often come from **better prompts, improved retrieval systems, or improved orchestration pipelines** rather than changes to the model itself.

---

## 4.3 Side-by-Side Comparison

The differences between traditional machine learning systems and LLM systems become clearer when viewed side by side.

| Dimension           | Traditional ML Systems                 | LLM Systems                                 |
| ------------------- | -------------------------------------- | ------------------------------------------- |
| Model scope         | Task-specific models                   | General-purpose models                      |
| Training approach   | Frequent retraining                    | Large-scale pretraining                     |
| Adaptation to tasks | Dataset updates and retraining         | Prompting and contextual inputs             |
| System architecture | Model-centric                          | System-centric                              |
| Engineering focus   | Feature engineering and model training | Prompt design, retrieval, and orchestration |
| Iteration cycle     | Training pipelines                     | Inference pipeline improvements             |

These differences influence how AI systems are designed, evaluated, and improved over time.

Traditional ML systems concentrate engineering effort on **training better models**, while LLM systems concentrate effort on **designing better system architectures around the model**.

---

## 4.4 Training vs Inference-Centered Systems

Another important difference lies in how systems evolve over time.

Traditional machine learning systems are typically **training-centered**. Engineers repeatedly improve datasets and retrain models to improve performance.

```
Data Collection
↓
Model Training
↓
Evaluation
↓
Deployment
```

LLM systems, by contrast, are often **inference-centered systems**.

Because the model is already pretrained, engineers focus primarily on improving the inference pipeline.

```
Prompt Design
↓
Context Retrieval
↓
Model Inference
↓
Evaluation
↓
Iteration
```

This shift significantly changes the day-to-day work of engineers. Instead of retraining models frequently, engineers iterate on:

- prompt templates
- retrieval strategies
- system orchestration
- evaluation datasets

In other words, engineering effort moves away from model training and toward **designing the system that surrounds the model**.

**Key Concept — LLM Systems Shift Effort from Training to System Design**

Traditional machine learning workflows focus heavily on training and improving models.

LLM-based systems shift much of this effort toward **designing the surrounding system architecture** that controls how the model is used.

---

## 4.5 Strengths and Limitations

Both approaches offer advantages depending on the type of problem being solved.

Traditional ML systems excel at:

- numerical prediction tasks
- structured data analysis
- high-precision classification
- large-scale recommendation systems

LLM systems excel at:

- language understanding
- reasoning over unstructured information
- summarization and explanation
- natural language interaction

Because these strengths differ, many real-world applications combine both approaches.

For example, an AI-powered search system might include:

- ranking models for search relevance
- recommendation systems
- a large language model for generating explanations
- retrieval pipelines for contextual information

Combining these components allows systems to benefit from both specialized models and general-purpose reasoning.

---

## 4.6 Hybrid AI Systems

Modern AI systems frequently integrate both traditional machine learning models and large language models.

```
User Query
↓
Application Service
↓
Search / Retrieval Models
↓
LLM Reasoning
↓
Generated Response
```

In this architecture:

- traditional ML models may handle ranking, filtering, or classification
- LLMs generate explanations or synthesize information
- retrieval systems provide contextual knowledge
- application services orchestrate the overall workflow

This hybrid architecture reflects the broader trend in AI Systems Engineering: combining deterministic software infrastructure with probabilistic models to create reliable and flexible systems.

**Key Concept — Modern AI Systems Combine Multiple Model Types**

Production AI systems often integrate specialized machine learning models with large language models.

Each model type contributes different capabilities, allowing systems to balance accuracy, scalability, and flexibility.

---

## Chapter Summary

- Traditional machine learning systems typically rely on models trained for specific tasks.
- Large language models are general-purpose models trained on large corpora of text.
- LLM systems can adapt to new tasks using prompts and contextual information rather than retraining.
- Traditional ML systems are typically training-centered, while LLM systems are often inference-centered.
- Modern AI architectures frequently combine traditional machine learning models with LLMs to leverage the strengths of both approaches.

---

## Comprehension Questions

1. What are the main differences between traditional machine learning models and large language models?
2. Why are traditional machine learning systems often trained separately for each task?
3. How do LLM systems adapt to new tasks without retraining?
4. Why are LLM-based systems often described as architecture-driven?
5. What engineering workflows differ between traditional ML systems and LLM systems?
6. Why do many production systems combine traditional machine learning models with LLMs?

---

## References

### Papers

Attention Is All You Need — Vaswani et al., 2017
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### Books

Designing Machine Learning Systems — Chip Huyen, 2022
[https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

---

## Key Takeaways

- Traditional machine learning models are typically trained for specific tasks.
- Large language models are general-purpose models capable of performing many tasks.
- LLM systems are **system-centric architectures**, where prompts, context, and orchestration guide model behavior.
- Engineering workflows for LLM systems focus on prompts, retrieval, and orchestration rather than frequent model retraining.
- Modern AI platforms often combine specialized ML models with large language models to build more capable systems.
