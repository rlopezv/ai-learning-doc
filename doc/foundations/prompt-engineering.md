# Prompt Engineering

[⬅ Back to Foundations](index.md)

---

## Context

Large language models are trained as general-purpose systems capable of performing a wide range of tasks. However, their behavior is not fixed. Instead, it is strongly influenced by the **prompts** provided as input.

Prompts define the instructions, context, and constraints that guide how the model interprets a request and generates a response. Even small changes in prompt structure can significantly alter model behavior.

For engineers building AI systems, prompts function as a critical interface between users and models. Designing effective prompts allows systems to shape model behavior without retraining the underlying model.

Understanding prompt engineering is therefore essential for AI Systems Engineering. It enables developers to control how models perform tasks, integrate external knowledge, and interact with other components of the system architecture.

---

Modern AI applications rarely rely on a single user input alone. Instead, prompts are often constructed dynamically by combining multiple elements such as instructions, retrieved context, conversation history, and tool outputs.

A typical prompt in a production system might include:

- system instructions
- user input
- retrieved documents
- conversation history
- tool results

Prompt engineering focuses on designing and organizing these components so that the model produces reliable and useful outputs.

---

## Concept Overview

A **prompt** is the structured input provided to a language model to guide its behavior.

Prompts typically contain instructions, contextual information, and user queries.

A simplified prompt structure might look like this:

```id="2v9p1n"
System Instructions
+
User Query
↓
Model
↓
Response
```

In production systems, prompts are usually more complex and include additional context:

```id="x7b6mr"
System Instructions
+
Retrieved Context
+
Conversation History
+
User Query
↓
LLM
↓
Response
```

Prompt engineering involves designing these structures so that the model:

- understands the task
- follows the intended instructions
- uses provided context effectively
- generates reliable responses

**Key Concept — Prompts Control Model Behavior**

Large language models are general-purpose systems. Prompts provide the instructions that guide how the model interprets inputs and generates outputs.

Well-designed prompts allow engineers to adapt a single model to many different tasks without retraining.

---

## 6.1 Prompt Structure

Effective prompts usually follow a structured format that clearly separates instructions from input data.

A common prompt structure includes:

```id="6xy8bq"
Instruction
+
Context
+
Input
```

Where:

- **Instruction** defines the task the model should perform
- **Context** provides relevant information needed to complete the task
- **Input** represents the specific query or problem

Example:

```id="f0m1m3"
Instruction:
Summarize the following document.

Context:
Document text...

Input:
User request
```

This structure helps the model distinguish between instructions, contextual knowledge, and user input.

### Example Prompt

A production prompt often combines multiple components:

```id="0r2g4h"
System Prompt:
You are a technical assistant that answers questions concisely.

Context:
{retrieved_documents}

User Query:
Explain how vector databases are used in RAG systems.
```

The system dynamically fills placeholders such as `{retrieved_documents}` during prompt construction.

---

## 6.2 System Prompts

Many AI systems include **system prompts** that define the behavior and constraints of the model.

System prompts typically appear at the beginning of the prompt and establish rules such as:

- tone of the response
- formatting requirements
- safety constraints
- domain expertise

Example:

```id="9z8b7d"
System Prompt:
You are an AI assistant that explains technical concepts clearly and concisely.

User Query:
Explain retrieval-augmented generation.
```

System prompts help ensure that model responses remain consistent across interactions.

**Key Concept — System Prompts Define Behavioral Constraints**

System prompts provide persistent instructions that shape how the model behaves across many interactions.

They allow engineers to enforce tone, formatting, and safety constraints within AI systems.

---

## 6.3 Prompt Templates

In production systems, prompts are rarely written manually for each request. Instead, applications use **prompt templates**.

Prompt templates are reusable prompt structures where dynamic values are inserted at runtime.

Example template:

```id="x3gk5p"
You are an expert assistant.

Answer the following question using the provided context.

Context:
{retrieved_documents}

Question:
{user_query}
```

At runtime, the system fills in placeholders with actual values.

```id="h5t3r9"
retrieved_documents → retrieved knowledge
user_query → user input
```

Prompt templates enable:

- consistent prompt structure
- easier prompt iteration
- integration with retrieval pipelines
- integration with application logic

Prompt templates are commonly managed within orchestration frameworks used in AI systems.

---

## 6.4 Prompt Patterns

Certain prompt design patterns consistently improve model performance.

Common prompt patterns include:

### Instruction Prompting

Provide explicit instructions describing the task.

```id="u4j6m1"
Explain the following concept in simple terms.
```

### Few-Shot Prompting

Provide examples demonstrating the desired output format.

```id="t8y2v5"
Example 1:
Question → Answer

Example 2:
Question → Answer
```

### Role Prompting

Assign a role to guide model behavior.

```id="k9w1x2"
You are a cybersecurity expert.
```

### Step-by-Step Reasoning

Encourage the model to reason through the task.

```id="j7r5l3"
Explain your reasoning step by step.
```

These patterns help the model interpret the task and generate more reliable responses.

---

## 6.5 Prompt Failures

Prompt design can significantly influence system behavior. Poorly designed prompts can lead to unreliable or incorrect outputs.

Common prompt-related issues include:

- ambiguous instructions
- conflicting instructions
- insufficient context
- excessive context
- unclear output format

For example, a vague prompt such as:

```id="a2c4v8"
Explain AI.
```

may produce inconsistent responses.

A more effective prompt provides clearer instructions:

```id="w6d9k1"
Explain artificial intelligence in three concise paragraphs suitable for a software engineer.
```

Effective prompt engineering reduces ambiguity and helps ensure that the model understands the intended task.

---

## 6.6 Prompt Engineering in System Architecture

In production AI systems, prompts are rarely static. Instead, they are generated dynamically by combining multiple system components.

Example architecture:

```id="p4n7s8"
User Query
↓
Retriever
↓
Vector Database
↓
Prompt Template
↓
LLM
↓
Response
```

In this architecture:

- retrieval systems provide contextual knowledge
- prompt templates structure the input
- orchestration logic assembles the final prompt

Prompt engineering therefore interacts closely with several components of AI systems:

- retrieval pipelines
- workflow orchestrators
- evaluation systems
- safety guardrails

Because prompts influence model behavior, they are often treated as **versioned system artifacts** within AI platforms.

---

## 6.7 Prompt Iteration and Evaluation

Prompt engineering is typically an **iterative process**.

Engineers improve prompts by:

- testing different prompt structures
- evaluating model outputs
- measuring performance on evaluation datasets

This process resembles traditional software iteration, but focuses on improving prompt design rather than modifying code.

Example workflow:

```id="r3t8g2"
Design Prompt
↓
Run Evaluation Dataset
↓
Measure Output Quality
↓
Refine Prompt
↓
Repeat
```

Evaluation frameworks help ensure that prompt updates improve system behavior without introducing regressions.

**Key Concept — Prompts Are Versioned System Artifacts**

In production AI systems, prompts are often treated as versioned artifacts similar to source code.

Prompt updates are evaluated, tested, and tracked to ensure that system behavior remains consistent and reliable.

---

## Chapter Summary

- Prompts are structured inputs used to guide the behavior of large language models.
- Prompt engineering focuses on designing instructions and context that produce reliable model outputs.
- Production systems typically use **prompt templates** rather than manually written prompts.
- Common prompt patterns include instruction prompting, few-shot prompting, role prompting, and step-by-step reasoning.
- Poor prompt design can lead to unreliable model behavior.
- Prompt engineering interacts closely with retrieval systems, orchestration workflows, and evaluation frameworks.

---

## Comprehension Questions

1. What role do prompts play in controlling the behavior of large language models?
2. Why are prompt templates commonly used in production systems?
3. What is the purpose of a system prompt?
4. How does few-shot prompting influence model behavior?
5. Why must prompt engineering be treated as an iterative process?
6. Why are prompts often treated as versioned artifacts in AI systems?

---

## References

### Papers

Language Models are Few-Shot Learners — Brown et al., 2020
[https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — Wei et al., 2022
[https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

### Documentation

OpenAI Prompt Engineering Guide
[https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering)

Anthropic Prompt Engineering Guide
[https://docs.anthropic.com](https://docs.anthropic.com)

---

## Key Takeaways

- Prompts define the instructions and context used to guide language model behavior.
- Prompt templates enable reusable and consistent prompt structures in production systems.
- System prompts establish behavioral constraints such as tone, format, and safety rules.
- Prompt design patterns help improve model reliability and task performance.
- Prompt engineering is an iterative process supported by evaluation frameworks.
