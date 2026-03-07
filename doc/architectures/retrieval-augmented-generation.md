# Retrieval-Augmented Generation

[⬅ Back to Architectures](index.md)

---

## Context

Prompt-based architectures rely entirely on the internal knowledge of language models. While this approach works well for many tasks, it introduces a major limitation: models cannot reliably access **external, private, or up-to-date information**.

Large language models are trained on static datasets and therefore cannot directly retrieve information from proprietary documents, databases, or recent events. As a result, systems that rely only on prompts often struggle with factual accuracy and domain-specific knowledge.

**Retrieval-Augmented Generation (RAG)** addresses this limitation by combining language models with external knowledge retrieval systems. Instead of relying solely on the model’s training data, RAG systems dynamically retrieve relevant documents and include them in the prompt as context.

This architecture allows AI systems to generate responses grounded in **retrieved knowledge sources**, significantly improving factual accuracy and enabling the use of private data.

RAG architectures have become one of the most widely used patterns for building production AI applications, especially for enterprise knowledge assistants, document analysis systems, and internal information retrieval tools.

---

## Concept Overview

Retrieval-Augmented Generation integrates **information retrieval systems** with language model inference.

Rather than generating responses from the model alone, the system retrieves relevant documents and injects them into the prompt.

A simplified RAG pipeline looks like this:

```id="rag-basic-flow"
User Query
↓
Retriever
↓
Vector Database
↓
Relevant Documents
↓
Prompt Construction
↓
LLM
↓
Response
```

In this architecture, the model receives both the user query and additional context retrieved from external sources.

A key concept in RAG systems is **grounding**. Grounding refers to the process of constraining model responses using external information retrieved from reliable knowledge sources. Instead of relying solely on learned patterns in model parameters, the model reasons over retrieved documents that provide factual context.

**Key Concept — Retrieval Extends Model Knowledge**

RAG systems expand the capabilities of language models by providing access to external knowledge sources. This allows the model to reason over information that is not stored in its parameters and to generate responses grounded in retrieved context.

---

## 1. Architecture Structure

A typical RAG architecture introduces several new components compared to prompt-based systems.

```id="rag-architecture"
User
↓
Application
↓
Retriever
↓
Vector Database
↓
Context Builder
↓
Prompt
↓
LLM
↓
Response
```

Key components include:

- **retriever** — searches for relevant documents
- **vector database** — stores embeddings used for semantic search
- **context builder** — assembles retrieved documents into the prompt
- **language model** — generates the final response

These components work together to transform a user query into a response grounded in external information.

---

## 2. End-to-End System Flow

In a production AI system, retrieval is only one stage within a larger execution pipeline.

A typical end-to-end flow in a RAG application looks like this:

```id="rag-end-to-end-flow"
User
↓
Application Service
↓
Retriever
↓
Vector Database
↓
Context Builder
↓
Prompt Construction
↓
LLM
↓
Response
```

This pipeline illustrates how multiple components interact to transform a user request into a response grounded in external knowledge.

Understanding this end-to-end execution flow helps engineers identify where errors occur and how system performance can be optimized.

---

## 3. The Retrieval Pipeline

The retrieval pipeline is responsible for identifying relevant information from the knowledge base.

Typical steps include:

```id="retrieval-pipeline"
User Query
↓
Embedding Model
↓
Vector Search
↓
Candidate Documents
↓
Document Selection
```

First, the query is converted into a vector representation using an **embedding model**.

The vector database then performs a similarity search to identify documents whose embeddings are closest to the query.

The retrieved documents are returned to the application for further processing.

---

## 4. Context Construction

After retrieval, the system must construct the prompt sent to the language model.

This step is known as **context construction**.

A typical prompt structure might look like:

```id="rag-prompt-composition"
System Prompt
+
Retrieved Documents
+
User Query
```

The retrieved documents provide the factual information the model should use when generating a response.

Because language models have limited context windows, systems must carefully select and format retrieved documents to ensure that the most relevant information is included.

---

## 5. Advantages of RAG Architectures

RAG architectures provide several important advantages compared to prompt-only systems.

### Access to External Knowledge

RAG systems can retrieve information from external documents, databases, and knowledge bases.

### Improved Factual Accuracy

Because responses are grounded in retrieved documents, hallucination rates can be reduced.

### Support for Private Data

Organizations can build AI systems that answer questions about internal documents or proprietary datasets.

### Updatable Knowledge

Knowledge bases can be updated without retraining the model.

These properties make RAG architectures particularly well suited for **enterprise AI applications**.

---

## 6. Limitations of RAG Systems

Although RAG significantly improves system capabilities, it also introduces new engineering challenges.

### Retrieval Quality

If the retriever returns irrelevant documents, the model may produce incorrect answers.

### Context Window Constraints

Only a limited number of documents can fit within the model’s context window.

### Increased Latency

Retrieval steps add additional processing time before generation occurs.

### System Complexity

Compared to prompt-based systems, RAG architectures require additional infrastructure such as vector databases and retrieval pipelines.

These challenges are addressed through techniques such as:

- better chunking strategies
- improved embedding models
- reranking algorithms
- multi-stage retrieval pipelines

---

## 7. Relationship to the AI Systems Reference Stack

RAG architectures span multiple layers of the **AI Systems Reference Stack**.

```id="rag-stack"
Interaction Layer
↓
Application Layer
↓
Orchestration Layer
↓
Prompt Layer
↓
Retrieval Layer
↓
Model Layer
↓
Data Layer
```

The most significant addition compared to prompt-based systems is the **Retrieval Layer**, which introduces new infrastructure for knowledge access.

This layered perspective helps engineers reason about where retrieval pipelines belong in the system architecture.

---

## 8. Example Applications

RAG architectures are commonly used in systems that must answer questions based on large collections of documents.

Examples include:

- enterprise knowledge assistants
- document question-answering systems
- research assistants
- legal document analysis tools
- technical documentation search systems

Example flow for an enterprise assistant:

```id="enterprise-rag-flow"
User Question
↓
Retriever searches company documents
↓
Relevant documents returned
↓
Context inserted into prompt
↓
LLM generates answer
↓
Response returned to user
```

In this scenario, the system combines the reasoning capabilities of the language model with the organization’s internal knowledge base.

---

## 9. Evolution Toward Advanced RAG Systems

Basic RAG architectures often evolve into more advanced retrieval systems.

Typical improvements include:

- multi-stage retrieval pipelines
- reranking models
- query rewriting
- hybrid search (vector + keyword)
- knowledge graph integration

These techniques improve retrieval quality and system reliability.

Later sections of the book explore these topics in detail within the **RAG Engineering** and **Advanced RAG** sections.

---

## Chapter Summary

- Retrieval-Augmented Generation combines language models with external information retrieval systems.
- RAG architectures retrieve relevant documents and include them in prompts sent to language models.
- This approach improves factual accuracy and enables the use of private or domain-specific knowledge.
- RAG systems introduce new architectural components such as vector databases, embedding models, and retrieval pipelines.
- Although powerful, RAG systems introduce additional engineering challenges related to retrieval quality, context management, and latency.

---

## Comprehension Questions

1. What problem does Retrieval-Augmented Generation solve in AI systems?
2. What components are typically introduced in a RAG architecture?
3. How does the retrieval pipeline identify relevant documents?
4. Why is context construction important in RAG systems?
5. What new engineering challenges arise when introducing retrieval systems?

---

## References

### Papers

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al., 2020
[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

REALM: Retrieval-Augmented Language Model Pre-Training — Guu et al., 2020
[https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)

### Documentation

LangChain Retrieval Documentation
[https://python.langchain.com/docs/use_cases/question_answering/](https://python.langchain.com/docs/use_cases/question_answering/)

LlamaIndex Documentation
[https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)

---

## Key Takeaways

- Retrieval-Augmented Generation extends language models with external knowledge retrieval.
- RAG architectures combine retrievers, vector databases, and language models.
- Retrieved documents are inserted into prompts to ground model responses.
- RAG systems introduce additional complexity but significantly improve factual accuracy and system capability.
- Grounding model responses in retrieved context is a key mechanism for improving reliability in AI systems.
