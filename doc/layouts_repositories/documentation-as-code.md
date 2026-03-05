## Chapter 4 — Documentation as Code 🧪

### 4.1 Living Documentation for AI Systems

AI systems suffer from a documentation anti-pattern: decisions are made verbally in meetings, committed as code without explanation, and the reasoning is lost within weeks. Six months later, no one can answer "why is the chunk size 512?" or "why did we choose Qdrant over Weaviate?"

Documentation as code treats documentation with the same version control, review, and quality discipline applied to source code. It has three practical components:

1. **Architecture Decision Records (ADRs)** — document significant decisions and their rationale
2. **Runbooks** — executable, testable operational procedures
3. **API documentation** — auto-generated from code, always current

The key discipline: documentation lives in the repository alongside the code it describes. It is updated in the same PR as the code change. It is reviewed by the same reviewers. It becomes outdated for the same reason code becomes outdated — and is visible in the same git blame.

---

### 4.2 Architecture Decision Records

An ADR records a significant architectural decision: its context, the options considered, the decision made, and the consequences.

```markdown
<!-- docs/adr/ADR-007-vector-database-selection.md -->
# ADR-007: Vector Database Selection

**Date:** 2024-11-15
**Status:** Accepted
**Deciders:** Platform Team (Alice, Bob, Carlos)
**Supersedes:** ADR-003 (FAISS in-process index)

---
[« Back to layouts_repositories Index](index.md) | [🏠 Home](../../index.md)