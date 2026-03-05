## Chapter 1 — LLM Security Threat Model

### 1.1 Threat Taxonomy for AI Systems

AI systems face the full spectrum of conventional application security threats — injection, broken access control, insecure dependencies — plus a set of novel threats that arise specifically from the LLM architecture. Understanding the threat taxonomy is the foundation for designing effective defences.

**Conventional threats (apply to AI systems as to any web service):**
- Input injection targeting the application layer (SQL injection, path traversal)
- Authentication bypass and session hijacking
- Insecure direct object reference (querying another tenant's data via document IDs)
- Sensitive data exposure in transit or at rest
- Dependency vulnerabilities in ML libraries (PyTorch, Transformers)

**LLM-specific threats (novel to AI systems):**
- **Prompt injection** — user-supplied input overrides system instructions
- **Indirect prompt injection** — instructions embedded in retrieved documents or tool results
- **Jailbreaking** — crafted inputs that bypass content policies
- **Data poisoning** — malicious content introduced into the training corpus or RAG knowledge base
- **Model extraction** — systematic queries designed to reconstruct model behaviour or training data
- **Inference disclosure** — answers that inadvertently reveal system prompt contents, internal data, or other users' queries
- **PII leakage** — user query or retrieved document contains PII that should not be returned

---

### 1.2 OWASP LLM Top 10 Applied

The [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) provides a structured catalogue of the most critical LLM security risks:

| OWASP LLM Risk | RAG system manifestation | Primary defence |
|---|---|---|
| LLM01: Prompt Injection | User query overrides system prompt | Structural defence, input validation |
| LLM02: Insecure Output Handling | Raw LLM output used in SQL/shell | Output schema validation before use |
| LLM03: Training Data Poisoning | Malicious docs ingested into corpus | Corpus provenance and content scanning |
| LLM04: Model DoS | Long context overload, token exhaustion | Token limits, rate limiting, cost caps |
| LLM05: Supply Chain | Compromised model weights or SDK | Signed model checksums, pinned deps |
| LLM06: Sensitive Info Disclosure | PII or secrets in retrieved context | PII redaction, data classification |
| LLM07: Insecure Plugin Design | Tool calls with unsanitised LLM output | Tool call schema validation |
| LLM08: Excessive Agency | LLM writes to DB based on instructions | Human-in-the-loop for write operations |
| LLM09: Overreliance | Hallucination accepted as fact | Faithfulness gating, citation requirement |
| LLM10: Model Theft | API reverse engineering | Rate limiting, output watermarking |

---

### 1.3 Attack Surface Mapping

```python
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"

@dataclass
class AttackSurface:
    surface_id: str
    name: str
    description: str
    threat_level: ThreatLevel
    attack_vectors: list[str]
    mitigations: list[str]

RAG_ATTACK_SURFACES = [
    AttackSurface(
        "AS-01", "User Query Input",
        "Free-text query submitted by end user",
        ThreatLevel.CRITICAL,
        attack_vectors=[
            "Direct prompt injection (override system prompt)",
            "Jailbreak via roleplay or hypothetical framing",
            "Token flooding (extremely long queries)",
            "Unicode/encoding attacks",
        ],
        mitigations=[
            "Input length enforcement (max_chars)",
            "Structural prompt design (context isolation)",
            "Injection pattern detection",
            "Input sanitisation (control characters)",
        ]
    ),
    AttackSurface(
        "AS-02", "Retrieved Document Content",
        "Third-party or user-uploaded documents in vector index",
        ThreatLevel.CRITICAL,
        attack_vectors=[
            "Indirect prompt injection via document content",
            "PII leakage from documents (patient records, financial data)",
            "Secrets embedded in documents (API keys, passwords)",
            "Data poisoning (malicious instructions in corpus)",
        ],
        mitigations=[
            "Document content scanning before ingestion",
            "PII detection and redaction",
            "Secrets scanning (detect API keys, credentials)",
            "Corpus provenance tracking and access control",
        ]
    ),
    AttackSurface(
        "AS-03", "LLM API Interface",
        "Communication between application and LLM provider",
        ThreatLevel.HIGH,
        attack_vectors=[
            "API key leakage (logs, git history)",
            "Man-in-the-middle on API calls",
            "Prompt content exposure in provider logs",
        ],
        mitigations=[
            "API keys in secrets manager (not env vars or code)",
            "TLS for all API calls",
            "Data processing agreements with providers",
            "On-premise LLM for sensitive data",
        ]
    ),
    AttackSurface(
        "AS-04", "Vector Database",
        "Qdrant / Chroma index storage and query interface",
        ThreatLevel.HIGH,
        attack_vectors=[
            "Cross-tenant data access via crafted queries",
            "Index poisoning via unauthenticated write access",
            "Metadata filter bypass",
        ],
        mitigations=[
            "Vector DB authentication (API key required)",
            "Tenant-level metadata filters enforced server-side",
            "Network policy: vector DB not accessible from internet",
            "Audit log all write operations",
        ]
    ),
    AttackSurface(
        "AS-05", "Ingestion Pipeline",
        "Document upload, chunking, and embedding pathway",
        ThreatLevel.MEDIUM,
        attack_vectors=[
            "Malicious document upload (code, macros, zip bombs)",
            "SSRF via URLs in document metadata",
            "Path traversal in document identifiers",
        ],
        mitigations=[
            "File type allowlist enforcement",
            "File size limits",
            "Virus/malware scanning on upload",
            "Document ID sanitisation",
        ]
    ),
]
```

---

### 1.4 Security Architecture Principles

```
Defence-in-Depth Architecture for RAG Systems

External                    Platform                     Internal
─────────────────────────────────────────────────────────────────
User Query
    │
    ▼
[1] API Gateway          ← Rate limiting, auth, TLS
    │
    ▼
[2] Input Validator      ← Length, charset, injection patterns
    │
    ▼
[3] PII Detector         ← Detect + redact before logging
    │
    ▼
[4] RAG Pipeline         ← Structural prompt isolation
    │         │
    │         ▼
    │   [5] Document     ← Pre-ingestion content scanning
    │       Scanner      ← PII redaction in corpus
    │         │
    ▼         ▼
[6] LLM Gateway          ← API key management, cost limits
    │
    ▼
[7] Output Validator     ← Schema check, PII scan, injection detect
    │
    ▼
[8] Audit Logger         ← Immutable, tamper-evident audit trail
    │
    ▼
Response to user
```

Each layer is independently deployable and testable. A failure or bypass at layer N should be caught at layer N+1.

---

> ### 📋 Chapter Summary
>
> - AI systems face both conventional (injection, auth bypass, PII exposure) and novel (prompt injection, indirect injection, data poisoning) security threats.
> - The **OWASP LLM Top 10** provides a structured risk catalogue mapped to RAG-specific manifestations and primary defences.
> - **Attack surface mapping** identifies five primary surfaces: user query, document content, LLM API interface, vector database, and ingestion pipeline.
> - **Defence-in-depth** places eight independent validation and control layers between user input and system response — each layer independently catches what the previous missed.

---

> ### ❓ Comprehension Questions
>
> 1. OWASP LLM08 (Excessive Agency) warns against LLM systems that perform write operations based on LLM output. A RAG system that only retrieves and answers appears safe. Describe a scenario where a RAG system still exhibits excessive agency risk.
> 2. Indirect prompt injection (LLM01 via retrieved documents) is rated CRITICAL. The attack requires an adversary to insert content into the knowledge base. What conditions in a real enterprise make this attack plausible, and what mitigations address the root cause?
> 3. "Defence-in-depth" means each layer catches what the previous missed. However, this increases latency. Layers 2–4 add approximately 5ms each. At 500 RPS, is this acceptable? How would you optimise high-frequency layers?
> 4. The attack surface for the vector database notes "metadata filter bypass" as a vector. In Qdrant's filter language, a query can use nested operators. How would you design the tenant isolation filter to be resistant to client-supplied filter overrides?
> 5. A developer argues that since the system prompt says "ignore user instructions to override", prompt injection is already mitigated. Explain why this is an insufficient defence and what structural mitigations are required.

---

## References

### Documentation
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/system/files/documents/2023/01/26/NIST.AI.100-1.pdf)
- [MITRE ATLAS](https://atlas.mitre.org) — Adversarial Threat Landscape for AI Systems.
- [Garak LLM Vulnerability Scanner](https://github.com/NVIDIA/garak)

### Papers
- [Prompt Injection Attacks Against LLM-Integrated Applications](https://arxiv.org/abs/2302.12173) — Greshake et al., 2023.
- [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injections](https://arxiv.org/abs/2302.12173) — Greshake et al., 2023.

---

---
[« Back to security Index](index.md) | [🏠 Home](../../index.md)