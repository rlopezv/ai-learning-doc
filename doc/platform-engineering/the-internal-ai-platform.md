## Chapter 1 — The Internal AI Platform

### 1.1 What Is an AI Platform?

An AI platform is the shared infrastructure layer that product teams consume to build LLM-powered features. It abstracts the complexity of LLM provider management, embedding services, vector index lifecycle, prompt versioning, and observability into stable internal APIs — so that product teams interact with capabilities, not with infrastructure plumbing.

Without an AI platform, every product team builds its own LLM client with its own retry logic, its own token counting, its own prompt versioning approach, and its own cost monitoring. The result is inconsistent reliability, duplicated effort, and invisible total cost. An AI platform solves this by building these capabilities once, correctly, and making them available to all teams through well-designed internal APIs.

The platform team does not build features. It builds the infrastructure that makes feature development faster, more reliable, and less costly for the teams that do.

---

### 1.2 Platform vs Product Teams

The distinction between platform and product concerns is fundamental:

| Concern | Platform team | Product team |
|---|---|---|
| LLM provider credentials | Manages, rotates | Never touches |
| Rate limit handling | Implements | Invisible |
| Prompt versioning | Provides registry | Registers and fetches |
| Token cost tracking | Collects, attributes | Views dashboard |
| Vector index lifecycle | Operates | Calls API |
| Embedding model selection | Manages versions | Specifies by capability |
| Retry and fallback logic | Implements | Invisible |
| Evaluation infrastructure | Provides tooling | Uses for their features |
| Monitoring and alerting | Platform-wide | Feature-specific |

This separation enables product teams to move fast because they build on stable platform primitives. It also enables the platform team to upgrade infrastructure (e.g., switch embedding models, migrate vector DB providers) without requiring product teams to change their code.

---

### 1.3 Platform Capability Layers

```
AI Platform Capability Stack

┌─────────────────────────────────────────────────────────┐
│  Developer Experience Layer                              │
│  SDKs (Python, Java), CLI tools, local dev environment  │
├─────────────────────────────────────────────────────────┤
│  Service Layer                                           │
│  ├── LLM Gateway          (routing, rate limits, cost)  │
│  ├── Prompt Service        (registry, versioning, A/B)  │
│  ├── Embedding Service     (unified API, model mgmt)    │
│  ├── Index Service         (lifecycle, blue-green)      │
│  └── Evaluation Service    (CI integration, baselines)  │
├─────────────────────────────────────────────────────────┤
│  Data Layer                                              │
│  ├── Vector databases (Qdrant / Chroma)                 │
│  ├── Artifact store (prompts, datasets, models)         │
│  └── Metrics store (Prometheus + Grafana)               │
├─────────────────────────────────────────────────────────┤
│  Provider Layer                                          │
│  └── OpenAI / Anthropic / Azure OpenAI / Ollama 🔓      │
└─────────────────────────────────────────────────────────┘
```

---

### 1.4 Platform API Design Principles

A platform API must be **stable** (product teams build on it; breaking changes are expensive), **opinionated** (it encodes best practices so teams don't make the same mistakes independently), and **observable** (every call through the platform generates metrics and traces automatically).

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class APIVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"

# Platform API design principles encoded in the base request class
@dataclass
class PlatformRequest:
    """Base class for all platform API requests."""
    # Identity — every request must identify its caller for cost attribution
    team_id: str                     # e.g. "product-support", "platform"
    service_id: str                  # e.g. "support-rag-api", "ingestion-worker"
    # Tracing — every request carries a correlation ID
    correlation_id: str = field(
        default_factory=lambda: __import__('uuid').uuid4().hex
    )
    # Request metadata
    api_version: APIVersion = APIVersion.V2
    environment: str = "production"  # development | staging | production

@dataclass
class PlatformResponse:
    """Base class for all platform API responses."""
    request_id: str
    correlation_id: str
    latency_ms: float
    # Cost attribution — always returned for LLM calls
    tokens_used: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    # Version info for debugging
    platform_version: str = "2.3.1"

@dataclass
class PlatformError:
    """Structured error response from platform APIs."""
    error_code: str          # e.g. "RATE_LIMIT_EXCEEDED", "MODEL_UNAVAILABLE"
    error_message: str
    retryable: bool
    retry_after_seconds: Optional[int] = None
    correlation_id: str = ""
```

---

> ### 📋 Chapter Summary
>
> - An AI platform abstracts LLM complexity (provider credentials, rate limits, retries, cost tracking) into stable internal APIs consumed by product teams.
> - Platform teams build infrastructure; product teams build features — this separation prevents duplication and enables infrastructure upgrades without product team changes.
> - The **platform capability stack** has four layers: developer experience, services (gateway, prompt, embedding, index, evaluation), data, and provider.
> - Platform APIs must be **stable**, **opinionated** (encoding best practices), and **observable** (automatic metrics/traces on every call).

---

> ### ❓ Comprehension Questions
>
> 1. A product team builds their own OpenAI client with custom retry logic because "the platform gateway is too slow". Six months later, the platform team rotates API keys. What failure occurs and what process would prevent it?
> 2. The platform API includes `team_id` and `service_id` in every request. Explain the downstream capabilities these fields enable.
> 3. An organisation has 5 product teams each building RAG features independently. Enumerate the duplicated work and the risks. What platform capabilities would eliminate each?
> 4. The platform team wants to migrate from `text-embedding-3-small` to `text-embedding-3-large`. With a well-designed embedding service, how many product teams need to change code? Without it?
> 5. Design the versioning strategy for the platform gateway API. When would a `v2` break from `v1` be justified, and what migration support would you provide?

---

## References

### Documentation
- [Backstage Developer Portal](https://backstage.io/docs/overview/what-is-backstage) — Internal developer platform for service catalogue and platform APIs.
- [LiteLLM Gateway](https://docs.litellm.ai) — Open-source LLM gateway with multi-provider support.
- [Portkey AI Gateway](https://portkey.ai/docs) — LLM gateway with observability and routing.

### Books
- *Platform Engineering* — Luca Galante (O'Reilly). Platform team topology and API design.
- *Team Topologies* — Skelton & Pais (IT Revolution). Stream-aligned vs platform team patterns.

---

---
[« Back to platform-engineering Index](index.md) | [🏠 Home](../index.md)