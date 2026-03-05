# Part XI — AI Platform Engineering

---

> **Navigation**
> [← Part X — Testing LLM Systems](part_10_testing.md) | [→ Part XII — Infrastructure](part_12_infrastructure.md)

---

## Contents

- [Chapter 1 — The Internal AI Platform](#chapter-1--the-internal-ai-platform)
  - [1.1 What Is an AI Platform?](#11-what-is-an-ai-platform)
  - [1.2 Platform vs Product Teams](#12-platform-vs-product-teams)
  - [1.3 Platform Capability Layers](#13-platform-capability-layers)
  - [1.4 Platform API Design Principles](#14-platform-api-design-principles)
- [Chapter 2 — LLM Gateway Design](#chapter-2--llm-gateway-design)
  - [2.1 Why a Gateway?](#21-why-a-gateway)
  - [2.2 Gateway Core Capabilities](#22-gateway-core-capabilities)
  - [2.3 Implementing a Production Gateway](#23-implementing-a-production-gateway)
  - [2.4 Rate Limiting and Quota Management](#24-rate-limiting-and-quota-management)
  - [2.5 Multi-Provider Routing and Fallback](#25-multi-provider-routing-and-fallback)
- [Chapter 3 — Prompt Management Platform](#chapter-3--prompt-management-platform)
  - [3.1 Centralised Prompt Service](#31-centralised-prompt-service)
  - [3.2 Prompt Serving with Caching](#32-prompt-serving-with-caching)
  - [3.3 Prompt Governance Workflows](#33-prompt-governance-workflows)
- [Chapter 4 — Embedding and Index Platform 🧪](#chapter-4--embedding-and-index-platform-)
  - [4.1 Shared Embedding Service](#41-shared-embedding-service)
  - [4.2 Index Lifecycle Service](#42-index-lifecycle-service)
  - [4.3 Multi-Tenant Index Isolation](#43-multi-tenant-index-isolation)
  - [4.4 Java Integration Patterns](#44-java-integration-patterns)
  - [🧪 Hands-on Lab: Minimal AI Platform](#-hands-on-lab-minimal-ai-platform)
- [Chapter 5 — Platform Observability and Cost Management](#chapter-5--platform-observability-and-cost-management)
  - [5.1 Platform Metrics Taxonomy](#51-platform-metrics-taxonomy)
  - [5.2 Cost Attribution and Budgets](#52-cost-attribution-and-budgets)
  - [5.3 Token Usage Tracking](#53-token-usage-tracking)
  - [5.4 Platform SLOs](#54-platform-slos)

---

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

## Chapter 2 — LLM Gateway Design

### 2.1 Why a Gateway?

A gateway sits between all product service LLM calls and the external LLM providers. It provides:

- **Unified authentication** — API keys stored in one place, rotated in one place
- **Rate limiting** — per-team quotas enforced centrally before hitting provider limits
- **Retry and fallback** — exponential backoff, provider fallback (OpenAI → Anthropic → Ollama)
- **Cost tracking** — token usage attributed to team and service on every call
- **Request/response logging** — audit trail for compliance and debugging
- **Caching** — semantic caching for identical or near-identical queries
- **Model version pinning** — "gpt-4o" always resolves to the same model until the platform team decides to upgrade

Without a gateway, each of these must be implemented by every product team independently — and usually is not.

---

### 2.2 Gateway Core Capabilities

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time

class LLMProvider(str, Enum):
    OPENAI     = "openai"
    ANTHROPIC  = "anthropic"
    AZURE      = "azure_openai"
    OLLAMA     = "ollama"      # 🔓 On-premise

@dataclass
class GatewayRequest:
    messages: list[dict]
    model_alias: str             # Platform alias, e.g. "default", "powerful", "fast"
    team_id: str
    service_id: str
    correlation_id: str = field(default_factory=lambda: __import__('uuid').uuid4().hex)
    temperature: float = 0.0
    max_tokens: int = 1000
    response_format: Optional[dict] = None
    stream: bool = False

@dataclass
class GatewayResponse:
    content: str
    model_used: str
    provider_used: LLMProvider
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    correlation_id: str
    cached: bool = False

@dataclass
class ModelAliasConfig:
    """Maps a platform model alias to provider-specific models."""
    alias: str
    primary_provider: LLMProvider
    primary_model: str
    fallback_provider: Optional[LLMProvider] = None
    fallback_model: Optional[str] = None
    max_tokens_limit: int = 4096
    # Cost per 1M tokens (input / output)
    cost_per_1m_input: float = 0.15
    cost_per_1m_output: float = 0.60

MODEL_ALIASES = {
    "default":   ModelAliasConfig("default",   LLMProvider.OPENAI, "gpt-4o-mini",
                                   LLMProvider.OLLAMA, "llama3",
                                   cost_per_1m_input=0.15, cost_per_1m_output=0.60),
    "powerful":  ModelAliasConfig("powerful",  LLMProvider.OPENAI, "gpt-4o",
                                   LLMProvider.ANTHROPIC, "claude-3-5-sonnet-20241022",
                                   cost_per_1m_input=2.50, cost_per_1m_output=10.00),
    "fast":      ModelAliasConfig("fast",      LLMProvider.OPENAI, "gpt-4o-mini",
                                   cost_per_1m_input=0.15, cost_per_1m_output=0.60),
    "local":     ModelAliasConfig("local",     LLMProvider.OLLAMA, "llama3",
                                   cost_per_1m_input=0.0, cost_per_1m_output=0.0),
}
```

---

### 2.3 Implementing a Production Gateway

```python
import time
import logging
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError
import anthropic

logger = logging.getLogger("llm-gateway")

class LLMGateway:
    """
    Production LLM gateway with retry, fallback, cost tracking, and metrics.
    Deployed as a shared internal service — all product teams call this.
    """
    def __init__(
        self,
        secret_manager,
        cost_tracker: "CostTracker",
        rate_limiter: "RateLimiter",
        cache: "SemanticCache" = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0
    ):
        self.secrets = secret_manager
        self.cost_tracker = cost_tracker
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def complete(self, request: GatewayRequest) -> GatewayResponse:
        t0 = time.perf_counter()

        # 1. Rate limit check
        allowed, retry_after = self.rate_limiter.check(request.team_id, request.service_id)
        if not allowed:
            raise RateLimitExceededException(
                f"Team {request.team_id} rate limit exceeded",
                retry_after_seconds=retry_after
            )

        # 2. Cache lookup (semantic cache on message content)
        if self.cache:
            cached = self.cache.lookup(request.messages)
            if cached:
                logger.info(f"Cache hit [{request.correlation_id}]")
                return GatewayResponse(
                    content=cached["content"],
                    model_used=cached["model"],
                    provider_used=LLMProvider(cached["provider"]),
                    prompt_tokens=cached["prompt_tokens"],
                    completion_tokens=0,
                    total_tokens=cached["prompt_tokens"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    estimated_cost_usd=0.0,
                    correlation_id=request.correlation_id,
                    cached=True
                )

        # 3. Resolve model alias
        alias = MODEL_ALIASES.get(request.model_alias, MODEL_ALIASES["default"])

        # 4. Call primary provider with retries
        response = self._call_with_retry(request, alias, t0)

        # 5. Track cost
        cost = self._estimate_cost(alias, response.prompt_tokens, response.completion_tokens)
        response.estimated_cost_usd = cost
        self.cost_tracker.record(
            team_id=request.team_id,
            service_id=request.service_id,
            model=response.model_used,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=cost,
            correlation_id=request.correlation_id
        )

        # 6. Store in cache
        if self.cache:
            self.cache.store(request.messages, {
                "content": response.content,
                "model": response.model_used,
                "provider": response.provider_used.value,
                "prompt_tokens": response.prompt_tokens,
            })

        return response

    def _call_with_retry(
        self,
        request: GatewayRequest,
        alias: ModelAliasConfig,
        t0: float
    ) -> GatewayResponse:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Try primary provider
                if attempt < 2:
                    return self._call_provider(
                        request, alias.primary_provider, alias.primary_model, t0
                    )
                # On final attempt, try fallback
                elif alias.fallback_provider:
                    logger.warning(
                        f"Falling back to {alias.fallback_provider.value} "
                        f"[{request.correlation_id}]"
                    )
                    return self._call_provider(
                        request, alias.fallback_provider, alias.fallback_model, t0
                    )
            except RateLimitError as e:
                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(f"Rate limit on attempt {attempt+1}, retrying in {delay:.1f}s")
                time.sleep(delay)
                last_error = e
            except APIConnectionError as e:
                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(f"Connection error on attempt {attempt+1}, retrying in {delay:.1f}s")
                time.sleep(delay)
                last_error = e
            except APIStatusError as e:
                if e.status_code >= 500:
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    last_error = e
                else:
                    raise  # 4xx errors are not retried

        raise GatewayException(
            f"All {self.max_retries} attempts failed",
            last_error=last_error
        )

    def _call_provider(
        self,
        request: GatewayRequest,
        provider: LLMProvider,
        model: str,
        t0: float
    ) -> GatewayResponse:
        t_call = time.perf_counter()

        if provider == LLMProvider.OPENAI or provider == LLMProvider.AZURE:
            client = OpenAI(api_key=self.secrets.get("OPENAI_API_KEY"))
            params = {
                "model": model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            if request.response_format:
                params["response_format"] = request.response_format
            raw = client.chat.completions.create(**params)
            return GatewayResponse(
                content=raw.choices[0].message.content,
                model_used=raw.model,
                provider_used=provider,
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
                latency_ms=(time.perf_counter() - t0) * 1000,
                estimated_cost_usd=0.0,
                correlation_id=request.correlation_id
            )

        elif provider == LLMProvider.ANTHROPIC:
            client = anthropic.Anthropic(api_key=self.secrets.get("ANTHROPIC_API_KEY"))
            system = next((m["content"] for m in request.messages if m["role"] == "system"), None)
            user_messages = [m for m in request.messages if m["role"] != "system"]
            kwargs = dict(model=model, max_tokens=request.max_tokens, messages=user_messages)
            if system:
                kwargs["system"] = system
            raw = client.messages.create(**kwargs)
            return GatewayResponse(
                content=raw.content[0].text,
                model_used=raw.model,
                provider_used=provider,
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
                latency_ms=(time.perf_counter() - t0) * 1000,
                estimated_cost_usd=0.0,
                correlation_id=request.correlation_id
            )

        elif provider == LLMProvider.OLLAMA:
            # 🔓 On-premise: call Ollama local server
            import httpx
            base_url = self.secrets.get("OLLAMA_BASE_URL") or "http://localhost:11434"
            resp = httpx.post(
                f"{base_url}/api/chat",
                json={"model": model, "messages": request.messages, "stream": False},
                timeout=60.0
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            return GatewayResponse(
                content=content,
                model_used=model,
                provider_used=LLMProvider.OLLAMA,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
                estimated_cost_usd=0.0,
                correlation_id=request.correlation_id
            )

        raise ValueError(f"Unsupported provider: {provider}")

    def _estimate_cost(
        self, alias: ModelAliasConfig, prompt_tokens: int, completion_tokens: int
    ) -> float:
        return (prompt_tokens / 1_000_000 * alias.cost_per_1m_input +
                completion_tokens / 1_000_000 * alias.cost_per_1m_output)


class RateLimitExceededException(Exception):
    def __init__(self, message, retry_after_seconds=None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds

class GatewayException(Exception):
    def __init__(self, message, last_error=None):
        super().__init__(message)
        self.last_error = last_error
```

---

### 2.4 Rate Limiting and Quota Management

```python
import time
import threading
from dataclasses import dataclass, field
from collections import deque

@dataclass
class QuotaConfig:
    """Per-team quota configuration."""
    team_id: str
    requests_per_minute: int = 100
    tokens_per_day: int = 5_000_000
    tokens_per_minute: int = 50_000
    max_concurrent_requests: int = 20
    burst_multiplier: float = 1.5   # Allow short bursts above rpm limit

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter per team.
    Thread-safe for concurrent gateway requests.
    """
    def __init__(self, quotas: dict[str, QuotaConfig]):
        self.quotas = quotas
        self._windows: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, team_id: str, service_id: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        quota = self.quotas.get(team_id)
        if not quota:
            # Unknown team — apply default restrictive quota
            quota = QuotaConfig(team_id, requests_per_minute=10)

        with self._lock:
            now = time.time()
            window_key = f"{team_id}:rpm"
            if window_key not in self._windows:
                self._windows[window_key] = deque()

            window = self._windows[window_key]
            # Evict entries older than 60 seconds
            while window and window[0] < now - 60:
                window.popleft()

            burst_limit = int(quota.requests_per_minute * quota.burst_multiplier)
            if len(window) >= burst_limit:
                # Calculate when the oldest entry expires
                oldest = window[0]
                retry_after = int(60 - (now - oldest)) + 1
                logger.warning(
                    f"Rate limit hit: team={team_id} service={service_id} "
                    f"requests_in_window={len(window)} limit={burst_limit}"
                )
                return False, retry_after

            window.append(now)
            return True, 0

class TokenBudgetTracker:
    """Tracks token usage against daily budgets per team."""
    def __init__(self, quotas: dict[str, QuotaConfig]):
        self.quotas = quotas
        self._daily_usage: dict[str, int] = {}
        self._day_key: str = ""
        self._lock = threading.Lock()

    def check_and_record(self, team_id: str, estimated_tokens: int) -> bool:
        """Returns False if daily budget would be exceeded."""
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")

        with self._lock:
            if today != self._day_key:
                self._daily_usage.clear()
                self._day_key = today

            quota = self.quotas.get(team_id)
            if not quota:
                return True

            current = self._daily_usage.get(team_id, 0)
            if current + estimated_tokens > quota.tokens_per_day:
                logger.error(
                    f"Daily token budget exceeded: team={team_id} "
                    f"used={current:,} limit={quota.tokens_per_day:,}"
                )
                return False

            self._daily_usage[team_id] = current + estimated_tokens
            return True

    def get_usage_summary(self) -> dict[str, dict]:
        with self._lock:
            return {
                team: {
                    "used_today": used,
                    "budget": self.quotas.get(team, QuotaConfig(team)).tokens_per_day,
                    "pct_used": used / self.quotas.get(team, QuotaConfig(team)).tokens_per_day * 100
                }
                for team, used in self._daily_usage.items()
            }
```

---

### 2.5 Multi-Provider Routing and Fallback

Beyond per-request fallback, the gateway supports routing policies that direct different workloads to different providers.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class RoutingStrategy(str, Enum):
    PRIMARY_WITH_FALLBACK = "primary_with_fallback"
    COST_OPTIMISED        = "cost_optimised"     # Cheapest model that meets requirements
    LOAD_BALANCED         = "load_balanced"       # Round-robin across providers
    LATENCY_OPTIMISED     = "latency_optimised"   # Fastest provider based on recent P50

@dataclass
class RoutingRule:
    """Route requests matching a predicate to a specific model alias."""
    name: str
    predicate: Callable[[GatewayRequest], bool]
    model_alias: str
    priority: int = 0  # Higher priority rules evaluated first

class RequestRouter:
    def __init__(self, rules: list[RoutingRule], default_alias: str = "default"):
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self.default_alias = default_alias

    def resolve_alias(self, request: GatewayRequest) -> str:
        """Apply routing rules in priority order. Return first match."""
        for rule in self.rules:
            try:
                if rule.predicate(request):
                    logger.debug(
                        f"Routing rule matched: {rule.name} → {rule.model_alias} "
                        f"[{request.correlation_id}]"
                    )
                    return rule.model_alias
            except Exception:
                continue
        return self.default_alias

# Example routing configuration
ROUTING_RULES = [
    RoutingRule(
        name="high_priority_teams_get_powerful_model",
        predicate=lambda r: r.team_id in ("legal", "finance"),
        model_alias="powerful",
        priority=100
    ),
    RoutingRule(
        name="long_contexts_use_powerful_model",
        predicate=lambda r: sum(len(m.get("content", "")) for m in r.messages) > 10000,
        model_alias="powerful",
        priority=90
    ),
    RoutingRule(
        name="development_uses_local",
        predicate=lambda r: r.environment == "development",
        model_alias="local",     # 🔓 Ollama for dev
        priority=80
    ),
    RoutingRule(
        name="classification_tasks_use_fast",
        predicate=lambda r: r.max_tokens <= 50,
        model_alias="fast",
        priority=50
    ),
]
```

---

> ### 📋 Chapter Summary
>
> - The **LLM gateway** centralises authentication, rate limiting, retry/fallback, cost tracking, caching, and model version pinning for all product teams.
> - **Model aliases** (`default`, `powerful`, `fast`, `local`) abstract provider-specific model names — product teams never embed model name strings in their code.
> - **Sliding window rate limiting** enforces per-team quotas with burst allowances before provider rate limits are hit.
> - **Routing rules** direct workloads to appropriate models based on team priority, context length, environment, or task type.

---

> ### ❓ Comprehension Questions
>
> 1. A product team calls `model_alias="gpt-4o"` directly, bypassing the model alias system. The platform team migrates to `gpt-4o-2024-11-20`. What happens to the product team's service, and how does the alias system prevent this?
> 2. The gateway's fallback logic retries twice on the primary provider then falls back. A developer argues that fallback should happen immediately on any error. Why is retrying the primary first the correct behaviour for transient errors?
> 3. Token budget tracking resets daily at midnight UTC. A batch job runs at 23:45 and exhausts the budget. A real-time user-facing service then fails at 23:50. How would you design the quota system to protect interactive workloads from batch workloads?
> 4. The semantic cache stores responses keyed by message content similarity. What is the risk of caching responses for queries that include time-sensitive information (e.g., "What is today's exchange rate?"), and how would you mitigate it?
> 5. The routing rule `development_uses_local` redirects all development traffic to Ollama. A developer is testing a feature that depends on GPT-4o's structured output mode, which Ollama does not support. How would you handle this exception in the routing system?

---

## References

### Documentation
- [LiteLLM](https://docs.litellm.ai) — Open-source LLM gateway with 100+ provider support.
- [Portkey AI Gateway](https://portkey.ai/docs) — Production LLM gateway with observability.
- [Kong AI Gateway](https://docs.konghq.com/hub/kong-inc/ai-proxy/) — Enterprise API gateway with LLM plugin.
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md) — On-premise LLM serving API.

### Papers
- [Efficiently Serving LLMs](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023. vLLM scheduling and throughput optimisation.

---

## Chapter 3 — Prompt Management Platform

### 3.1 Centralised Prompt Service

The prompt service provides a centralised API for all prompt template operations. Product teams register, fetch, and A/B test prompts through this API — never by reading files directly.

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import json
import hashlib
from datetime import datetime
from pathlib import Path

app = FastAPI(
    title="Prompt Management Service",
    version="2.0.0",
    description="Centralised prompt registry for all AI product teams"
)

# ── API schemas ──────────────────────────────────────────────────────────────
class PromptRegistrationRequest(BaseModel):
    template_id: str = Field(..., description="Unique identifier for this prompt template")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version")
    description: str
    system_template: str
    user_template: str
    variables: list[str]
    optional_variables: list[str] = []
    defaults: dict = {}
    author: str
    team_id: str
    tags: list[str] = []
    change_summary: str = Field(..., description="Human-readable description of changes")

class PromptFetchRequest(BaseModel):
    template_id: str
    version: str = "production"   # "production" | "staging" | specific version
    team_id: str
    service_id: str

class PromptRenderRequest(BaseModel):
    template_id: str
    version: str = "production"
    variables: dict
    team_id: str
    service_id: str

class PromptVersionInfo(BaseModel):
    template_id: str
    version: str
    stage: str
    description: str
    author: str
    team_id: str
    created_at: str
    content_hash: str
    change_summary: str

# ── Service implementation ────────────────────────────────────────────────────
class PromptStore:
    def __init__(self, base_path: str = "./prompt_store"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, req: PromptRegistrationRequest, stage: str = "development") -> str:
        template_dir = self.base / req.template_id
        template_dir.mkdir(exist_ok=True)

        data = {**req.dict(), "stage": stage, "created_at": datetime.utcnow().isoformat()}
        content = json.dumps(data, indent=2, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        data["content_hash"] = content_hash

        (template_dir / f"{req.version}.json").write_text(json.dumps(data, indent=2))
        return content_hash

    def load(self, template_id: str, version: str = "production") -> dict:
        template_dir = self.base / template_id
        if version in ("production", "staging", "development"):
            # Find latest version with that stage
            candidates = []
            for f in template_dir.glob("*.json"):
                if f.stem == "latest":
                    continue
                try:
                    d = json.loads(f.read_text())
                    if d.get("stage") == version:
                        candidates.append(d)
                except Exception:
                    pass
            if not candidates:
                raise HTTPException(404, f"No {version} version of {template_id}")
            return max(candidates, key=lambda d: d["created_at"])

        version_file = template_dir / f"{version}.json"
        if not version_file.exists():
            raise HTTPException(404, f"{template_id} v{version} not found")
        return json.loads(version_file.read_text())

    def set_stage(self, template_id: str, version: str, stage: str):
        data = self.load(template_id, version)
        data["stage"] = stage
        data["stage_updated_at"] = datetime.utcnow().isoformat()
        (self.base / template_id / f"{version}.json").write_text(json.dumps(data, indent=2))

store = PromptStore()

# ── REST endpoints ─────────────────────────────────────────────────────────
@app.post("/v2/prompts", response_model=dict, tags=["prompts"])
def register_prompt(req: PromptRegistrationRequest):
    """Register a new prompt version in the registry."""
    content_hash = store.save(req)
    return {"template_id": req.template_id, "version": req.version,
            "content_hash": content_hash, "stage": "development"}

@app.get("/v2/prompts/{template_id}/versions", response_model=list, tags=["prompts"])
def list_versions(template_id: str):
    """List all registered versions of a prompt template."""
    template_dir = store.base / template_id
    if not template_dir.exists():
        raise HTTPException(404, f"Template {template_id} not found")
    versions = []
    for f in sorted(template_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            versions.append({"version": d["version"], "stage": d.get("stage", "development"),
                             "created_at": d.get("created_at"), "author": d.get("author")})
        except Exception:
            pass
    return versions

@app.post("/v2/prompts/render", response_model=dict, tags=["prompts"])
def render_prompt(req: PromptRenderRequest):
    """Fetch and render a prompt template with variables substituted."""
    data = store.load(req.template_id, req.version)
    try:
        system = data["system_template"].format(
            **{**data.get("defaults", {}), **req.variables}
        )
        user = data["user_template"].format(
            **{**data.get("defaults", {}), **req.variables}
        )
    except KeyError as e:
        raise HTTPException(400, f"Missing required variable: {e}")
    return {
        "template_id": req.template_id,
        "version": data["version"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "content_hash": data.get("content_hash")
    }

@app.post("/v2/prompts/{template_id}/{version}/promote", tags=["governance"])
def promote_prompt(template_id: str, version: str, target_stage: str):
    """Promote a prompt version to a new stage (staging → production)."""
    allowed_transitions = {
        "development": ["staging"],
        "staging": ["production"],
        "production": []
    }
    current = store.load(template_id, version)
    current_stage = current.get("stage", "development")
    if target_stage not in allowed_transitions.get(current_stage, []):
        raise HTTPException(
            400,
            f"Cannot promote from {current_stage} to {target_stage}. "
            f"Allowed: {allowed_transitions.get(current_stage, [])}"
        )
    store.set_stage(template_id, version, target_stage)
    return {"template_id": template_id, "version": version,
            "previous_stage": current_stage, "new_stage": target_stage}
```

---

### 3.2 Prompt Serving with Caching

```python
import functools
from typing import Optional

class PromptCache:
    """
    In-memory LRU cache for frequently accessed prompts.
    Prevents repeated disk/DB reads for high-traffic templates.
    """
    def __init__(self, maxsize: int = 200):
        self._cache: dict = {}
        self._maxsize = maxsize
        self._access_order: list = []

    def get(self, key: str) -> Optional[dict]:
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: dict):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._maxsize:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        self._cache[key] = value
        self._access_order.append(key)

    def invalidate(self, template_id: str):
        """Invalidate all cached versions of a template (on promotion)."""
        keys_to_remove = [k for k in self._cache if k.startswith(f"{template_id}:")]
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

class CachingPromptClient:
    """
    Client-side wrapper that product services use to fetch prompts.
    Handles caching, fallback, and telemetry transparently.
    """
    def __init__(self, prompt_service_url: str, cache: PromptCache):
        self.url = prompt_service_url
        self.cache = cache

    def get_messages(
        self,
        template_id: str,
        variables: dict,
        version: str = "production",
        team_id: str = "unknown",
        service_id: str = "unknown"
    ) -> list[dict]:
        import httpx
        cache_key = f"{template_id}:{version}"
        cached_template = self.cache.get(cache_key)

        if not cached_template:
            resp = httpx.post(
                f"{self.url}/v2/prompts/render",
                json={
                    "template_id": template_id,
                    "version": version,
                    "variables": variables,
                    "team_id": team_id,
                    "service_id": service_id
                },
                timeout=5.0
            )
            resp.raise_for_status()
            return resp.json()["messages"]

        # Render from cached template
        try:
            system = cached_template["system_template"].format(
                **{**cached_template.get("defaults", {}), **variables}
            )
            user = cached_template["user_template"].format(
                **{**cached_template.get("defaults", {}), **variables}
            )
            return [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        except KeyError:
            # Cache miss on variable — re-fetch
            self.cache.invalidate(template_id)
            return self.get_messages(template_id, variables, version, team_id, service_id)
```

---

### 3.3 Prompt Governance Workflows

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class ApprovalStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"
    WITHDRAWN = "withdrawn"

@dataclass
class PromotionRequest:
    request_id: str
    template_id: str
    version: str
    from_stage: str
    to_stage: str
    requester: str
    team_id: str
    change_summary: str
    eval_recall_at_5: Optional[float] = None
    prompt_test_pass_rate: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: Optional[str] = None
    review_comment: Optional[str] = None
    reviewed_at: Optional[str] = None

class PromotionWorkflow:
    """
    Governs prompt promotion with configurable approval requirements.
    Staging → Production requires: eval gate + tech lead approval.
    Development → Staging requires: eval gate pass only (auto-approved).
    """
    def __init__(self, store: PromptStore):
        self.store = store
        self.pending_requests: dict[str, PromotionRequest] = {}

    def request_promotion(self, req: PromotionRequest) -> str:
        # Auto-approve dev → staging if eval gate passes
        if req.from_stage == "development" and req.to_stage == "staging":
            if req.prompt_test_pass_rate == 1.0:
                self._execute_promotion(req)
                req.status = ApprovalStatus.APPROVED
                req.reviewer = "auto-approval"
                return f"Auto-approved: {req.template_id} v{req.version} → staging"

        # Staging → production requires human approval
        self.pending_requests[req.request_id] = req
        return f"Promotion request {req.request_id} created, awaiting review"

    def approve(self, request_id: str, reviewer: str, comment: str = "") -> str:
        req = self.pending_requests.get(request_id)
        if not req:
            raise ValueError(f"Request {request_id} not found")

        # Validate eval gate before approving
        if req.eval_recall_at_5 and req.eval_recall_at_5 < 0.80:
            raise ValueError(
                f"Cannot approve: Recall@5 {req.eval_recall_at_5:.2%} below threshold 80%"
            )
        if req.prompt_test_pass_rate is not None and req.prompt_test_pass_rate < 1.0:
            raise ValueError(
                f"Cannot approve: prompt tests {req.prompt_test_pass_rate:.0%} — must be 100%"
            )

        self._execute_promotion(req)
        req.status = ApprovalStatus.APPROVED
        req.reviewer = reviewer
        req.review_comment = comment
        req.reviewed_at = datetime.utcnow().isoformat()
        del self.pending_requests[request_id]
        return f"Approved: {req.template_id} v{req.version} → {req.to_stage}"

    def reject(self, request_id: str, reviewer: str, reason: str) -> str:
        req = self.pending_requests.get(request_id)
        if not req:
            raise ValueError(f"Request {request_id} not found")
        req.status = ApprovalStatus.REJECTED
        req.reviewer = reviewer
        req.review_comment = reason
        req.reviewed_at = datetime.utcnow().isoformat()
        del self.pending_requests[request_id]
        return f"Rejected: {req.template_id} v{req.version} — {reason}"

    def _execute_promotion(self, req: PromotionRequest):
        self.store.set_stage(req.template_id, req.version, req.to_stage)
```

---

> ### 📋 Chapter Summary
>
> - The **prompt service** provides a REST API for registering, fetching, rendering, and promoting prompt templates — product teams never manage prompt files directly.
> - **Prompt caching** at the client side prevents high-frequency repeated fetches for production templates.
> - **Governance workflows** enforce approval gates: development → staging auto-approves when tests pass; staging → production requires human tech lead approval.
> - Stage transitions are one-directional (development → staging → production) and validated at every step.

---

> ### ❓ Comprehension Questions
>
> 1. The prompt service returns rendered messages (system + user turn). An alternative design returns the raw template and renders client-side. What are the trade-offs of each approach in terms of caching, observability, and feature consistency?
> 2. A product team registers a prompt version directly to the `production` stage by calling the API with `stage="production"`. What process control prevents this, and how would you implement it?
> 3. The `PromotionWorkflow` auto-approves development → staging when `prompt_test_pass_rate == 1.0`. Should evaluation Recall@5 also be required for this transition? Argue both positions.
> 4. The prompt cache has `maxsize=200`. The system has 50 unique templates each with 3 active versions. Is 200 sufficient? How would cache eviction affect production traffic if popular templates are evicted?
> 5. A product team needs to test a new prompt version against 5% of production traffic. How would you extend the prompt service's `render` endpoint to support A/B routing at the platform level?

---

## References

### Documentation
- [LangSmith Prompt Hub](https://docs.smith.langchain.com/prompt_engineering) — Managed prompt registry.
- [Helicone Prompts](https://docs.helicone.ai/features/prompts) — Prompt versioning with observability.
- [FastAPI Documentation](https://fastapi.tiangolo.com) — REST API framework.

---

## Chapter 4 — Embedding and Index Platform 🧪

### 4.1 Shared Embedding Service

A shared embedding service provides a consistent embedding API for all product teams, abstracting model selection and version management.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time
import hashlib

class EmbeddingModelCapability(str, Enum):
    FAST   = "fast"    # MiniLM / small models — low latency, lower quality
    STANDARD = "standard"  # text-embedding-3-small equivalent
    HIGH_QUALITY = "high_quality"  # text-embedding-3-large equivalent
    LOCAL  = "local"   # 🔓 On-premise model

@dataclass
class EmbeddingModel:
    capability: EmbeddingModelCapability
    model_name: str
    provider: str
    dimensions: int
    max_input_tokens: int
    cost_per_1m_tokens: float

EMBEDDING_MODELS = {
    EmbeddingModelCapability.FAST: EmbeddingModel(
        EmbeddingModelCapability.FAST,
        "text-embedding-3-small", "openai", 1536, 8191, 0.02
    ),
    EmbeddingModelCapability.STANDARD: EmbeddingModel(
        EmbeddingModelCapability.STANDARD,
        "text-embedding-3-small", "openai", 1536, 8191, 0.02
    ),
    EmbeddingModelCapability.HIGH_QUALITY: EmbeddingModel(
        EmbeddingModelCapability.HIGH_QUALITY,
        "text-embedding-3-large", "openai", 3072, 8191, 0.13
    ),
    EmbeddingModelCapability.LOCAL: EmbeddingModel(
        EmbeddingModelCapability.LOCAL,
        "all-MiniLM-L6-v2", "sentence_transformers", 384, 512, 0.0
    ),
}

@dataclass
class EmbeddingRequest:
    texts: list[str]
    capability: EmbeddingModelCapability = EmbeddingModelCapability.STANDARD
    team_id: str = "unknown"
    service_id: str = "unknown"
    # Normalise output vectors to unit length
    normalise: bool = True

@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    model_name: str
    dimensions: int
    tokens_used: int
    latency_ms: float
    estimated_cost_usd: float

class EmbeddingService:
    """
    Shared embedding service. Supports OpenAI and local SentenceTransformers.
    Product teams specify capability, not model name.
    """
    def __init__(self, secret_manager, cost_tracker):
        self.secrets = secret_manager
        self.cost_tracker = cost_tracker
        self._local_model = None  # Lazy-loaded

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_config = EMBEDDING_MODELS[request.capability]
        t0 = time.perf_counter()

        if model_config.provider == "openai":
            embeddings, tokens = self._embed_openai(request.texts, model_config.model_name)
        elif model_config.provider == "sentence_transformers":
            embeddings, tokens = self._embed_local(request.texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {model_config.provider}")

        if request.normalise:
            embeddings = [self._normalise(e) for e in embeddings]

        latency_ms = (time.perf_counter() - t0) * 1000
        cost = tokens / 1_000_000 * model_config.cost_per_1m_tokens

        self.cost_tracker.record(
            team_id=request.team_id, service_id=request.service_id,
            model=model_config.model_name, prompt_tokens=tokens,
            completion_tokens=0, cost_usd=cost, correlation_id=""
        )
        return EmbeddingResponse(
            embeddings=embeddings,
            model_name=model_config.model_name,
            dimensions=model_config.dimensions,
            tokens_used=tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost
        )

    def _embed_openai(self, texts: list[str], model: str) -> tuple[list, int]:
        from openai import OpenAI
        client = OpenAI(api_key=self.secrets.get("OPENAI_API_KEY"))
        response = client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        return embeddings, response.usage.total_tokens

    def _embed_local(self, texts: list[str]) -> tuple[list, int]:
        # 🔓 On-premise: SentenceTransformers, no API call
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self._local_model.encode(texts, normalize_embeddings=True).tolist()
        tokens = sum(len(t.split()) for t in texts)  # Approximate
        return embeddings, tokens

    def _normalise(self, vec: list[float]) -> list[float]:
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec
```

---

### 4.2 Index Lifecycle Service

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json
import time

class IndexBuildStatus(str, Enum):
    QUEUED      = "queued"
    BUILDING    = "building"
    VALIDATING  = "validating"
    READY       = "ready"
    FAILED      = "failed"

@dataclass
class IndexBuildRequest:
    index_id: str
    corpus_id: str
    corpus_version: str
    team_id: str
    embedding_capability: EmbeddingModelCapability = EmbeddingModelCapability.STANDARD
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_recall_threshold: float = 0.80

@dataclass
class IndexBuildJob:
    job_id: str
    request: IndexBuildRequest
    status: IndexBuildStatus
    collection_name: str
    started_at: str
    completed_at: Optional[str] = None
    eval_recall_at_5: Optional[float] = None
    error_message: Optional[str] = None
    vector_count: int = 0

class IndexLifecycleService:
    """
    Manages the full lifecycle of vector indexes:
    Build → Validate → Stage → Promote (blue-green).
    Product teams request builds; the service handles everything else.
    """
    def __init__(self, embedding_service: EmbeddingService, vector_db_client):
        self.embedding = embedding_service
        self.vdb = vector_db_client
        self.jobs: dict[str, IndexBuildJob] = {}
        self._active_collections: dict[str, str] = {}  # index_id → active collection

    def request_build(self, req: IndexBuildRequest) -> str:
        import uuid
        job_id = uuid.uuid4().hex[:10]
        collection_name = f"{req.index_id}_{job_id}"
        job = IndexBuildJob(
            job_id=job_id, request=req,
            status=IndexBuildStatus.QUEUED,
            collection_name=collection_name,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        self.jobs[job_id] = job
        # In production: submit to async job queue (Celery, RQ, Argo)
        # For simplicity: run synchronously
        self._run_build_job(job)
        return job_id

    def get_job_status(self, job_id: str) -> IndexBuildJob:
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")
        return self.jobs[job_id]

    def get_active_collection(self, index_id: str) -> Optional[str]:
        return self._active_collections.get(index_id)

    def _run_build_job(self, job: IndexBuildJob):
        try:
            job.status = IndexBuildStatus.BUILDING
            # 1. Load corpus documents
            documents = self._load_corpus(job.request.corpus_id, job.request.corpus_version)
            # 2. Chunk documents
            chunks = self._chunk_documents(documents, job.request)
            # 3. Embed chunks
            embed_req = EmbeddingRequest(
                texts=[c["content"] for c in chunks],
                capability=job.request.embedding_capability,
                team_id=job.request.team_id,
                service_id="index-lifecycle-service"
            )
            embed_resp = self.embedding.embed(embed_req)
            # 4. Store in new collection
            self._store_chunks(job.collection_name, chunks, embed_resp.embeddings)
            job.vector_count = len(chunks)

            # 5. Validate
            job.status = IndexBuildStatus.VALIDATING
            recall = self._validate_index(job.collection_name, job.request)
            job.eval_recall_at_5 = recall

            if recall < job.request.min_recall_threshold:
                job.status = IndexBuildStatus.FAILED
                job.error_message = (
                    f"Validation failed: Recall@5={recall:.2%} < "
                    f"threshold {job.request.min_recall_threshold:.2%}"
                )
                return

            # 6. Promote (blue-green switch)
            old_collection = self._active_collections.get(job.request.index_id)
            self._active_collections[job.request.index_id] = job.collection_name
            job.status = IndexBuildStatus.READY
            job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Schedule old collection deletion after 7-day retention window
            if old_collection:
                self._schedule_deletion(old_collection, delay_days=7)

        except Exception as e:
            job.status = IndexBuildStatus.FAILED
            job.error_message = str(e)

    def _load_corpus(self, corpus_id, version):
        return []  # Load from corpus store

    def _chunk_documents(self, documents, req: IndexBuildRequest):
        return []  # Chunking pipeline

    def _store_chunks(self, collection_name, chunks, embeddings):
        pass  # Store in vector DB

    def _validate_index(self, collection_name, req: IndexBuildRequest) -> float:
        return 0.85  # Run eval dataset against collection

    def _schedule_deletion(self, collection_name, delay_days):
        pass  # Schedule async deletion job
```

---

### 4.3 Multi-Tenant Index Isolation

```python
@dataclass
class TenantIndexConfig:
    tenant_id: str
    index_id: str
    allowed_corpus_ids: list[str]  # Corpora this tenant can search
    max_vectors: int = 1_000_000
    metadata_filters: dict = None  # Always-applied tenant filter

class MultiTenantIndexRouter:
    """
    Routes queries to the correct tenant collection and applies
    mandatory metadata filters to prevent cross-tenant data access.
    """
    def __init__(self, lifecycle_service: IndexLifecycleService):
        self.lifecycle = lifecycle_service
        self.tenant_configs: dict[str, TenantIndexConfig] = {}

    def register_tenant(self, config: TenantIndexConfig):
        self.tenant_configs[config.tenant_id] = config

    def search(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict = None
    ) -> list[dict]:
        config = self.tenant_configs.get(tenant_id)
        if not config:
            raise PermissionError(f"Unknown tenant: {tenant_id}")

        collection = self.lifecycle.get_active_collection(config.index_id)
        if not collection:
            raise RuntimeError(f"No active index for tenant {tenant_id}")

        # Merge tenant mandatory filter with query filter
        effective_filter = {**(config.metadata_filters or {}), **(metadata_filter or {})}
        # In production: pass effective_filter to vector DB where clause
        return []  # Vector DB query with filter
```

---

### 4.4 Java Integration Patterns

```java
// Java client for the embedding service and index lifecycle API
@Service
public class PlatformEmbeddingClient {

    private final WebClient webClient;
    private final String embeddingServiceUrl;

    @Value("${platform.team-id}")
    private String teamId;

    @Value("${platform.service-id}")
    private String serviceId;

    public List<List<Double>> embed(List<String> texts, String capability) {
        var request = Map.of(
            "texts", texts,
            "capability", capability,
            "team_id", teamId,
            "service_id", serviceId
        );

        return webClient.post()
            .uri(embeddingServiceUrl + "/v2/embeddings")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(EmbeddingResponse.class)
            .map(EmbeddingResponse::embeddings)
            .block(Duration.ofSeconds(30));
    }

    public record EmbeddingResponse(
        List<List<Double>> embeddings,
        String modelName,
        int dimensions,
        int tokensUsed
    ) {}
}

// Using the platform gateway from Java
@Service
public class PlatformLLMClient {

    private final WebClient gatewayClient;

    @Value("${platform.team-id}")
    private String teamId;

    public String complete(List<Message> messages, String modelAlias) {
        var request = Map.of(
            "messages", messages.stream()
                .map(m -> Map.of("role", m.role(), "content", m.content()))
                .toList(),
            "model_alias", modelAlias,
            "team_id", teamId,
            "service_id", "java-rag-service"
        );

        return gatewayClient.post()
            .uri("/v2/complete")
            .bodyValue(request)
            .retrieve()
            .onStatus(HttpStatusCode::is4xxClientError, response ->
                response.bodyToMono(GatewayError.class)
                    .flatMap(err -> Mono.error(new LLMGatewayException(err.message())))
            )
            .bodyToMono(GatewayResponse.class)
            .map(GatewayResponse::content)
            .retryWhen(Retry.backoff(3, Duration.ofSeconds(1))
                .filter(e -> e instanceof WebClientResponseException.ServiceUnavailable))
            .block(Duration.ofSeconds(60));
    }
}
```

---

### 🧪 Hands-on Lab: Minimal AI Platform

**Objective:** Assemble a minimal but functional AI platform using the components from this part. Run a RAG query routed through the gateway, rendered by the prompt service, and retrieved from the index service.

**Prerequisites:** `openai`, `fastapi`, `uvicorn`, `httpx`, `sentence-transformers`

```python
#!/usr/bin/env python3
"""
minimal_platform.py — Runs a self-contained AI platform in a single process.
Demonstrates: Gateway → Prompt Service → Embedding Service → RAG pipeline.
"""

import json
import time
import hashlib
from pathlib import Path
from openai import OpenAI

oai_client = OpenAI()

# ── 1. Minimal gateway (in-process) ────────────────────────────────
class MinimalGateway:
    def __init__(self):
        self.call_log = []

    def complete(self, messages: list[dict], model_alias: str,
                 team_id: str, service_id: str) -> dict:
        model_map = {"default": "gpt-4o-mini", "powerful": "gpt-4o", "fast": "gpt-4o-mini"}
        model = model_map.get(model_alias, "gpt-4o-mini")
        t0 = time.perf_counter()
        resp = oai_client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=300
        )
        latency = (time.perf_counter() - t0) * 1000
        self.call_log.append({
            "team_id": team_id, "service_id": service_id, "model": model,
            "tokens": resp.usage.total_tokens, "latency_ms": round(latency, 1)
        })
        return {"content": resp.choices[0].message.content,
                "model": model, "tokens": resp.usage.total_tokens}

# ── 2. Minimal prompt service (in-process) ─────────────────────────
PROMPT_REGISTRY = {}

def register_prompt(template_id: str, version: str, system: str, user: str,
                    variables: list, defaults: dict, description: str):
    PROMPT_REGISTRY[f"{template_id}:production"] = {
        "template_id": template_id, "version": version,
        "system_template": system, "user_template": user,
        "variables": variables, "defaults": defaults, "description": description
    }
    print(f"  ✓ Registered prompt: {template_id} v{version}")

def render_prompt(template_id: str, variables: dict) -> list[dict]:
    tmpl = PROMPT_REGISTRY.get(f"{template_id}:production")
    if not tmpl:
        raise KeyError(f"Prompt {template_id} not in registry")
    merged = {**tmpl["defaults"], **variables}
    return [
        {"role": "system", "content": tmpl["system_template"].format(**merged)},
        {"role": "user",   "content": tmpl["user_template"].format(**merged)},
    ]

# ── 3. Minimal embedding + index service (in-process) ──────────────
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX = {"docs": [], "embeddings": []}

def ingest_documents(documents: list[dict]):
    texts = [d["content"] for d in documents]
    embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()
    INDEX["docs"].extend(documents)
    INDEX["embeddings"].extend(embeddings)
    print(f"  ✓ Indexed {len(documents)} documents ({len(INDEX['docs'])} total)")

def search_index(query: str, top_k: int = 5) -> list[dict]:
    if not INDEX["docs"]:
        return []
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, INDEX["embeddings"])[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [{"id": INDEX["docs"][i]["id"],
             "content": INDEX["docs"][i]["content"],
             "score": float(sims[i])} for i in top_idx]

# ── 4. Assemble and run ────────────────────────────────────────────
gateway = MinimalGateway()

print("\n" + "=" * 55)
print("  Minimal AI Platform — Setup")
print("=" * 55)

# Register the RAG prompt
register_prompt(
    template_id="support_rag",
    version="2.0.0",
    description="Support RAG QA prompt",
    system="You are a support assistant for {org}. Answer using only the context below. "
           "Cite sources [N]. If not found, say 'I cannot find this information.' Tone: {tone}.",
    user="Context:\n{context}\n\nQuestion: {question}",
    variables=["context", "question"],
    defaults={"org": "Acme Corp", "tone": "professional"}
)

# Ingest knowledge base
ingest_documents([
    {"id": "d1", "content": "Enterprise plan: 90-day return window. Standard plan: 30 days."},
    {"id": "d2", "content": "Professional plan costs $150/month. Includes 25 users and priority support."},
    {"id": "d3", "content": "API authentication uses OAuth 2.0. Access tokens expire after 3600 seconds."},
    {"id": "d4", "content": "Refunds are processed within 5-7 business days for all payment methods."},
    {"id": "d5", "content": "Enterprise plans include dedicated support manager and 99.99% SLA."},
])

# ── 5. Run queries through the platform ────────────────────────────
print("\n" + "=" * 55)
print("  Platform RAG Queries")
print("=" * 55)

TEST_QUERIES = [
    ("product-support", "How long can enterprise customers return items?"),
    ("product-billing",  "What does the Professional plan cost?"),
    ("product-api",      "How long do API tokens last?"),
    ("product-billing",  "What is the process for government procurement?"),
]

for team_id, query in TEST_QUERIES:
    print(f"\n  [{team_id}] Q: {query}")

    # Retrieve
    retrieved = search_index(query, top_k=3)
    context = "\n".join([f"[{i+1}] {r['content']}" for i, r in enumerate(retrieved)])

    # Render prompt via prompt service
    messages = render_prompt("support_rag", {"context": context, "question": query})

    # Call LLM via gateway
    response = gateway.complete(messages, "default", team_id=team_id, service_id="rag-app")
    print(f"     A: {response['content'][:100]}")
    print(f"     [model={response['model']} tokens={response['tokens']}]")

# ── 6. Cost report ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Platform Usage Report")
print("=" * 55)
by_team = {}
total_tokens = 0
for log in gateway.call_log:
    t = log["team_id"]
    by_team[t] = by_team.get(t, 0) + log["tokens"]
    total_tokens += log["tokens"]
for team, tokens in sorted(by_team.items()):
    est_cost = tokens / 1_000_000 * 0.15
    print(f"  {team:<25}: {tokens:>6} tokens  (~${est_cost:.4f})")
print(f"  {'TOTAL':<25}: {total_tokens:>6} tokens  (~${total_tokens/1e6*0.15:.4f})")
print()
```

**Run the lab:**
```bash
python minimal_platform.py
```

**Extensions:**
- Add a `quota_config` per team and enforce it in `MinimalGateway.complete()`, rejecting calls over budget
- Extend the prompt registry to support stage promotion (`development → staging → production`)
- Add a `metrics_report()` that shows P50/P95/P99 latency per team from `gateway.call_log`

---

> ### 📋 Chapter Summary
>
> - The **embedding service** abstracts model selection behind capabilities (`fast`, `standard`, `high_quality`, `local`) — product teams never embed model name strings.
> - The **index lifecycle service** manages build → validate → blue-green promotion atomically, with automatic rollback on validation failure.
> - **Multi-tenant index routing** applies mandatory metadata filters per tenant, preventing cross-tenant data access at the platform layer.
> - The **hands-on lab** demonstrates a complete end-to-end platform integration: gateway → prompt service → embedding → index → generation with usage reporting.

---

> ### ❓ Comprehension Questions
>
> 1. The embedding service accepts `capability` not `model_name`. A product team needs the exact `text-embedding-3-large` with 3072 dimensions for a specific downstream use. How would you accommodate this without breaking the capability abstraction?
> 2. The `IndexLifecycleService._run_build_job` runs synchronously. For a corpus of 500,000 documents, this could take 4+ hours. Redesign the method signature and workflow to support asynchronous execution with status polling.
> 3. Multi-tenant index routing applies `config.metadata_filters` to every query. A tenant's filter is `{"tenant_id": "acme"}`. An attacker modifies their query to include `{"tenant_id": {"$ne": "acme"}}`. What class of attack is this and how does the router defend against it?
> 4. The lab cost report estimates cost as `tokens / 1M * 0.15`. This is the input token price. Why is this estimate incorrect, and how would you compute a more accurate cost?
> 5. Design the interface between the platform embedding service and the index lifecycle service. When the embedding model is upgraded from `STANDARD` to a new version, what happens to existing indexes, and what does the platform need to communicate to product teams?

---

## References

### Documentation
- [SentenceTransformers Documentation](https://www.sbert.net/docs/) — Local embedding models.
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) — Cloud embedding service.
- [Qdrant Multi-tenancy](https://qdrant.tech/documentation/guides/multiple-partitions/) — Vector DB tenant isolation.
- [FastAPI](https://fastapi.tiangolo.com) — REST API for platform services.

### Papers
- [Text Embeddings Reveal Almost As Much As Text](https://arxiv.org/abs/2310.06816) — Morris et al., 2023. Embedding model capability trade-offs.

---

## Chapter 5 — Platform Observability and Cost Management

### 5.1 Platform Metrics Taxonomy

```python
from dataclasses import dataclass
from enum import Enum

class MetricType(str, Enum):
    COUNTER   = "counter"
    GAUGE     = "gauge"
    HISTOGRAM = "histogram"

# Platform metrics — collected automatically on every API call
PLATFORM_METRICS = {
    # Gateway metrics
    "gateway_requests_total":         (MetricType.COUNTER, ["team_id", "service_id", "model", "status"]),
    "gateway_tokens_total":           (MetricType.COUNTER, ["team_id", "service_id", "model", "token_type"]),
    "gateway_latency_ms":             (MetricType.HISTOGRAM, ["team_id", "model", "provider"]),
    "gateway_cost_usd_total":         (MetricType.COUNTER, ["team_id", "service_id", "model"]),
    "gateway_cache_hits_total":       (MetricType.COUNTER, ["team_id"]),
    "gateway_fallback_total":         (MetricType.COUNTER, ["from_provider", "to_provider"]),
    # Rate limiting
    "gateway_rate_limit_hits_total":  (MetricType.COUNTER, ["team_id"]),
    "gateway_quota_pct_used":         (MetricType.GAUGE,   ["team_id", "quota_type"]),
    # Embedding service
    "embedding_requests_total":       (MetricType.COUNTER, ["team_id", "capability", "status"]),
    "embedding_latency_ms":           (MetricType.HISTOGRAM, ["capability"]),
    "embedding_tokens_total":         (MetricType.COUNTER, ["team_id", "model"]),
    # Index service
    "index_build_duration_minutes":   (MetricType.HISTOGRAM, ["team_id", "index_id"]),
    "index_validation_recall":        (MetricType.GAUGE,   ["team_id", "index_id"]),
    "index_vector_count":             (MetricType.GAUGE,   ["index_id", "collection"]),
    # Prompt service
    "prompt_fetches_total":           (MetricType.COUNTER, ["team_id", "template_id", "stage"]),
    "prompt_render_errors_total":     (MetricType.COUNTER, ["team_id", "template_id", "error_type"]),
}
```

---

### 5.2 Cost Attribution and Budgets

```python
import json
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional
import threading

@dataclass
class CostRecord:
    timestamp: str
    team_id: str
    service_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    correlation_id: str

@dataclass
class TeamBudget:
    team_id: str
    daily_budget_usd: float
    monthly_budget_usd: float
    alert_threshold_pct: float = 0.80   # Alert at 80% of budget
    hard_cutoff: bool = False           # If True, block requests at 100%

class CostTracker:
    def __init__(self, budgets: dict[str, TeamBudget], store_path: str = "metrics/costs.jsonl"):
        self.budgets = budgets
        self.store = Path(store_path)
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, team_id: str, service_id: str, model: str,
               prompt_tokens: int, completion_tokens: int,
               cost_usd: float, correlation_id: str):
        record = CostRecord(
            timestamp=datetime.utcnow().isoformat(),
            team_id=team_id, service_id=service_id, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost_usd=cost_usd, correlation_id=correlation_id
        )
        with self._lock:
            with open(self.store, "a") as f:
                f.write(json.dumps(record.__dict__) + "\n")

    def get_daily_spend(self, team_id: str, for_date: Optional[date] = None) -> float:
        target = (for_date or date.today()).isoformat()
        with self._lock:
            if not self.store.exists():
                return 0.0
            total = 0.0
            with open(self.store) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["team_id"] == team_id and r["timestamp"].startswith(target):
                        total += r["cost_usd"]
            return total

    def get_cost_breakdown(
        self,
        start_date: str,
        end_date: str,
        group_by: str = "team_id"  # team_id | service_id | model
    ) -> dict:
        breakdown = {}
        with self._lock:
            if not self.store.exists():
                return breakdown
            with open(self.store) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    ts = r["timestamp"][:10]
                    if start_date <= ts <= end_date:
                        key = r.get(group_by, "unknown")
                        breakdown[key] = breakdown.get(key, 0.0) + r["cost_usd"]
        return {k: round(v, 6) for k, v in sorted(breakdown.items(), key=lambda x: -x[1])}

    def check_budget(self, team_id: str) -> tuple[bool, str]:
        """Returns (within_budget, message). Call before expensive requests."""
        budget = self.budgets.get(team_id)
        if not budget:
            return True, "no budget configured"
        daily_spend = self.get_daily_spend(team_id)
        pct = daily_spend / budget.daily_budget_usd if budget.daily_budget_usd > 0 else 0
        if pct >= 1.0 and budget.hard_cutoff:
            return False, f"Daily budget ${budget.daily_budget_usd:.2f} exhausted (${daily_spend:.2f} used)"
        if pct >= budget.alert_threshold_pct:
            return True, f"WARNING: {pct:.0%} of daily budget used (${daily_spend:.2f} / ${budget.daily_budget_usd:.2f})"
        return True, "ok"
```

---

### 5.3 Token Usage Tracking

```python
import tiktoken
from typing import Optional

class TokenCounter:
    """
    Pre-call token estimation to catch context overflow before API call.
    Supports GPT-4 family and Claude tokenisation.
    """
    def __init__(self):
        self._encoders = {}

    def count_messages(self, messages: list[dict], model: str = "gpt-4o") -> int:
        """Count tokens in a message list. Matches OpenAI's actual counting."""
        encoding = self._get_encoding(model)
        # OpenAI per-message overhead: 3 tokens for role + content delimiter
        tokens_per_message = 3
        tokens = 0
        for msg in messages:
            tokens += tokens_per_message
            for key, value in msg.items():
                if isinstance(value, str):
                    tokens += len(encoding.encode(value))
        tokens += 3  # Response priming
        return tokens

    def count_text(self, text: str, model: str = "gpt-4o") -> int:
        encoding = self._get_encoding(model)
        return len(encoding.encode(text))

    def _get_encoding(self, model: str):
        if model not in self._encoders:
            try:
                self._encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self._encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self._encoders[model]

    def will_exceed_context(
        self,
        messages: list[dict],
        model: str,
        max_response_tokens: int,
        context_limits: dict[str, int] = None
    ) -> tuple[bool, int]:
        """Returns (will_exceed, total_tokens)."""
        limits = context_limits or {
            "gpt-4o": 128000, "gpt-4o-mini": 128000,
            "claude-3-5-sonnet-20241022": 200000
        }
        limit = limits.get(model, 8192)
        input_tokens = self.count_messages(messages, model)
        total = input_tokens + max_response_tokens
        return total > limit, total

token_counter = TokenCounter()

def truncate_context_to_fit(
    messages: list[dict],
    model: str,
    max_response_tokens: int,
    context_limits: dict[str, int] = None
) -> list[dict]:
    """
    Truncate retrieved context chunks until the message fits within context window.
    Preserves system message and user question; truncates context chunks.
    """
    will_exceed, total = token_counter.will_exceed_context(
        messages, model, max_response_tokens, context_limits
    )
    if not will_exceed:
        return messages

    # Find the user message with context
    user_idx = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if user_idx is None:
        return messages

    user_msg = messages[user_idx]["content"]
    # Simple strategy: truncate context section
    if "Context:" in user_msg:
        context_start = user_msg.index("Context:") + len("Context:\n")
        question_part = user_msg[user_msg.rfind("Question:"):]
        context_part = user_msg[context_start:user_msg.rfind("Question:")]
        lines = context_part.strip().split("\n")
        # Remove context lines until it fits
        while len(lines) > 1:
            lines = lines[:-1]
            truncated_content = f"Context:\n{chr(10).join(lines)}\n\n{question_part}"
            test_messages = list(messages)
            test_messages[user_idx] = {"role": "user", "content": truncated_content}
            will_exceed, _ = token_counter.will_exceed_context(
                test_messages, model, max_response_tokens, context_limits
            )
            if not will_exceed:
                return test_messages
    return messages
```

---

### 5.4 Platform SLOs

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class SLODefinition:
    name: str
    description: str
    metric: str             # Prometheus metric name
    threshold: float        # Value that defines "good"
    comparison: str         # ">" | "<" | ">=" | "<="
    target_pct: float       # e.g. 0.999 = 99.9% of requests must meet threshold
    window_days: int = 30   # Rolling window

PLATFORM_SLOS = [
    SLODefinition(
        "gateway_availability",
        "Gateway processes ≥99.9% of requests without 5xx error",
        "gateway_requests_total{status='success'}",
        threshold=0.999, comparison=">=", target_pct=0.999
    ),
    SLODefinition(
        "gateway_latency_p99",
        "99% of gateway requests complete within 3000ms",
        "gateway_latency_ms_p99",
        threshold=3000, comparison="<=", target_pct=0.99
    ),
    SLODefinition(
        "embedding_latency_p95",
        "95% of embedding requests complete within 500ms",
        "embedding_latency_ms_p95",
        threshold=500, comparison="<=", target_pct=0.95
    ),
    SLODefinition(
        "index_build_success_rate",
        "≥95% of index builds complete successfully (pass validation)",
        "index_build_success_ratio",
        threshold=0.95, comparison=">=", target_pct=0.95
    ),
    SLODefinition(
        "prompt_service_availability",
        "Prompt service responds with 2xx on ≥99.9% of fetch requests",
        "prompt_fetches_total{status='2xx'}",
        threshold=0.999, comparison=">=", target_pct=0.999
    ),
]

def compute_error_budget(slo: SLODefinition, current_pct: float) -> dict:
    """Compute remaining error budget for an SLO."""
    allowed_failures = 1.0 - slo.target_pct
    actual_failures  = 1.0 - current_pct
    budget_remaining = max(0.0, 1.0 - (actual_failures / allowed_failures))
    days_remaining   = slo.window_days * budget_remaining

    return {
        "slo_name":         slo.name,
        "target":           f"{slo.target_pct:.1%}",
        "current":          f"{current_pct:.3%}",
        "budget_remaining": f"{budget_remaining:.1%}",
        "days_remaining":   round(days_remaining, 1),
        "status":           "OK" if budget_remaining > 0.20 else
                            ("WARNING" if budget_remaining > 0 else "EXHAUSTED")
    }
```

---

> ### 📋 Chapter Summary
>
> - Platform metrics cover every API surface: gateway (requests, tokens, latency, cost, cache, fallbacks), embedding service, index service, and prompt service — all with `team_id` labels for attribution.
> - **Cost attribution** records per-call cost with team/service labels, enabling per-team spend dashboards and budget enforcement.
> - **Token counting** before API calls prevents context overflow errors in production and enables proactive context truncation.
> - **Platform SLOs** define formal availability and latency commitments for each platform service, with error budget tracking.

---

> ### ❓ Comprehension Questions
>
> 1. The cost tracker records every LLM call in a JSONL file. At 10,000 calls per day, this file grows rapidly. Design a tiered storage strategy: hot (recent 7 days), warm (7–90 days), cold (90+ days). What query patterns does each tier need to support?
> 2. `TokenCounter.will_exceed_context` estimates tokens before the API call. Why is pre-call estimation preferable to handling the API's context length error, and what inaccuracies might the estimation have?
> 3. The gateway SLO targets 99.9% availability over 30 days. The error budget allows 43.2 minutes of downtime. A scheduled maintenance window requires 2 hours. How would you handle planned downtime in SLO accounting?
> 4. The `get_cost_breakdown` method reads the full JSONL file on every call. With 1 million records, this becomes a performance problem. What data structures or storage backends would you use to make this query efficient?
> 5. A product team's daily spend reaches the `alert_threshold_pct` (80%) by 2pm. The `hard_cutoff` is False. Design the alerting and escalation flow that should follow, including what information the alert should contain and who should receive it.

---

## References

### Documentation
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/) — Metrics collection and alerting.
- [Grafana Documentation](https://grafana.com/docs/) — Metrics visualisation and SLO dashboards.
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI tokeniser for token counting.
- [OpenCost](https://www.opencost.io) — Kubernetes cost attribution.
- [Google SRE Workbook — SLOs](https://sre.google/workbook/alerting-on-slos/) — SLO methodology and error budgets.

### Papers
- [Borg, Omega, and Kubernetes](https://research.google/pubs/pub44843/) — Google cluster management and SLO principles.

---

> **Navigation**
> [← Part X — Testing LLM Systems](part_10_testing.md) | [→ Part XII — Infrastructure](part_12_infrastructure.md)
