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

---
[« Back to platform_engineering Index](index.md) | [🏠 Home](../index.md)