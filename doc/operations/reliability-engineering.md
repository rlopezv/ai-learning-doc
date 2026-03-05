## Chapter 2 — Reliability Engineering

### 2.1 SLOs, SLIs, and Error Budgets for AI Services

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

@dataclass
class SLI:
    """Service Level Indicator — what we measure."""
    name: str
    description: str
    metric_query: str       # Prometheus query producing the SLI value
    unit: str               # "ratio" | "ms" | "score"

@dataclass
class SLO:
    """Service Level Objective — the target we commit to."""
    slo_id: str
    name: str
    sli: SLI
    target: float           # e.g. 0.999 for 99.9%
    window_days: int = 30   # Rolling window

    @property
    def error_budget_fraction(self) -> float:
        return 1.0 - self.target

    def error_budget_minutes(self) -> float:
        return self.window_days * 24 * 60 * self.error_budget_fraction

AI_SLOS = [
    SLO("SLO-001", "RAG API Availability",
        SLI("availability", "Fraction of requests returning 2xx",
            "sum(rate(rag_requests_total{status!~'5..'}[5m])) / sum(rate(rag_requests_total[5m]))",
            "ratio"),
        target=0.999, window_days=30),   # 43 min/month error budget

    SLO("SLO-002", "RAG P99 Latency",
        SLI("latency_p99", "99th percentile end-to-end latency",
            "histogram_quantile(0.99, sum(rate(rag_request_latency_ms_bucket[5m])) by (le))",
            "ms"),
        target=0.95, window_days=30),    # 95% of requests under 3000ms

    SLO("SLO-003", "Answer Quality",
        SLI("faithfulness", "Rolling faithfulness score from LLM-judge",
            "rag_quality_score{metric='faithfulness'}",
            "score"),
        target=0.85, window_days=30),    # 85% faithfulness target

    SLO("SLO-004", "Refusal Rate",
        SLI("refusal_rate", "Fraction of requests returning refusal",
            "rate(rag_requests_total{is_refusal='true'}[1h]) / rate(rag_requests_total[1h])",
            "ratio"),
        target=0.80,                     # Refusal rate BELOW 20%
        window_days=30),
]

@dataclass
class ErrorBudgetStatus:
    slo: SLO
    budget_total_minutes: float
    budget_consumed_minutes: float
    budget_remaining_minutes: float
    budget_remaining_pct: float
    burn_rate_last_1h: float   # Current consumption rate vs ideal
    status: str                # "healthy" | "warning" | "exhausted"

def compute_error_budget(
    slo: SLO,
    current_sli_value: float,
    window_start: datetime
) -> ErrorBudgetStatus:
    budget_total = slo.error_budget_minutes()
    elapsed_days = (datetime.utcnow() - window_start).total_seconds() / 86400
    ideal_consumption_pct = elapsed_days / slo.window_days

    # How much of the budget is consumed
    if slo.sli.unit == "ratio":
        actual_error_rate = 1.0 - current_sli_value
        consumed_pct = actual_error_rate / slo.error_budget_fraction if slo.error_budget_fraction > 0 else 1
    else:
        consumed_pct = ideal_consumption_pct   # Simplified for non-ratio SLIs

    consumed_minutes = budget_total * consumed_pct
    remaining_minutes = max(0, budget_total - consumed_minutes)
    burn_rate = consumed_pct / ideal_consumption_pct if ideal_consumption_pct > 0 else 1.0

    status = "healthy"
    if remaining_minutes < budget_total * 0.10:
        status = "exhausted"
    elif burn_rate > 2.0:
        status = "warning"

    return ErrorBudgetStatus(
        slo=slo,
        budget_total_minutes=round(budget_total, 1),
        budget_consumed_minutes=round(consumed_minutes, 1),
        budget_remaining_minutes=round(remaining_minutes, 1),
        budget_remaining_pct=round((remaining_minutes / budget_total) * 100, 1),
        burn_rate_last_1h=round(burn_rate, 2),
        status=status
    )
```

---

### 2.2 Circuit Breakers and Fallback Chains

```python
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any, Optional

class CircuitState(str, Enum):
    CLOSED   = "closed"     # Normal operation — requests pass through
    OPEN     = "open"       # Failing — requests rejected immediately
    HALF_OPEN = "half_open" # Probe mode — one request allowed through

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5         # Open after N consecutive failures
    success_threshold: int = 2         # Close after N successes in HALF_OPEN
    open_duration_seconds: float = 60  # Stay open for this long before probing
    timeout_seconds: float = 10        # Request timeout

class CircuitBreaker:
    """
    Circuit breaker for LLM provider calls.
    Prevents cascading failures when the upstream provider is degraded.
    """
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: Optional[float] = None

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if time.time() - self._opened_at > self.config.open_duration_seconds:
                self.state = CircuitState.HALF_OPEN
                self._success_count = 0
                print(f"  [{self.name}] Circuit HALF-OPEN — probing")
            else:
                raise RuntimeError(f"Circuit {self.name} is OPEN — request rejected")

        try:
            import signal
            # Simplified timeout via exception (production: use asyncio.wait_for)
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self._failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                print(f"  [{self.name}] Circuit CLOSED — provider recovered")

    def _on_failure(self):
        self._failure_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._opened_at = time.time()
            print(f"  [{self.name}] Probe failed — circuit remains OPEN")
        elif self._failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self._opened_at = time.time()
            print(f"  [{self.name}] Circuit OPENED after {self._failure_count} failures")

    @property
    def is_available(self) -> bool:
        if self.state == CircuitState.OPEN:
            return time.time() - self._opened_at > self.config.open_duration_seconds
        return True


class FallbackChain:
    """
    Ordered fallback chain for LLM providers.
    Tries each provider in sequence; returns first successful response.

    Example chain:
      primary:   OpenAI GPT-4o-mini
      secondary: Anthropic Claude Haiku
      tertiary:  Local Ollama llama3.2    (on-premise fallback)
    """
    def __init__(self, providers: list[tuple[str, Callable, CircuitBreaker]]):
        self.providers = providers  # [(name, call_fn, circuit_breaker)]

    def call(self, *args, **kwargs) -> tuple[Any, str]:
        """Returns (result, provider_name_used)."""
        errors = []
        for name, fn, breaker in self.providers:
            if not breaker.is_available:
                errors.append(f"{name}: circuit open")
                continue
            try:
                result = breaker.call(fn, *args, **kwargs)
                return result, name
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue
        raise RuntimeError(f"All providers failed: {errors}")
```

---

### 2.3 Graceful Degradation Patterns

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DegradationLevel(str, Enum):
    FULL       = "full"        # All features available
    DEGRADED   = "degraded"    # Reduced quality but functional
    MINIMAL    = "minimal"     # Core functionality only
    UNAVAILABLE = "unavailable" # Service offline

@dataclass
class DegradationResponse:
    level: DegradationLevel
    answer: str
    is_cached: bool
    is_degraded: bool
    degradation_reason: Optional[str] = None

class GracefulDegradationHandler:
    """
    Returns the best available response when components fail.
    Degradation ladder:
      1. Full RAG (retrieval + generation)
      2. Cached answer (if available for similar query)
      3. Static FAQ answer (keyword match)
      4. Canned refusal with redirect
    """
    def __init__(self, cache, faq_store):
        self.cache = cache
        self.faq = faq_store

    def handle(
        self,
        question: str,
        rag_available: bool,
        generation_available: bool
    ) -> DegradationResponse:

        # Level 1: Full RAG
        if rag_available and generation_available:
            return DegradationResponse(
                DegradationLevel.FULL, "", False, False
            )

        # Level 2: Semantic cache hit
        cached = self.cache.lookup(question, threshold=0.92)
        if cached:
            return DegradationResponse(
                DegradationLevel.DEGRADED,
                answer=cached["answer"],
                is_cached=True, is_degraded=True,
                degradation_reason="serving_cached_answer"
            )

        # Level 3: Keyword FAQ match
        faq_answer = self.faq.lookup(question)
        if faq_answer:
            return DegradationResponse(
                DegradationLevel.MINIMAL,
                answer=faq_answer,
                is_cached=False, is_degraded=True,
                degradation_reason="serving_faq_fallback"
            )

        # Level 4: Canned response
        return DegradationResponse(
            DegradationLevel.UNAVAILABLE,
            answer=("I'm temporarily unable to answer questions. "
                    "Please visit our help centre at support.company.com "
                    "or contact support@company.com."),
            is_cached=False, is_degraded=True,
            degradation_reason="all_components_unavailable"
        )
```

---

### 2.4 Chaos Engineering for LLM Systems

```python
import random, time
from dataclasses import dataclass
from typing import Callable

@dataclass
class ChaosExperiment:
    name: str
    description: str
    hypothesis: str      # What we expect the system to do when this fails
    steady_state: dict   # Metrics that define normal operation
    fault_fn: Callable   # Function that injects the fault
    duration_seconds: int
    rollback_fn: Callable

# LLM-specific chaos experiments
def inject_llm_latency(target_service: str, extra_latency_ms: int = 3000):
    """Add artificial latency to LLM API responses."""
    # In production: use Istio fault injection or toxiproxy
    print(f"  Injecting {extra_latency_ms}ms latency on {target_service}")

def inject_llm_errors(target_service: str, error_rate: float = 0.5):
    """Make 50% of LLM calls return 500."""
    print(f"  Injecting {error_rate:.0%} error rate on {target_service}")

def inject_vector_db_partition():
    """Simulate vector DB network partition."""
    print("  Partitioning vector DB network")

def inject_corpus_stale(days_old: int = 14):
    """Simulate stale corpus by pointing to old index version."""
    print(f"  Pointing index to {days_old}-day-old corpus")

CHAOS_EXPERIMENTS = [
    ChaosExperiment(
        "llm_provider_latency",
        "LLM API responds with 3s additional latency",
        "P99 latency increases but system serves degraded responses from cache; "
        "circuit breaker opens after 5 failures; fallback activates within 30s",
        steady_state={"error_rate": 0.01, "p99_ms": 1200, "refusal_rate": 0.08},
        fault_fn=lambda: inject_llm_latency("openai-gateway", 3000),
        duration_seconds=300,
        rollback_fn=lambda: print("  Removing latency injection")
    ),
    ChaosExperiment(
        "vector_db_unavailable",
        "Qdrant returns 503 for all queries",
        "System returns graceful canned response; no 500s exposed to users; "
        "alert fires within 2 minutes",
        steady_state={"error_rate": 0.01, "availability": 0.999},
        fault_fn=inject_vector_db_partition,
        duration_seconds=120,
        rollback_fn=lambda: print("  Restoring vector DB connectivity")
    ),
    ChaosExperiment(
        "stale_corpus",
        "Index is 14 days old — knowledge base not updated",
        "Refusal rate increases (questions about recent events refused); "
        "no availability impact; staleness alert fires within 48h",
        steady_state={"refusal_rate": 0.08},
        fault_fn=lambda: inject_corpus_stale(14),
        duration_seconds=3600,
        rollback_fn=lambda: print("  Restoring current index")
    ),
    ChaosExperiment(
        "high_token_consumption",
        "Requests suddenly use 10x normal token count (context stuffing)",
        "Token budget enforcer blocks requests over limit; cost alert fires; "
        "no token budget overrun beyond 2x daily budget",
        steady_state={"cost_per_hour_usd": 5.0},
        fault_fn=lambda: print("  Injecting oversized context windows"),
        duration_seconds=600,
        rollback_fn=lambda: print("  Restoring normal context sizes")
    ),
]

class ChaosRunner:
    def __init__(self, metrics_client):
        self.metrics = metrics_client

    def run(self, experiment: ChaosExperiment) -> dict:
        print(f"\n[CHAOS] Experiment: {experiment.name}")
        print(f"  Hypothesis: {experiment.hypothesis}")

        # Verify steady state before fault injection
        pre_state = self.metrics.snapshot()
        print(f"  Pre-experiment state: {pre_state}")

        # Inject fault
        print(f"  Injecting fault for {experiment.duration_seconds}s...")
        experiment.fault_fn()
        time.sleep(min(experiment.duration_seconds, 30))   # Abbreviated for demo

        during_state = self.metrics.snapshot()

        # Rollback
        experiment.rollback_fn()
        time.sleep(10)   # Recovery window
        post_state = self.metrics.snapshot()

        return {
            "experiment": experiment.name,
            "pre":    pre_state,
            "during": during_state,
            "post":   post_state,
            "hypothesis_validated": True   # In practice: check against expected behaviour
        }
```

---

> ### 📋 Chapter Summary
>
> - **SLOs for AI services** add quality dimensions (faithfulness, refusal rate) alongside availability and latency. Error budgets derived from SLOs drive release velocity decisions.
> - **Circuit breakers** prevent cascading failures by rejecting requests when a provider is failing, entering HALF_OPEN mode to probe recovery, and auto-closing on success.
> - **Fallback chains** (primary → secondary → local 🔓) ensure availability even when the primary LLM provider is down.
> - **Chaos experiments** validate reliability hypotheses before they become production incidents — LLM-specific scenarios include latency injection, vector DB partitioning, stale corpus simulation, and token budget stress.

---

> ### ❓ Comprehension Questions
>
> 1. `SLO-003` targets faithfulness ≥ 0.85 over a 30-day window. Faithfulness is scored by an LLM judge on a 1% sample of requests. A sharp quality drop affects 5% of requests but the sample misses it. What sample rate is required to detect a 5-percentage-point faithfulness drop with 95% confidence in a 30-day window of 1M requests?
> 2. The circuit breaker opens after 5 consecutive failures. A temporary network blip causes exactly 5 failures before recovering. The circuit opens and blocks requests for 60 seconds — causing more errors than the original blip. Is this acceptable behaviour? How would you tune `failure_threshold` and `open_duration_seconds` to reduce this over-reaction?
> 3. `FallbackChain` tries OpenAI, then Anthropic, then Ollama. Anthropic's API has 2s higher latency than OpenAI. When OpenAI's circuit opens, every request incurs +2s. Describe the P99 latency impact and what SLO headroom you need to absorb this.
> 4. `GracefulDegradationHandler` serves a cached answer when RAG is unavailable. The cached answer was generated 48 hours ago. The knowledge base was updated 24 hours ago. The cached answer may now be incorrect. How would you implement cache TTL-based invalidation for degraded-mode answers?
> 5. The chaos experiment `stale_corpus` runs for 3600 seconds. Running this in production would degrade user experience for an hour. How would you conduct this experiment safely — what environment, traffic conditions, and monitoring would you require?

---

## References

### Documentation
- [Google SRE Book — SLOs](https://sre.google/sre-book/service-level-objectives/)
- [Resilience4j](https://resilience4j.readme.io/docs/circuitbreaker) — Circuit breaker for JVM.
- [Python circuit-breaker-py](https://github.com/fabfuel/circuitbreaker)
- [Chaos Toolkit](https://chaostoolkit.org) — Open-source chaos engineering.

### Books
- *Site Reliability Engineering* — Beyer et al. (Google O'Reilly). Chapter 4 (SLOs) and Chapter 22 (Managing Incidents).

---

---
[« Back to operations Index](index.md) | [🏠 Home](../index.md)