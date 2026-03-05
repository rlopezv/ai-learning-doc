# Part XVI — Operations

---

> **Navigation**
> [← Part XV — Governance](part_15_governance.md) | [→ Part XVII — Reference Architectures](part_17_reference_architectures.md)

---

## Contents

- [Chapter 1 — Deployment Patterns and Release Engineering](#chapter-1--deployment-patterns-and-release-engineering)
  - [1.1 Release Strategies for AI Systems](#11-release-strategies-for-ai-systems)
  - [1.2 Blue-Green Deployment for RAG Services](#12-blue-green-deployment-for-rag-services)
  - [1.3 Canary Releases with Quality Gating](#13-canary-releases-with-quality-gating)
  - [1.4 Feature Flags for LLM Experiments](#14-feature-flags-for-llm-experiments)
- [Chapter 2 — Reliability Engineering](#chapter-2--reliability-engineering)
  - [2.1 SLOs, SLIs, and Error Budgets for AI Services](#21-slos-slis-and-error-budgets-for-ai-services)
  - [2.2 Circuit Breakers and Fallback Chains](#22-circuit-breakers-and-fallback-chains)
  - [2.3 Graceful Degradation Patterns](#23-graceful-degradation-patterns)
  - [2.4 Chaos Engineering for LLM Systems](#24-chaos-engineering-for-llm-systems)
- [Chapter 3 — Cost Management in Production](#chapter-3--cost-management-in-production)
  - [3.1 Token Budget Enforcement](#31-token-budget-enforcement)
  - [3.2 Semantic Caching for Cost Reduction](#32-semantic-caching-for-cost-reduction)
  - [3.3 Model Routing for Cost Optimisation](#33-model-routing-for-cost-optimisation)
  - [3.4 Cost Anomaly Detection](#34-cost-anomaly-detection)
- [Chapter 4 — Knowledge Base Operations 🧪](#chapter-4--knowledge-base-operations-)
  - [4.1 Corpus Maintenance Workflows](#41-corpus-maintenance-workflows)
  - [4.2 Index Refresh Strategies](#42-index-refresh-strategies)
  - [4.3 Quality Regression Detection on Updates](#43-quality-regression-detection-on-updates)
  - [🧪 Hands-on Lab: Operational Runbook Simulator](#-hands-on-lab-operational-runbook-simulator)

---

## Chapter 1 — Deployment Patterns and Release Engineering

### 1.1 Release Strategies for AI Systems

AI systems have deployment requirements that differ from stateless microservices. A release may involve any combination of: application code, prompt templates, model versions, embedding models, and corpus content. Each component has a different rollback cost and quality impact.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ReleaseStrategy(str, Enum):
    RECREATE     = "recreate"       # Stop old, start new — downtime accepted
    ROLLING      = "rolling"        # Replace pods one at a time
    BLUE_GREEN   = "blue_green"     # Full parallel environment, instant cutover
    CANARY       = "canary"         # Gradual traffic shift with quality gates
    FEATURE_FLAG = "feature_flag"   # Code deployed, behaviour toggled at runtime
    SHADOW       = "shadow"         # New version runs in parallel, not serving traffic

@dataclass
class ReleaseComponent:
    component: str       # "api_code" | "prompt" | "embedding_model" | "corpus" | "llm"
    change_type: str     # "patch" | "minor" | "major"
    rollback_minutes: int
    quality_impact: str  # "low" | "medium" | "high"

# Recommended strategy by (change_type, quality_impact)
RELEASE_STRATEGY_MATRIX = {
    ("patch",  "low"):    ReleaseStrategy.ROLLING,
    ("patch",  "medium"): ReleaseStrategy.CANARY,
    ("patch",  "high"):   ReleaseStrategy.BLUE_GREEN,
    ("minor",  "low"):    ReleaseStrategy.CANARY,
    ("minor",  "medium"): ReleaseStrategy.BLUE_GREEN,
    ("minor",  "high"):   ReleaseStrategy.BLUE_GREEN,
    ("major",  "low"):    ReleaseStrategy.BLUE_GREEN,
    ("major",  "medium"): ReleaseStrategy.BLUE_GREEN,
    ("major",  "high"):   ReleaseStrategy.BLUE_GREEN,
}

def recommend_strategy(component: ReleaseComponent) -> ReleaseStrategy:
    return RELEASE_STRATEGY_MATRIX.get(
        (component.change_type, component.quality_impact),
        ReleaseStrategy.BLUE_GREEN   # Default to safest
    )

# Examples
COMPONENT_PROFILES = [
    ReleaseComponent("api_code",         "patch", rollback_minutes=5,   quality_impact="low"),
    ReleaseComponent("prompt_template",  "minor", rollback_minutes=2,   quality_impact="high"),
    ReleaseComponent("embedding_model",  "major", rollback_minutes=120, quality_impact="high"),
    ReleaseComponent("llm_provider",     "major", rollback_minutes=10,  quality_impact="high"),
    ReleaseComponent("corpus_content",   "minor", rollback_minutes=60,  quality_impact="medium"),
]
for c in COMPONENT_PROFILES:
    print(f"{c.component:<25} → {recommend_strategy(c).value}")
```

---

### 1.2 Blue-Green Deployment for RAG Services

```yaml
# kubernetes/rag-api/blue-green.yaml
# Two identical environments; service selector determines active slot

apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api-blue
  namespace: ai-platform
  labels: { app: rag-api, slot: blue }
spec:
  replicas: 3
  selector:
    matchLabels: { app: rag-api, slot: blue }
  template:
    metadata:
      labels: { app: rag-api, slot: blue }
    spec:
      containers:
        - name: api
          image: registry.company.com/rag-api:2.3.1
          env:
            - { name: PROMPT_VERSION,    value: "v14" }
            - { name: EMBEDDING_MODEL,   value: "all-MiniLM-L6-v2" }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api-green
  namespace: ai-platform
  labels: { app: rag-api, slot: green }
spec:
  replicas: 3
  selector:
    matchLabels: { app: rag-api, slot: green }
  template:
    metadata:
      labels: { app: rag-api, slot: green }
    spec:
      containers:
        - name: api
          image: registry.company.com/rag-api:2.4.0
          env:
            - { name: PROMPT_VERSION,    value: "v15" }
            - { name: EMBEDDING_MODEL,   value: "all-MiniLM-L6-v2" }
---
# Service: change selector slot to cut over instantly
apiVersion: v1
kind: Service
metadata:
  name: rag-api
  namespace: ai-platform
spec:
  selector:
    app: rag-api
    slot: blue        # kubectl patch svc rag-api -p '{"spec":{"selector":{"slot":"green"}}}'
  ports:
    - port: 80
      targetPort: 8080
```

```python
# scripts/blue_green_cutover.py — automated blue-green cutover with smoke test
import subprocess, time, httpx, sys

def get_active_slot(namespace: str = "ai-platform") -> str:
    result = subprocess.run(
        ["kubectl", "get", "svc", "rag-api", "-n", namespace,
         "-o", "jsonpath={.spec.selector.slot}"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def cutover(target_slot: str, namespace: str = "ai-platform"):
    print(f"Cutting over to slot: {target_slot}")
    subprocess.run([
        "kubectl", "patch", "svc", "rag-api", "-n", namespace,
        "-p", f'{{"spec":{{"selector":{{"slot":"{target_slot}"}}}}}}'
    ], check=True)
    print(f"  ✓ Service now pointing to {target_slot}")

def smoke_test(api_url: str) -> bool:
    """Run a quick smoke test against the live endpoint."""
    try:
        resp = httpx.post(
            f"{api_url}/v2/query",
            json={"question": "What is the return policy?", "team_id": "smoke-test"},
            timeout=15
        )
        return resp.status_code == 200 and "answer" in resp.json()
    except Exception as e:
        print(f"  Smoke test failed: {e}")
        return False

def safe_cutover(api_url: str, namespace: str = "ai-platform"):
    current = get_active_slot(namespace)
    target = "green" if current == "blue" else "blue"
    print(f"Current slot: {current} → target slot: {target}")

    # Smoke test new slot before cutting over (direct pod IP or internal test)
    print("Running pre-cutover smoke test on green pods...")
    # In practice: test via internal service pointing to green only

    cutover(target, namespace)
    time.sleep(3)

    # Post-cutover smoke test
    print("Running post-cutover smoke test...")
    if smoke_test(api_url):
        print(f"  ✓ Cutover to {target} successful")
    else:
        print(f"  ✗ Smoke test failed — rolling back to {current}")
        cutover(current, namespace)
        sys.exit(1)
```

---

### 1.3 Canary Releases with Quality Gating

```python
import time, statistics
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CanaryConfig:
    canary_weight_pct: int = 5          # Initial traffic to canary (5%)
    step_pct: int = 10                  # Increment per step
    step_interval_minutes: int = 30     # Wait between steps
    max_weight_pct: int = 100           # Full rollout target
    # Quality gates — canary must meet ALL thresholds to advance
    min_faithfulness: float = 0.85
    max_error_rate: float = 0.02
    max_latency_p99_ms: float = 3000
    max_refusal_rate: float = 0.20
    min_sample_size: int = 100          # Min requests before evaluating gates

@dataclass
class CanaryMetrics:
    """Metrics collected from canary traffic window."""
    request_count: int = 0
    error_rate: float = 0.0
    latency_p99_ms: float = 0.0
    refusal_rate: float = 0.0
    faithfulness_score: Optional[float] = None

class CanaryController:
    """
    Manages a canary rollout with automated quality gating.
    Advances canary traffic weight only when all quality gates pass.
    Rolls back automatically if gates fail.
    """
    def __init__(self, config: CanaryConfig, metrics_client, traffic_controller):
        self.config = config
        self.metrics = metrics_client
        self.traffic = traffic_controller
        self.current_weight = 0
        self.history: list[dict] = []

    def start(self):
        print(f"Starting canary at {self.config.canary_weight_pct}% traffic")
        self.traffic.set_canary_weight(self.config.canary_weight_pct)
        self.current_weight = self.config.canary_weight_pct

    def evaluate_gates(self, window_metrics: CanaryMetrics) -> tuple[bool, list[str]]:
        """Check all quality gates. Returns (passed, list_of_failures)."""
        failures = []
        if window_metrics.request_count < self.config.min_sample_size:
            return False, [f"Insufficient sample: {window_metrics.request_count} < {self.config.min_sample_size}"]
        if window_metrics.error_rate > self.config.max_error_rate:
            failures.append(f"error_rate={window_metrics.error_rate:.3f} > {self.config.max_error_rate}")
        if window_metrics.latency_p99_ms > self.config.max_latency_p99_ms:
            failures.append(f"p99={window_metrics.latency_p99_ms:.0f}ms > {self.config.max_latency_p99_ms}ms")
        if window_metrics.refusal_rate > self.config.max_refusal_rate:
            failures.append(f"refusal_rate={window_metrics.refusal_rate:.3f} > {self.config.max_refusal_rate}")
        if (window_metrics.faithfulness_score is not None and
                window_metrics.faithfulness_score < self.config.min_faithfulness):
            failures.append(f"faithfulness={window_metrics.faithfulness_score:.3f} < {self.config.min_faithfulness}")
        return len(failures) == 0, failures

    def advance(self) -> bool:
        """Attempt to advance to next step. Returns False if rollback triggered."""
        window_metrics = self.metrics.get_canary_window(minutes=self.config.step_interval_minutes)
        passed, failures = self.evaluate_gates(window_metrics)

        step_record = {
            "weight": self.current_weight,
            "metrics": window_metrics.__dict__,
            "passed": passed,
            "failures": failures,
        }
        self.history.append(step_record)

        if not passed:
            print(f"  ✗ Quality gates FAILED at {self.current_weight}%: {failures}")
            self._rollback()
            return False

        next_weight = min(self.current_weight + self.config.step_pct,
                          self.config.max_weight_pct)
        self.traffic.set_canary_weight(next_weight)
        self.current_weight = next_weight
        print(f"  ✓ Quality gates passed → advancing to {next_weight}%")
        return True

    def _rollback(self):
        print(f"  Rolling back canary to 0% traffic")
        self.traffic.set_canary_weight(0)
        self.current_weight = 0

    def run_to_completion(self):
        self.start()
        while self.current_weight < self.config.max_weight_pct:
            time.sleep(self.config.step_interval_minutes * 60)
            if not self.advance():
                print("Canary rollout aborted — rollback complete")
                return False
        print("Canary rollout complete — 100% traffic on new version")
        return True
```

---

### 1.4 Feature Flags for LLM Experiments

```python
from dataclasses import dataclass, field
from typing import Any, Optional
import hashlib

@dataclass
class FeatureFlag:
    flag_id: str
    description: str
    default_value: Any
    rollout_pct: float = 0.0          # 0–100
    team_overrides: dict = field(default_factory=dict)    # team_id → value
    user_overrides: dict = field(default_factory=dict)    # user_id → value

class FeatureFlagClient:
    """
    Evaluates feature flags for A/B testing LLM configurations.
    Deterministic assignment: same user always gets same variant.
    """
    def __init__(self, flags: dict[str, FeatureFlag]):
        self.flags = flags

    def get(self, flag_id: str, context: dict) -> Any:
        flag = self.flags.get(flag_id)
        if not flag:
            return None

        user_id  = context.get("user_id", "")
        team_id  = context.get("team_id", "")

        # User-level override (highest priority)
        if user_id in flag.user_overrides:
            return flag.user_overrides[user_id]

        # Team-level override
        if team_id in flag.team_overrides:
            return flag.team_overrides[team_id]

        # Percentage rollout (deterministic via hash)
        if flag.rollout_pct > 0:
            bucket = int(hashlib.md5(
                f"{flag_id}:{user_id or team_id}".encode()
            ).hexdigest(), 16) % 100
            if bucket < flag.rollout_pct:
                return True   # Or flag's non-default value

        return flag.default_value

# LLM experiment flags
LLM_FLAGS = {
    "use_gpt4o_for_complex":  FeatureFlag(
        "use_gpt4o_for_complex",
        "Use GPT-4o instead of GPT-4o-mini for queries > 500 chars",
        default_value=False,
        rollout_pct=20.0,
        team_overrides={"platform-team": True}
    ),
    "prompt_v15":  FeatureFlag(
        "prompt_v15",
        "Use prompt template v15 (new citation format)",
        default_value=False,
        rollout_pct=50.0,
    ),
    "hybrid_search":  FeatureFlag(
        "hybrid_search",
        "Enable BM25 + vector hybrid retrieval",
        default_value=False,
        rollout_pct=10.0,
    ),
    "context_window_8k":  FeatureFlag(
        "context_window_8k",
        "Increase context window from 4K to 8K tokens",
        default_value=False,
        rollout_pct=0.0,   # Not yet rolling out
        team_overrides={"beta-testers": True}
    ),
}
```

---

> ### 📋 Chapter Summary
>
> - **Release strategy selection** depends on change type and quality impact. Prompt and embedding model changes always warrant blue-green or canary — their quality impact is high and hard to predict.
> - **Blue-green deployment** achieves zero-downtime cutover by maintaining two full environments and switching a Kubernetes Service selector atomically.
> - **Canary releases** advance traffic weight only when quality gates (error rate, latency, faithfulness, refusal rate) pass — automatically rolling back on gate failure.
> - **Feature flags** enable runtime A/B testing of LLM configurations with deterministic, hash-based user assignment — no redeploy required to start or stop an experiment.

---

> ### ❓ Comprehension Questions
>
> 1. A prompt template change (`minor`, `high` quality impact) is recommended for blue-green deployment. The rollback time is 2 minutes. An embedding model change (`major`, `high`) has a 120-minute rollback. How does rollback cost affect the pre-deployment testing rigour required for each?
> 2. The canary controller requires `min_sample_size=100` before evaluating quality gates. At 10 RPS with 5% canary traffic (0.5 RPS to canary), how long must the canary run before the first gate evaluation? Is this acceptable for a 30-minute step interval?
> 3. `FeatureFlagClient.get` uses `hashlib.md5` for bucket assignment. An attacker who knows the flag_id and their user_id can compute which bucket they fall into and predict their assignment. Is this a security concern for LLM experiment flags? Argue your position.
> 4. Blue-green deployment keeps both environments running simultaneously, doubling compute cost during the transition. For an embedding server with `nvidia.com/gpu: 1` resource limit, what is the GPU cost of maintaining both slots? How would you minimise this?
> 5. The canary config has `max_refusal_rate=0.20`. The new prompt version reduces refusal rate from 0.25 to 0.18 — an improvement, but gate passes because 0.18 < 0.20. The gate cannot distinguish improvement from deterioration. Redesign the gate logic to compare against a baseline (stable) refusal rate rather than an absolute threshold.

---

## References

### Documentation
- [Argo Rollouts](https://argoproj.github.io/rollouts/) — Kubernetes-native canary and blue-green deployments.
- [Flagger](https://flagger.app) — Progressive delivery with automated analysis.
- [LaunchDarkly](https://docs.launchdarkly.com) — Feature flag management platform.
- [Unleash](https://docs.getunleash.io) — 🔓 On-premise open-source feature flags.

---

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

## Chapter 3 — Cost Management in Production

### 3.1 Token Budget Enforcement

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import threading

@dataclass
class TeamBudget:
    team_id: str
    daily_token_limit: int        # Hard limit in tokens/day
    daily_cost_limit_usd: float   # Hard limit in USD/day
    alert_threshold_pct: float = 0.80
    hard_cutoff: bool = True      # Block requests if limit reached

class TokenBudgetEnforcer:
    """
    Enforces per-team token and cost budgets.
    Thread-safe with daily reset at midnight UTC.
    Implements both soft alerts and hard cutoffs.
    """
    def __init__(self, budgets: dict[str, TeamBudget]):
        self.budgets = budgets
        self._lock = threading.Lock()
        self._daily: dict[str, dict] = {}
        self._reset_date = datetime.utcnow().date()

    def _get_or_init(self, team_id: str) -> dict:
        today = datetime.utcnow().date()
        if today != self._reset_date:
            with self._lock:
                self._daily = {}
                self._reset_date = today
        if team_id not in self._daily:
            self._daily[team_id] = {"tokens": 0, "cost_usd": 0.0}
        return self._daily[team_id]

    def check_and_record(
        self,
        team_id: str,
        estimated_tokens: int,
        estimated_cost_usd: float
    ) -> tuple[bool, str]:
        """
        Check if request is within budget and record usage.
        Returns (allowed, reason).
        """
        budget = self.budgets.get(team_id)
        if not budget:
            return True, "no_budget_configured"

        with self._lock:
            usage = self._get_or_init(team_id)

            projected_tokens = usage["tokens"] + estimated_tokens
            projected_cost   = usage["cost_usd"] + estimated_cost_usd

            # Hard cutoff check
            if budget.hard_cutoff:
                if projected_tokens > budget.daily_token_limit:
                    return False, (f"TOKEN_BUDGET_EXCEEDED: "
                                   f"{projected_tokens:,} > {budget.daily_token_limit:,}")
                if projected_cost > budget.daily_cost_limit_usd:
                    return False, (f"COST_BUDGET_EXCEEDED: "
                                   f"${projected_cost:.2f} > ${budget.daily_cost_limit_usd:.2f}")

            # Soft alert (log but allow)
            token_pct = projected_tokens / budget.daily_token_limit
            cost_pct  = projected_cost / budget.daily_cost_limit_usd
            alert_msg = None
            if token_pct >= budget.alert_threshold_pct or cost_pct >= budget.alert_threshold_pct:
                alert_msg = (f"BUDGET_WARNING: team={team_id} "
                             f"tokens={token_pct:.0%} cost={cost_pct:.0%}")

            # Record usage
            usage["tokens"]   += estimated_tokens
            usage["cost_usd"] += estimated_cost_usd

        if alert_msg:
            print(f"[BUDGET ALERT] {alert_msg}")

        return True, "allowed"

    def get_usage(self, team_id: str) -> dict:
        budget = self.budgets.get(team_id, TeamBudget(team_id, 0, 0))
        with self._lock:
            usage = self._get_or_init(team_id)
            return {
                "team_id":     team_id,
                "tokens_used": usage["tokens"],
                "tokens_limit": budget.daily_token_limit,
                "cost_usd":    round(usage["cost_usd"], 4),
                "cost_limit":  budget.daily_cost_limit_usd,
                "token_pct":   usage["tokens"] / budget.daily_token_limit
                               if budget.daily_token_limit > 0 else 0,
            }
```

---

### 3.2 Semantic Caching for Cost Reduction

```python
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CacheEntry:
    query_text: str
    query_embedding: list[float]
    answer: str
    model_used: str
    prompt_version: str
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0

class SemanticCache:
    """
    Caches LLM responses by semantic query similarity.
    A cache hit occurs when a new query is semantically close to a cached query
    (cosine similarity ≥ threshold), avoiding a new LLM call entirely.
    Typical hit rates: 15–30% for customer support use cases.
    """
    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.92,
        max_entries: int = 5000,
        ttl_seconds: int = 3600   # 1-hour TTL
    ):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._entries: list[CacheEntry] = []

    def lookup(self, query: str) -> Optional[CacheEntry]:
        """Find a semantically similar cached entry."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        self._evict_expired()
        if not self._entries:
            return None

        q_emb = self.embedder.encode([query])
        cached_embs = np.array([e.query_embedding for e in self._entries])
        sims = cosine_similarity(q_emb, cached_embs)[0]

        best_idx = int(sims.argmax())
        if sims[best_idx] >= self.threshold:
            entry = self._entries[best_idx]
            entry.hit_count += 1
            return entry
        return None

    def store(self, query: str, answer: str, model: str, prompt_version: str):
        """Cache a query-answer pair."""
        embedding = self.embedder.encode([query])[0].tolist()
        entry = CacheEntry(
            query_text=query,
            query_embedding=embedding,
            answer=answer,
            model_used=model,
            prompt_version=prompt_version
        )
        self._entries.append(entry)
        # Evict oldest entries if over limit (LRU approximation)
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: e.created_at)
            self._entries = self._entries[-self.max_entries:]

    def _evict_expired(self):
        cutoff = time.time() - self.ttl
        self._entries = [e for e in self._entries if e.created_at > cutoff]

    def stats(self) -> dict:
        total_hits = sum(e.hit_count for e in self._entries)
        return {
            "entries":    len(self._entries),
            "total_hits": total_hits,
            "avg_hits":   total_hits / len(self._entries) if self._entries else 0,
        }
```

---

### 3.3 Model Routing for Cost Optimisation

```python
from dataclasses import dataclass
from enum import Enum

class QueryComplexity(str, Enum):
    SIMPLE   = "simple"    # FAQ-style, short answer expected
    MEDIUM   = "medium"    # Multi-sentence, needs context synthesis
    COMPLEX  = "complex"   # Multi-hop, reasoning required
    EXPERT   = "expert"    # Highly technical or sensitive

@dataclass
class RoutingDecision:
    model_alias: str
    reason: str
    estimated_cost_usd: float

class CostOptimisedRouter:
    """
    Routes queries to the most cost-effective model that meets
    quality requirements for the detected complexity level.

    Cost hierarchy (approximate, 2025 pricing):
      fast (gpt-4o-mini):    $0.15/1M input,  $0.60/1M output
      default (gpt-4o):      $2.50/1M input, $10.00/1M output
      powerful (gpt-4o):     $2.50/1M input, $10.00/1M output
      local (ollama):        $0.00 (compute cost amortised)
    """
    COMPLEXITY_SIGNALS = {
        QueryComplexity.SIMPLE: [
            "what is", "how much", "when does", "where is",
            "what are the hours", "do you offer"
        ],
        QueryComplexity.COMPLEX: [
            "compare", "difference between", "analyse", "evaluate",
            "recommend", "should i", "pros and cons", "explain why"
        ],
        QueryComplexity.EXPERT: [
            "legal", "medical", "financial advice", "compliance",
            "regulation", "liability", "HIPAA", "GDPR"
        ]
    }

    def route(
        self,
        query: str,
        context_length_chars: int,
        team_id: str,
        budget_pct_used: float
    ) -> RoutingDecision:
        complexity = self._classify(query, context_length_chars)

        # Budget pressure: downgrade model when team budget is >80% used
        if budget_pct_used > 0.80 and complexity != QueryComplexity.EXPERT:
            return RoutingDecision("fast", f"budget_pressure:{budget_pct_used:.0%}",
                                   self._estimate_cost("fast", context_length_chars))

        routing_map = {
            QueryComplexity.SIMPLE:  "fast",
            QueryComplexity.MEDIUM:  "default",
            QueryComplexity.COMPLEX: "powerful",
            QueryComplexity.EXPERT:  "powerful",
        }
        alias = routing_map[complexity]
        return RoutingDecision(alias, f"complexity:{complexity.value}",
                               self._estimate_cost(alias, context_length_chars))

    def _classify(self, query: str, context_chars: int) -> QueryComplexity:
        q_lower = query.lower()
        for signal in self.COMPLEXITY_SIGNALS[QueryComplexity.EXPERT]:
            if signal in q_lower:
                return QueryComplexity.EXPERT
        for signal in self.COMPLEXITY_SIGNALS[QueryComplexity.COMPLEX]:
            if signal in q_lower:
                return QueryComplexity.COMPLEX
        for signal in self.COMPLEXITY_SIGNALS[QueryComplexity.SIMPLE]:
            if signal in q_lower:
                return QueryComplexity.SIMPLE
        if context_chars > 6000:
            return QueryComplexity.COMPLEX
        return QueryComplexity.MEDIUM

    def _estimate_cost(self, alias: str, context_chars: int) -> float:
        tokens = context_chars / 4   # Rough approximation
        costs = {
            "fast":    tokens / 1e6 * 0.15 + 200 / 1e6 * 0.60,
            "default": tokens / 1e6 * 2.50 + 200 / 1e6 * 10.00,
            "powerful":tokens / 1e6 * 2.50 + 400 / 1e6 * 10.00,
            "local":   0.0,
        }
        return round(costs.get(alias, 0.001), 6)
```

---

### 3.4 Cost Anomaly Detection

```python
import statistics
from dataclasses import dataclass
from typing import Optional
from collections import deque
from datetime import datetime

@dataclass
class CostAnomaly:
    detected_at: str
    team_id: str
    anomaly_type: str    # "spike" | "drift" | "runaway"
    current_rate: float  # USD/hour
    baseline_rate: float
    multiplier: float
    severity: str        # "warning" | "critical"

class CostAnomalyDetector:
    """
    Detects unusual cost patterns:
    - Spike: sudden jump vs recent baseline (2x+ in <10 min)
    - Drift: gradual increase over hours/days
    - Runaway: single request or team consuming anomalous tokens
    """
    def __init__(self, window_size: int = 60, spike_multiplier: float = 3.0):
        self.window = window_size
        self.spike_multiplier = spike_multiplier
        self._hourly: deque = deque(maxlen=24 * 7)     # 7 days of hourly data
        self._per_team: dict[str, deque] = {}

    def record(self, team_id: str, cost_usd: float, timestamp: str):
        if team_id not in self._per_team:
            self._per_team[team_id] = deque(maxlen=self.window)
        self._per_team[team_id].append({"cost": cost_usd, "ts": timestamp})

    def check_spike(self, team_id: str) -> Optional[CostAnomaly]:
        data = list(self._per_team.get(team_id, []))
        if len(data) < 10:
            return None

        # Compare last 5 records to prior 5
        recent  = statistics.mean(r["cost"] for r in data[-5:])
        prior   = statistics.mean(r["cost"] for r in data[-10:-5])
        if prior == 0:
            return None

        multiplier = recent / prior
        if multiplier >= self.spike_multiplier:
            return CostAnomaly(
                detected_at=datetime.utcnow().isoformat(),
                team_id=team_id,
                anomaly_type="spike",
                current_rate=round(recent, 4),
                baseline_rate=round(prior, 4),
                multiplier=round(multiplier, 2),
                severity="critical" if multiplier > 5 else "warning"
            )
        return None

    def check_all(self) -> list[CostAnomaly]:
        anomalies = []
        for team_id in self._per_team:
            anomaly = self.check_spike(team_id)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies
```

---

> ### 📋 Chapter Summary
>
> - **Token budget enforcement** applies daily per-team limits with a hard cutoff and 80% soft alert — preventing cost overruns from runaway requests or misconfigured prompts.
> - **Semantic caching** reuses answers for semantically similar queries (cosine similarity ≥ 0.92), typically achieving 15–30% hit rates in customer support use cases.
> - **Cost-optimised routing** assigns model aliases by query complexity (simple→fast, expert→powerful), with automatic downgrade under budget pressure.
> - **Cost anomaly detection** distinguishes spikes (sudden jumps), drifts (gradual increases), and runaways (single-request outliers) to catch cost incidents before they become invoices.

---

> ### ❓ Comprehension Questions
>
> 1. `TokenBudgetEnforcer` resets daily usage at midnight UTC. A team based in UTC+9 makes their heaviest requests between 09:00–18:00 local time (00:00–09:00 UTC). Their budget resets mid-working-day. How would you implement timezone-aware budget windows?
> 2. `SemanticCache` uses cosine similarity ≥ 0.92 as the hit threshold. At threshold 0.85, more requests hit the cache but some returned answers are for subtly different questions. At 0.98, the cache is rarely hit. How would you empirically determine the optimal threshold for a specific use case?
> 3. `CostOptimisedRouter` classifies "should I renew my contract?" as COMPLEX (matches "should i"). The answer is a simple yes/no based on a policy document — a `fast` model would suffice. What signals beyond keyword matching would improve complexity classification accuracy?
> 4. `CostAnomalyDetector.check_spike` compares the last 5 records to the prior 5. At 1 request/minute, this window is 10 minutes. A legitimate batch job generates 50 requests in 1 minute, triggering a false positive. How would you distinguish batch workloads from genuine cost spikes?
> 5. Semantic caching stores `query_embedding` and `answer` in memory. At 5000 entries with 1536-dimensional float32 embeddings, calculate the memory footprint of the embedding store alone. How would you offload this to a vector DB while maintaining sub-10ms cache lookup latency?

---

## References

### Documentation
- [OpenAI Token Usage](https://platform.openai.com/docs/guides/production-best-practices)
- [tiktoken](https://github.com/openai/tiktoken) — Token counting library.
- [GPTCache](https://github.com/zilliztech/GPTCache) — Semantic caching for LLM calls.
- [OpenCost](https://www.opencost.io) — Kubernetes cost monitoring.

---

## Chapter 4 — Knowledge Base Operations 🧪

### 4.1 Corpus Maintenance Workflows

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

class MaintenanceTaskType(str, Enum):
    INCREMENTAL_UPDATE = "incremental_update"  # Add/update new documents
    FULL_REBUILD       = "full_rebuild"         # Rebuild entire index from scratch
    DOCUMENT_DELETE    = "document_delete"      # Remove specific documents
    QUALITY_SCAN       = "quality_scan"         # Scan for low-quality chunks
    DEDUPLICATION      = "deduplication"        # Find and remove duplicate vectors
    PII_REMEDIATION    = "pii_remediation"      # Re-scan and re-redact PII

@dataclass
class MaintenanceTask:
    task_id: str
    task_type: MaintenanceTaskType
    corpus_id: str
    scheduled_at: str
    triggered_by: str           # "schedule" | "quality_drop" | "manual" | "event"
    priority: str               # "critical" | "high" | "normal"
    estimated_duration_minutes: int
    parameters: dict = field(default_factory=dict)
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None

class CorpusMaintenanceScheduler:
    """Schedules and executes corpus maintenance tasks."""

    SCHEDULED_TASKS = [
        {"type": MaintenanceTaskType.INCREMENTAL_UPDATE,
         "cron": "0 */4 * * *",   # Every 4 hours
         "priority": "normal", "estimated_minutes": 30},
        {"type": MaintenanceTaskType.QUALITY_SCAN,
         "cron": "0 2 * * 0",     # Weekly Sunday 02:00
         "priority": "normal", "estimated_minutes": 120},
        {"type": MaintenanceTaskType.DEDUPLICATION,
         "cron": "0 3 1 * *",     # Monthly, 1st at 03:00
         "priority": "normal", "estimated_minutes": 240},
    ]

    def __init__(self, ingestion_pipeline, evaluator, pii_detector):
        self.pipeline = ingestion_pipeline
        self.evaluator = evaluator
        self.pii = pii_detector

    def run_incremental_update(self, corpus_id: str, since_hours: int = 4) -> dict:
        """Ingest documents modified or created in the last N hours."""
        new_docs = self.pipeline.fetch_changed_documents(since_hours=since_hours)
        if not new_docs:
            return {"status": "no_changes", "documents_processed": 0}
        results = []
        for doc in new_docs:
            pii_result = self.pii.scan(doc.get("content", ""))
            if pii_result.has_pii:
                doc["content"] = pii_result.redacted_text
            result = self.pipeline.ingest_one(doc)
            results.append(result)
        return {
            "status": "completed",
            "documents_processed": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }

    def run_quality_scan(self, corpus_id: str) -> dict:
        """Identify low-quality chunks that should be removed or revised."""
        low_quality = []
        chunks = self.pipeline.get_all_chunks(corpus_id)
        for chunk in chunks:
            # Flag chunks shorter than 50 chars (likely noise)
            if len(chunk["content"]) < 50:
                low_quality.append({"id": chunk["id"], "reason": "too_short",
                                    "length": len(chunk["content"])})
            # Flag chunks with very low embedding norm (degenerate vectors)
            if chunk.get("embedding_norm", 1.0) < 0.1:
                low_quality.append({"id": chunk["id"], "reason": "degenerate_vector"})
        return {
            "total_chunks": len(chunks),
            "low_quality_count": len(low_quality),
            "low_quality_pct": len(low_quality) / len(chunks) if chunks else 0,
            "items": low_quality[:50]   # Return first 50 for review
        }
```

---

### 4.2 Index Refresh Strategies

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class IndexRefreshStrategy(str, Enum):
    FULL_REBUILD     = "full_rebuild"      # Rebuild everything — safe but slow
    INCREMENTAL      = "incremental"       # Add/update changed docs only
    SEGMENT_REFRESH  = "segment_refresh"   # Refresh one corpus segment at a time
    HOT_SWAP         = "hot_swap"          # Build new index in parallel, then swap

@dataclass
class IndexRefreshPolicy:
    strategy: IndexRefreshStrategy
    trigger: str          # "schedule" | "staleness" | "quality_drop" | "size_threshold"
    max_staleness_hours: int = 24
    quality_drop_threshold: float = 0.05  # Trigger rebuild if recall drops by 5%
    size_increase_pct: float = 0.20       # Rebuild when corpus grows by 20%

class HotSwapIndexManager:
    """
    Builds a new index in the background while the current index
    serves production traffic. Swaps atomically when new index is validated.
    Zero downtime, no query degradation during rebuild.
    """
    def __init__(self, vector_db, embedding_service, evaluator):
        self.vdb = vector_db
        self.embedder = embedding_service
        self.evaluator = evaluator

    def build_shadow_index(
        self,
        corpus_id: str,
        new_version: str,
        active_collection: str
    ) -> str:
        """Build new index in shadow collection. Returns shadow collection name."""
        shadow_collection = f"{corpus_id}_shadow_{new_version}"
        print(f"  Building shadow index: {shadow_collection}")

        # Create shadow collection
        self.vdb.create_collection(shadow_collection)

        # Fetch all documents, re-embed, store in shadow
        all_docs = self._fetch_corpus(corpus_id)
        batch_size = 100
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i+batch_size]
            texts = [d["content"] for d in batch]
            embeddings = self.embedder.embed(texts)
            self.vdb.upsert(shadow_collection,
                            ids=[d["id"] for d in batch],
                            vectors=embeddings,
                            payloads=[d.get("metadata", {}) for d in batch])

        print(f"  ✓ Shadow index built: {len(all_docs)} documents")
        return shadow_collection

    def validate_shadow(
        self,
        shadow_collection: str,
        eval_dataset: list[dict],
        min_recall: float = 0.80
    ) -> tuple[bool, dict]:
        """Run evaluation against shadow index. Returns (passed, metrics)."""
        metrics = self.evaluator.evaluate_retrieval(
            collection=shadow_collection,
            eval_dataset=eval_dataset
        )
        passed = metrics.get("recall_at_5", 0) >= min_recall
        return passed, metrics

    def swap(self, active_collection: str, shadow_collection: str,
             retire_after_hours: int = 24):
        """Atomic swap: shadow becomes active, old active is retired."""
        print(f"  Swapping {active_collection} → {shadow_collection}")
        self.vdb.alias_collection(shadow_collection, alias=active_collection + "_live")
        # Schedule deletion of old collection after retire_after_hours
        print(f"  Old collection scheduled for deletion in {retire_after_hours}h")

    def _fetch_corpus(self, corpus_id: str) -> list[dict]:
        return []  # Delegate to corpus store
```

---

### 4.3 Quality Regression Detection on Updates

```python
from dataclasses import dataclass
from typing import Optional
import statistics

@dataclass
class IndexQualityBaseline:
    corpus_id: str
    index_version: str
    recall_at_5: float
    mrr: float               # Mean Reciprocal Rank
    faithfulness: float
    eval_date: str
    eval_dataset_size: int

class QualityRegressionDetector:
    """
    Compares new index quality against baseline.
    Blocks hot-swap if quality regresses beyond threshold.
    """
    def __init__(self, regression_threshold: float = 0.05):
        self.threshold = regression_threshold   # 5% relative regression
        self._baselines: dict[str, IndexQualityBaseline] = {}

    def set_baseline(self, baseline: IndexQualityBaseline):
        self._baselines[baseline.corpus_id] = baseline

    def check_regression(
        self,
        corpus_id: str,
        new_metrics: dict
    ) -> tuple[bool, list[str]]:
        """
        Returns (no_regression, list_of_regressions).
        True = safe to swap. False = regression detected, block swap.
        """
        baseline = self._baselines.get(corpus_id)
        if not baseline:
            print(f"  No baseline for {corpus_id} — allowing swap")
            return True, []

        regressions = []
        for metric, new_value in new_metrics.items():
            baseline_value = getattr(baseline, metric, None)
            if baseline_value is None or baseline_value == 0:
                continue
            drop = (baseline_value - new_value) / baseline_value
            if drop > self.threshold:
                regressions.append(
                    f"{metric}: {baseline_value:.4f} → {new_value:.4f} "
                    f"(dropped {drop:.1%})"
                )

        return len(regressions) == 0, regressions

    def update_baseline(self, corpus_id: str, new_metrics: dict,
                        index_version: str, eval_date: str):
        """Update baseline after successful swap."""
        self._baselines[corpus_id] = IndexQualityBaseline(
            corpus_id=corpus_id,
            index_version=index_version,
            recall_at_5=new_metrics.get("recall_at_5", 0),
            mrr=new_metrics.get("mrr", 0),
            faithfulness=new_metrics.get("faithfulness", 0),
            eval_date=eval_date,
            eval_dataset_size=new_metrics.get("eval_dataset_size", 0)
        )
```

---

### 🧪 Hands-on Lab: Operational Runbook Simulator

**Objective:** Simulate key operational scenarios — budget enforcement, cache hits, cost routing, and canary gate evaluation — in a single runnable script.

```python
#!/usr/bin/env python3
"""
ops_runbook_simulator.py — Operational patterns demo.
Simulates: budget enforcement, semantic cache, model routing, canary gates.
No external dependencies required.
"""

import hashlib, time, statistics
from dataclasses import dataclass, field
from typing import Optional

# ── 1. Token budget enforcer ─────────────────────────────────────
print("=" * 55)
print("  1. Token Budget Enforcement")
print("=" * 55)

BUDGETS = {
    "team-a": {"daily_tokens": 100_000, "daily_cost": 5.0, "used_tokens": 0, "used_cost": 0.0},
    "team-b": {"daily_tokens": 50_000,  "daily_cost": 2.0, "used_tokens": 0, "used_cost": 0.0},
}

def check_budget(team_id: str, tokens: int, cost: float) -> tuple[bool, str]:
    b = BUDGETS.get(team_id)
    if not b:
        return True, "no_limit"
    if b["used_tokens"] + tokens > b["daily_tokens"]:
        return False, f"TOKEN_EXCEEDED: {b['used_tokens']+tokens:,} > {b['daily_tokens']:,}"
    if b["used_cost"] + cost > b["daily_cost"]:
        return False, f"COST_EXCEEDED: ${b['used_cost']+cost:.2f} > ${b['daily_cost']:.2f}"
    if (b["used_tokens"] + tokens) / b["daily_tokens"] >= 0.80:
        print(f"  ⚠  [{team_id}] Budget at 80% — alert triggered")
    b["used_tokens"] += tokens
    b["used_cost"]   += cost
    return True, "allowed"

requests = [
    ("team-a", 2_000, 0.10),
    ("team-a", 80_000, 4.50),   # Will push to 82% — warning
    ("team-a", 25_000, 1.20),   # Will exceed token limit
    ("team-b", 10_000, 0.50),
    ("team-b", 45_000, 1.80),   # Will exceed cost limit
]
for team, tokens, cost in requests:
    allowed, reason = check_budget(team, tokens, cost)
    icon = "✓" if allowed else "✗"
    print(f"  {icon} [{team}] {tokens:,} tokens ${cost:.2f} → {reason}")

# ── 2. Semantic cache simulator ──────────────────────────────────
print("\n" + "=" * 55)
print("  2. Semantic Cache")
print("=" * 55)

CACHE = {}    # query_hash → answer

def simple_similarity(q1: str, q2: str) -> float:
    """Toy similarity: word overlap / max_words."""
    s1, s2 = set(q1.lower().split()), set(q2.lower().split())
    return len(s1 & s2) / max(len(s1 | s2), 1)

def cache_lookup(query: str, threshold: float = 0.6) -> Optional[str]:
    best_sim, best_ans = 0.0, None
    for cached_q, cached_a in CACHE.items():
        sim = simple_similarity(query, cached_q)
        if sim > best_sim:
            best_sim, best_ans = sim, cached_a
    return best_ans if best_sim >= threshold else None

def cache_store(query: str, answer: str):
    CACHE[query] = answer

queries = [
    ("What is the enterprise return policy?",      "Enterprise: 90-day returns."),
    ("How long for enterprise returns?",           None),  # Should hit cache
    ("enterprise return window duration?",         None),  # Should hit cache
    ("What is the Professional plan pricing?",     "$150/month, 25 users."),
    ("How much does Professional plan cost?",      None),  # Should hit cache
    ("What is the cancellation policy?",           None),  # Should miss
]

hits, misses = 0, 0
for query, ground_truth_answer in queries:
    cached = cache_lookup(query)
    if cached:
        print(f"  HIT  '{query[:45]}...' → served from cache")
        hits += 1
    else:
        answer = ground_truth_answer or f"[LLM answer for: {query[:30]}]"
        cache_store(query, answer)
        print(f"  MISS '{query[:45]}' → LLM called, cached")
        misses += 1

print(f"\n  Cache hit rate: {hits}/{hits+misses} = {hits/(hits+misses):.0%}")

# ── 3. Cost-optimised routing ────────────────────────────────────
print("\n" + "=" * 55)
print("  3. Cost-Optimised Model Routing")
print("=" * 55)

ROUTES = {
    "simple":  ("fast",     0.000040),
    "medium":  ("default",  0.000650),
    "complex": ("powerful", 0.001200),
    "expert":  ("powerful", 0.001500),
}

def classify_query(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["legal","medical","compliance","gdpr","hipaa"]):
        return "expert"
    if any(w in q for w in ["compare","difference","analyse","recommend","should i"]):
        return "complex"
    if any(w in q for w in ["what is","how much","when","where"]):
        return "simple"
    return "medium"

test_queries = [
    "What are the office hours?",
    "Compare enterprise and professional plans",
    "Should I choose annual or monthly billing?",
    "What are GDPR implications for storing user data in the EU?",
    "How do I reset my password?",
]

total_cost = 0.0
for q in test_queries:
    complexity = classify_query(q)
    model, cost = ROUTES[complexity]
    total_cost += cost
    print(f"  [{complexity:<8}] {model:<10} ${cost:.6f}  {q[:50]}")

print(f"\n  Total estimated cost: ${total_cost:.4f} | Avg: ${total_cost/len(test_queries):.6f}")

# ── 4. Canary gate evaluation ────────────────────────────────────
print("\n" + "=" * 55)
print("  4. Canary Quality Gate Simulation")
print("=" * 55)

@dataclass
class CanaryWindow:
    weight_pct: int
    error_rate: float
    latency_p99_ms: float
    refusal_rate: float
    faithfulness: float
    request_count: int

CANARY_WINDOWS = [
    CanaryWindow(5,   0.005, 950,  0.08, 0.91, 120),
    CanaryWindow(15,  0.008, 1100, 0.09, 0.90, 350),
    CanaryWindow(25,  0.012, 1400, 0.12, 0.88, 600),
    CanaryWindow(50,  0.025, 2800, 0.18, 0.83, 1200),  # latency+faithfulness fail
    CanaryWindow(75,  0.010, 1200, 0.10, 0.91, 1800),
    CanaryWindow(100, 0.006, 980,  0.08, 0.92, 2400),
]

GATES = {
    "error_rate":     (0.02,   "<="),
    "latency_p99_ms": (2000,   "<="),
    "refusal_rate":   (0.20,   "<="),
    "faithfulness":   (0.85,   ">="),
}

current_weight = 0
for window in CANARY_WINDOWS:
    failures = []
    for metric, (threshold, op) in GATES.items():
        value = getattr(window, metric)
        if op == "<=" and value > threshold:
            failures.append(f"{metric}={value} > {threshold}")
        elif op == ">=" and value < threshold:
            failures.append(f"{metric}={value} < {threshold}")

    if failures:
        print(f"  ✗ GATE FAILED at {window.weight_pct}%: {failures}")
        print(f"    → Rolling back to {current_weight}%")
        break
    else:
        current_weight = window.weight_pct
        print(f"  ✓ {window.weight_pct:>3}% gates passed "
              f"(err={window.error_rate:.3f} p99={window.latency_p99_ms:.0f}ms "
              f"faith={window.faithfulness:.2f})")

print("\n" + "=" * 55)
print(f"  Simulation complete. Final canary weight: {current_weight}%")
```

**Run the lab:**
```bash
python ops_runbook_simulator.py
```

**Extensions:**
- Add a `circuit_breaker` simulation: after 5 consecutive LLM call failures, circuit opens; subsequent requests use cache or fallback
- Extend the canary simulation with an `auto_rollback_fn` that patches a Kubernetes service selector (mock it with a print statement)
- Add a `cost_anomaly_check` that flags any team whose per-request cost is > 3x its 10-request rolling average

---

> ### 📋 Chapter Summary
>
> - **Corpus maintenance** runs on predictable schedules (incremental every 4h, quality scan weekly, deduplication monthly) plus event-triggered runs (quality drop, consent withdrawal).
> - **Hot-swap index refresh** builds a new index in a shadow collection, validates it against the eval dataset, and swaps atomically — production traffic is never interrupted.
> - **Quality regression detection** blocks index swaps when recall, MRR, or faithfulness drop more than 5% relative to the established baseline.
> - The **operations lab** demonstrates all four operational patterns in a single executable: budget enforcement, semantic cache, model routing, and canary gate evaluation.

---

> ### ❓ Comprehension Questions
>
> 1. `CorpusMaintenanceScheduler` runs incremental updates every 4 hours. A document is deleted from the source system at 09:00 and the next ingestion run is at 12:00. For 3 hours, users can receive answers citing this deleted document. Design a deletion propagation mechanism with sub-1-hour latency.
> 2. `HotSwapIndexManager.validate_shadow` requires recall ≥ 0.80 before swapping. The shadow index passes with recall=0.81 (borderline). The active index has recall=0.86. The hot swap would degrade quality by 5 percentage points — within the regression threshold. How would you add a comparison gate (shadow vs active, not shadow vs absolute threshold)?
> 3. `QualityRegressionDetector` updates the baseline after a successful swap. If quality silently degrades across 5 successive swaps (each drop is small enough to pass the threshold), the baseline drifts downward. How would you implement a floor — a minimum absolute quality below which no swap is allowed regardless of relative regression?
> 4. The `CostOptimisedRouter` routes to `fast` under budget pressure. A high-stakes legal query arrives when the team is at 85% budget. The router downgrades it to `fast`. What safeguard prevents budget pressure from downgrading expert-tier queries?
> 5. The canary simulation fails at 50% due to `faithfulness=0.83 < 0.85`. The new prompt version has a different citation format that scores lower on the LLM judge but users prefer it. How would you adjust the evaluation methodology to separate judge-format sensitivity from genuine quality regression?

---

## References

### Documentation
- [Argo Workflows](https://argoproj.github.io/workflows/) — Kubernetes-native workflow orchestration.
- [Prefect](https://docs.prefect.io) — Python-native workflow orchestration.
- [Qdrant Collections API](https://qdrant.tech/documentation/concepts/collections/) — Collection management for hot-swap.
- [Google SRE — Release Engineering](https://sre.google/sre-book/release-engineering/)

### Books
- *The DevOps Handbook* — Kim, Humble, Debois & Willis (IT Revolution). Release engineering and deployment patterns.

---

> **Navigation**
> [← Part XV — Governance](part_15_governance.md) | [→ Part XVII — Reference Architectures](part_17_reference_architectures.md)
