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

---
[« Back to operations Index](index.md) | [🏠 Home](../index.md)