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
> [← Part X — Testing LLM Systems](../testing/index.md) | [→ Part XII — Infrastructure](../infrastructure/index.md)

---
[« Back to platform_engineering Index](index.md) | [🏠 Home](../index.md)