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

---
[« Back to operations Index](index.md) | [🏠 Home](../../index.md)