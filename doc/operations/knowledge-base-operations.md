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
> [← Part XV — Governance](../governance/index.md) | [→ Part XVII — Reference Architectures](../reference_architectures/index.md)

---
[« Back to operations Index](index.md) | [🏠 Home](../index.md)