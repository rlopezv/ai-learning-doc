## Chapter 3 — Structured Logging

### 3.1 Log Schema for RAG Systems

Unstructured log messages ("Answering query for user") cannot be queried, aggregated, or analysed at scale. Structured logs encode every field as a typed key-value pair, enabling exact and range queries in log analytics systems.

```python
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
from enum import Enum
import json

class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"

class EventType(str, Enum):
    REQUEST_RECEIVED       = "request.received"
    RETRIEVAL_COMPLETED    = "retrieval.completed"
    GENERATION_COMPLETED   = "generation.completed"
    REQUEST_COMPLETED      = "request.completed"
    QUALITY_SCORED         = "quality.scored"
    REFUSAL_RETURNED       = "refusal.returned"
    ERROR_OCCURRED         = "error.occurred"
    CACHE_HIT              = "cache.hit"

@dataclass
class StructuredLogRecord:
    """
    Canonical log schema for all RAG service events.
    All fields are typed; no free-form message strings for queryable data.
    """
    # Mandatory fields — every log record
    timestamp: str
    level: str
    event_type: str
    service_id: str
    team_id: str
    request_id: str
    trace_id: str

    # Message (human-readable, not parsed programmatically)
    message: str = ""

    # Request context
    query_length_chars: Optional[int] = None
    session_id: Optional[str] = None

    # Retrieval context
    retrieved_count: Optional[int] = None
    retrieval_latency_ms: Optional[float] = None
    top_score: Optional[float] = None

    # Generation context
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    generation_latency_ms: Optional[float] = None
    is_refusal: Optional[bool] = None

    # Response context
    answer_length_chars: Optional[int] = None
    citation_count: Optional[int] = None
    total_latency_ms: Optional[float] = None
    http_status: Optional[int] = None

    # Cost
    estimated_cost_usd: Optional[float] = None

    # Quality
    faithfulness_score: Optional[float] = None
    relevancy_score: Optional[float] = None
    user_rating: Optional[float] = None

    # Error context
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Versions
    app_version: Optional[str] = None
    index_version: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {k: v for k, v in asdict(self).items() if v is not None},
            default=str
        )
```

---

### 3.2 Implementing Structured Logging

```python
import logging
import sys
from datetime import datetime

class StructuredLogger:
    """
    Wraps Python's logging with structured JSON output.
    Produces one JSON object per line — compatible with
    Elasticsearch, Loki, CloudWatch Logs Insights, and GCP Logs.
    """
    def __init__(self, service_id: str, app_version: str):
        self.service_id = service_id
        self.app_version = app_version
        self._logger = logging.getLogger(service_id)
        self._configure_json_handler()

    def _configure_json_handler(self):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

    def emit(self, record: StructuredLogRecord):
        record.service_id = record.service_id or self.service_id
        record.app_version = record.app_version or self.app_version
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }
        self._logger.log(
            level_map.get(record.level, logging.INFO),
            record.to_json()
        )

    def request_completed(self, trace: "RAGRequestTrace"):
        self.emit(StructuredLogRecord(
            timestamp=datetime.utcnow().isoformat(),
            level=LogLevel.INFO,
            event_type=EventType.REQUEST_COMPLETED,
            service_id=trace.service_id,
            team_id=trace.team_id,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            message="RAG request completed",
            query_length_chars=trace.query_length_chars,
            retrieved_count=len(trace.retrieved_doc_ids),
            retrieval_latency_ms=round(trace.retrieval_latency_ms, 2),
            model_name=trace.model_name,
            prompt_version=trace.prompt_version,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            generation_latency_ms=round(trace.generation_latency_ms, 2),
            is_refusal=trace.is_refusal,
            answer_length_chars=trace.answer_length_chars,
            total_latency_ms=round(trace.total_latency_ms, 2),
            http_status=trace.http_status,
            estimated_cost_usd=trace.estimated_cost_usd,
            index_version=trace.index_version,
        ))

    def error(self, request_id: str, trace_id: str,
              team_id: str, error_type: str, error_message: str):
        self.emit(StructuredLogRecord(
            timestamp=datetime.utcnow().isoformat(),
            level=LogLevel.ERROR,
            event_type=EventType.ERROR_OCCURRED,
            service_id=self.service_id,
            team_id=team_id,
            request_id=request_id,
            trace_id=trace_id,
            message=f"Error: {error_type}",
            error_type=error_type,
            error_message=error_message[:500]  # Truncate for log safety
        ))


class JsonFormatter(logging.Formatter):
    """Output log records as raw JSON (already formatted by StructuredLogRecord)."""
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


# Example log query patterns (Elasticsearch / Loki / CloudWatch)
LOG_QUERY_EXAMPLES = {
    "high_latency_requests": {
        "description": "Requests with P99 > 3000ms in the last hour",
        "loki": '{service_id="rag-api"} | json | total_latency_ms > 3000',
        "elasticsearch": {
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"total_latency_ms": {"gt": 3000}}},
                        {"range": {"@timestamp": {"gte": "now-1h"}}}
                    ]
                }
            }
        }
    },
    "refusal_rate_by_team": {
        "description": "Refusal rate per team in last 24h",
        "loki": 'sum by (team_id) (count_over_time({service_id="rag-api"} | json | is_refusal="true" [24h])) / sum by (team_id) (count_over_time({service_id="rag-api"} | json [24h]))',
    },
    "expensive_requests": {
        "description": "Top 10 most expensive requests today",
        "loki": '{service_id="rag-api"} | json | sort by (estimated_cost_usd) desc | limit 10',
    },
}
```

---

### 3.3 Log Sampling and Cost Control

At high traffic volumes, logging every request becomes prohibitively expensive. Sampling reduces log volume while preserving visibility into important events.

```python
import random
import hashlib
from enum import Enum

class SamplingDecision(str, Enum):
    ALWAYS   = "always"    # Always log regardless of sampling rate
    SAMPLED  = "sampled"   # Log at configured rate
    NEVER    = "never"     # Drop this log record

class AdaptiveLogSampler:
    """
    Adaptive log sampler:
    - Errors:           always logged (rate=1.0)
    - Refusals:         always logged (quality signal)
    - High latency:     always logged (performance signal)
    - High cost:        always logged (cost signal)
    - Normal requests:  sampled at configured base_rate
    """
    def __init__(
        self,
        base_rate: float = 0.10,         # Sample 10% of normal requests
        high_latency_threshold_ms: float = 2000,
        high_cost_threshold_usd: float = 0.01
    ):
        self.base_rate = base_rate
        self.high_latency_ms = high_latency_threshold_ms
        self.high_cost_usd = high_cost_threshold_usd

    def should_log(self, trace: "RAGRequestTrace") -> tuple[bool, str]:
        """Returns (should_log, reason)."""
        # Always log errors
        if trace.http_status >= 500:
            return True, "always:error"

        # Always log refusals (quality monitoring)
        if trace.is_refusal:
            return True, "always:refusal"

        # Always log high-latency requests (performance debugging)
        if trace.total_latency_ms > self.high_latency_ms:
            return True, "always:high_latency"

        # Always log high-cost requests (cost monitoring)
        if trace.estimated_cost_usd > self.high_cost_usd:
            return True, "always:high_cost"

        # Deterministic sampling based on request_id for reproducibility
        # (same request always makes same sampling decision across retries)
        sample_hash = int(hashlib.md5(trace.request_id.encode()).hexdigest(), 16)
        if (sample_hash % 10000) < int(self.base_rate * 10000):
            return True, f"sampled:{self.base_rate:.0%}"

        return False, "dropped"

    def get_sample_rate_for_metric(self, reason: str) -> float:
        """Return effective sample rate for metric adjustment."""
        if reason.startswith("always"):
            return 1.0
        return self.base_rate
```

---

### 3.4 Log-Based Quality Signals

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

class LogQualityAnalyser:
    """
    Computes quality signals from recent log records.
    Acts as a lightweight alternative to full evaluation pipeline
    for real-time quality monitoring.
    """
    def __init__(self, window_minutes: int = 60):
        self.window = timedelta(minutes=window_minutes)
        self._records: deque = deque()

    def ingest(self, log_record: dict):
        """Ingest a parsed log record."""
        self._records.append({
            **log_record,
            "_ingested_at": datetime.utcnow()
        })
        self._evict_old()

    def compute_signals(self) -> dict:
        """Compute quality signals over the current window."""
        records = list(self._records)
        if not records:
            return {}

        total = len(records)
        refusals = sum(1 for r in records if r.get("is_refusal"))
        errors = sum(1 for r in records if (r.get("http_status") or 0) >= 500)
        high_latency = sum(1 for r in records
                           if (r.get("total_latency_ms") or 0) > 2000)

        latencies = [r["total_latency_ms"] for r in records
                     if r.get("total_latency_ms")]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        by_team = defaultdict(lambda: {"total": 0, "refusals": 0, "cost": 0.0})
        for r in records:
            team = r.get("team_id", "unknown")
            by_team[team]["total"] += 1
            if r.get("is_refusal"):
                by_team[team]["refusals"] += 1
            by_team[team]["cost"] += r.get("estimated_cost_usd") or 0

        return {
            "window_minutes": self.window.seconds // 60,
            "total_requests": total,
            "error_rate": round(errors / total, 4) if total else 0,
            "refusal_rate": round(refusals / total, 4) if total else 0,
            "high_latency_rate": round(high_latency / total, 4) if total else 0,
            "latency_p50_ms": round(latencies_sorted[n // 2], 1) if n else None,
            "latency_p99_ms": round(latencies_sorted[int(n * 0.99)], 1) if n else None,
            "total_cost_usd": round(sum(r.get("estimated_cost_usd") or 0
                                        for r in records), 4),
            "by_team": {
                team: {
                    "requests": v["total"],
                    "refusal_rate": round(v["refusals"] / v["total"], 4)
                                    if v["total"] else 0,
                    "cost_usd": round(v["cost"], 4)
                }
                for team, v in by_team.items()
            }
        }

    def _evict_old(self):
        cutoff = datetime.utcnow() - self.window
        while self._records and self._records[0]["_ingested_at"] < cutoff:
            self._records.popleft()
```

---

> ### 📋 Chapter Summary
>
> - **Structured logs** encode every field as typed key-value pairs — enabling exact queries, aggregations, and anomaly detection at scale.
> - `StructuredLogRecord` with `EventType` enumeration provides a consistent schema across all RAG service events.
> - **Adaptive sampling** always logs errors, refusals, high-latency, and high-cost requests; samples normal requests at a configurable rate — typically 10%.
> - `LogQualityAnalyser` computes quality signals (refusal rate, error rate, latency percentiles, cost) from recent logs, enabling lightweight real-time quality monitoring.

---

> ### ❓ Comprehension Questions
>
> 1. A log record omits `query_text` by default for privacy but includes it when the request is sampled. An engineer needs to debug a specific refusal. How would the `should_log` method ensure the query text is available for all refusals?
> 2. `AdaptiveLogSampler` uses `hashlib.md5(trace.request_id)` for deterministic sampling. Why is determinism important here, and what failure mode does it prevent compared to `random.random()`?
> 3. Log storage costs $0.50/GB. At 1000 RPS with an average log record size of 1KB, calculate the daily log volume and cost at: (a) 100% sampling rate, (b) 10% base rate with always-log for errors (2%), refusals (5%), and high-latency (3%).
> 4. `LogQualityAnalyser.compute_signals()` holds all window records in memory. At 1000 RPS with a 60-minute window, how many records could this deque hold, and what is the approximate memory cost?
> 5. The log schema includes `index_version`. After correlating logs from the past week, you find that refusal rate is 8% on `index_version=v23` but 22% on `index_version=v24`. What investigation steps follow this finding?

---

## References

### Documentation
- [Grafana Loki](https://grafana.com/docs/loki/latest/) — Log aggregation system.
- [OpenSearch / Elasticsearch](https://opensearch.org/docs/latest/) — Log analytics.
- [Structlog](https://www.structlog.org/en/stable/) — Python structured logging library.
- [AWS CloudWatch Logs Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AnalyzingLogData.html)

---

---
[« Back to observability Index](index.md) | [🏠 Home](../index.md)