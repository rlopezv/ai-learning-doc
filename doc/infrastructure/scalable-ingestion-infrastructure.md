## Chapter 3 — Scalable Ingestion Infrastructure

### 3.1 Ingestion Architecture Patterns

Document ingestion is typically the most resource-intensive offline workload in a RAG system. Three patterns accommodate different scale requirements:

**Synchronous in-process** — suitable for small corpora (< 10K documents) or development. The API service ingests documents as part of a request handler. Simple but blocking.

**Queue-backed async** — suitable for production systems with continuous or batch updates. Documents are queued; worker processes consume and ingest asynchronously. Scales by adding workers.

**Streaming** — suitable for near-real-time knowledge base updates. Documents are published to a message stream (Kafka, Pub/Sub); ingestion workers consume in near-real time.

```python
from enum import Enum

class IngestionPattern(str, Enum):
    SYNCHRONOUS = "synchronous"
    ASYNC_QUEUE = "async_queue"   # Celery/RQ
    STREAMING   = "streaming"     # Kafka/Pub/Sub

def select_pattern(
    documents_per_day: int,
    max_latency_minutes: int,
    operational_complexity: str   # "low" | "medium" | "high"
) -> IngestionPattern:
    if documents_per_day < 1000 and max_latency_minutes > 60:
        return IngestionPattern.SYNCHRONOUS
    if max_latency_minutes > 15 or operational_complexity == "low":
        return IngestionPattern.ASYNC_QUEUE
    return IngestionPattern.STREAMING
```

---

### 3.2 Queue-Backed Ingestion with Celery

```python
# apps/ingestion/src/celery_app.py
from celery import Celery
from celery.utils.log import get_task_logger

app = Celery(
    "ingestion",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
    include=["src.tasks"]
)
app.conf.update(
    task_routes={
        "src.tasks.ingest_document":  {"queue": "ingestion"},
        "src.tasks.rebuild_index":    {"queue": "index_rebuild"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,              # Acknowledge AFTER completion
    task_reject_on_worker_lost=True,  # Re-queue if worker dies
    worker_prefetch_multiplier=1,     # One task per worker at a time
    task_soft_time_limit=300,         # Warn at 5 min
    task_time_limit=600,              # Kill at 10 min
    task_max_retries=3,
    task_default_retry_delay=60,
)

logger = get_task_logger(__name__)

@app.task(bind=True, max_retries=3, default_retry_delay=60, queue="ingestion")
def ingest_document(self, document: dict, config: dict) -> dict:
    """Ingest one document. Idempotent — safe to retry."""
    doc_id = document.get("id", "unknown")
    logger.info(f"Ingesting {doc_id}")
    try:
        from src.pipeline import IngestionPipeline
        pipeline = IngestionPipeline.from_config(config)
        result = pipeline.ingest_one(document)
        return result
    except Exception as exc:
        logger.error(f"Failed {doc_id}: {exc}")
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))

@app.task(queue="ingestion")
def ingest_batch(documents: list[dict], config: dict) -> dict:
    """Fan out to individual tasks for parallelism."""
    from celery import group
    subtasks = group(ingest_document.s(doc, config) for doc in documents)
    results = subtasks.apply_async()
    return {"submitted": len(documents), "group_id": results.id}
```

```python
# API endpoint — submit to Celery queue, return task ID for polling
from fastapi import APIRouter
from src.tasks import ingest_document, ingest_batch, app as celery_app

router = APIRouter(prefix="/v2/ingest", tags=["ingestion"])

@router.post("/document")
async def submit_document(doc: dict, config: dict):
    task = ingest_document.apply_async(args=[doc, config])
    return {"task_id": task.id, "status": "submitted"}

@router.post("/batch")
async def submit_batch(documents: list[dict], config: dict):
    result = ingest_batch.apply_async(args=[documents, config])
    return {"batch_id": result.id, "count": len(documents)}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": result.status,
            "result": result.result if result.ready() else None}
```

---

### 3.3 Streaming Ingestion with Kafka 🔓

```python
# apps/ingestion/src/kafka_consumer.py
# On-premise alternative: Apache Kafka for near-real-time document updates

import json
from dataclasses import dataclass

@dataclass
class DocumentEvent:
    event_type: str          # "created" | "updated" | "deleted"
    document_id: str
    content: str
    metadata: dict
    source_system: str
    timestamp: str

class KafkaIngestionConsumer:
    """
    Consumes document events from Kafka.
    At-least-once semantics: commit offsets only after successful batch.
    """
    def __init__(self, bootstrap_servers: str, topic: str,
                 group_id: str, pipeline):
        from confluent_kafka import Consumer
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,    # Manual commit after processing
            "max.poll.interval.ms": 300000,
        })
        self.topic = topic
        self.pipeline = pipeline

    def run(self, batch_size: int = 10):
        self.consumer.subscribe([self.topic])
        batch = []
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    if batch:
                        self._process_batch(batch)
                        self.consumer.commit()
                        batch = []
                    continue
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue
                batch.append(DocumentEvent(**json.loads(msg.value().decode())))
                if len(batch) >= batch_size:
                    self._process_batch(batch)
                    self.consumer.commit()  # Only after successful batch
                    batch = []
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()

    def _process_batch(self, batch: list[DocumentEvent]):
        creates = [e for e in batch if e.event_type in ("created", "updated")]
        deletes = [e for e in batch if e.event_type == "deleted"]
        if creates:
            self.pipeline.ingest([{"id": e.document_id, "content": e.content,
                                   **e.metadata} for e in creates])
        for e in deletes:
            self.pipeline.delete(e.document_id)
        print(f"  Batch processed: {len(creates)} upserts, {len(deletes)} deletes")
```

---

### 3.4 Idempotency and Exactly-Once Ingestion

```python
import hashlib, json
from pathlib import Path

class IdempotentIngestionPipeline:
    """
    Content-hash based idempotency: re-ingesting unchanged documents
    is a no-op. Enables safe retries and prevents duplicate vectors.
    """
    def __init__(self, base_pipeline, state_store_path: str = "ingestion/state.json"):
        self.pipeline = base_pipeline
        self.state_path = Path(state_store_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, str] = {}
        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text())

    def ingest_one(self, document: dict) -> dict:
        doc_id = document["id"]
        content_hash = hashlib.sha256(
            document.get("content", "").encode()
        ).hexdigest()[:16]

        if self._state.get(doc_id) == content_hash:
            return {"doc_id": doc_id, "status": "skipped", "reason": "content unchanged"}

        result = self.pipeline.ingest_one(document)
        self._state[doc_id] = content_hash
        self._save_state()
        return {**result, "status": "ingested"}

    def delete(self, doc_id: str):
        self.pipeline.delete(doc_id)
        self._state.pop(doc_id, None)
        self._save_state()

    def _save_state(self):
        self.state_path.write_text(json.dumps(self._state, indent=2))
```

---

> ### 📋 Chapter Summary
>
> - Choose ingestion pattern based on volume and latency: synchronous for dev/small corpora; Celery queue for production batch; Kafka for near-real-time.
> - `task_acks_late=True` and `task_reject_on_worker_lost=True` are required for at-least-once Celery delivery.
> - **Idempotent ingestion** using content hashes prevents duplicate vectors and makes retries safe.
> - Kafka consumers must commit offsets only after the batch is successfully processed.

---

> ### ❓ Comprehension Questions
>
> 1. A Celery worker is processing 50 documents when the Kubernetes node fails. With `task_acks_late=True`, what happens to the task? What would happen without it?
> 2. `IdempotentIngestionPipeline` stores state in a local JSON file. Two ingestion workers run in parallel. What race condition exists, and how would you fix it with a distributed state store?
> 3. The Kafka consumer commits offsets after a batch of 10. Message 10 fails after 9 succeed. What is re-processed on restart? What does "at-least-once" imply for vector DB state?
> 4. A document is updated 5 times within a minute. The Kafka topic receives 5 events. How would you prevent 5 redundant re-embeddings without sacrificing update freshness?
> 5. `worker_prefetch_multiplier=1` prevents workers from taking more than one task at a time. A DevOps engineer sets it to 4 to increase throughput. What risks does this introduce for memory-intensive ingestion tasks?

---

## References

### Documentation
- [Celery Documentation](https://docs.celeryq.dev/en/stable/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Redis Documentation](https://redis.io/docs/)
- [Confluent Kafka Python](https://docs.confluent.io/kafka-clients/python/current/overview.html)

---

---
[« Back to infrastructure Index](index.md) | [🏠 Home](../../index.md)