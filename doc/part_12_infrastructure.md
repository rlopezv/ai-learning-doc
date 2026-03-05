# Part XII — Infrastructure

---

> **Navigation**
> [← Part XI — AI Platform Engineering](part_11_platform_engineering.md) | [→ Part XIII — Observability](part_13_observability.md)

---

## Contents

- [Chapter 1 — Containerisation and Orchestration](#chapter-1--containerisation-and-orchestration)
- [Chapter 2 — Vector Database Infrastructure](#chapter-2--vector-database-infrastructure)
- [Chapter 3 — Scalable Ingestion Infrastructure](#chapter-3--scalable-ingestion-infrastructure)
- [Chapter 4 — On-Premise LLM Platform Design 🔓](#chapter-4--on-premise-llm-platform-design-)
- [Chapter 5 — Infrastructure as Code 🧪](#chapter-5--infrastructure-as-code-)

---

## Chapter 1 — Containerisation and Orchestration

### 1.1 Why Containers Matter for AI Workloads

AI workloads have infrastructure requirements that make container discipline particularly important.

**Dependency complexity.** LLM services depend on specific Python versions, CUDA toolkit versions, model libraries with complex native dependencies (`torch`, `transformers`, `sentence-transformers`), and provider SDKs with tight version coupling. Without containers, dependency conflicts across services are nearly inevitable. A single container image encapsulates the exact combination that was tested.

**Reproducibility.** A model serving container built today must behave identically when deployed six months later. The container image is the reproducibility artifact — pinned base image, pinned Python dependencies, pinned model weights (or a deterministic download path).

**Isolation of GPU workloads.** Services that use GPUs (embedding servers, fine-tuning jobs, local LLM serving) must be scheduled on GPU-equipped nodes. Containers with NVIDIA device plugin labels enable the Kubernetes scheduler to place these workloads correctly.

**Environment parity.** The same container that runs locally via Docker Compose runs in CI and in production Kubernetes. "It works on my machine" failures are eliminated by construction.

---

### 1.2 Dockerfile Patterns for LLM Services

```dockerfile
# apps/api/Dockerfile — Production RAG API (Python/FastAPI)
# Multi-stage build: separate build and runtime environments

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*
COPY requirements.lock ./
RUN pip install --no-cache-dir --target=/app/deps -r requirements.lock

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
COPY --from=builder /app/deps /app/deps
ENV PYTHONPATH=/app/deps
COPY apps/api/src ./src
COPY libs/ ./libs

LABEL org.opencontainers.image.title="AI RAG API"
USER appuser
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "2", "--timeout-graceful-shutdown", "30"]

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

```dockerfile
# apps/ingestion/Dockerfile — GPU-accelerated ingestion worker
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS runtime
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_HOME=/app/.torch
RUN pip install --no-cache-dir \
    sentence-transformers==3.0.1 celery==5.4.0 chromadb==0.5.0
WORKDIR /app
COPY apps/ingestion/src ./src
COPY libs/ ./libs
RUN useradd -r -u 1001 worker
USER worker
CMD ["celery", "-A", "src.celery_app", "worker", \
     "--loglevel=info", "--concurrency=4", "--queues=ingestion"]
```

---

### 1.3 Kubernetes Deployment for RAG Services

```yaml
# kubernetes/rag-api/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: ai-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0        # Zero-downtime rolling updates
  template:
    metadata:
      labels:
        app: rag-api
        version: "2.3.1"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: rag-api
      containers:
        - name: api
          image: registry.company.com/rag-api:2.3.1
          ports:
            - containerPort: 8080
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-credentials
                  key: openai-api-key
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 15
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 2
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 5"]
      terminationGracePeriodSeconds: 60
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: ai-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

---

### 1.4 Resource Limits and GPU Scheduling

```yaml
# kubernetes/embedding-server/deployment-gpu.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-server
  namespace: ai-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: embedding-server
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-t4
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: server
          image: registry.company.com/embedding-server:3.0.1
          resources:
            requests:
              cpu: "2000m"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4000m"
              memory: "16Gi"
              nvidia.com/gpu: "1"
          env:
            - name: MODEL_NAME
              value: "sentence-transformers/all-MiniLM-L6-v2"
            - name: BATCH_SIZE
              value: "64"
```

```python
# Resource profiles by service type
RESOURCE_PROFILES = {
    "rag_api": {
        "description": "Stateless API — calls external LLM, light compute",
        "cpu_request": "500m", "cpu_limit": "2000m",
        "memory_request": "512Mi", "memory_limit": "2Gi",
        "gpu": False, "replicas_min": 3
    },
    "embedding_server_cpu": {
        "description": "CPU embedding with MiniLM",
        "cpu_request": "2000m", "cpu_limit": "4000m",
        "memory_request": "2Gi", "memory_limit": "4Gi",
        "gpu": False, "replicas_min": 2
    },
    "embedding_server_gpu": {
        "description": "GPU embedding with large model",
        "cpu_request": "2000m", "cpu_limit": "4000m",
        "memory_request": "8Gi", "memory_limit": "16Gi",
        "gpu": True, "gpu_count": 1, "replicas_min": 2
    },
    "local_llm_ollama": {
        "description": "On-premise Ollama serving quantised 7B model",
        "cpu_request": "4000m", "cpu_limit": "8000m",
        "memory_request": "16Gi", "memory_limit": "32Gi",
        "gpu": False, "replicas_min": 1,
        "note": "Q4 quantisation: 7B fits in ~4GB RAM without GPU"
    },
}
```

---

### 1.5 Health Checks and Readiness Probes

```python
# apps/api/src/health.py
from fastapi import FastAPI
from pydantic import BaseModel
import time

class HealthStatus(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    checks: dict

START_TIME = time.time()

def check_vector_db(host: str, port: int) -> dict:
    import socket
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def check_llm_gateway(gateway_url: str) -> dict:
    import httpx
    try:
        resp = httpx.get(f"{gateway_url}/health", timeout=3)
        return {"status": "ok" if resp.status_code == 200 else "degraded"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def register_health_endpoints(app: FastAPI, config):

    @app.get("/health", response_model=HealthStatus, tags=["ops"])
    async def liveness():
        """
        Liveness: is the process alive?
        Fails only on catastrophic issues. Kubernetes restarts on failure.
        """
        return HealthStatus(
            status="healthy", version=config.version,
            uptime_seconds=round(time.time() - START_TIME, 1),
            checks={}
        )

    @app.get("/ready", response_model=HealthStatus, tags=["ops"])
    async def readiness():
        """
        Readiness: is the service ready for traffic?
        Fails if dependencies are unavailable.
        Kubernetes removes pod from load balancer on failure (no restart).
        """
        checks = {}
        overall = "healthy"

        vdb = check_vector_db(config.vector_db.host, config.vector_db.port)
        checks["vector_db"] = vdb
        if vdb["status"] != "ok":
            overall = "unhealthy"

        gw = check_llm_gateway(config.llm_gateway_url)
        checks["llm_gateway"] = gw
        if gw["status"] == "error" and overall == "healthy":
            overall = "degraded"

        return HealthStatus(
            status=overall, version=config.version,
            uptime_seconds=round(time.time() - START_TIME, 1),
            checks=checks
        )

    @app.get("/startup", tags=["ops"])
    async def startup_probe():
        """
        Startup probe: has initialisation completed?
        Kubernetes delays liveness/readiness until this passes.
        Use for slow-starting processes: model loading, index warmup.
        """
        if not getattr(app.state, "embedder_ready", False):
            return {"status": "starting", "message": "Embedder loading..."}, 503
        return {"status": "ready"}
```

---

> ### 📋 Chapter Summary
>
> - **Multi-stage Dockerfiles** separate build and runtime layers, reducing image size and attack surface. Non-root users are mandatory for production images.
> - Kubernetes deployments use `topologySpreadConstraints` for node-level resilience and `HPA` for traffic-based scaling with CPU and custom metrics.
> - GPU workloads require `nvidia.com/gpu` resource requests, `nodeSelector`, and `tolerations` — all three must be present for correct scheduling.
> - **Liveness** detects stuck processes (restart); **readiness** detects unavailable dependencies (remove from load balancer); **startup** gives slow-starters time to initialise.

---

> ### ❓ Comprehension Questions
>
> 1. The RAG API Dockerfile copies `libs/` in a single layer. A developer changes `libs/rag_core/utils.py`. Which Docker layers are invalidated? How would you restructure the Dockerfile to optimise rebuild time for frequent lib changes?
> 2. The HPA scales on CPU utilisation (70%). An LLM-backed service spends most time waiting for API responses (I/O bound) with low CPU. Why is CPU a poor scaling metric here, and what metric better reflects actual load?
> 3. Liveness `failureThreshold=3` and readiness `failureThreshold=2`. Explain the reasoning. What happens to in-flight requests when a pod fails its readiness probe?
> 4. The GPU deployment sets `nvidia.com/gpu: "1"` in both requests and limits. What happens if you set the limit higher than the request? Why does Kubernetes enforce equality for GPU resources?
> 5. A service has `terminationGracePeriodSeconds: 60` and a `preStop` sleep of 5 seconds. Describe the complete sequence from Kubernetes signalling termination to the new pod receiving 100% of traffic.

---

## References

### Documentation
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Kubernetes Health Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)

---

## Chapter 2 — Vector Database Infrastructure

### 2.1 Sizing and Capacity Planning

Vector database sizing is driven by four parameters: vector count, vector dimensions, metadata size, and query throughput.

```python
from dataclasses import dataclass

@dataclass
class VectorDBSizingModel:
    """
    Estimates storage and memory requirements.
    Based on Qdrant capacity planning guidelines.
    """
    vector_count: int
    vector_dimensions: int
    metadata_bytes_per_vector: int = 200
    quantisation: str = "none"       # none | scalar | product

    def vector_storage_gb(self) -> float:
        bytes_per_vector = self.vector_dimensions * 4    # float32
        if self.quantisation == "scalar":
            bytes_per_vector = self.vector_dimensions * 1   # int8
        elif self.quantisation == "product":
            bytes_per_vector = self.vector_dimensions // 8  # ~95% compression
        return (self.vector_count * bytes_per_vector) / 1e9

    def total_storage_gb(self, replication_factor: int = 2) -> float:
        raw = self.vector_storage_gb() + (
            self.vector_count * self.metadata_bytes_per_vector) / 1e9
        return raw * replication_factor * 1.3  # 30% overhead for HNSW index

    def recommended_ram_gb(self) -> float:
        """HNSW index: ~2x raw vector storage in RAM for sub-10ms search."""
        return self.vector_storage_gb() * 2 * 1.2

    def report(self):
        print(f"\n── Vector DB Sizing Report ──────────────────────")
        print(f"  Vectors        : {self.vector_count:,}")
        print(f"  Dimensions     : {self.vector_dimensions}")
        print(f"  Quantisation   : {self.quantisation}")
        print(f"  Vector storage : {self.vector_storage_gb():.2f} GB")
        print(f"  Total disk     : {self.total_storage_gb():.2f} GB")
        print(f"  Recommended RAM: {self.recommended_ram_gb():.1f} GB")


# Example: 500K vectors, 1536-dim, scalar quantisation (4x memory reduction)
VectorDBSizingModel(500_000, 1536, quantisation="scalar").report()
# Vector storage:  0.77 GB   (vs 3.07 GB without quantisation)
# Total disk:      1.13 GB
# Recommended RAM: 1.85 GB
```

---

### 2.2 High Availability and Replication

```yaml
# kubernetes/qdrant/qdrant-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: ai-platform
spec:
  serviceName: qdrant-headless
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: qdrant
              topologyKey: topology.kubernetes.io/zone
      containers:
        - name: qdrant
          image: qdrant/qdrant:v1.11.3
          ports:
            - containerPort: 6333   # REST
            - containerPort: 6334   # gRPC
            - containerPort: 6335   # Cluster P2P
          env:
            - name: QDRANT__CLUSTER__ENABLED
              value: "true"
            - name: QDRANT__CLUSTER__P2P__PORT
              value: "6335"
          volumeMounts:
            - name: qdrant-storage
              mountPath: /qdrant/storage
          resources:
            requests:
              cpu: "2000m"
              memory: "8Gi"
            limits:
              cpu: "4000m"
              memory: "16Gi"
  volumeClaimTemplates:
    - metadata:
        name: qdrant-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-headless
  namespace: ai-platform
spec:
  clusterIP: None
  selector:
    app: qdrant
  ports:
    - port: 6335
      name: p2p
```

---

### 2.3 Backup and Disaster Recovery

```python
import httpx
from datetime import datetime
from pathlib import Path

class VectorDBBackupManager:
    """
    Manages Qdrant snapshots: create → upload to S3 → restore.
    Run as Kubernetes CronJob — daily snapshots, 7-day retention.
    """
    def __init__(self, qdrant_url: str, s3_bucket: str,
                 s3_prefix: str = "qdrant-snapshots"):
        self.qdrant_url = qdrant_url
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix

    def create_snapshot(self, collection_name: str) -> str:
        resp = httpx.post(
            f"{self.qdrant_url}/collections/{collection_name}/snapshots",
            timeout=300
        )
        resp.raise_for_status()
        snapshot_name = resp.json()["result"]["name"]
        print(f"  ✓ Snapshot created: {snapshot_name}")
        return snapshot_name

    def upload_to_s3(self, collection_name: str, snapshot_name: str) -> str:
        import boto3
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        s3_key = f"{self.s3_prefix}/{date_prefix}/{collection_name}/{snapshot_name}"
        resp = httpx.get(
            f"{self.qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}",
            timeout=600
        )
        resp.raise_for_status()
        boto3.client("s3").put_object(
            Bucket=self.s3_bucket, Key=s3_key, Body=resp.content
        )
        print(f"  ✓ Uploaded to s3://{self.s3_bucket}/{s3_key}")
        return s3_key

    def restore_from_s3(self, collection_name: str, s3_key: str):
        import boto3
        obj = boto3.client("s3").get_object(
            Bucket=self.s3_bucket, Key=s3_key
        )
        snapshot_data = obj["Body"].read()
        resp = httpx.post(
            f"{self.qdrant_url}/collections/{collection_name}/snapshots/upload",
            content=snapshot_data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600
        )
        resp.raise_for_status()
        print(f"  ✓ Restored {collection_name} from {s3_key}")

    def backup_all_collections(self) -> dict:
        resp = httpx.get(f"{self.qdrant_url}/collections", timeout=10)
        collections = [c["name"] for c in resp.json()["result"]["collections"]]
        results = {}
        for coll in collections:
            try:
                snap = self.create_snapshot(coll)
                s3_key = self.upload_to_s3(coll, snap)
                results[coll] = {"status": "ok", "s3_key": s3_key}
            except Exception as e:
                results[coll] = {"status": "error", "error": str(e)}
        return results
```

```yaml
# kubernetes/qdrant/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: qdrant-backup
  namespace: ai-platform
spec:
  schedule: "0 2 * * *"     # Daily at 02:00 UTC
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 7
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: backup
              image: registry.company.com/qdrant-backup:1.0.0
              command: ["python", "backup_manager.py", "--backup-all"]
              env:
                - name: QDRANT_URL
                  value: "http://qdrant.ai-platform.svc:6333"
                - name: S3_BUCKET
                  valueFrom:
                    configMapKeyRef:
                      name: backup-config
                      key: s3-bucket
```

---

### 2.4 Production Qdrant Configuration

```yaml
# qdrant/config/production.yaml
storage:
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  optimizers_config:
    deleted_threshold: 0.2          # Vacuum at 20% deleted vectors
    vacuum_min_vector_number: 1000
    max_segment_size_kb: 200000
    indexing_threshold_kb: 20000    # Build HNSW above this size

service:
  grpc_port: 6334
  http_port: 6333
  max_request_size_mb: 32

cluster:
  enabled: true
  p2p:
    port: 6335

# HNSW index — tune recall vs speed
hnsw_config:
  m: 16                   # Connections per layer (higher = better recall)
  ef_construct: 100       # Build-time depth (higher = better recall, slower build)
  full_scan_threshold: 10000

# Scalar quantisation: float32 → int8 (4x less memory, <5% recall loss)
quantization_config:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: true
```

---

### 2.5 On-Premise Alternatives 🔓

```yaml
# docker-compose.onprem.yml — Vector DB options for on-premise deployments

services:
  # Qdrant — recommended: best balance of features, performance, and simplicity
  qdrant:
    image: qdrant/qdrant:v1.11.3
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_data:/qdrant/storage]
    environment:
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}

  # Chroma — simplest setup, good for prototyping and small corpora (<1M vectors)
  chroma:
    image: chromadb/chroma:0.5.0
    ports: ["8000:8000"]
    volumes: [chroma_data:/chroma/chroma]

  # Weaviate — strong GraphQL API, good hybrid search, module ecosystem
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.26.4
    ports: ["8080:8080"]
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
      DEFAULT_VECTORIZER_MODULE: none
    volumes: [weaviate_data:/var/lib/weaviate]

  # Milvus — enterprise-grade, highest throughput, most operational complexity
  milvus:
    image: milvusdb/milvus:v2.4.13
    command: ["milvus", "run", "standalone"]
    ports: ["19530:19530"]
    volumes: [milvus_data:/var/lib/milvus]

volumes:
  qdrant_data:
  chroma_data:
  weaviate_data:
  milvus_data:
```

---

> ### 📋 Chapter Summary
>
> - Vector DB sizing requires computing storage (vector bytes × count) and RAM (2× storage for HNSW index). **Scalar quantisation** (float32→int8) reduces memory 4x with <5% recall loss.
> - Production Qdrant runs as a 3-node **StatefulSet** with pod anti-affinity across availability zones and a PVC per pod.
> - **Daily snapshots** to S3 with lifecycle-based expiry provide RPO. Restore procedures must be tested before a disaster, not during one.
> - HNSW `m` and `ef_construct` are the primary recall vs. speed tuning levers — increase them if Recall@5 < 0.80.

---

> ### ❓ Comprehension Questions
>
> 1. Your corpus grows from 500K to 2M vectors. Using `VectorDBSizingModel`, calculate recommended RAM at 2M vectors (1536-dim, scalar quantisation). What Kubernetes resource change does this require?
> 2. The StatefulSet uses `requiredDuringSchedulingIgnoredDuringExecution` pod anti-affinity across zones. This blocks scheduling if only one zone is available. Should this be `required` or `preferred` for enterprise deployments? Argue both positions.
> 3. The backup CronJob runs at 02:00 UTC. A corruption event occurs at 01:55 UTC. What is the worst-case RPO, and how would you reduce it?
> 4. A team reports Recall@5 = 0.78 (below 0.80 threshold). Which HNSW parameter would you increase first, and what is the trade-off?
> 5. A finance team requires vector data encrypted at rest. What Kubernetes-level and Qdrant-level mechanisms apply?

---

## References

### Documentation
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [Qdrant Capacity Planning](https://qdrant.tech/documentation/guides/capacity-planning/)
- [Weaviate Kubernetes Deployment](https://weaviate.io/developers/weaviate/installation/kubernetes)
- [Chroma Self-Hosting](https://docs.trychroma.com/guides/running-chroma)
- [Milvus Architecture](https://milvus.io/docs/architecture_overview.md)

---

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

## Chapter 4 — On-Premise LLM Platform Design 🔓

### 4.1 When to Go On-Premise

On-premise LLM deployment is warranted when one or more of the following hold:

**Data residency.** EU GDPR, HIPAA, banking regulations, or classified environments prohibit sending data to external providers. All inference must run on organisation-controlled infrastructure.

**Air-gapped environments.** Critical infrastructure and secure research environments have no internet connectivity. The entire model serving stack must run internally.

**Cost at scale.** For organisations running > 10M tokens/day continuously, on-premise hardware can be significantly cheaper than pay-per-token cloud APIs despite higher operational complexity.

**Latency requirements.** Some applications require sub-50ms P99 inference. A local serving stack eliminates the network round-trip to a cloud provider.

---

### 4.2 Ollama for Development and Small Workloads

[Ollama](https://ollama.com/docs) provides the simplest on-premise serving with an OpenAI-compatible API.

```bash
# Install and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text   # Embedding model

# Start server (default: localhost:11434)
ollama serve
```

```python
# Drop-in replacement for OpenAI client — same API surface
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # Required by SDK, not validated by Ollama
)

# LLM completion
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is RAG?"}],
    temperature=0
)
print(response.choices[0].message.content)

# Embeddings
embed = client.embeddings.create(
    model="nomic-embed-text",
    input="Document content"
)
print(f"Dimensions: {len(embed.data[0].embedding)}")
```

```
# Ollama resource requirements by model size
Model       Quantisation   RAM required   Notes
─────────   ────────────   ────────────   ─────────────────────────────
3B params   Q4_K_M         ~2 GB          Fast, limited capability
7B params   Q4_K_M         ~4 GB          Good quality, runs on laptop
13B params  Q4_K_M         ~8 GB          Better quality, 16GB system
70B params  Q4_K_M         ~40 GB         Near-GPT-4 quality, needs GPU
```

---

### 4.3 vLLM for High-Throughput Serving

[vLLM](https://docs.vllm.ai) provides production-grade serving with PagedAttention — GPU memory managed as virtual memory pages, dramatically increasing concurrent request throughput.

```bash
pip install vllm

# Serve Mistral 7B on GPU
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90
```

```yaml
# kubernetes/vllm/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-mistral-7b
  namespace: ai-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-mistral
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-a100
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.6.4
          args:
            - "--model"
            - "mistralai/Mistral-7B-Instruct-v0.3"
            - "--max-model-len"
            - "8192"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "40Gi"
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-weights-pvc
```

```python
# vLLM uses the same OpenAI client — change only base_url
from openai import OpenAI

vllm_client = OpenAI(
    base_url="http://vllm-mistral.ai-platform.svc:8000/v1",
    api_key="vllm-local"
)
response = vllm_client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Summarise the key points."}],
    temperature=0, max_tokens=500
)
```

---

### 4.4 On-Premise Infrastructure Stack

```
On-Premise AI Platform (single data centre)

┌───────────────────────────────────────────────────────┐
│  Load Balancer (Nginx / HAProxy)                       │
├───────────────────────────────────────────────────────┤
│  API Layer (FastAPI / Spring Boot)                     │
│  ├── LLM Gateway  → vLLM or Ollama                    │
│  ├── Prompt Service                                    │
│  └── Embedding Service → local SentenceTransformers   │
├───────────────────────────────────────────────────────┤
│  Compute                                               │
│  ├── CPU nodes: API, ingestion, embeddings             │
│  └── GPU nodes: vLLM serving (optional)               │
├───────────────────────────────────────────────────────┤
│  Data                                                  │
│  ├── Qdrant (vector DB) — 3-node StatefulSet           │
│  ├── Redis (Celery) — HA cluster                       │
│  ├── PostgreSQL (metadata, prompts) — primary+standby  │
│  └── MinIO (object store: models, datasets, snapshots) │
├───────────────────────────────────────────────────────┤
│  Secrets & Config                                      │
│  └── HashiCorp Vault + Consul                         │
└───────────────────────────────────────────────────────┘
```

```python
# On-premise gateway config: all traffic goes to local models
from part_11_platform_engineering import ModelAliasConfig, LLMProvider

ON_PREMISE_ALIASES = {
    "default":   ModelAliasConfig("default",  LLMProvider.OLLAMA, "llama3.2",
                                   cost_per_1m_input=0.0, cost_per_1m_output=0.0),
    "powerful":  ModelAliasConfig("powerful", LLMProvider.OLLAMA, "mistral",
                                   cost_per_1m_input=0.0, cost_per_1m_output=0.0),
    "fast":      ModelAliasConfig("fast",     LLMProvider.OLLAMA, "llama3.2:3b",
                                   cost_per_1m_input=0.0, cost_per_1m_output=0.0),
}
# Note: cost=0.0 is misleading — track GPU/CPU hours instead
```

---

> ### 📋 Chapter Summary
>
> - On-premise is warranted for data residency, air-gapped environments, cost at scale (>10M tokens/day), and strict latency requirements.
> - **Ollama** (development/small workloads) and **vLLM** (production GPU serving) both expose an OpenAI-compatible API — product code changes only `base_url`.
> - PagedAttention in vLLM manages GPU KV-cache as virtual memory pages, enabling far higher concurrent request throughput than naive memory management.
> - On-premise "cost" is not zero — GPU/CPU hours, hardware amortisation, and operational overhead must be tracked.

---

> ### ❓ Comprehension Questions
>
> 1. A financial services firm needs GPT-4o quality but cannot send data to OpenAI. What on-premise model family would you evaluate first, and what evaluation methodology would confirm quality parity?
> 2. vLLM's PagedAttention manages GPU KV-cache like virtual memory pages. Why does this increase throughput for concurrent requests compared to pre-allocating fixed-size KV buffers per request?
> 3. An Ollama server with 16GB RAM serves `llama3:8b` (Q4_K_M ~4.5GB). Three concurrent requests arrive simultaneously. What happens to memory and latency versus a single request?
> 4. The on-premise gateway sets `cost_per_1m_input=0.0`. What real costs should be tracked instead, and how would you modify `CostTracker` to capture compute hours?
> 5. A regulated firm requires model weights never leave their data centre. vLLM downloads from Hugging Face at startup. Design the model weight distribution workflow satisfying this requirement.

---

## References

### Documentation
- [Ollama Documentation](https://ollama.com/docs)
- [vLLM Documentation](https://docs.vllm.ai)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)
- [MinIO Documentation](https://min.io/docs/minio/container/index.html)
- [HashiCorp Vault](https://developer.hashicorp.com/vault/docs)

### Papers
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023.

---

## Chapter 5 — Infrastructure as Code 🧪

### 5.1 Terraform for AI Infrastructure

```hcl
# infrastructure/terraform/main.tf

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.30" }
  }
  backend "s3" {
    bucket = "company-terraform-state"
    key    = "ai-platform/terraform.tfstate"
    region = "eu-west-1"
  }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "ai-platform-${var.environment}"
  cluster_version = "1.30"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  eks_managed_node_groups = {
    cpu_workers = {
      instance_types = ["m6i.xlarge"]
      min_size = 3; max_size = 20; desired_size = 3
      labels = { workload = "cpu" }
    }
    gpu_workers = {
      instance_types = ["g4dn.xlarge"]  # 1x T4 GPU, 16GB VRAM
      min_size = 0; max_size = 4; desired_size = 1
      labels = { workload = "gpu", accelerator = "nvidia-tesla-t4" }
      taints = [{ key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }]
    }
  }
}

resource "aws_s3_bucket" "qdrant_backups" {
  bucket = "company-qdrant-backups-${var.environment}"
}

resource "aws_s3_bucket_lifecycle_configuration" "backup_retention" {
  bucket = aws_s3_bucket.qdrant_backups.id
  rule {
    id = "7-day-retention"; status = "Enabled"
    filter { prefix = "qdrant-snapshots/" }
    expiration { days = 7 }
  }
}

resource "aws_secretsmanager_secret" "llm_credentials" {
  name = "ai-platform/${var.environment}/llm-credentials"
  recovery_window_in_days = 7
}

variable "environment" {
  type    = string
  default = "staging"
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "environment must be development, staging, or production."
  }
}
```

---

### 5.2 Kubernetes Manifests and Helm Charts

```yaml
# infrastructure/helm/ai-platform/values.yaml
global:
  imageRegistry: registry.company.com
  environment: production
  namespace: ai-platform

ragApi:
  image: rag-api
  tag: "2.3.1"
  replicas: { min: 3, max: 20 }
  resources:
    requests: { cpu: 500m, memory: 512Mi }
    limits: { cpu: 2000m, memory: 2Gi }

qdrant:
  replicas: 3
  storage: { size: 100Gi, storageClass: gp3 }
  resources:
    requests: { cpu: 2000m, memory: 8Gi }
    limits: { cpu: 4000m, memory: 16Gi }

embeddingServer:
  enabled: true
  model: all-MiniLM-L6-v2
  useGpu: false
  replicas: 2

# Disabled by default — enable for on-premise deployments
ollamaServer:
  enabled: false
  model: llama3.2
```

```yaml
# infrastructure/helm/ai-platform/values.onpremise.yaml
# Override: disable cloud LLM calls, use local Ollama

llmGateway:
  defaultProvider: ollama
  ollamaUrl: http://ollama.ai-platform.svc:11434

ollamaServer:
  enabled: true
  model: mistral
  resources:
    requests: { memory: 8Gi }
    limits: { memory: 16Gi }
```

---

### 5.3 Local Development Stack

```yaml
# docker-compose.yml — Complete local development environment
services:
  qdrant:
    image: qdrant/qdrant:v1.11.3
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_data:/qdrant/storage]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7.2-alpine
    ports: ["6379:6379"]
    volumes: [redis_data:/data]
    command: redis-server --appendonly yes

  # On-premise LLM — only with: docker-compose --profile onpremise up
  ollama:
    image: ollama/ollama:0.3.12
    ports: ["11434:11434"]
    volumes: [ollama_models:/root/.ollama]
    profiles: ["onpremise"]

  # Mock LLM for dev without real API key
  mock-llm:
    image: wiremock/wiremock:3.9.1
    ports: ["9090:8080"]
    volumes: [./tests/fixtures/wiremock:/home/wiremock]

  prometheus:
    image: prom/prometheus:v2.54.1
    ports: ["9090:9090"]
    volumes:
      - ./infrastructure/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:11.2.0
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: dev-password
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  qdrant_data:
  redis_data:
  ollama_models:
  prometheus_data:
  grafana_data:
```

---

### 🧪 Hands-on Lab: Full Local Stack with Docker Compose

**Objective:** Launch the complete local development stack and run an end-to-end ingestion and retrieval smoke test.

**Prerequisites:** Docker, Docker Compose, Python ≥ 3.11

```python
#!/usr/bin/env python3
"""
smoke_test.py — End-to-end smoke test for local development stack.
Run after: docker-compose up -d
"""

import sys
import time

def check_service(name: str, check_fn, retries: int = 5) -> bool:
    for i in range(retries):
        try:
            check_fn()
            print(f"  ✓ {name} healthy")
            return True
        except Exception as e:
            if i < retries - 1:
                time.sleep(2)
            else:
                print(f"  ✗ {name} unreachable: {e}")
    return False

def check_qdrant():
    import httpx
    r = httpx.get("http://localhost:6333/health", timeout=3)
    assert r.status_code == 200

def check_redis():
    import redis
    redis.Redis(host="localhost", port=6379).ping()

# 1. Verify services
print("
=== Infrastructure Smoke Test ===
")
ok = all([
    check_service("Qdrant", check_qdrant),
    check_service("Redis", check_redis),
])
if not ok:
    print("
Some services are not healthy. Run: docker-compose up -d")
    sys.exit(1)

# 2. Ingest test documents
print("
[2] Testing vector ingestion...")
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.HttpClient(host="localhost", port=8000)
try:
    client.delete_collection("smoke_test")
except Exception:
    pass
collection = client.create_collection("smoke_test")

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = [
    "Enterprise customers have a 90-day return window.",
    "Professional plan costs $150 per month with 25 users.",
    "API tokens expire after 3600 seconds using OAuth 2.0.",
    "Refunds processed within 5-7 business days.",
]
embeddings = model.encode(docs).tolist()
collection.add(
    ids=[f"d{i}" for i in range(len(docs))],
    embeddings=embeddings,
    documents=docs
)
print(f"  ✓ Ingested {len(docs)} documents")

# 3. Query
print("
[3] Testing retrieval...")
queries = [
    "enterprise return policy",
    "professional plan pricing",
    "API authentication token lifetime",
]
for q in queries:
    q_emb = model.encode([q]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=1)
    top = results["documents"][0][0][:70] if results["documents"][0] else "NO RESULT"
    print(f"  Q: {q}")
    print(f"  A: {top}")

# 4. Cleanup
client.delete_collection("smoke_test")
print("
=== Smoke test passed ===")
print("
Services running:")
print("  Qdrant:     http://localhost:6333")
print("  Redis:      localhost:6379")
print("  Grafana:    http://localhost:3000  (admin/dev-password)")
print("
On-premise LLM:")
print("  docker-compose --profile onpremise up -d ollama")
print("  docker-compose exec ollama ollama pull llama3.2")
```

**Run the lab:**
```bash
docker-compose up -d
pip install chromadb sentence-transformers httpx redis
python smoke_test.py
```

**Extensions:**
- Add a Celery worker smoke test: submit an ingestion task and poll until `COMPLETED`
- Extend to assert Grafana is reachable and the AI Platform dashboard loads
- Add a Makefile target `make stack-up` / `make stack-down` / `make smoke-test`

---

> ### 📋 Chapter Summary
>
> - Terraform manages EKS with separate CPU and GPU node groups, S3 buckets with lifecycle policies, and Secrets Manager — all parameterised by `environment`.
> - **Helm charts** with environment-specific `values.yaml` overrides deploy the same chart to dev, staging, production, and on-premise.
> - Docker Compose provides a reproducible local environment with Qdrant, Redis, Ollama (optional via profile), mock LLM, Prometheus, and Grafana.
> - The smoke test verifies infrastructure health and end-to-end ingestion + retrieval in under 2 minutes.

---

> ### ❓ Comprehension Questions
>
> 1. Terraform stores state in S3. Two engineers run `terraform apply` simultaneously. What prevents state corruption, and what happens if the lock is not released after a failed apply?
> 2. The GPU node group has `min_size=0`. The cluster autoscaler scales it to 0 after inactivity. A batch embedding job is submitted. Describe the sequence of events and latency impact.
> 3. The Helm chart has `values.onpremise.yaml`. A developer deploys to production using this file by mistake. What breaks, and what GitOps process prevents this?
> 4. Docker Compose binds Qdrant to `localhost:6333`. The API service runs in a separate container. Why can it not reach `localhost:6333`, and what hostname should it use?
> 5. The S3 lifecycle deletes snapshots after 7 days. A compliance audit requires 90-day retention for production. Modify the Terraform resource to apply different retention per environment.

---

## References

### Documentation
- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs)
- [Helm Documentation](https://helm.sh/docs/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/latest/userguide/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Prometheus](https://prometheus.io/docs/introduction/overview/)
- [Grafana](https://grafana.com/docs/)

### Books
- *Kubernetes Patterns* — Ibryam & Huß (O'Reilly).
- *Terraform: Up and Running* — Brikman (O'Reilly).

---

> **Navigation**
> [← Part XI — AI Platform Engineering](part_11_platform_engineering.md) | [→ Part XIII — Observability](part_13_observability.md)
