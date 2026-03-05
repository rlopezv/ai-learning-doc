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

---
[« Back to infrastructure Index](index.md) | [🏠 Home](../../index.md)