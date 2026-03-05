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

---
[« Back to infrastructure Index](index.md) | [🏠 Home](../../index.md)