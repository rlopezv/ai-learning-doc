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

---
[« Back to infrastructure Index](index.md) | [🏠 Home](../../index.md)