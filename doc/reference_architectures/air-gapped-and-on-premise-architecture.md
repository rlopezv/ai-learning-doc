## Chapter 4 — Air-Gapped and On-Premise Architecture 🔓🧪

### 4.1 Constraints and Design Principles

An air-gapped deployment has zero outbound internet connectivity. Every component must run locally: LLM inference, embeddings, vector DB, object storage, secrets management, container registry, and monitoring. This constraint eliminates cloud-managed services and requires careful component selection.

```python
AIR_GAP_CONSTRAINTS = {
    "network":    "No outbound internet. Internal network only.",
    "models":     "All models must be downloaded before deployment and stored on local registry.",
    "secrets":    "No cloud KMS. Use HashiCorp Vault or SOPS + local key.",
    "storage":    "No S3. Use MinIO (S3-compatible, self-hosted).",
    "registry":   "No Docker Hub or ECR. Use Harbor or Nexus.",
    "updates":    "Software updates require manual air-gap transfer process.",
    "licensing":  "All components must be open-source or have offline licensing.",
}

AIR_GAP_DESIGN_PRINCIPLES = [
    "Prefer OSS components with no telemetry or phone-home behaviour",
    "Pin all container image digests — no 'latest' tags",
    "Store all model weights in local object storage (MinIO)",
    "All secret references point to local Vault — never hardcoded",
    "Monitoring stack is self-contained (Prometheus + Grafana + Loki, no cloud agent)",
    "Document every external dependency that was replaced and its local equivalent",
]
```

---

### 4.2 Component Selection for Air-Gapped Environments

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ComponentMapping:
    cloud_component: str
    on_premise_replacement: str
    notes: str
    license: str
    install_command: str

AIR_GAP_COMPONENT_MAP = [
    ComponentMapping(
        "OpenAI GPT-4o",
        "Ollama + Llama 3.1 / Mistral / Phi-3",
        "Pull models: ollama pull llama3.1:8b. OpenAI-compatible API.",
        "MIT / Meta Llama Community License",
        "curl -fsSL https://ollama.ai/install.sh | sh"
    ),
    ComponentMapping(
        "OpenAI text-embedding-3-small",
        "sentence-transformers/all-MiniLM-L6-v2 (local)",
        "Served via HuggingFace Inference Server or custom FastAPI.",
        "Apache 2.0",
        "pip install sentence-transformers"
    ),
    ComponentMapping(
        "Pinecone / Weaviate Cloud",
        "Qdrant OSS (self-hosted)",
        "Docker or Kubernetes. No internet required after image pull.",
        "Apache 2.0",
        "docker run -p 6333:6333 qdrant/qdrant"
    ),
    ComponentMapping(
        "AWS S3",
        "MinIO",
        "S3-compatible API. Use for model weights, backups, corpus storage.",
        "GNU AGPL v3",
        "docker run -p 9000:9000 minio/minio server /data"
    ),
    ComponentMapping(
        "AWS Secrets Manager",
        "HashiCorp Vault OSS",
        "Full secrets management with dynamic secrets and audit log.",
        "BUSL 1.1 (free for self-hosted)",
        "https://developer.hashicorp.com/vault/install"
    ),
    ComponentMapping(
        "AWS ECR / Docker Hub",
        "Harbor",
        "Container registry with vulnerability scanning and RBAC.",
        "Apache 2.0",
        "helm install harbor harbor/harbor"
    ),
    ComponentMapping(
        "Datadog / CloudWatch",
        "Prometheus + Grafana + Loki + Jaeger",
        "Full observability stack. All OSS, no external endpoints.",
        "Apache 2.0",
        "helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack"
    ),
    ComponentMapping(
        "GitHub Actions / CircleCI",
        "Gitea + Gitea Actions or Jenkins",
        "Self-hosted CI/CD. Gitea is lightweight Git + Actions compatible.",
        "MIT",
        "helm install gitea gitea-charts/gitea"
    ),
]
```

---

### 4.3 Full On-Premise Stack

```yaml
# docker-compose.airgap.yml — Complete air-gapped RAG stack
# All images must be pre-pulled and pushed to Harbor registry
# Run: docker-compose -f docker-compose.airgap.yml up -d

version: "3.9"

services:

  # ── LLM Inference ───────────────────────────────────────────────
  ollama:
    image: harbor.internal/ollama/ollama:0.4.7
    container_name: ollama
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - OLLAMA_HOST=0.0.0.0
    ports: ["11434:11434"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ── Embedding Service ────────────────────────────────────────────
  embedding:
    image: harbor.internal/ai-platform/embedding-service:2.3.1
    container_name: embedding
    volumes:
      - model_cache:/app/models      # Models pre-downloaded and mounted
    environment:
      - MODEL_NAME=all-MiniLM-L6-v2
      - MODEL_PATH=/app/models/all-MiniLM-L6-v2
    ports: ["8001:8001"]
    restart: unless-stopped

  # ── Vector Database ──────────────────────────────────────────────
  qdrant:
    image: harbor.internal/qdrant/qdrant:v1.12.0
    container_name: qdrant
    volumes:
      - qdrant_data:/qdrant/storage
    ports: ["6333:6333"]
    environment:
      - QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 15s

  # ── Object Storage (S3-compatible) ───────────────────────────────
  minio:
    image: harbor.internal/minio/minio:RELEASE.2024-11-07T00-52-20Z
    container_name: minio
    volumes:
      - minio_data:/data
    environment:
      - MINIO_ROOT_USER=admin
      - MINIO_ROOT_PASSWORD_FILE=/run/secrets/minio_password
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    secrets: [minio_password]
    restart: unless-stopped

  # ── Secrets Management ───────────────────────────────────────────
  vault:
    image: harbor.internal/hashicorp/vault:1.18.1
    container_name: vault
    cap_add: [IPC_LOCK]
    volumes:
      - vault_data:/vault/data
      - ./vault/config.hcl:/vault/config/config.hcl
    ports: ["8200:8200"]
    command: vault server -config=/vault/config/config.hcl
    restart: unless-stopped

  # ── RAG API ──────────────────────────────────────────────────────
  rag-api:
    image: harbor.internal/ai-platform/rag-api:2.4.0
    container_name: rag-api
    environment:
      - LLM_BASE_URL=http://ollama:11434/v1
      - EMBEDDING_URL=http://embedding:8001
      - QDRANT_URL=http://qdrant:6333
      - VAULT_ADDR=http://vault:8200
      - VAULT_ROLE=rag-api
    ports: ["8080:8080"]
    depends_on: [ollama, embedding, qdrant, vault]
    restart: unless-stopped

  # ── Observability ────────────────────────────────────────────────
  prometheus:
    image: harbor.internal/prometheus/prometheus:v3.0.1
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]
    restart: unless-stopped

  grafana:
    image: harbor.internal/grafana/grafana:11.3.1
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
    secrets: [grafana_password]
    restart: unless-stopped

volumes:
  ollama_models:
  qdrant_data:
  minio_data:
  vault_data:
  model_cache:
  grafana_data:

secrets:
  minio_password:
    file: ./secrets/minio_password.txt
  grafana_password:
    file: ./secrets/grafana_password.txt
```

---

### 🧪 Hands-on Lab: Architecture Decision Record Generator

**Objective:** Generate a structured ADR (Architecture Decision Record) for a RAG system architecture choice, covering context, options, decision, and consequences.

```python
#!/usr/bin/env python3
"""
adr_generator.py — Architecture Decision Record generator.
Produces a structured ADR.md for a given architecture choice.
No external dependencies required.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class ArchitectureOption:
    name: str
    description: str
    pros: list[str]
    cons: list[str]
    cost_estimate: str
    implementation_effort: str    # "low" | "medium" | "high"

@dataclass
class ADR:
    adr_id: str
    title: str
    status: str                   # "proposed" | "accepted" | "deprecated" | "superseded"
    context: str
    decision_drivers: list[str]
    options: list[ArchitectureOption]
    chosen_option: str            # Name of chosen option
    decision_rationale: str
    consequences: dict            # {"positive": [...], "negative": [...], "neutral": [...]}
    links: list[str] = field(default_factory=list)
    date: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    authors: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        chosen = next((o for o in self.options if o.name == self.chosen_option), None)
        lines = [
            f"# ADR {self.adr_id}: {self.title}",
            f"",
            f"**Status:** {self.status.upper()}  ",
            f"**Date:** {self.date}  ",
            f"**Authors:** {', '.join(self.authors) or 'TBD'}",
            f"",
            f"---",
            f"",
            f"## Context",
            f"",
            self.context,
            f"",
            f"## Decision Drivers",
            f"",
        ] + [f"- {d}" for d in self.decision_drivers] + [
            f"",
            f"## Options Considered",
            f"",
        ]
        for opt in self.options:
            marker = "✅ **CHOSEN**" if opt.name == self.chosen_option else ""
            lines += [
                f"### {opt.name} {marker}",
                f"",
                f"{opt.description}",
                f"",
                f"**Pros:**",
            ] + [f"- {p}" for p in opt.pros] + [
                f"",
                f"**Cons:**",
            ] + [f"- {c}" for c in opt.cons] + [
                f"",
                f"*Cost: {opt.cost_estimate} | Effort: {opt.implementation_effort}*",
                f"",
            ]
        lines += [
            f"## Decision",
            f"",
            f"**Chosen: {self.chosen_option}**",
            f"",
            self.decision_rationale,
            f"",
            f"## Consequences",
            f"",
            f"**Positive:**",
        ] + [f"- {c}" for c in self.consequences.get("positive", [])] + [
            f"",
            f"**Negative:**",
        ] + [f"- {c}" for c in self.consequences.get("negative", [])] + [
            f"",
            f"**Neutral:**",
        ] + [f"- {c}" for c in self.consequences.get("neutral", [])]

        if self.links:
            lines += ["", "## Links", ""] + [f"- {l}" for l in self.links]
        return "\n".join(lines)


# ── Example ADR: Vector Database Selection ────────────────────────
vector_db_adr = ADR(
    adr_id="0003",
    title="Vector Database Selection for Production RAG Platform",
    status="accepted",
    context=(
        "The RAG platform requires a vector database to store and query document "
        "embeddings. The system must support 500K+ vectors, multi-tenancy via "
        "metadata filters, horizontal scaling, and HA. We evaluated three options."
    ),
    decision_drivers=[
        "Support for 500K+ vectors with sub-100ms P99 retrieval",
        "Multi-tenancy via payload filters enforced server-side",
        "Kubernetes StatefulSet deployment with HA (≥3 replicas)",
        "Active OSS community and recent releases",
        "On-premise deployment support (air-gap compatible)",
        "S3-compatible snapshot backup",
    ],
    options=[
        ArchitectureOption(
            "Qdrant",
            "Written in Rust. Distributed mode with P2P replication. "
            "Rich payload filtering. REST + gRPC API.",
            pros=[
                "Excellent filtering performance on payload metadata",
                "Distributed mode with shard replication",
                "Active development (weekly releases)",
                "OpenAI-compatible REST API",
                "Built-in snapshot to S3",
            ],
            cons=[
                "Smaller ecosystem than Weaviate",
                "Distributed mode requires 3+ nodes for quorum",
            ],
            cost_estimate="Free OSS; ~$0.10/GB/month self-hosted storage",
            implementation_effort="medium"
        ),
        ArchitectureOption(
            "Weaviate",
            "Go-based. GraphQL + REST API. Strong hybrid search (BM25 + vector). "
            "Module ecosystem for automatic vectorisation.",
            pros=[
                "Best-in-class hybrid search (BM25 + vector)",
                "GraphQL API for complex filtered queries",
                "Large community and extensive documentation",
            ],
            cons=[
                "Higher memory footprint than Qdrant",
                "Cluster setup more complex",
                "Module system adds operational complexity",
            ],
            cost_estimate="Free OSS; Cloud from $25/month",
            implementation_effort="medium"
        ),
        ArchitectureOption(
            "pgvector (PostgreSQL extension)",
            "Adds vector similarity search to PostgreSQL. "
            "No separate infrastructure required.",
            pros=[
                "No new infrastructure — extends existing PostgreSQL",
                "ACID transactions; combine vector and relational queries",
                "Familiar operational model for teams with PG expertise",
            ],
            cons=[
                "Poor performance at scale (>500K vectors, high QPS)",
                "No native distributed mode",
                "ANN accuracy lower than dedicated vector DBs at scale",
            ],
            cost_estimate="Free (uses existing PostgreSQL)",
            implementation_effort="low"
        ),
    ],
    chosen_option="Qdrant",
    decision_rationale=(
        "Qdrant is selected because it provides the best balance of filtering "
        "performance, distributed mode for HA, and operational simplicity. "
        "Its Rust implementation delivers consistent sub-50ms P99 retrieval at "
        "our target corpus size. The payload filter system maps directly to our "
        "multi-tenant isolation requirement. Weaviate's hybrid search is superior "
        "but our retrieval pipeline handles BM25 at the application layer. "
        "pgvector is rejected due to scale limitations — expected corpus growth "
        "to 2M vectors within 12 months exceeds pgvector's performance envelope."
    ),
    consequences={
        "positive": [
            "Sub-50ms P99 vector retrieval at 500K vectors, validated in benchmarks",
            "Tenant isolation enforced server-side via payload filters",
            "Daily S3 snapshots satisfy data durability requirements",
        ],
        "negative": [
            "Teams must learn Qdrant-specific client library and filter syntax",
            "Distributed mode requires minimum 3 nodes — minimum infrastructure cost",
        ],
        "neutral": [
            "Migration from Chroma (used in development) requires re-embedding ~10K dev corpus",
            "Qdrant version upgrades must be coordinated across all 3 nodes",
        ]
    },
    links=[
        "https://qdrant.tech/documentation/",
        "Benchmark: qdrant.tech/benchmarks/",
        "ADR-0001: Kubernetes as deployment platform",
        "ADR-0002: Embedding model selection",
    ],
    authors=["platform-team@company.com"]
)

# Generate and save ADR
adr_md = vector_db_adr.to_markdown()
output_path = Path("docs/adr/ADR-0003-vector-database.md")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(adr_md)
print(adr_md[:2000])
print(f"\n... (full ADR saved to {output_path})")

# ── Architecture comparison summary ───────────────────────────────
print("\n" + "=" * 55)
print("  Architecture Tier Comparison")
print("=" * 55)

COMPARISON = [
    ("Component",           "Minimal",         "Production",          "Air-Gapped"),
    ("LLM",                 "OpenAI API",       "OpenAI + fallback",   "Ollama/vLLM"),
    ("Embeddings",          "In-process",       "Svc ×2",              "In-process (local)"),
    ("Vector DB",           "ChromaDB local",   "Qdrant ×3",           "Qdrant OSS"),
    ("Cache",               "None",             "Redis ×3",            "Redis OSS"),
    ("Object Storage",      "Local disk",       "S3",                  "MinIO"),
    ("Secrets",             "Env vars",         "Vault/Secrets Mgr",   "Vault OSS"),
    ("Container Registry",  "Docker Hub",       "ECR",                 "Harbor"),
    ("CI/CD",               "GitHub Actions",   "GitHub Actions",      "Gitea"),
    ("Monitoring",          "Logs to stdout",   "Prometheus+Grafana",  "Prometheus+Grafana"),
    ("Monthly cost (est.)", "~$50–150",         "~$800–2000",          "Capex only"),
]

col_w = [24, 18, 22, 22]
header = COMPARISON[0]
print("  " + "".join(f"{h:<{w}}" for h, w in zip(header, col_w)))
print("  " + "-" * sum(col_w))
for row in COMPARISON[1:]:
    print("  " + "".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
```

**Run the lab:**
```bash
python adr_generator.py
```

**Extensions:**
- Add a fourth tier "Enterprise SaaS" to the comparison table with multi-tenant components
- Generate ADRs for two more decisions: (a) embedding model selection, (b) LLM gateway retry strategy
- Build a simple ADR index (`docs/adr/README.md`) that lists all ADRs with status and date, auto-generated from the ADR files in the directory

---

> ### 📋 Chapter Summary
>
> - The air-gapped architecture replaces every cloud-managed service with a self-hosted OSS equivalent: Ollama for LLM, Qdrant for vectors, MinIO for object storage, Vault for secrets, Harbor for container registry.
> - All three tiers (minimal, production, air-gapped) share the same application logic — only the infrastructure layer changes, validating the abstraction design from Part XI.
> - **ADRs** document architecture decisions with options, rationale, and consequences — they are living governance artefacts that must be updated when decisions change.
> - The architecture comparison table provides a concise decision matrix across tiers, enabling teams to plan their migration path from minimal to production to air-gapped.

---

> ### ❓ Comprehension Questions
>
> 1. The air-gapped stack uses Ollama for LLM inference. Ollama runs a single model at a time by default. A production workload needs to serve `llama3.1:8b` for general queries and `codellama:13b` for code questions. How would you extend the on-premise LLM platform to support multiple concurrent models?
> 2. The ADR for vector database selection lists "migration from Chroma requires re-embedding ~10K dev corpus" as a neutral consequence. In production, the corpus has 500K documents. Estimate the re-embedding time using `all-MiniLM-L6-v2` on CPU (approximate: 500 docs/minute on a 4-core server), and identify what else must be migrated beyond embeddings.
> 3. The `docker-compose.airgap.yml` uses Docker secrets for MinIO and Grafana passwords. In a true air-gapped environment, how would you bootstrap Vault without internet access, and how would you rotate secrets after initial deployment?
> 4. The multi-tenant architecture provision creates two collections per tenant: `corpus_main` and `corpus_archive`. A tenant with 3 years of history has 50 corpora variants. How would you design a corpus versioning scheme that keeps the collection namespace manageable?
> 5. An enterprise customer requires a hybrid architecture: standard queries use the cloud production stack, but queries containing specific keywords (patient names, account numbers) must be routed to an on-premise air-gapped deployment. Design the routing logic and describe the dual-write corpus maintenance challenge this creates.

---

## References

### Documentation
- [Ollama Model Library](https://ollama.ai/library) — Available models for local deployment.
- [vLLM Documentation](https://docs.vllm.ai) — GPU-optimised LLM serving.
- [MinIO Documentation](https://min.io/docs/minio/container/index.html)
- [HashiCorp Vault](https://developer.hashicorp.com/vault/docs)
- [Harbor Container Registry](https://goharbor.io/docs/)
- [Architecture Decision Records](https://adr.github.io) — ADR format and tooling.

### Papers
- [Architectural Decision Records in Practice](https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — Nygard, 2011. Original ADR proposal.

---

> **Navigation**
> [← Part XVI — Operations](part_16_operations.md) | [→ Part XVIII — Practical Case Studies](part_18_case_studies.md)

---
[« Back to reference_architectures Index](index.md) | [🏠 Home](../../index.md)