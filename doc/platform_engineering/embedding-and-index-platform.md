## Chapter 4 — Embedding and Index Platform 🧪

### 4.1 Shared Embedding Service

A shared embedding service provides a consistent embedding API for all product teams, abstracting model selection and version management.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time
import hashlib

class EmbeddingModelCapability(str, Enum):
    FAST   = "fast"    # MiniLM / small models — low latency, lower quality
    STANDARD = "standard"  # text-embedding-3-small equivalent
    HIGH_QUALITY = "high_quality"  # text-embedding-3-large equivalent
    LOCAL  = "local"   # 🔓 On-premise model

@dataclass
class EmbeddingModel:
    capability: EmbeddingModelCapability
    model_name: str
    provider: str
    dimensions: int
    max_input_tokens: int
    cost_per_1m_tokens: float

EMBEDDING_MODELS = {
    EmbeddingModelCapability.FAST: EmbeddingModel(
        EmbeddingModelCapability.FAST,
        "text-embedding-3-small", "openai", 1536, 8191, 0.02
    ),
    EmbeddingModelCapability.STANDARD: EmbeddingModel(
        EmbeddingModelCapability.STANDARD,
        "text-embedding-3-small", "openai", 1536, 8191, 0.02
    ),
    EmbeddingModelCapability.HIGH_QUALITY: EmbeddingModel(
        EmbeddingModelCapability.HIGH_QUALITY,
        "text-embedding-3-large", "openai", 3072, 8191, 0.13
    ),
    EmbeddingModelCapability.LOCAL: EmbeddingModel(
        EmbeddingModelCapability.LOCAL,
        "all-MiniLM-L6-v2", "sentence_transformers", 384, 512, 0.0
    ),
}

@dataclass
class EmbeddingRequest:
    texts: list[str]
    capability: EmbeddingModelCapability = EmbeddingModelCapability.STANDARD
    team_id: str = "unknown"
    service_id: str = "unknown"
    # Normalise output vectors to unit length
    normalise: bool = True

@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    model_name: str
    dimensions: int
    tokens_used: int
    latency_ms: float
    estimated_cost_usd: float

class EmbeddingService:
    """
    Shared embedding service. Supports OpenAI and local SentenceTransformers.
    Product teams specify capability, not model name.
    """
    def __init__(self, secret_manager, cost_tracker):
        self.secrets = secret_manager
        self.cost_tracker = cost_tracker
        self._local_model = None  # Lazy-loaded

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_config = EMBEDDING_MODELS[request.capability]
        t0 = time.perf_counter()

        if model_config.provider == "openai":
            embeddings, tokens = self._embed_openai(request.texts, model_config.model_name)
        elif model_config.provider == "sentence_transformers":
            embeddings, tokens = self._embed_local(request.texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {model_config.provider}")

        if request.normalise:
            embeddings = [self._normalise(e) for e in embeddings]

        latency_ms = (time.perf_counter() - t0) * 1000
        cost = tokens / 1_000_000 * model_config.cost_per_1m_tokens

        self.cost_tracker.record(
            team_id=request.team_id, service_id=request.service_id,
            model=model_config.model_name, prompt_tokens=tokens,
            completion_tokens=0, cost_usd=cost, correlation_id=""
        )
        return EmbeddingResponse(
            embeddings=embeddings,
            model_name=model_config.model_name,
            dimensions=model_config.dimensions,
            tokens_used=tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost
        )

    def _embed_openai(self, texts: list[str], model: str) -> tuple[list, int]:
        from openai import OpenAI
        client = OpenAI(api_key=self.secrets.get("OPENAI_API_KEY"))
        response = client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        return embeddings, response.usage.total_tokens

    def _embed_local(self, texts: list[str]) -> tuple[list, int]:
        # 🔓 On-premise: SentenceTransformers, no API call
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self._local_model.encode(texts, normalize_embeddings=True).tolist()
        tokens = sum(len(t.split()) for t in texts)  # Approximate
        return embeddings, tokens

    def _normalise(self, vec: list[float]) -> list[float]:
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec
```

---

### 4.2 Index Lifecycle Service

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json
import time

class IndexBuildStatus(str, Enum):
    QUEUED      = "queued"
    BUILDING    = "building"
    VALIDATING  = "validating"
    READY       = "ready"
    FAILED      = "failed"

@dataclass
class IndexBuildRequest:
    index_id: str
    corpus_id: str
    corpus_version: str
    team_id: str
    embedding_capability: EmbeddingModelCapability = EmbeddingModelCapability.STANDARD
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_recall_threshold: float = 0.80

@dataclass
class IndexBuildJob:
    job_id: str
    request: IndexBuildRequest
    status: IndexBuildStatus
    collection_name: str
    started_at: str
    completed_at: Optional[str] = None
    eval_recall_at_5: Optional[float] = None
    error_message: Optional[str] = None
    vector_count: int = 0

class IndexLifecycleService:
    """
    Manages the full lifecycle of vector indexes:
    Build → Validate → Stage → Promote (blue-green).
    Product teams request builds; the service handles everything else.
    """
    def __init__(self, embedding_service: EmbeddingService, vector_db_client):
        self.embedding = embedding_service
        self.vdb = vector_db_client
        self.jobs: dict[str, IndexBuildJob] = {}
        self._active_collections: dict[str, str] = {}  # index_id → active collection

    def request_build(self, req: IndexBuildRequest) -> str:
        import uuid
        job_id = uuid.uuid4().hex[:10]
        collection_name = f"{req.index_id}_{job_id}"
        job = IndexBuildJob(
            job_id=job_id, request=req,
            status=IndexBuildStatus.QUEUED,
            collection_name=collection_name,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        self.jobs[job_id] = job
        # In production: submit to async job queue (Celery, RQ, Argo)
        # For simplicity: run synchronously
        self._run_build_job(job)
        return job_id

    def get_job_status(self, job_id: str) -> IndexBuildJob:
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")
        return self.jobs[job_id]

    def get_active_collection(self, index_id: str) -> Optional[str]:
        return self._active_collections.get(index_id)

    def _run_build_job(self, job: IndexBuildJob):
        try:
            job.status = IndexBuildStatus.BUILDING
            # 1. Load corpus documents
            documents = self._load_corpus(job.request.corpus_id, job.request.corpus_version)
            # 2. Chunk documents
            chunks = self._chunk_documents(documents, job.request)
            # 3. Embed chunks
            embed_req = EmbeddingRequest(
                texts=[c["content"] for c in chunks],
                capability=job.request.embedding_capability,
                team_id=job.request.team_id,
                service_id="index-lifecycle-service"
            )
            embed_resp = self.embedding.embed(embed_req)
            # 4. Store in new collection
            self._store_chunks(job.collection_name, chunks, embed_resp.embeddings)
            job.vector_count = len(chunks)

            # 5. Validate
            job.status = IndexBuildStatus.VALIDATING
            recall = self._validate_index(job.collection_name, job.request)
            job.eval_recall_at_5 = recall

            if recall < job.request.min_recall_threshold:
                job.status = IndexBuildStatus.FAILED
                job.error_message = (
                    f"Validation failed: Recall@5={recall:.2%} < "
                    f"threshold {job.request.min_recall_threshold:.2%}"
                )
                return

            # 6. Promote (blue-green switch)
            old_collection = self._active_collections.get(job.request.index_id)
            self._active_collections[job.request.index_id] = job.collection_name
            job.status = IndexBuildStatus.READY
            job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Schedule old collection deletion after 7-day retention window
            if old_collection:
                self._schedule_deletion(old_collection, delay_days=7)

        except Exception as e:
            job.status = IndexBuildStatus.FAILED
            job.error_message = str(e)

    def _load_corpus(self, corpus_id, version):
        return []  # Load from corpus store

    def _chunk_documents(self, documents, req: IndexBuildRequest):
        return []  # Chunking pipeline

    def _store_chunks(self, collection_name, chunks, embeddings):
        pass  # Store in vector DB

    def _validate_index(self, collection_name, req: IndexBuildRequest) -> float:
        return 0.85  # Run eval dataset against collection

    def _schedule_deletion(self, collection_name, delay_days):
        pass  # Schedule async deletion job
```

---

### 4.3 Multi-Tenant Index Isolation

```python
@dataclass
class TenantIndexConfig:
    tenant_id: str
    index_id: str
    allowed_corpus_ids: list[str]  # Corpora this tenant can search
    max_vectors: int = 1_000_000
    metadata_filters: dict = None  # Always-applied tenant filter

class MultiTenantIndexRouter:
    """
    Routes queries to the correct tenant collection and applies
    mandatory metadata filters to prevent cross-tenant data access.
    """
    def __init__(self, lifecycle_service: IndexLifecycleService):
        self.lifecycle = lifecycle_service
        self.tenant_configs: dict[str, TenantIndexConfig] = {}

    def register_tenant(self, config: TenantIndexConfig):
        self.tenant_configs[config.tenant_id] = config

    def search(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict = None
    ) -> list[dict]:
        config = self.tenant_configs.get(tenant_id)
        if not config:
            raise PermissionError(f"Unknown tenant: {tenant_id}")

        collection = self.lifecycle.get_active_collection(config.index_id)
        if not collection:
            raise RuntimeError(f"No active index for tenant {tenant_id}")

        # Merge tenant mandatory filter with query filter
        effective_filter = {**(config.metadata_filters or {}), **(metadata_filter or {})}
        # In production: pass effective_filter to vector DB where clause
        return []  # Vector DB query with filter
```

---

### 4.4 Java Integration Patterns

```java
// Java client for the embedding service and index lifecycle API
@Service
public class PlatformEmbeddingClient {

    private final WebClient webClient;
    private final String embeddingServiceUrl;

    @Value("${platform.team-id}")
    private String teamId;

    @Value("${platform.service-id}")
    private String serviceId;

    public List<List<Double>> embed(List<String> texts, String capability) {
        var request = Map.of(
            "texts", texts,
            "capability", capability,
            "team_id", teamId,
            "service_id", serviceId
        );

        return webClient.post()
            .uri(embeddingServiceUrl + "/v2/embeddings")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(EmbeddingResponse.class)
            .map(EmbeddingResponse::embeddings)
            .block(Duration.ofSeconds(30));
    }

    public record EmbeddingResponse(
        List<List<Double>> embeddings,
        String modelName,
        int dimensions,
        int tokensUsed
    ) {}
}

// Using the platform gateway from Java
@Service
public class PlatformLLMClient {

    private final WebClient gatewayClient;

    @Value("${platform.team-id}")
    private String teamId;

    public String complete(List<Message> messages, String modelAlias) {
        var request = Map.of(
            "messages", messages.stream()
                .map(m -> Map.of("role", m.role(), "content", m.content()))
                .toList(),
            "model_alias", modelAlias,
            "team_id", teamId,
            "service_id", "java-rag-service"
        );

        return gatewayClient.post()
            .uri("/v2/complete")
            .bodyValue(request)
            .retrieve()
            .onStatus(HttpStatusCode::is4xxClientError, response ->
                response.bodyToMono(GatewayError.class)
                    .flatMap(err -> Mono.error(new LLMGatewayException(err.message())))
            )
            .bodyToMono(GatewayResponse.class)
            .map(GatewayResponse::content)
            .retryWhen(Retry.backoff(3, Duration.ofSeconds(1))
                .filter(e -> e instanceof WebClientResponseException.ServiceUnavailable))
            .block(Duration.ofSeconds(60));
    }
}
```

---

### 🧪 Hands-on Lab: Minimal AI Platform

**Objective:** Assemble a minimal but functional AI platform using the components from this part. Run a RAG query routed through the gateway, rendered by the prompt service, and retrieved from the index service.

**Prerequisites:** `openai`, `fastapi`, `uvicorn`, `httpx`, `sentence-transformers`

```python
#!/usr/bin/env python3
"""
minimal_platform.py — Runs a self-contained AI platform in a single process.
Demonstrates: Gateway → Prompt Service → Embedding Service → RAG pipeline.
"""

import json
import time
import hashlib
from pathlib import Path
from openai import OpenAI

oai_client = OpenAI()

# ── 1. Minimal gateway (in-process) ────────────────────────────────
class MinimalGateway:
    def __init__(self):
        self.call_log = []

    def complete(self, messages: list[dict], model_alias: str,
                 team_id: str, service_id: str) -> dict:
        model_map = {"default": "gpt-4o-mini", "powerful": "gpt-4o", "fast": "gpt-4o-mini"}
        model = model_map.get(model_alias, "gpt-4o-mini")
        t0 = time.perf_counter()
        resp = oai_client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=300
        )
        latency = (time.perf_counter() - t0) * 1000
        self.call_log.append({
            "team_id": team_id, "service_id": service_id, "model": model,
            "tokens": resp.usage.total_tokens, "latency_ms": round(latency, 1)
        })
        return {"content": resp.choices[0].message.content,
                "model": model, "tokens": resp.usage.total_tokens}

# ── 2. Minimal prompt service (in-process) ─────────────────────────
PROMPT_REGISTRY = {}

def register_prompt(template_id: str, version: str, system: str, user: str,
                    variables: list, defaults: dict, description: str):
    PROMPT_REGISTRY[f"{template_id}:production"] = {
        "template_id": template_id, "version": version,
        "system_template": system, "user_template": user,
        "variables": variables, "defaults": defaults, "description": description
    }
    print(f"  ✓ Registered prompt: {template_id} v{version}")

def render_prompt(template_id: str, variables: dict) -> list[dict]:
    tmpl = PROMPT_REGISTRY.get(f"{template_id}:production")
    if not tmpl:
        raise KeyError(f"Prompt {template_id} not in registry")
    merged = {**tmpl["defaults"], **variables}
    return [
        {"role": "system", "content": tmpl["system_template"].format(**merged)},
        {"role": "user",   "content": tmpl["user_template"].format(**merged)},
    ]

# ── 3. Minimal embedding + index service (in-process) ──────────────
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX = {"docs": [], "embeddings": []}

def ingest_documents(documents: list[dict]):
    texts = [d["content"] for d in documents]
    embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()
    INDEX["docs"].extend(documents)
    INDEX["embeddings"].extend(embeddings)
    print(f"  ✓ Indexed {len(documents)} documents ({len(INDEX['docs'])} total)")

def search_index(query: str, top_k: int = 5) -> list[dict]:
    if not INDEX["docs"]:
        return []
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, INDEX["embeddings"])[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [{"id": INDEX["docs"][i]["id"],
             "content": INDEX["docs"][i]["content"],
             "score": float(sims[i])} for i in top_idx]

# ── 4. Assemble and run ────────────────────────────────────────────
gateway = MinimalGateway()

print("\n" + "=" * 55)
print("  Minimal AI Platform — Setup")
print("=" * 55)

# Register the RAG prompt
register_prompt(
    template_id="support_rag",
    version="2.0.0",
    description="Support RAG QA prompt",
    system="You are a support assistant for {org}. Answer using only the context below. "
           "Cite sources [N]. If not found, say 'I cannot find this information.' Tone: {tone}.",
    user="Context:\n{context}\n\nQuestion: {question}",
    variables=["context", "question"],
    defaults={"org": "Acme Corp", "tone": "professional"}
)

# Ingest knowledge base
ingest_documents([
    {"id": "d1", "content": "Enterprise plan: 90-day return window. Standard plan: 30 days."},
    {"id": "d2", "content": "Professional plan costs $150/month. Includes 25 users and priority support."},
    {"id": "d3", "content": "API authentication uses OAuth 2.0. Access tokens expire after 3600 seconds."},
    {"id": "d4", "content": "Refunds are processed within 5-7 business days for all payment methods."},
    {"id": "d5", "content": "Enterprise plans include dedicated support manager and 99.99% SLA."},
])

# ── 5. Run queries through the platform ────────────────────────────
print("\n" + "=" * 55)
print("  Platform RAG Queries")
print("=" * 55)

TEST_QUERIES = [
    ("product-support", "How long can enterprise customers return items?"),
    ("product-billing",  "What does the Professional plan cost?"),
    ("product-api",      "How long do API tokens last?"),
    ("product-billing",  "What is the process for government procurement?"),
]

for team_id, query in TEST_QUERIES:
    print(f"\n  [{team_id}] Q: {query}")

    # Retrieve
    retrieved = search_index(query, top_k=3)
    context = "\n".join([f"[{i+1}] {r['content']}" for i, r in enumerate(retrieved)])

    # Render prompt via prompt service
    messages = render_prompt("support_rag", {"context": context, "question": query})

    # Call LLM via gateway
    response = gateway.complete(messages, "default", team_id=team_id, service_id="rag-app")
    print(f"     A: {response['content'][:100]}")
    print(f"     [model={response['model']} tokens={response['tokens']}]")

# ── 6. Cost report ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Platform Usage Report")
print("=" * 55)
by_team = {}
total_tokens = 0
for log in gateway.call_log:
    t = log["team_id"]
    by_team[t] = by_team.get(t, 0) + log["tokens"]
    total_tokens += log["tokens"]
for team, tokens in sorted(by_team.items()):
    est_cost = tokens / 1_000_000 * 0.15
    print(f"  {team:<25}: {tokens:>6} tokens  (~${est_cost:.4f})")
print(f"  {'TOTAL':<25}: {total_tokens:>6} tokens  (~${total_tokens/1e6*0.15:.4f})")
print()
```

**Run the lab:**
```bash
python minimal_platform.py
```

**Extensions:**
- Add a `quota_config` per team and enforce it in `MinimalGateway.complete()`, rejecting calls over budget
- Extend the prompt registry to support stage promotion (`development → staging → production`)
- Add a `metrics_report()` that shows P50/P95/P99 latency per team from `gateway.call_log`

---

> ### 📋 Chapter Summary
>
> - The **embedding service** abstracts model selection behind capabilities (`fast`, `standard`, `high_quality`, `local`) — product teams never embed model name strings.
> - The **index lifecycle service** manages build → validate → blue-green promotion atomically, with automatic rollback on validation failure.
> - **Multi-tenant index routing** applies mandatory metadata filters per tenant, preventing cross-tenant data access at the platform layer.
> - The **hands-on lab** demonstrates a complete end-to-end platform integration: gateway → prompt service → embedding → index → generation with usage reporting.

---

> ### ❓ Comprehension Questions
>
> 1. The embedding service accepts `capability` not `model_name`. A product team needs the exact `text-embedding-3-large` with 3072 dimensions for a specific downstream use. How would you accommodate this without breaking the capability abstraction?
> 2. The `IndexLifecycleService._run_build_job` runs synchronously. For a corpus of 500,000 documents, this could take 4+ hours. Redesign the method signature and workflow to support asynchronous execution with status polling.
> 3. Multi-tenant index routing applies `config.metadata_filters` to every query. A tenant's filter is `{"tenant_id": "acme"}`. An attacker modifies their query to include `{"tenant_id": {"$ne": "acme"}}`. What class of attack is this and how does the router defend against it?
> 4. The lab cost report estimates cost as `tokens / 1M * 0.15`. This is the input token price. Why is this estimate incorrect, and how would you compute a more accurate cost?
> 5. Design the interface between the platform embedding service and the index lifecycle service. When the embedding model is upgraded from `STANDARD` to a new version, what happens to existing indexes, and what does the platform need to communicate to product teams?

---

## References

### Documentation
- [SentenceTransformers Documentation](https://www.sbert.net/docs/) — Local embedding models.
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) — Cloud embedding service.
- [Qdrant Multi-tenancy](https://qdrant.tech/documentation/guides/multiple-partitions/) — Vector DB tenant isolation.
- [FastAPI](https://fastapi.tiangolo.com) — REST API for platform services.

### Papers
- [Text Embeddings Reveal Almost As Much As Text](https://arxiv.org/abs/2310.06816) — Morris et al., 2023. Embedding model capability trade-offs.

---

---
[« Back to platform_engineering Index](index.md) | [🏠 Home](../../index.md)