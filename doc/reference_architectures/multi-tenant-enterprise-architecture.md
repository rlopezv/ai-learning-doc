## Chapter 3 — Multi-Tenant Enterprise Architecture

### 3.1 Tenancy Model Options

```python
from dataclasses import dataclass
from enum import Enum

class TenancyModel(str, Enum):
    SILO      = "silo"       # Fully isolated stack per tenant
    POOL      = "pool"       # Shared infrastructure, logical isolation
    BRIDGE    = "bridge"     # Shared compute, isolated data stores

@dataclass
class TenancyOption:
    model: TenancyModel
    description: str
    isolation_level: str
    cost_per_tenant: str
    operational_complexity: str
    use_case: str
    data_residency_compliant: bool

TENANCY_OPTIONS = [
    TenancyOption(
        TenancyModel.SILO,
        "Separate Kubernetes namespace, vector DB collection set, and LLM gateway per tenant",
        isolation_level="complete",
        cost_per_tenant="high — dedicated resources",
        operational_complexity="high — N × operational burden",
        use_case="Regulated industries (financial, health), government, highest-value tenants",
        data_residency_compliant=True
    ),
    TenancyOption(
        TenancyModel.POOL,
        "Shared infrastructure; tenant isolation via metadata filters and RBAC",
        isolation_level="logical",
        cost_per_tenant="low — shared resources",
        operational_complexity="medium — single platform to operate",
        use_case="SaaS product with many small-medium tenants, internal enterprise teams",
        data_residency_compliant=False  # Data co-located across tenants
    ),
    TenancyOption(
        TenancyModel.BRIDGE,
        "Shared compute (API, embedding, LLM gateway); isolated vector collections per tenant",
        isolation_level="data-isolated, compute-shared",
        cost_per_tenant="medium",
        operational_complexity="medium",
        use_case="Enterprise with compliance requirements but cost sensitivity",
        data_residency_compliant=True  # Vectors isolated per tenant collection
    ),
]

def recommend_tenancy(
    tenant_count: int,
    has_regulated_tenants: bool,
    data_residency_required: bool,
    cost_sensitive: bool
) -> TenancyModel:
    if has_regulated_tenants and data_residency_required:
        return TenancyModel.SILO
    if tenant_count > 50 and cost_sensitive:
        return TenancyModel.POOL
    return TenancyModel.BRIDGE
```

---

### 3.2 Shared Platform with Tenant Isolation

```
Multi-Tenant Bridge Architecture

                        ┌─────────────────────┐
                        │   Control Plane      │
                        │  Tenant registry     │
                        │  RBAC / Auth         │
                        │  Billing / Quotas    │
                        │  Governance store    │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │         Shared Data Plane (Kubernetes namespace)     │
        │                                                     │
        │  [API Gateway]  [RAG API ×3]  [LLM Gateway ×2]     │
        │       │              │               │              │
        │       │         [Embedding Svc ×2]   │              │
        │       │                              │              │
        └───────┼──────────────────────────────┼──────────────┘
                │                              │
     ┌──────────▼──────────────────────────────▼──────────┐
     │            Qdrant — Multi-collection                │
     │                                                     │
     │   tenant_acme/           tenant_globex/             │
     │   ├── corpus_main        ├── corpus_main            │
     │   ├── corpus_archive     └── corpus_archive         │
     │   └── corpus_hr                                     │
     └─────────────────────────────────────────────────────┘
     
     Every query: filter = {tenant_id: "acme"} — enforced server-side
```

```python
class MultiTenantRAGRouter:
    """
    Routes requests to the correct tenant-isolated data plane.
    Enforces:
    - Tenant namespace isolation
    - Per-tenant token budgets
    - Per-tenant prompt versions
    - Cross-tenant audit logging
    """
    def __init__(self, tenant_registry, budget_enforcer,
                 prompt_client, vector_db, llm_gateway):
        self.registry  = tenant_registry
        self.budgets   = budget_enforcer
        self.prompts   = prompt_client
        self.vdb       = vector_db
        self.gateway   = llm_gateway

    def query(self, tenant_id: str, user_id: str, question: str) -> dict:
        # 1. Resolve tenant configuration
        tenant = self.registry.get(tenant_id)
        if not tenant:
            raise ValueError(f"Unknown tenant: {tenant_id}")

        # 2. Budget check
        allowed, reason = self.budgets.check_and_record(
            tenant_id, estimated_tokens=800, estimated_cost_usd=0.001
        )
        if not allowed:
            return {"error": reason, "code": "BUDGET_EXCEEDED"}

        # 3. Retrieve from tenant-isolated collection
        q_emb = self._embed(question)
        results = self.vdb.search(
            collection=f"tenant_{tenant_id}/corpus_main",
            query_vector=q_emb,
            filter={"tenant_id": tenant_id},   # Mandatory server-side filter
            top_k=tenant.get("top_k", 5)
        )

        # 4. Fetch tenant-specific prompt template
        prompt_id = tenant.get("prompt_template", "default_support")
        messages  = self.prompts.get_messages(prompt_id, {
            "context":  "\n".join(r["content"] for r in results),
            "question": question
        })

        # 5. Generate with tenant model preference
        model_alias = tenant.get("model_alias", "default")
        response    = self.gateway.complete(messages, model_alias)

        return {
            "tenant_id": tenant_id,
            "answer":    response["answer"],
            "model":     response["model"],
            "sources":   [r["id"] for r in results]
        }

    def _embed(self, text: str) -> list[float]:
        return []  # Delegate to embedding service
```

---

### 3.3 Control Plane and Data Plane Separation

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TenantRecord:
    """Tenant registration in the control plane."""
    tenant_id: str
    name: str
    tier: str                        # "starter" | "professional" | "enterprise"
    # Data plane configuration
    vector_collection_prefix: str
    prompt_template_id: str
    model_alias: str
    top_k: int = 5
    # Quotas (from control plane, enforced in data plane)
    daily_token_limit: int = 100_000
    daily_cost_limit_usd: float = 5.0
    # Compliance
    data_residency: str = "any"      # "EU" | "US" | "any"
    eu_ai_act_tier: str = "minimal_risk"
    # Operational
    active: bool = True
    created_at: str = ""

class ControlPlane:
    """
    Manages tenant lifecycle and propagates configuration to data plane.
    Separated from data plane: control plane changes do not require
    data plane restarts.
    """
    def __init__(self, db, config_store):
        self.db     = db            # PostgreSQL — tenant records
        self.config = config_store  # Redis — fast config cache for data plane

    def provision_tenant(self, record: TenantRecord):
        """Provision a new tenant: create DB record + vector collection."""
        # Persist to control plane DB
        self.db.upsert_tenant(record)

        # Create isolated vector collections
        collections = [
            f"tenant_{record.tenant_id}/corpus_main",
            f"tenant_{record.tenant_id}/corpus_archive",
        ]
        for coll in collections:
            self.db.create_vector_collection(coll)

        # Push config to cache for data plane hot-reload
        self.config.set(f"tenant:{record.tenant_id}", record.__dict__)
        print(f"  ✓ Tenant {record.tenant_id} provisioned")

    def update_quota(self, tenant_id: str, daily_tokens: int, daily_cost: float):
        self.db.update_quota(tenant_id, daily_tokens, daily_cost)
        # Live update: data plane reads from cache, no restart needed
        record = self.db.get_tenant(tenant_id)
        record["daily_token_limit"] = daily_tokens
        record["daily_cost_limit_usd"] = daily_cost
        self.config.set(f"tenant:{tenant_id}", record)

    def deprovision_tenant(self, tenant_id: str, retain_data_days: int = 30):
        """Mark tenant inactive; schedule data deletion after retention period."""
        self.db.set_active(tenant_id, False)
        self.config.set(f"tenant:{tenant_id}:active", False)
        self.db.schedule_deletion(tenant_id, retain_data_days)
        print(f"  ✓ Tenant {tenant_id} deprovisioned; data deleted in {retain_data_days}d")
```

---

### 3.4 Cross-Tenant Governance and Cost Allocation

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TenantUsageReport:
    tenant_id: str
    period: str            # ISO date range
    total_queries: int
    total_tokens: int
    total_cost_usd: float
    avg_faithfulness: Optional[float]
    avg_latency_ms: float
    refusal_rate: float
    incidents: int
    compliance_status: str   # "compliant" | "non_compliant" | "review_required"

class CrossTenantGovernanceReport:
    """Generates platform-wide and per-tenant governance reports."""

    def __init__(self, metrics_store, audit_store, compliance_runner):
        self.metrics    = metrics_store
        self.audit      = audit_store
        self.compliance = compliance_runner

    def generate_monthly_report(self, period: str) -> dict:
        tenant_ids = self.metrics.list_active_tenants(period)
        tenant_reports = []

        for tid in tenant_ids:
            usage    = self.metrics.get_usage(tid, period)
            quality  = self.metrics.get_quality(tid, period)
            incidents = self.audit.count_incidents(tid, period)
            comp_result = self.compliance.run({"system_id": tid})

            tenant_reports.append(TenantUsageReport(
                tenant_id=tid,
                period=period,
                total_queries=usage["total_queries"],
                total_tokens=usage["total_tokens"],
                total_cost_usd=usage["total_cost_usd"],
                avg_faithfulness=quality.get("faithfulness"),
                avg_latency_ms=quality.get("avg_latency_ms", 0),
                refusal_rate=quality.get("refusal_rate", 0),
                incidents=incidents,
                compliance_status=comp_result.overall_status
            ))

        total_cost = sum(r.total_cost_usd for r in tenant_reports)
        return {
            "period":         period,
            "tenant_count":   len(tenant_reports),
            "platform_cost":  round(total_cost, 2),
            "tenants":        [r.__dict__ for r in tenant_reports],
            "non_compliant":  [r.tenant_id for r in tenant_reports
                               if r.compliance_status == "non_compliant"],
            "generated_at":   datetime.utcnow().isoformat()
        }
```

---

> ### 📋 Chapter Summary
>
> - Three tenancy models (silo, pool, bridge) trade isolation level for cost and complexity. The bridge model — shared compute, isolated data — is the most common enterprise choice.
> - **Multi-tenant isolation** is enforced at three levels: Qdrant collection namespace, mandatory server-side metadata filters, and RBAC at the API gateway.
> - **Control plane / data plane separation** allows quota and configuration changes to propagate via a config cache without data plane restarts.
> - Cross-tenant governance reports combine usage, quality, incidents, and compliance status per tenant — providing the platform team a single view for SLA management and billing.

---

> ### ❓ Comprehension Questions
>
> 1. The bridge model uses shared compute but isolated Qdrant collections. A "noisy neighbour" tenant submits 100 large requests simultaneously, saturating the embedding service. How would you implement request-level tenant fairness without isolating compute entirely?
> 2. `ControlPlane.deprovision_tenant` retains data for 30 days before deletion. GDPR Art. 17 requires erasure "without undue delay". Is 30 days acceptable under GDPR? What factors determine the acceptable retention period after deprovisioning?
> 3. The cross-tenant report includes `avg_faithfulness` per tenant. Tenant A has 0.91 faithfulness; Tenant B has 0.73. Both use the same RAG platform. What tenant-specific factors (not platform bugs) could explain this 18-point gap?
> 4. `MultiTenantRAGRouter` enforces a server-side `tenant_id` filter on every Qdrant query. A developer bypasses the router and calls Qdrant directly with a forged `tenant_id` filter. What infrastructure-level control prevents this, and how would you implement it?
> 5. A new enterprise tenant requires data residency in the EU. The platform currently runs in `us-east-1`. Design the minimum architecture change to support EU-resident tenants without rebuilding the entire platform.

---

## References

### Documentation
- [Qdrant Multi-tenancy Guide](https://qdrant.tech/documentation/guides/multiple-partitions/)
- [Kubernetes Namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/)
- [AWS Organizations for Multi-Account](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_introduction.html)

---

---
[« Back to reference_architectures Index](index.md) | [🏠 Home](../../index.md)