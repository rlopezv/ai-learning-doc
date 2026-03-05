## Chapter 3 — Prompt Management Platform

### 3.1 Centralised Prompt Service

The prompt service provides a centralised API for all prompt template operations. Product teams register, fetch, and A/B test prompts through this API — never by reading files directly.

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import json
import hashlib
from datetime import datetime
from pathlib import Path

app = FastAPI(
    title="Prompt Management Service",
    version="2.0.0",
    description="Centralised prompt registry for all AI product teams"
)

# ── API schemas ──────────────────────────────────────────────────────────────
class PromptRegistrationRequest(BaseModel):
    template_id: str = Field(..., description="Unique identifier for this prompt template")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version")
    description: str
    system_template: str
    user_template: str
    variables: list[str]
    optional_variables: list[str] = []
    defaults: dict = {}
    author: str
    team_id: str
    tags: list[str] = []
    change_summary: str = Field(..., description="Human-readable description of changes")

class PromptFetchRequest(BaseModel):
    template_id: str
    version: str = "production"   # "production" | "staging" | specific version
    team_id: str
    service_id: str

class PromptRenderRequest(BaseModel):
    template_id: str
    version: str = "production"
    variables: dict
    team_id: str
    service_id: str

class PromptVersionInfo(BaseModel):
    template_id: str
    version: str
    stage: str
    description: str
    author: str
    team_id: str
    created_at: str
    content_hash: str
    change_summary: str

# ── Service implementation ────────────────────────────────────────────────────
class PromptStore:
    def __init__(self, base_path: str = "./prompt_store"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, req: PromptRegistrationRequest, stage: str = "development") -> str:
        template_dir = self.base / req.template_id
        template_dir.mkdir(exist_ok=True)

        data = {**req.dict(), "stage": stage, "created_at": datetime.utcnow().isoformat()}
        content = json.dumps(data, indent=2, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        data["content_hash"] = content_hash

        (template_dir / f"{req.version}.json").write_text(json.dumps(data, indent=2))
        return content_hash

    def load(self, template_id: str, version: str = "production") -> dict:
        template_dir = self.base / template_id
        if version in ("production", "staging", "development"):
            # Find latest version with that stage
            candidates = []
            for f in template_dir.glob("*.json"):
                if f.stem == "latest":
                    continue
                try:
                    d = json.loads(f.read_text())
                    if d.get("stage") == version:
                        candidates.append(d)
                except Exception:
                    pass
            if not candidates:
                raise HTTPException(404, f"No {version} version of {template_id}")
            return max(candidates, key=lambda d: d["created_at"])

        version_file = template_dir / f"{version}.json"
        if not version_file.exists():
            raise HTTPException(404, f"{template_id} v{version} not found")
        return json.loads(version_file.read_text())

    def set_stage(self, template_id: str, version: str, stage: str):
        data = self.load(template_id, version)
        data["stage"] = stage
        data["stage_updated_at"] = datetime.utcnow().isoformat()
        (self.base / template_id / f"{version}.json").write_text(json.dumps(data, indent=2))

store = PromptStore()

# ── REST endpoints ─────────────────────────────────────────────────────────
@app.post("/v2/prompts", response_model=dict, tags=["prompts"])
def register_prompt(req: PromptRegistrationRequest):
    """Register a new prompt version in the registry."""
    content_hash = store.save(req)
    return {"template_id": req.template_id, "version": req.version,
            "content_hash": content_hash, "stage": "development"}

@app.get("/v2/prompts/{template_id}/versions", response_model=list, tags=["prompts"])
def list_versions(template_id: str):
    """List all registered versions of a prompt template."""
    template_dir = store.base / template_id
    if not template_dir.exists():
        raise HTTPException(404, f"Template {template_id} not found")
    versions = []
    for f in sorted(template_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            versions.append({"version": d["version"], "stage": d.get("stage", "development"),
                             "created_at": d.get("created_at"), "author": d.get("author")})
        except Exception:
            pass
    return versions

@app.post("/v2/prompts/render", response_model=dict, tags=["prompts"])
def render_prompt(req: PromptRenderRequest):
    """Fetch and render a prompt template with variables substituted."""
    data = store.load(req.template_id, req.version)
    try:
        system = data["system_template"].format(
            **{**data.get("defaults", {}), **req.variables}
        )
        user = data["user_template"].format(
            **{**data.get("defaults", {}), **req.variables}
        )
    except KeyError as e:
        raise HTTPException(400, f"Missing required variable: {e}")
    return {
        "template_id": req.template_id,
        "version": data["version"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "content_hash": data.get("content_hash")
    }

@app.post("/v2/prompts/{template_id}/{version}/promote", tags=["governance"])
def promote_prompt(template_id: str, version: str, target_stage: str):
    """Promote a prompt version to a new stage (staging → production)."""
    allowed_transitions = {
        "development": ["staging"],
        "staging": ["production"],
        "production": []
    }
    current = store.load(template_id, version)
    current_stage = current.get("stage", "development")
    if target_stage not in allowed_transitions.get(current_stage, []):
        raise HTTPException(
            400,
            f"Cannot promote from {current_stage} to {target_stage}. "
            f"Allowed: {allowed_transitions.get(current_stage, [])}"
        )
    store.set_stage(template_id, version, target_stage)
    return {"template_id": template_id, "version": version,
            "previous_stage": current_stage, "new_stage": target_stage}
```

---

### 3.2 Prompt Serving with Caching

```python
import functools
from typing import Optional

class PromptCache:
    """
    In-memory LRU cache for frequently accessed prompts.
    Prevents repeated disk/DB reads for high-traffic templates.
    """
    def __init__(self, maxsize: int = 200):
        self._cache: dict = {}
        self._maxsize = maxsize
        self._access_order: list = []

    def get(self, key: str) -> Optional[dict]:
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: dict):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._maxsize:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        self._cache[key] = value
        self._access_order.append(key)

    def invalidate(self, template_id: str):
        """Invalidate all cached versions of a template (on promotion)."""
        keys_to_remove = [k for k in self._cache if k.startswith(f"{template_id}:")]
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

class CachingPromptClient:
    """
    Client-side wrapper that product services use to fetch prompts.
    Handles caching, fallback, and telemetry transparently.
    """
    def __init__(self, prompt_service_url: str, cache: PromptCache):
        self.url = prompt_service_url
        self.cache = cache

    def get_messages(
        self,
        template_id: str,
        variables: dict,
        version: str = "production",
        team_id: str = "unknown",
        service_id: str = "unknown"
    ) -> list[dict]:
        import httpx
        cache_key = f"{template_id}:{version}"
        cached_template = self.cache.get(cache_key)

        if not cached_template:
            resp = httpx.post(
                f"{self.url}/v2/prompts/render",
                json={
                    "template_id": template_id,
                    "version": version,
                    "variables": variables,
                    "team_id": team_id,
                    "service_id": service_id
                },
                timeout=5.0
            )
            resp.raise_for_status()
            return resp.json()["messages"]

        # Render from cached template
        try:
            system = cached_template["system_template"].format(
                **{**cached_template.get("defaults", {}), **variables}
            )
            user = cached_template["user_template"].format(
                **{**cached_template.get("defaults", {}), **variables}
            )
            return [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        except KeyError:
            # Cache miss on variable — re-fetch
            self.cache.invalidate(template_id)
            return self.get_messages(template_id, variables, version, team_id, service_id)
```

---

### 3.3 Prompt Governance Workflows

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class ApprovalStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"
    WITHDRAWN = "withdrawn"

@dataclass
class PromotionRequest:
    request_id: str
    template_id: str
    version: str
    from_stage: str
    to_stage: str
    requester: str
    team_id: str
    change_summary: str
    eval_recall_at_5: Optional[float] = None
    prompt_test_pass_rate: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: Optional[str] = None
    review_comment: Optional[str] = None
    reviewed_at: Optional[str] = None

class PromotionWorkflow:
    """
    Governs prompt promotion with configurable approval requirements.
    Staging → Production requires: eval gate + tech lead approval.
    Development → Staging requires: eval gate pass only (auto-approved).
    """
    def __init__(self, store: PromptStore):
        self.store = store
        self.pending_requests: dict[str, PromotionRequest] = {}

    def request_promotion(self, req: PromotionRequest) -> str:
        # Auto-approve dev → staging if eval gate passes
        if req.from_stage == "development" and req.to_stage == "staging":
            if req.prompt_test_pass_rate == 1.0:
                self._execute_promotion(req)
                req.status = ApprovalStatus.APPROVED
                req.reviewer = "auto-approval"
                return f"Auto-approved: {req.template_id} v{req.version} → staging"

        # Staging → production requires human approval
        self.pending_requests[req.request_id] = req
        return f"Promotion request {req.request_id} created, awaiting review"

    def approve(self, request_id: str, reviewer: str, comment: str = "") -> str:
        req = self.pending_requests.get(request_id)
        if not req:
            raise ValueError(f"Request {request_id} not found")

        # Validate eval gate before approving
        if req.eval_recall_at_5 and req.eval_recall_at_5 < 0.80:
            raise ValueError(
                f"Cannot approve: Recall@5 {req.eval_recall_at_5:.2%} below threshold 80%"
            )
        if req.prompt_test_pass_rate is not None and req.prompt_test_pass_rate < 1.0:
            raise ValueError(
                f"Cannot approve: prompt tests {req.prompt_test_pass_rate:.0%} — must be 100%"
            )

        self._execute_promotion(req)
        req.status = ApprovalStatus.APPROVED
        req.reviewer = reviewer
        req.review_comment = comment
        req.reviewed_at = datetime.utcnow().isoformat()
        del self.pending_requests[request_id]
        return f"Approved: {req.template_id} v{req.version} → {req.to_stage}"

    def reject(self, request_id: str, reviewer: str, reason: str) -> str:
        req = self.pending_requests.get(request_id)
        if not req:
            raise ValueError(f"Request {request_id} not found")
        req.status = ApprovalStatus.REJECTED
        req.reviewer = reviewer
        req.review_comment = reason
        req.reviewed_at = datetime.utcnow().isoformat()
        del self.pending_requests[request_id]
        return f"Rejected: {req.template_id} v{req.version} — {reason}"

    def _execute_promotion(self, req: PromotionRequest):
        self.store.set_stage(req.template_id, req.version, req.to_stage)
```

---

> ### 📋 Chapter Summary
>
> - The **prompt service** provides a REST API for registering, fetching, rendering, and promoting prompt templates — product teams never manage prompt files directly.
> - **Prompt caching** at the client side prevents high-frequency repeated fetches for production templates.
> - **Governance workflows** enforce approval gates: development → staging auto-approves when tests pass; staging → production requires human tech lead approval.
> - Stage transitions are one-directional (development → staging → production) and validated at every step.

---

> ### ❓ Comprehension Questions
>
> 1. The prompt service returns rendered messages (system + user turn). An alternative design returns the raw template and renders client-side. What are the trade-offs of each approach in terms of caching, observability, and feature consistency?
> 2. A product team registers a prompt version directly to the `production` stage by calling the API with `stage="production"`. What process control prevents this, and how would you implement it?
> 3. The `PromotionWorkflow` auto-approves development → staging when `prompt_test_pass_rate == 1.0`. Should evaluation Recall@5 also be required for this transition? Argue both positions.
> 4. The prompt cache has `maxsize=200`. The system has 50 unique templates each with 3 active versions. Is 200 sufficient? How would cache eviction affect production traffic if popular templates are evicted?
> 5. A product team needs to test a new prompt version against 5% of production traffic. How would you extend the prompt service's `render` endpoint to support A/B routing at the platform level?

---

## References

### Documentation
- [LangSmith Prompt Hub](https://docs.smith.langchain.com/prompt_engineering) — Managed prompt registry.
- [Helicone Prompts](https://docs.helicone.ai/features/prompts) — Prompt versioning with observability.
- [FastAPI Documentation](https://fastapi.tiangolo.com) — REST API framework.

---

---
[« Back to platform_engineering Index](index.md) | [🏠 Home](../../index.md)