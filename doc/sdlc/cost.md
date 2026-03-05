## Cost
- [ ] Has the token count impact of this change been estimated?
- [ ] For prompt changes: estimated daily cost delta at current traffic?
```

---

### 3.4 Staging Environment Design

A staging environment for AI systems must mirror production across four dimensions:

```python
from dataclasses import dataclass

@dataclass
class EnvironmentSpec:
    name: str
    # LLM configuration
    llm_model: str
    llm_temperature: float
    # Vector DB
    vector_db_provider: str
    vector_db_collection: str
    # Knowledge base
    kb_document_count: int
    kb_freshness_hours: int      # Max age of knowledge base content
    # Traffic
    request_rate_per_minute: int
    # Cost constraints
    max_daily_token_spend_usd: float

PRODUCTION = EnvironmentSpec(
    name="production",
    llm_model="gpt-4o",
    llm_temperature=0.0,
    vector_db_provider="qdrant",
    vector_db_collection="kb_prod_active",
    kb_document_count=250000,
    kb_freshness_hours=24,
    request_rate_per_minute=1000,
    max_daily_token_spend_usd=500.0
)

STAGING = EnvironmentSpec(
    name="staging",
    llm_model="gpt-4o",         # Same model as production — critical
    llm_temperature=0.0,
    vector_db_provider="qdrant",
    vector_db_collection="kb_staging",
    kb_document_count=250000,   # Same corpus size as production — critical
    kb_freshness_hours=48,      # Slightly relaxed
    request_rate_per_minute=100,
    max_daily_token_spend_usd=50.0
)

DEVELOPMENT = EnvironmentSpec(
    name="development",
    llm_model="gpt-4o-mini",    # Cheaper model for development
    llm_temperature=0.0,
    vector_db_provider="chroma",
    vector_db_collection="kb_dev",
    kb_document_count=1000,     # Subset of corpus
    kb_freshness_hours=168,     # Weekly refresh acceptable
    request_rate_per_minute=10,
    max_daily_token_spend_usd=5.0
)
```

**Critical staging parity requirements:**
- Staging must use the **same LLM model** as production. Testing with `gpt-4o-mini` in staging and deploying to `gpt-4o` invalidates all staging evaluation results.
- Staging must use a **representative corpus** — not a tiny subset. Retrieval quality on 1,000 documents does not predict retrieval quality on 250,000 documents.
- Staging evaluation must use the **same evaluation dataset** as CI. Using a different dataset in staging produces non-comparable metrics.

---

> ### 📋 Chapter Summary
>
> - **Prompt-first development** writes the prompt, validates it manually, and writes evaluation test cases before implementing the surrounding code.
> - **Experiment tracking** with [MLflow](https://mlflow.org/docs/latest/index.html) or [Weights & Biases](https://docs.wandb.ai) ensures all experiments are reproducible and comparable.
> - Code review checklists must explicitly include prompt safety, retrieval correctness, LLM error handling, and cost impact.
> - Staging must match production on LLM model and corpus size — these are the two most common sources of staging-to-production quality discrepancy.

---

> ### ❓ Comprehension Questions
>
> 1. A team implements the RAG pipeline first and writes the prompt last ("we'll tune it after"). What problems does this create compared to the prompt-first approach?
> 2. Two experiment runs have identical parameters but different Recall@5 scores (0.83 vs 0.87). What sources of non-determinism could explain this, and how would you control for them?
> 3. A code reviewer approves a prompt change without running the test suite, relying on visual inspection. The change ships to production and causes a 15% quality regression. What process control is missing?
> 4. Staging uses `gpt-4o-mini` for cost efficiency. The evaluation gate passes. Production uses `gpt-4o`. What are three ways the quality in production could differ from staging?
> 5. Design an experiment tracking schema that captures all parameters needed to reproduce a RAG experiment six months later, even after library upgrades.

---

## References

### SCM Workflow and Code Review
- [GitHub Pull Requests](https://docs.github.com/en/pull-requests) — PR workflow, reviewers, required status checks, and branch protection rules.
- [GitLab Merge Requests](https://docs.gitlab.com/ee/user/project/merge_requests/) — MR workflow with inline code review and approval rules.
- [Bitbucket Pull Requests](https://support.atlassian.com/bitbucket-cloud/docs/create-a-pull-request/) — Atlassian's PR workflow with Jira issue linking.
- [Conventional Commits](https://www.conventionalcommits.org) — Commit message standard enabling automated changelog generation and SemVer bumps.
- [Semantic Release](https://semantic-release.gitbook.io) — Automated versioning and changelog from commit history; integrates with GitHub/GitLab CI.

### Experiment Tracking
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) — Experiment tracking and comparison.
- [Weights & Biases](https://docs.wandb.ai) — Experiment tracking with LLM support.
- [LangSmith](https://docs.smith.langchain.com) — LangChain experiment tracking and evaluation.
- [DVC Experiments](https://dvc.org/doc/user-guide/experiment-management) — Experiment tracking integrated with DVC.

### Papers
- [Experiment Tracking for Machine Learning](https://arxiv.org/abs/2108.12048) — Survey of experiment tracking practices.

---

---
[« Back to sdlc Index](index.md) | [🏠 Home](../../index.md)