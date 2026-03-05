## Chapter 1 — The AI Systems Development Lifecycle

### 1.1 How AI SDLC Differs from Traditional SDLC

Traditional software development follows a well-understood lifecycle: write code, write tests, merge, deploy. The feedback loop is deterministic — a function either returns the correct value or it does not. Tests are binary. Regressions are detectable with exact matching.

AI systems break every one of these assumptions.

**Behaviour is probabilistic.** The same prompt with the same inputs may produce different outputs across runs. A test that checks for an exact string will be flaky by design. Evaluation requires statistical measures over ensembles of inputs.

**Quality is multi-dimensional.** A change may improve answer accuracy while degrading latency, citation quality, or refusal rate. There is no single "it passes" criterion.

**Artifacts beyond code must be versioned and tested.** A prompt change is a release. A knowledge base update is a release. A new embedding model is a release. All of these require the same pipeline discipline as a code release — but traditional CI/CD pipelines are not designed for them.

**Regressions are silent.** A code bug typically manifests as a crash or a wrong value. An LLM regression manifests as subtly worse answers — harder to detect without explicit evaluation infrastructure.

**Experiments are a primary activity.** Data scientists and engineers spend significant time running experiments (comparing prompt versions, chunking strategies, embedding models) that do not produce production code but do produce knowledge. This activity requires infrastructure — experiment tracking, evaluation datasets, result storage — that traditional SDLC does not account for.

---

### 1.2 The Dual-Track Development Model

AI systems engineering benefits from a **dual-track model** that separates exploratory work from production work:

```
Exploration Track                    Production Track
─────────────────                    ────────────────
Goal: Learn                          Goal: Ship

Activities:                          Activities:
• Prompt experiments                 • Feature implementation
• Chunking strategy comparison       • Integration tests
• Embedding model evaluation         • Evaluation gate passage
• RAG pipeline A/B tests             • Deployment pipeline

Artifacts:                           Artifacts:
• Experiment reports                 • Versioned code
• Evaluation metrics                 • Versioned prompts
• ADRs for decisions                 • Dataset versions
                                     • Release notes

Timebox: 1–2 sprints max             Timebox: Normal sprint cycle
```

The critical discipline: **exploration does not extend indefinitely**. When an experiment produces a conclusion (positive or negative), it feeds into an ADR and either becomes a production task or is archived. An experiment that runs for three months without a conclusion is a process failure.

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class ExperimentStatus(str, Enum):
    ACTIVE = "active"
    CONCLUDED_POSITIVE = "concluded_positive"   # → Production task created
    CONCLUDED_NEGATIVE = "concluded_negative"   # → ADR records the learning
    ARCHIVED = "archived"                       # → Time-boxed out

@dataclass
class Experiment:
    experiment_id: str
    hypothesis: str                  # What we believe will be true
    success_metric: str              # How we will measure it
    success_threshold: float         # Minimum value to consider positive
    owner: str
    timebox_days: int = 14           # Max duration before forced conclusion
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: ExperimentStatus = ExperimentStatus.ACTIVE
    result_metric: Optional[float] = None
    conclusion: Optional[str] = None
    adr_reference: Optional[str] = None
    production_ticket: Optional[str] = None

    def is_overdue(self) -> bool:
        started = datetime.fromisoformat(self.started_at)
        elapsed = (datetime.utcnow() - started).days
        return elapsed > self.timebox_days

    def conclude(self, result_metric: float, conclusion: str):
        self.result_metric = result_metric
        self.conclusion = conclusion
        self.status = (
            ExperimentStatus.CONCLUDED_POSITIVE
            if result_metric >= self.success_threshold
            else ExperimentStatus.CONCLUDED_NEGATIVE
        )

class ExperimentRegistry:
    def __init__(self):
        self.experiments: dict[str, Experiment] = {}

    def register(self, experiment: Experiment):
        self.experiments[experiment.experiment_id] = experiment

    def overdue_experiments(self) -> list[Experiment]:
        return [e for e in self.experiments.values()
                if e.status == ExperimentStatus.ACTIVE and e.is_overdue()]

    def summary(self) -> dict:
        by_status = {}
        for e in self.experiments.values():
            by_status[e.status.value] = by_status.get(e.status.value, 0) + 1
        return by_status
```

---

### 1.3 Definition of Done for AI Features

"Definition of Done" (DoD) must be explicitly extended for AI features. A feature is done when:

```
Standard DoD (inherited from traditional SDLC):
☐ Code reviewed and approved
☐ Unit tests pass
☐ Integration tests pass
☐ Security scan passes
☐ Documentation updated

AI-specific DoD extensions:
☐ Evaluation gate passes (Recall@K, faithfulness, answer relevancy)
☐ Prompt changes reviewed and regression tests pass
☐ Any new prompt version registered in prompt registry
☐ Any knowledge base changes result in index rebuild + validation
☐ Latency benchmarks within SLA budget (P50, P95, P99)
☐ Experiment results documented (ADR or experiment report)
☐ Runbook updated if operational procedure changed
☐ Cost estimate reviewed (token count impact of prompt changes)
```

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FeatureDoD:
    """Checklist enforcement for AI feature completion."""
    feature_id: str
    # Standard items
    code_reviewed: bool = False
    unit_tests_pass: bool = False
    integration_tests_pass: bool = False
    security_scan_pass: bool = False
    # AI-specific items
    eval_gate_pass: bool = False
    eval_score: Optional[float] = None
    eval_threshold: float = 0.80
    prompt_regression_pass: bool = False
    latency_p99_ms: Optional[float] = None
    latency_budget_ms: float = 500.0
    cost_estimate_reviewed: bool = False
    experiment_documented: bool = False
    # Optional (required for specific change types)
    index_rebuilt_and_validated: Optional[bool] = None   # Required if corpus changed
    prompt_registered: Optional[bool] = None             # Required if prompt changed
    runbook_updated: Optional[bool] = None               # Required if ops procedure changed

    def is_complete(self) -> tuple[bool, list[str]]:
        failures = []
        if not self.code_reviewed:
            failures.append("Code not reviewed")
        if not self.unit_tests_pass:
            failures.append("Unit tests failing")
        if not self.eval_gate_pass:
            failures.append(f"Eval gate not passed (score: {self.eval_score}, threshold: {self.eval_threshold})")
        if not self.prompt_regression_pass:
            failures.append("Prompt regression tests not passed")
        if self.latency_p99_ms and self.latency_p99_ms > self.latency_budget_ms:
            failures.append(f"P99 latency {self.latency_p99_ms}ms exceeds budget {self.latency_budget_ms}ms")
        if not self.cost_estimate_reviewed:
            failures.append("Cost estimate not reviewed")
        # Conditional checks
        if self.index_rebuilt_and_validated is False:
            failures.append("Index rebuild/validation required but not completed")
        if self.prompt_registered is False:
            failures.append("Prompt version not registered in registry")
        return len(failures) == 0, failures

    def report(self):
        complete, failures = self.is_complete()
        status = "✓ DONE" if complete else "✗ NOT DONE"
        print(f"\nFeature {self.feature_id}: {status}")
        for f in failures:
            print(f"  ✗ {f}")
        if complete:
            print("  All DoD criteria satisfied.")
```

---

### 1.4 Sprint Ceremonies for AI Teams

Standard Agile ceremonies require adaptation for AI teams:

**Sprint Planning for AI:**
- Distinguish exploration tasks (timeboxed, outcome: knowledge) from delivery tasks (standard, outcome: working software)
- Estimate exploration tasks by timebox, not by story points
- Ensure each sprint has evaluation dataset tasks — evaluation infrastructure is a first-class deliverable, not a "nice to have"

**Backlog refinement for AI:**
```python
from enum import Enum

class TaskType(str, Enum):
    DELIVERY = "delivery"         # Standard feature/fix → story points
    EXPLORATION = "exploration"   # Experiment → timebox in days
    INFRASTRUCTURE = "infra"      # Platform, tooling → story points
    DATA = "data"                 # Dataset creation/curation → story points

@dataclass
class AIBacklogItem:
    id: str
    title: str
    task_type: TaskType
    description: str
    acceptance_criteria: list[str]
    # Delivery tasks
    story_points: Optional[int] = None
    # Exploration tasks
    timebox_days: Optional[int] = None
    hypothesis: Optional[str] = None
    success_metric: Optional[str] = None
    # Both types
    dependencies: list[str] = field(default_factory=list)
    ai_dod_extensions: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors = []
        if self.task_type == TaskType.EXPLORATION:
            if not self.timebox_days:
                errors.append("Exploration tasks require timebox_days")
            if not self.hypothesis:
                errors.append("Exploration tasks require hypothesis")
            if not self.success_metric:
                errors.append("Exploration tasks require success_metric")
        elif self.task_type == TaskType.DELIVERY and not self.story_points:
            errors.append("Delivery tasks require story_points")
        return errors
```

**Sprint retrospective additions for AI:**
- Measure proportion of sprint capacity consumed by rework from silent regressions
- Review overdue experiments — why did they run over timebox?
- Review evaluation score trend — is quality improving sprint-over-sprint?

---

> ### 📋 Chapter Summary
>
> - AI SDLC differs from traditional SDLC in four fundamental ways: probabilistic behaviour, multi-dimensional quality, non-code artifacts requiring pipeline discipline, and silent regressions.
> - The **dual-track model** separates exploration (timeboxed, goal: learn) from production (standard delivery, goal: ship).
> - **Definition of Done** must be explicitly extended for AI: evaluation gates, prompt regression, latency budgets, cost review, and experiment documentation.
> - Agile ceremonies require adaptation: exploration tasks are timeboxed not pointed; evaluation infrastructure is a first-class sprint deliverable.

---

> ### ❓ Comprehension Questions
>
> 1. A traditional developer joins an AI team and applies standard DoD to an LLM feature ("tests pass, PR approved"). What quality dimensions are missed and what failures could result?
> 2. An exploration experiment has been running for 6 weeks without a conclusion. Using the dual-track model, what intervention is required and what process should have prevented this?
> 3. Explain why story points are inappropriate for estimating exploration tasks. What does a timebox estimate represent that a story point estimate does not?
> 4. A team finds that 40% of sprint velocity is consumed by fixing silent regressions discovered in production. What process changes to the SDLC would reduce this?
> 5. Design a sprint planning template that includes slots for: delivery tasks, exploration timeboxes, evaluation infrastructure work, and dataset curation. How would you ensure evaluation infrastructure is not deprioritised in favour of features?

---

## References

### Source Code Management
- [GitHub](https://docs.github.com) — Industry-standard SCM; GitHub Actions provides native CI/CD tightly integrated with the repository.
- [GitLab](https://docs.gitlab.com) — Unified DevSecOps platform: SCM + built-in CI/CD pipelines + container registry + package registry in a single tool. 🔓 Self-hosted via GitLab CE.
- [Bitbucket](https://support.atlassian.com/bitbucket-cloud/) — Atlassian's SCM; integrates natively with Jira for ticket-to-commit traceability.
- [Gitea](https://docs.gitea.com) — 🔓 Lightweight self-hosted Git service; supports Gitea Actions (GitHub Actions-compatible syntax) for air-gapped environments.
- [Azure DevOps Repos](https://learn.microsoft.com/en-us/azure/devops/repos/) — Microsoft's enterprise SCM; integrates with Azure Pipelines and Azure Boards.

### Documentation
- [LangSmith Experiment Tracking](https://docs.smith.langchain.com) — Experiment management for LLM systems.
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) — Experiment tracking and comparison.
- [Weights & Biases](https://docs.wandb.ai) — Experiment tracking with LLM evaluation support.

### Books
- *Accelerate* — Forsgren, Humble & Kim (IT Revolution). DORA metrics applicable to AI SDLC.
- *Continuous Delivery* — Humble & Farley (Addison-Wesley). CD principles for AI artifact pipelines.
- *The Lean Startup* — Ries. Dual-track model parallels Build-Measure-Learn cycle.

---

---
[« Back to sdlc Index](index.md) | [🏠 Home](../../index.md)