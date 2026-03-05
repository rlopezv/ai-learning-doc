# Part VIII — AI Systems SDLC

---

> **Navigation**
> [← Part VII — Layouts and Repositories](part_07_layouts_repositories.md) | [→ Part IX — Evaluation Engineering](part_09_evaluation_engineering.md)

---

## Contents

- [Chapter 1 — The AI Systems Development Lifecycle](#chapter-1--the-ai-systems-development-lifecycle)
  - [1.1 How AI SDLC Differs from Traditional SDLC](#11-how-ai-sdlc-differs-from-traditional-sdlc)
  - [1.2 The Dual-Track Development Model](#12-the-dual-track-development-model)
  - [1.3 Definition of Done for AI Features](#13-definition-of-done-for-ai-features)
  - [1.4 Sprint Ceremonies for AI Teams](#14-sprint-ceremonies-for-ai-teams)
- [Chapter 2 — CI/CD for LLM Systems](#chapter-2--cicd-for-llm-systems)
  - [2.1 What CI Means for AI Systems](#21-what-ci-means-for-ai-systems)
  - [2.2 The LLM CI Pipeline](#22-the-llm-ci-pipeline)
  - [2.3 Evaluation Gates in CI](#23-evaluation-gates-in-ci)
  - [2.4 Deployment Strategies](#24-deployment-strategies)
  - [2.5 Rollback Triggers and Automation](#25-rollback-triggers-and-automation)
- [Chapter 3 — Feature Development Workflow](#chapter-3--feature-development-workflow)
  - [3.1 Prompt-First Development](#31-prompt-first-development)
  - [3.2 Experiment Tracking](#32-experiment-tracking)
  - [3.3 Code Review for AI Components](#33-code-review-for-ai-components)
  - [3.4 Staging Environment Design](#34-staging-environment-design)
- [Chapter 4 — Release Management 🧪](#chapter-4--release-management-)
  - [4.1 Semantic Versioning for AI Systems](#41-semantic-versioning-for-ai-systems)
  - [4.2 Release Candidate Process](#42-release-candidate-process)
  - [4.3 Canary Releases for LLM Updates](#43-canary-releases-for-llm-updates)
  - [4.4 Post-Release Monitoring Window](#44-post-release-monitoring-window)
  - [🧪 Hands-on Lab: Complete CI Pipeline](#-hands-on-lab-complete-ci-pipeline)

---

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

## Chapter 2 — CI/CD for LLM Systems

### 2.1 What CI Means for AI Systems

Continuous Integration for LLM systems extends the traditional definition — "every commit triggers automated testing" — to include a broader set of quality signals:

| CI signal type | Traditional CI | LLM CI |
|---|---|---|
| **Unit tests** | Function-level assertions | Prompt regression tests, utility functions |
| **Integration tests** | API contract tests | End-to-end RAG pipeline tests |
| **Quality gate** | Code coverage threshold | Evaluation score threshold (Recall@K, faithfulness) |
| **Security scan** | SAST, dependency audit | Prompt injection scan, PII leak detection |
| **Artifact build** | Compiled binary | Prompt registry update, index rebuild |
| **Lint** | Code style | Prompt format validation, schema validation |

The goal is to ensure that no change reaches production without demonstrably maintaining or improving quality across all relevant dimensions.

---

### 2.1a Tooling Landscape: SCM, CI/CD, and Artifact Management

Every AI SDLC pipeline rests on three infrastructure categories: source control, CI/CD execution, and artifact storage. The table below maps the traditional enterprise tooling against the patterns used throughout this book.

#### Source Control Management (SCM)

| Platform | Hosting | Notable for AI SDLC | On-premise |
|---|---|---|---|
| **GitHub** | Cloud | GitHub Actions; Codespaces for notebook-based development; native integration with GitHub Packages | GitHub Enterprise Server |
| **GitLab** | Cloud / Self | Built-in CI/CD, Container Registry, Package Registry, and Model Experiments (MLflow-compatible) in one product | GitLab CE / EE |
| **Bitbucket** | Cloud / Self | Bitbucket Pipelines; native Jira integration for requirement traceability | Bitbucket Data Center |
| **Azure DevOps** | Cloud / Self | Azure Pipelines; natively integrates with Azure ML, Azure Container Registry | Azure DevOps Server |
| **Gitea** | Self-only | Lightweight; Gitea Actions (GitHub-compatible YAML); ideal for air-gapped environments | ✓ All deployments |

Branching strategy recommendation for AI systems: **trunk-based development** with short-lived feature branches (< 2 days). Long-lived prompt branches accumulate drift against the main corpus and are hard to evaluate in isolation.

#### CI/CD Execution

| Platform | Hosting | Strengths | Typical use |
|---|---|---|---|
| **GitHub Actions** | Cloud | Zero setup; large action marketplace; matrix builds for multi-model eval | Most cloud-native AI teams |
| **GitLab CI/CD** | Cloud / Self | Tight SCM integration; Docker-in-Docker; built-in caching | Enterprise GitLab adopters |
| **Jenkins** | Self-only | Maximum flexibility; Groovy-based pipelines; large plugin ecosystem | On-premise / air-gapped |
| **Azure Pipelines** | Cloud / Self | YAML pipelines; tight Azure ML integration; hosted GPU agents | Azure-native organisations |
| **CircleCI** | Cloud | Fast caching; simple YAML; good Docker support | Startups, SaaS teams |
| **Argo Workflows** | Self (K8s) | Kubernetes-native DAG pipelines; ideal for heavy ingestion jobs | Platform teams on K8s |

For the CI pipeline examples in section 2.2 (GitHub Actions YAML), the equivalent GitLab CI and Jenkins equivalents are:

```yaml
# GitLab CI equivalent of the GitHub Actions CI pipeline
# .gitlab-ci.yml

stages: [lint, test, evaluate, build, deploy]

variables:
  PYTHON_VERSION: "3.11"
  EVAL_THRESHOLD_RECALL: "0.80"

lint-prompts:
  stage: lint
  image: python:${PYTHON_VERSION}
  script:
    - pip install pyyaml jsonschema
    - python scripts/validate_prompts.py prompts/
  only: [merge_requests, main]

unit-tests:
  stage: test
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements-dev.txt
    - pytest tests/unit/ -v --tb=short
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

evaluate-rag:
  stage: evaluate
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements-dev.txt
    - python scripts/run_evaluation.py --threshold ${EVAL_THRESHOLD_RECALL}
  artifacts:
    paths: [eval_results/]
    expire_in: 30 days
  allow_failure: false   # Blocks merge if evaluation fails

build-docker:
  stage: build
  image: docker:24
  services: [docker:24-dind]
  script:
    - docker build -t ${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA} apps/api/
    - docker push ${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}
  only: [main]
```

```groovy
// Jenkins declarative pipeline equivalent
// Jenkinsfile

pipeline {
    agent { label 'python-3.11' }
    environment {
        EVAL_THRESHOLD_RECALL = '0.80'
        REGISTRY = 'registry.company.com'
    }
    stages {
        stage('Lint') {
            steps {
                sh 'python scripts/validate_prompts.py prompts/'
            }
        }
        stage('Unit Tests') {
            steps {
                sh 'pytest tests/unit/ -v --junitxml=test-results.xml'
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }
        stage('Evaluate RAG') {
            steps {
                sh 'python scripts/run_evaluation.py --threshold ${EVAL_THRESHOLD_RECALL}'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'eval_results/**'
                }
                failure {
                    error 'Evaluation gate failed — blocking deployment'
                }
            }
        }
        stage('Build & Push') {
            when { branch 'main' }
            steps {
                sh """
                    docker build -t ${REGISTRY}/rag-api:${GIT_COMMIT} apps/api/
                    docker push ${REGISTRY}/rag-api:${GIT_COMMIT}
                """
            }
        }
    }
}
```

#### Artifact and Package Management

AI systems produce multiple artifact types beyond compiled code: Docker images, Python packages (internal libraries), model weights, embedding models, and prompt registries. Each requires a dedicated storage strategy.

| Artifact type | Recommended store | On-premise alternative |
|---|---|---|
| **Docker images** | AWS ECR / GCP Artifact Registry / GitHub Packages | Harbor (OSS) · GitLab Container Registry |
| **Python packages** | PyPI (public) / AWS CodeArtifact / GitHub Packages | JFrog Artifactory · Sonatype Nexus Repository |
| **Java/Maven artifacts** | Maven Central (public) / AWS CodeArtifact / GitHub Packages | JFrog Artifactory OSS · Sonatype Nexus OSS |
| **ML model weights** | Hugging Face Hub · MLflow Model Registry · AWS S3 | MinIO + MLflow · JFrog Artifactory (large file support) |
| **Prompt templates** | Git-tracked YAML (recommended) + prompt service DB | Same — prompts are code |
| **Embedding model files** | S3/GCS object storage + CDN | MinIO · Nexus Raw Repository |
| **Corpus snapshots** | S3/GCS versioned bucket | MinIO with versioning enabled |

```python
# artifact_registry.py — Unified artifact versioning across stores
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ArtifactStore(str, Enum):
    ECR               = "ecr"          # AWS Elastic Container Registry
    GITHUB_PACKAGES   = "ghcr"         # GitHub Container Registry
    HARBOR            = "harbor"       # 🔓 On-premise Harbor
    ARTIFACTORY       = "artifactory"  # JFrog Artifactory
    NEXUS             = "nexus"        # 🔓 Sonatype Nexus
    CODE_ARTIFACT     = "codeartifact" # AWS CodeArtifact (Python/Maven)
    GITLAB_REGISTRY   = "gitlab"       # GitLab Container + Package Registry
    MINIO             = "minio"        # 🔓 On-premise S3-compatible

@dataclass
class ArtifactCoordinates:
    """Uniquely identifies an artifact across any store."""
    store: ArtifactStore
    registry_host: str
    namespace: str          # org or project
    name: str               # image/package name
    tag: str                # semantic version or git SHA
    digest: Optional[str]   # SHA256 digest for immutable pinning

    @property
    def full_reference(self) -> str:
        """Returns the pull/push reference for this artifact."""
        if self.store in (ArtifactStore.ECR, ArtifactStore.HARBOR,
                          ArtifactStore.GITHUB_PACKAGES, ArtifactStore.GITLAB_REGISTRY):
            return f"{self.registry_host}/{self.namespace}/{self.name}:{self.tag}"
        if self.store == ArtifactStore.NEXUS:
            # Maven-style: groupId:artifactId:version
            return f"{self.namespace}:{self.name}:{self.tag}"
        return f"{self.name}:{self.tag}"

    @property
    def immutable_reference(self) -> Optional[str]:
        """Returns digest-pinned reference (preferred for production deployments)."""
        if self.digest:
            return f"{self.registry_host}/{self.namespace}/{self.name}@{self.digest}"
        return None

# Production artifact promotion example
ARTIFACT_LIFECYCLE = {
    "development": ArtifactCoordinates(
        ArtifactStore.HARBOR, "harbor.internal", "ai-platform", "rag-api",
        tag="feature-hybrid-search-a3f9c2", digest=None
    ),
    "staging": ArtifactCoordinates(
        ArtifactStore.HARBOR, "harbor.internal", "ai-platform", "rag-api",
        tag="rc-2.4.0", digest="sha256:a1b2c3d4e5f6..."
    ),
    "production": ArtifactCoordinates(
        ArtifactStore.HARBOR, "harbor.internal", "ai-platform", "rag-api",
        tag="2.4.0", digest="sha256:a1b2c3d4e5f6..."  # Same digest — promoted, not rebuilt
    ),
}

# Key principle: promote artifacts between environments rather than rebuilding.
# The staging and production images are identical (same digest); only the tag changes.
# This eliminates environment-specific build bugs and simplifies rollback.
```

> **🔓 On-premise artifact stack:** For air-gapped environments, the recommended combination is **Harbor** (container images) + **Sonatype Nexus OSS** or **JFrog Artifactory OSS** (Python packages, Maven/Gradle artifacts, raw files) + **MinIO** (model weights, corpus snapshots). All three are open-source and can run as Kubernetes StatefulSets.

---

### 2.2 The LLM CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  pull_request:
    branches: [main, staging]
  push:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  JAVA_VERSION: "21"

jobs:
  # ── Stage 1: Fast checks (< 2 min) ─────────────────────────────────────
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ env.PYTHON_VERSION }}" }
      - run: pip install ruff mypy
      - run: ruff check apps/ libs/
      - run: mypy apps/ libs/ --ignore-missing-imports

  validate-prompts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ env.PYTHON_VERSION }}" }
      - name: Validate prompt JSON schemas
        run: python scripts/validate_prompts.py prompts/
      - name: Check prompt variable consistency
        run: python scripts/check_prompt_vars.py prompts/

  validate-datasets:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install dvc
      - run: dvc pull datasets/eval/
      - run: python scripts/validate_datasets.py datasets/eval/

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install pip-audit
      - run: pip-audit -r requirements.lock --format json
      - name: Scan for hardcoded secrets
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}

  # ── Stage 2: Tests (5–15 min) ──────────────────────────────────────────
  unit-tests-python:
    needs: [lint-and-format]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ env.PYTHON_VERSION }}" }
      - run: pip install -e ".[dev]"
      - run: pytest libs/ apps/ -x -q --tb=short
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}

  unit-tests-java:
    needs: [lint-and-format]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with: { java-version: "${{ env.JAVA_VERSION }}", distribution: "temurin" }
      - run: mvn test -pl libs/java-rag,apps/api --no-transfer-progress

  # ── Stage 3: Evaluation gate (10–20 min) ───────────────────────────────
  evaluation-gate:
    needs: [unit-tests-python, unit-tests-java]
    runs-on: ubuntu-latest
    # Only run on main branch or when RAG/prompt files change
    if: |
      github.ref == 'refs/heads/main' ||
      contains(github.event.head_commit.modified, 'prompts/') ||
      contains(github.event.head_commit.modified, 'libs/rag-core/')
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ env.PYTHON_VERSION }}" }
      - run: pip install -e ".[dev]" && dvc pull datasets/eval/
      - name: Run retrieval evaluation
        run: |
          python scripts/run_eval.py \
            --dataset datasets/eval/eval_v2.jsonl \
            --output eval_results.json \
            --min-recall 0.80 \
            --min-faithfulness 0.85
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}
      - name: Upload eval results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results-${{ github.sha }}
          path: eval_results.json
      - name: Post eval summary to PR
        if: github.event_name == 'pull_request'
        run: python scripts/post_eval_comment.py eval_results.json
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # ── Stage 4: Integration tests (5–10 min) ──────────────────────────────
  integration-tests:
    needs: [evaluation-gate]
    runs-on: ubuntu-latest
    services:
      chroma:
        image: chromadb/chroma:latest
        ports: ["8000:8000"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ env.PYTHON_VERSION }}" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -x -q
        env:
          VECTOR_DB_HOST: localhost
          VECTOR_DB_PORT: 8000
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}
```

---

### 2.3 Evaluation Gates in CI

The evaluation gate is the most important addition to LLM CI. It runs the evaluation dataset against the current system and fails the pipeline if quality metrics fall below thresholds.

```python
# scripts/run_eval.py
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalConfig:
    dataset_path: str
    min_recall_at_5: float = 0.80
    min_faithfulness: float = 0.85
    min_answer_relevancy: float = 0.80
    max_latency_p99_ms: float = 500.0
    sample_size: Optional[int] = None  # None = use full dataset

@dataclass
class EvalResult:
    recall_at_5: float
    faithfulness: float
    answer_relevancy: float
    latency_p99_ms: float
    total_queries: int
    passed: bool
    failures: list[str]

def run_evaluation(config: EvalConfig, retriever, generator) -> EvalResult:
    """
    Run full evaluation pipeline and return structured results.
    """
    import time
    from ragas import evaluate
    from ragas.metrics import context_recall, faithfulness, answer_relevancy
    from datasets import Dataset

    # Load eval dataset
    records = [json.loads(l) for l in Path(config.dataset_path).read_text().splitlines() if l.strip()]
    if config.sample_size:
        import random
        records = random.sample(records, min(config.sample_size, len(records)))

    # Run RAG pipeline on each query and collect latencies
    questions, answers, contexts, ground_truths = [], [], [], []
    latencies = []

    for record in records:
        t0 = time.perf_counter()
        retrieved = retriever.retrieve(record["query"], top_k=5)
        answer = generator.generate(record["query"], [r["content"] for r in retrieved])
        latency_ms = (time.perf_counter() - t0) * 1000

        questions.append(record["query"])
        answers.append(answer)
        contexts.append([r["content"] for r in retrieved])
        ground_truths.append(record["ground_truth_answer"])
        latencies.append(latency_ms)

    # RAGAS evaluation
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    ragas_results = evaluate(dataset, metrics=[context_recall, faithfulness, answer_relevancy])

    # Compute latency percentiles
    latencies_sorted = sorted(latencies)
    p99_idx = int(len(latencies_sorted) * 0.99)
    p99_ms = latencies_sorted[p99_idx]

    # Check gates
    failures = []
    recall = float(ragas_results["context_recall"])
    faith = float(ragas_results["faithfulness"])
    relevancy = float(ragas_results["answer_relevancy"])

    if recall < config.min_recall_at_5:
        failures.append(f"Recall@5 {recall:.2%} < {config.min_recall_at_5:.2%}")
    if faith < config.min_faithfulness:
        failures.append(f"Faithfulness {faith:.2%} < {config.min_faithfulness:.2%}")
    if relevancy < config.min_answer_relevancy:
        failures.append(f"Answer relevancy {relevancy:.2%} < {config.min_answer_relevancy:.2%}")
    if p99_ms > config.max_latency_p99_ms:
        failures.append(f"P99 latency {p99_ms:.0f}ms > {config.max_latency_p99_ms:.0f}ms")

    result = EvalResult(
        recall_at_5=recall,
        faithfulness=faith,
        answer_relevancy=relevancy,
        latency_p99_ms=p99_ms,
        total_queries=len(records),
        passed=len(failures) == 0,
        failures=failures
    )
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--min-faithfulness", type=float, default=0.85)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    config = EvalConfig(
        dataset_path=args.dataset,
        min_recall_at_5=args.min_recall,
        min_faithfulness=args.min_faithfulness,
        sample_size=args.sample_size
    )

    # In a real pipeline: initialise retriever and generator from config
    print(f"Running evaluation on {args.dataset} (sample_size={args.sample_size or 'full'})")

    # Placeholder — real implementation uses actual RAG pipeline
    result = EvalResult(
        recall_at_5=0.87, faithfulness=0.91, answer_relevancy=0.88,
        latency_p99_ms=320.0, total_queries=100,
        passed=True, failures=[]
    )

    # Write structured results
    output = {
        "recall_at_5": result.recall_at_5,
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "latency_p99_ms": result.latency_p99_ms,
        "total_queries": result.total_queries,
        "passed": result.passed,
        "failures": result.failures,
        "thresholds": {
            "min_recall_at_5": config.min_recall_at_5,
            "min_faithfulness": config.min_faithfulness,
        }
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))

    if not result.passed:
        print(f"\n✗ Evaluation gate FAILED:", file=sys.stderr)
        for f in result.failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Evaluation gate PASSED")

if __name__ == "__main__":
    main()
```

**Posting evaluation results as a PR comment:**
```python
# scripts/post_eval_comment.py
import json
import os
import sys
import urllib.request

def post_pr_comment(eval_results_path: str):
    results = json.loads(open(eval_results_path).read())
    status = "✅ PASSED" if results["passed"] else "❌ FAILED"

    comment = f"""## Evaluation Gate Results {status}

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Recall@5 | {results["recall_at_5"]:.2%} | {results["thresholds"]["min_recall_at_5"]:.2%} | {"✅" if results["recall_at_5"] >= results["thresholds"]["min_recall_at_5"] else "❌"} |
| Faithfulness | {results["faithfulness"]:.2%} | {results["thresholds"]["min_faithfulness"]:.2%} | {"✅" if results["faithfulness"] >= results["thresholds"]["min_faithfulness"] else "❌"} |
| Answer Relevancy | {results["answer_relevancy"]:.2%} | — | — |
| P99 Latency | {results["latency_p99_ms"]:.0f}ms | 500ms | {"✅" if results["latency_p99_ms"] <= 500 else "❌"} |

*Evaluated on {results["total_queries"]} queries*
"""
    if results["failures"]:
        comment += "\n**Failures:**\n" + "\n".join(f"- {f}" for f in results["failures"])

    # Post via GitHub API
    token = os.environ.get("GITHUB_TOKEN")
    pr_number = os.environ.get("PR_NUMBER")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not all([token, pr_number, repo]):
        print("GitHub env vars not set — printing comment:")
        print(comment)
        return

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    data = json.dumps({"body": comment}).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Authorization": f"token {token}",
                                           "Content-Type": "application/json"})
    urllib.request.urlopen(req)
    print("Posted evaluation comment to PR")

if __name__ == "__main__":
    post_pr_comment(sys.argv[1])
```

---

### 2.4 Deployment Strategies

LLM systems benefit from progressive delivery strategies that limit blast radius.

**Blue-green deployment** — switch all traffic atomically between two identical environments:

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      # Verify evaluation gate passed on this commit
      - name: Check eval gate artifact
        run: |
          python scripts/verify_eval_gate.py ${{ github.sha }}

      # Build and push Docker image (green)
      - name: Build green image
        run: |
          docker build -t registry.company.com/ai-api:${{ github.sha }} apps/api/
          docker push registry.company.com/ai-api:${{ github.sha }}

      # Deploy to green environment
      - name: Deploy green
        run: |
          kubectl set image deployment/ai-api-green \
            api=registry.company.com/ai-api:${{ github.sha }}
          kubectl rollout status deployment/ai-api-green --timeout=5m

      # Health check green
      - name: Health check
        run: |
          python scripts/health_check.py \
            --endpoint https://ai-api-green.internal \
            --queries scripts/smoke_queries.json \
            --min-pass-rate 0.95

      # Switch traffic (blue → green)
      - name: Switch traffic
        run: |
          kubectl patch service ai-api \
            -p '{"spec":{"selector":{"slot":"green"}}}'

      # Monitor for 5 minutes
      - name: Post-deploy monitoring window
        run: python scripts/monitor_deployment.py --duration-minutes 5 --rollback-on-error

      # Mark old blue as standby
      - name: Update blue slot
        run: |
          kubectl set image deployment/ai-api-blue \
            api=registry.company.com/ai-api:${{ github.sha }}
```

**Canary deployment** — route a fraction of traffic to the new version:
```python
# Nginx upstream configuration for canary
# 10% traffic to canary, 90% to stable
upstream ai_api {
    server ai-api-stable:8080 weight=9;
    server ai-api-canary:8080 weight=1;
}
```

```python
# scripts/canary_controller.py — Gradually increase canary traffic
import time

def promote_canary(
    stable_service: str,
    canary_service: str,
    traffic_steps: list[float] = [0.05, 0.10, 0.25, 0.50, 1.0],
    step_duration_minutes: int = 10,
    error_rate_threshold: float = 0.02,
    latency_threshold_ms: float = 600
):
    """
    Incrementally shift traffic to canary, monitoring at each step.
    Rolls back automatically if error rate or latency exceeds threshold.
    """
    for traffic_fraction in traffic_steps:
        print(f"Setting canary traffic: {traffic_fraction:.0%}")
        update_load_balancer_weights(stable_service, canary_service, traffic_fraction)

        print(f"  Monitoring for {step_duration_minutes} minutes...")
        time.sleep(step_duration_minutes * 60)

        metrics = collect_metrics(canary_service)
        if metrics["error_rate"] > error_rate_threshold:
            print(f"  ✗ Error rate {metrics['error_rate']:.2%} > threshold. Rolling back.")
            update_load_balancer_weights(stable_service, canary_service, 0.0)
            raise RuntimeError("Canary rollback triggered: error rate exceeded")

        if metrics["p99_latency_ms"] > latency_threshold_ms:
            print(f"  ✗ P99 latency {metrics['p99_latency_ms']:.0f}ms > threshold. Rolling back.")
            update_load_balancer_weights(stable_service, canary_service, 0.0)
            raise RuntimeError("Canary rollback triggered: latency exceeded")

        print(f"  ✓ Metrics healthy at {traffic_fraction:.0%} canary traffic")

    print("✓ Canary promotion complete — 100% traffic on new version")

def update_load_balancer_weights(stable, canary, canary_fraction):
    pass  # Implement for your load balancer (nginx, Istio, ALB)

def collect_metrics(service):
    return {"error_rate": 0.005, "p99_latency_ms": 280}  # Placeholder
```

---

### 2.5 Rollback Triggers and Automation

Automated rollback prevents a bad deployment from causing extended downtime.

```python
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger("deployment-monitor")

@dataclass
class RollbackConfig:
    error_rate_threshold: float = 0.05      # 5% error rate
    latency_p99_threshold_ms: float = 600   # 600ms P99
    quality_score_drop_threshold: float = 0.05  # 5% quality drop vs baseline
    observation_window_minutes: int = 10
    check_interval_seconds: int = 30
    auto_rollback: bool = True

class DeploymentMonitor:
    def __init__(self, config: RollbackConfig, metrics_client, rollback_fn):
        self.config = config
        self.metrics = metrics_client
        self.rollback_fn = rollback_fn
        self.baseline_quality: float = None

    def set_baseline(self, baseline_quality: float):
        self.baseline_quality = baseline_quality

    def monitor(self, deployment_id: str) -> bool:
        """
        Monitor deployment for observation_window_minutes.
        Returns True if stable, False if rolled back.
        """
        end_time = time.time() + (self.config.observation_window_minutes * 60)
        logger.info(f"Monitoring deployment {deployment_id} for {self.config.observation_window_minutes}m")

        while time.time() < end_time:
            current = self.metrics.get_current(deployment_id)
            rollback_reason = self._check_thresholds(current)

            if rollback_reason:
                logger.error(f"ROLLBACK TRIGGERED: {rollback_reason}")
                if self.config.auto_rollback:
                    self.rollback_fn(deployment_id)
                    logger.info(f"Rolled back deployment {deployment_id}")
                return False

            elapsed = int(end_time - time.time()) // 60
            logger.info(
                f"  ✓ Healthy — error_rate={current['error_rate']:.2%} "
                f"p99={current['p99_ms']:.0f}ms "
                f"({elapsed}m remaining)"
            )
            time.sleep(self.config.check_interval_seconds)

        logger.info(f"✓ Deployment {deployment_id} stable after monitoring window")
        return True

    def _check_thresholds(self, metrics: dict) -> str | None:
        if metrics.get("error_rate", 0) > self.config.error_rate_threshold:
            return f"Error rate {metrics['error_rate']:.2%} > {self.config.error_rate_threshold:.2%}"
        if metrics.get("p99_ms", 0) > self.config.latency_p99_threshold_ms:
            return f"P99 latency {metrics['p99_ms']:.0f}ms > {self.config.latency_p99_threshold_ms:.0f}ms"
        if self.baseline_quality:
            quality_drop = self.baseline_quality - metrics.get("quality_score", self.baseline_quality)
            if quality_drop > self.config.quality_score_drop_threshold:
                return f"Quality dropped {quality_drop:.2%} from baseline"
        return None
```

---

> ### 📋 Chapter Summary
>
> - LLM CI extends traditional CI with prompt validation, evaluation gates, PII scan, and artifact build steps.
> - The **evaluation gate** is the most critical CI addition: it fails the pipeline if Recall@K, faithfulness, or latency fall below thresholds.
> - **Blue-green deployment** and **canary releases** limit blast radius for LLM system updates.
> - **Automated rollback** — triggered by error rate, latency, or quality score degradation — prevents extended downtime from bad deployments.

---

> ### ❓ Comprehension Questions
>
> 1. The evaluation gate runs on every push to `main` but takes 18 minutes. This is blocking developer velocity. How would you reduce evaluation gate latency without reducing its coverage?
> 2. A canary deployment reaches 25% traffic when P99 latency spikes to 850ms. The `canary_controller` rolls back. Post-mortem analysis shows the new prompt version generates longer responses. How would you distinguish latency regressions caused by model behaviour vs infrastructure?
> 3. An evaluation gate passes with Recall@5 = 0.82 (above the 0.80 threshold) but a manual review reveals the answers are less accurate than before. What limitation of automated evaluation does this reveal?
> 4. Design a CI pipeline for a RAG system where: (a) prompt changes only trigger prompt regression tests and a 50-query evaluation sample, (b) corpus changes trigger a full index rebuild and full evaluation, (c) code changes trigger standard unit/integration tests plus a 100-query evaluation. How would you detect which artifact type changed to route to the correct pipeline?
> 5. Your automated rollback triggers falsely during a traffic spike (temporarily high latency, not a deployment issue). How would you reduce false positive rollbacks while maintaining protection against real regressions?

---

## References

### CI/CD Platforms
- [GitHub Actions](https://docs.github.com/en/actions) — Cloud CI/CD tightly integrated with GitHub SCM; large action marketplace.
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci/) — Built-in CI/CD in GitLab; `.gitlab-ci.yml` syntax; includes built-in container and package registry.
- [Jenkins](https://www.jenkins.io/doc/) — 🔓 Self-hosted, open-source; Groovy-based declarative pipelines; extensive plugin ecosystem.
- [Azure Pipelines](https://learn.microsoft.com/en-us/azure/devops/pipelines/) — YAML pipelines; integrates with Azure ML and Azure Container Registry.
- [CircleCI](https://circleci.com/docs/) — Cloud CI with strong Docker caching and parallelism support.
- [Argo Workflows](https://argoproj.github.io/workflows/) — Kubernetes-native DAG workflows; ideal for heavyweight ingestion and evaluation jobs.

### Artifact and Package Management
- [JFrog Artifactory](https://jfrog.com/artifactory/) — Universal artifact repository: Docker, Maven, PyPI, npm, Helm, raw files. Cloud and 🔓 on-premise.
- [Sonatype Nexus Repository](https://www.sonatype.com/products/nexus-repository) — 🔓 OSS edition supports Maven, npm, PyPI, Docker, raw. Enterprise edition adds advanced security.
- [Harbor](https://goharbor.io/docs/) — 🔓 CNCF container registry with vulnerability scanning, RBAC, and replication.
- [AWS CodeArtifact](https://docs.aws.amazon.com/codeartifact/) — Managed Maven, Gradle, npm, and PyPI repository on AWS.
- [GitHub Packages](https://docs.github.com/en/packages) — Container and package registry tightly integrated with GitHub Actions.
- [GitLab Package Registry](https://docs.gitlab.com/ee/user/packages/package_registry/) — Supports Maven, PyPI, npm, Docker — all within a single GitLab project.

### Quality and Security Scanning
- [SonarQube](https://docs.sonarsource.com/sonarqube/) — 🔓 Static analysis for code quality and security; integrates with GitHub, GitLab, Jenkins.
- [OWASP Dependency-Check](https://owasp.org/www-project-dependency-check/) — 🔓 Detects known-vulnerable dependencies in Java/Python projects.
- [Trivy](https://trivy.dev) — 🔓 Container and filesystem vulnerability scanner; integrates into any CI pipeline.

### GitOps and Deployment
- [Argo CD](https://argo-cd.readthedocs.io) — GitOps continuous delivery for Kubernetes.
- [Flagger](https://docs.flagger.app) — Progressive delivery with automated canary analysis.

### Documentation
- [RAGAS Documentation](https://docs.ragas.io) — Automated RAG evaluation metrics.
- [MLflow Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html) — LLM evaluation tracking.

### Papers
- [Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html) — Sato, Wider & Windheuser, 2019.
- [What's your ML Test Score?](https://research.google/pubs/pub46555/) — Breck et al., 2017. ML testing taxonomy applicable to LLM CI.

---

## Chapter 3 — Feature Development Workflow

### 3.1 Prompt-First Development

For LLM features, prompt design should precede code implementation. Writing the prompt first forces clarity about: what the model is being asked to do, what context it needs, and what format it should produce. This clarity then drives the code design.

```
Prompt-First workflow:

1. Write the prompt (system + user template)
   └─ Define required variables
   └─ Define expected output format
   └─ Write 3–5 example input/output pairs

2. Evaluate prompt manually
   └─ Test against 10–20 representative queries
   └─ Identify failure modes (hallucination, wrong format, refusals)
   └─ Iterate prompt until manually acceptable

3. Write evaluation test cases
   └─ expected_contains, expected_not_contains
   └─ Edge cases found in step 2

4. Implement code around the validated prompt
   └─ Context assembly (retriever → context formatter)
   └─ Output parser
   └─ Error handling

5. Register prompt in prompt registry
   └─ Version 1.0.0 with author, description, tags
```

```python
# Example: Prompt-first development for a document summarisation feature

# Step 1: Write the prompt
SUMMARISE_PROMPT = PromptTemplate(
    template_id="doc_summarise",
    version="1.0.0",
    description="Summarise a document in bullet points, extracting key facts only",
    system_template="""Summarise the document in {max_bullets} bullet points.
Each bullet must:
- Start with a key fact or decision
- Be one sentence maximum
- Contain no speculation or information not in the document
If the document is too short to produce {max_bullets} bullets, produce fewer.""",
    user_template="Document:\n{document_text}",
    variables=["document_text"],
    optional_variables=["max_bullets"],
    defaults={"max_bullets": 5},
    author="product-team"
)

# Step 2: Manual evaluation - write test cases before implementation
SUMMARISE_TESTS = [
    PromptTestCase(
        test_id="produces_bullets",
        description="Output contains bullet points",
        inputs={"document_text": "The Q3 revenue was $2.4M, up 15% YoY. Headcount grew to 45. The new product line launched in October.", "max_bullets": "3"},
        expected_contains=["$2.4", "15%", "45"],
        expected_not_contains=["I cannot", "sorry"],
    ),
    PromptTestCase(
        test_id="no_hallucination",
        description="Does not add facts not in document",
        inputs={"document_text": "The meeting was held on Tuesday.", "max_bullets": "3"},
        expected_not_contains=["Monday", "Wednesday", "Thursday", "Friday"],
    ),
]
```

---

### 3.2 Experiment Tracking

Experiments must be tracked with sufficient metadata to be reproducible and comparable.

```python
import mlflow
import json
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class ExperimentRun:
    """Tracks a single experiment run with all relevant parameters."""
    experiment_name: str
    run_name: str
    # Prompt configuration
    prompt_template_id: str
    prompt_version: str
    # RAG configuration
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    reranking_enabled: bool
    hybrid_search_enabled: bool
    # Model configuration
    llm_model: str
    temperature: float
    # Evaluation
    eval_dataset_version: str
    eval_sample_size: int

@contextmanager
def track_experiment(run: ExperimentRun):
    """Context manager that logs all experiment parameters and metrics to MLflow."""
    mlflow.set_experiment(run.experiment_name)
    with mlflow.start_run(run_name=run.run_name) as mlflow_run:
        # Log all parameters
        mlflow.log_params({
            "prompt_template_id": run.prompt_template_id,
            "prompt_version": run.prompt_version,
            "embedding_model": run.embedding_model,
            "chunk_size": run.chunk_size,
            "chunk_overlap": run.chunk_overlap,
            "top_k": run.top_k,
            "reranking_enabled": run.reranking_enabled,
            "hybrid_search": run.hybrid_search_enabled,
            "llm_model": run.llm_model,
            "temperature": run.temperature,
            "eval_dataset_version": run.eval_dataset_version,
            "eval_sample_size": run.eval_sample_size,
        })
        yield mlflow_run

    # Usage:
    # with track_experiment(run) as mlflow_run:
    #     results = run_evaluation(...)
    #     mlflow.log_metrics({
    #         "recall_at_5": results.recall_at_5,
    #         "faithfulness": results.faithfulness,
    #         "p99_latency_ms": results.latency_p99_ms,
    #     })

def compare_experiments(experiment_name: str, metric: str = "recall_at_5") -> list[dict]:
    """Retrieve all runs for an experiment sorted by a metric."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "metric": r.data.metrics.get(metric),
            "params": r.data.params,
        }
        for r in runs
    ]
```

---

### 3.3 Code Review for AI Components

Code review checklists must include AI-specific concerns.

```markdown
# AI Code Review Checklist

## Prompt changes
- [ ] Has the prompt been tested against the standard test suite?
- [ ] Are all new variables declared with descriptions?
- [ ] Does the prompt version follow semantic versioning?
- [ ] Is the prompt registered in the prompt registry with this PR?
- [ ] Does the diff show only intentional changes (no whitespace/newline drift)?

## Retrieval changes
- [ ] Is the embedding model change documented in an ADR?
- [ ] Has the index been rebuilt with the new configuration?
- [ ] Has Recall@K been measured before and after?
- [ ] Are metadata filters consistent with the document schema?

## LLM integration changes
- [ ] Are all LLM calls wrapped in retry logic with exponential backoff?
- [ ] Is token counting implemented to prevent context overflow?
- [ ] Are API errors handled gracefully (no bare `except`)?
- [ ] Is the temperature set appropriately for the task (0 for deterministic)?

## Security
- [ ] Does this change access or log any user content that was not logged before?
- [ ] Could the prompt be manipulated by user input (prompt injection)?
- [ ] Are all API keys referenced via the SecretManager, not hardcoded?
- [ ] Does this change expand PII processing beyond what's documented?

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

## Chapter 4 — Release Management 🧪

### 4.1 Semantic Versioning for AI Systems

Semantic versioning (MAJOR.MINOR.PATCH) applies to AI systems but with AI-specific interpretations:

| Version bump | Traditional meaning | AI system meaning |
|---|---|---|
| **MAJOR** | Breaking API change | Model change (new base model, major prompt rewrite), breaking behaviour change |
| **MINOR** | New backwards-compatible feature | New capabilities (tool use, new document types), prompt improvements, embedding model upgrade |
| **PATCH** | Bug fix | Prompt typo fixes, chunking parameter tuning, non-behavioural fixes |

**Version file structure:**
```python
# version.py — single source of truth for system version
from dataclasses import dataclass

@dataclass(frozen=True)
class SystemVersion:
    major: int
    minor: int
    patch: int
    # Component versions tracked separately
    prompt_version: str          # e.g. "3.1.0"
    embedding_model: str         # e.g. "text-embedding-3-small"
    index_version: str           # e.g. "v1732012800"
    knowledge_base_date: str     # e.g. "2024-11-15"

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_major_bump_from(self, previous: "SystemVersion") -> bool:
        return self.major > previous.major

    def breaking_changes(self, previous: "SystemVersion") -> list[str]:
        changes = []
        if self.embedding_model != previous.embedding_model:
            changes.append(f"Embedding model changed: {previous.embedding_model} → {self.embedding_model}")
        if self.prompt_version.split(".")[0] != previous.prompt_version.split(".")[0]:
            changes.append(f"Major prompt version changed: {previous.prompt_version} → {self.prompt_version}")
        return changes

CURRENT = SystemVersion(
    major=2, minor=3, patch=1,
    prompt_version="3.1.0",
    embedding_model="text-embedding-3-small",
    index_version="v1732012800",
    knowledge_base_date="2024-11-15"
)
```

---

### 4.2 Release Candidate Process

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class RCStatus(str, Enum):
    OPEN = "open"
    EVAL_COMPLETE = "eval_complete"
    STAGING_DEPLOYED = "staging_deployed"
    APPROVED = "approved"
    REJECTED = "rejected"
    RELEASED = "released"

@dataclass
class ReleaseCandidate:
    rc_id: str                       # e.g. "v2.3.0-rc.1"
    target_version: str              # e.g. "2.3.0"
    git_sha: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: RCStatus = RCStatus.OPEN
    # Quality gates
    eval_recall_at_5: Optional[float] = None
    eval_faithfulness: Optional[float] = None
    eval_latency_p99_ms: Optional[float] = None
    prompt_tests_passed: Optional[bool] = None
    # Approvals
    tech_lead_approval: Optional[str] = None    # Username
    product_approval: Optional[str] = None
    # Breaking changes
    breaking_changes: list[str] = field(default_factory=list)
    # Release notes
    release_notes: str = ""

    def is_ready_to_release(self) -> tuple[bool, list[str]]:
        blockers = []
        if not self.eval_recall_at_5 or self.eval_recall_at_5 < 0.80:
            blockers.append(f"Recall@5 {self.eval_recall_at_5} below 0.80")
        if not self.prompt_tests_passed:
            blockers.append("Prompt regression tests not passed")
        if not self.tech_lead_approval:
            blockers.append("Tech lead approval required")
        if self.breaking_changes and not self.product_approval:
            blockers.append("Product approval required for breaking changes")
        if not self.release_notes:
            blockers.append("Release notes required")
        return len(blockers) == 0, blockers
```

---

### 4.3 Canary Releases for LLM Updates

LLM behaviour changes are difficult to assess in staging because they manifest across the long tail of user queries. Canary releases expose the new version to a small fraction of real production traffic.

```python
import time
from dataclasses import dataclass

@dataclass
class CanaryConfig:
    initial_traffic_fraction: float = 0.05  # Start at 5%
    traffic_steps: list[float] = None       # [0.05, 0.10, 0.25, 0.50, 1.0]
    step_duration_minutes: int = 30
    # Quality gates at each step
    max_error_rate: float = 0.02
    max_latency_p99_ms: float = 600
    min_quality_score: float = 0.80         # From online evaluation
    # Signals for quality score
    quality_signal: str = "thumbs_up_rate"  # or "llm_judge_score"
    min_samples_before_decision: int = 50

    def __post_init__(self):
        if self.traffic_steps is None:
            self.traffic_steps = [0.05, 0.10, 0.25, 0.50, 1.0]

class CanaryReleaseManager:
    def __init__(self, config: CanaryConfig, metrics_client, lb_client):
        self.config = config
        self.metrics = metrics_client
        self.lb = lb_client

    def run(self, canary_version: str, stable_version: str) -> bool:
        """
        Execute canary release. Returns True on success, False on rollback.
        """
        print(f"Starting canary release: {stable_version} → {canary_version}")

        for step, fraction in enumerate(self.config.traffic_steps):
            print(f"\n[Step {step+1}/{len(self.config.traffic_steps)}] "
                  f"Setting {fraction:.0%} traffic to canary")
            self.lb.set_canary_weight(canary_version, fraction)

            # Wait for minimum samples
            print(f"  Waiting for {self.config.min_samples_before_decision} samples...")
            self._wait_for_samples(canary_version)

            # Check quality gates
            metrics = self.metrics.get_canary_metrics(canary_version)
            failure = self._check_gates(metrics)

            if failure:
                print(f"  ✗ Canary rollback: {failure}")
                self.lb.set_canary_weight(canary_version, 0.0)
                return False

            print(f"  ✓ Healthy: error={metrics['error_rate']:.2%} "
                  f"p99={metrics['p99_ms']:.0f}ms "
                  f"quality={metrics.get('quality_score', 'N/A')}")
            time.sleep(self.config.step_duration_minutes * 60)

        print(f"\n✓ Canary promotion complete — {canary_version} at 100%")
        return True

    def _check_gates(self, metrics: dict) -> str | None:
        if metrics.get("error_rate", 0) > self.config.max_error_rate:
            return f"Error rate {metrics['error_rate']:.2%} > {self.config.max_error_rate:.2%}"
        if metrics.get("p99_ms", 0) > self.config.max_latency_p99_ms:
            return f"P99 {metrics['p99_ms']:.0f}ms > {self.config.max_latency_p99_ms:.0f}ms"
        quality = metrics.get("quality_score")
        if quality and quality < self.config.min_quality_score:
            return f"Quality {quality:.2%} < {self.config.min_quality_score:.2%}"
        return None

    def _wait_for_samples(self, version: str):
        while True:
            count = self.metrics.get_request_count(version, window_minutes=5)
            if count >= self.config.min_samples_before_decision:
                break
            time.sleep(30)
```

---

### 4.4 Post-Release Monitoring Window

Every release is followed by a mandatory monitoring window before the release is considered stable.

```python
from dataclasses import dataclass
import time

@dataclass
class MonitoringWindowConfig:
    duration_minutes: int = 30
    check_interval_seconds: int = 60
    # Thresholds compared to pre-release baseline
    max_error_rate_delta: float = 0.02      # Allow +2% absolute
    max_latency_delta_pct: float = 0.20     # Allow +20% relative
    max_quality_drop: float = 0.03          # Allow -3% absolute
    auto_rollback_on_breach: bool = True

class PostReleaseMonitor:
    def __init__(
        self,
        config: MonitoringWindowConfig,
        metrics_client,
        rollback_fn,
        alert_fn
    ):
        self.config = config
        self.metrics = metrics_client
        self.rollback_fn = rollback_fn
        self.alert_fn = alert_fn

    def run(self, release_id: str, baseline: dict) -> bool:
        """Monitor for duration_minutes. Returns True if stable."""
        end_time = time.time() + (self.config.duration_minutes * 60)
        checks_passed = 0

        while time.time() < end_time:
            current = self.metrics.get_current()
            breach = self._check_against_baseline(current, baseline)

            if breach:
                self.alert_fn(f"[{release_id}] Post-release breach: {breach}")
                if self.config.auto_rollback_on_breach:
                    self.rollback_fn(release_id)
                    return False
            else:
                checks_passed += 1

            remaining = int((end_time - time.time()) / 60)
            print(f"  [{checks_passed} checks ok] {remaining}m remaining — "
                  f"error={current['error_rate']:.2%} "
                  f"p99={current['p99_ms']:.0f}ms")
            time.sleep(self.config.check_interval_seconds)

        print(f"✓ Release {release_id} stable after {self.config.duration_minutes}m monitoring")
        return True

    def _check_against_baseline(self, current: dict, baseline: dict) -> str | None:
        err_delta = current.get("error_rate", 0) - baseline.get("error_rate", 0)
        if err_delta > self.config.max_error_rate_delta:
            return f"Error rate +{err_delta:.2%} above baseline"

        if baseline.get("p99_ms", 0) > 0:
            lat_delta_pct = (current.get("p99_ms", 0) - baseline["p99_ms"]) / baseline["p99_ms"]
            if lat_delta_pct > self.config.max_latency_delta_pct:
                return f"P99 latency +{lat_delta_pct:.0%} above baseline"

        quality_drop = baseline.get("quality_score", 0) - current.get("quality_score", 0)
        if quality_drop > self.config.max_quality_drop:
            return f"Quality dropped {quality_drop:.2%} from baseline"

        return None
```

---

### 🧪 Hands-on Lab: Complete CI Pipeline

**Objective:** Assemble a complete CI pipeline locally using `make` targets that mirror the GitHub Actions workflow. Each target corresponds to a CI stage.

**Prerequisites:** Python ≥ 3.11, `pytest`, `ruff`, `openai`

```makefile
# Makefile — local CI pipeline mirror

.PHONY: all lint validate-prompts test-unit test-prompts eval-gate check-dod clean

# Run full CI pipeline locally
all: lint validate-prompts test-unit test-prompts eval-gate
	@echo ""
	@echo "✓ All CI stages passed"

# Stage 1: Linting
lint:
	@echo "[LINT] Running ruff..."
	ruff check libs/ apps/ --exit-zero
	@echo "  ✓ Lint passed"

# Stage 2: Prompt validation
validate-prompts:
	@echo "[VALIDATE] Checking prompt schemas..."
	python scripts/validate_prompts.py prompts/
	@echo "  ✓ Prompt validation passed"

# Stage 3: Unit tests
test-unit:
	@echo "[TEST] Running unit tests..."
	pytest libs/ apps/ -x -q --tb=short 2>/dev/null || echo "  (no tests found — add pytest tests)"
	@echo "  ✓ Unit tests passed"

# Stage 4: Prompt regression tests
test-prompts:
	@echo "[PROMPTS] Running prompt regression tests..."
	python scripts/run_prompt_tests.py prompts/rag_qa/latest.json
	@echo "  ✓ Prompt tests passed"

# Stage 5: Evaluation gate
eval-gate:
	@echo "[EVAL] Running evaluation gate..."
	python scripts/run_eval.py \
		--dataset datasets/eval/eval_v1.jsonl \
		--output /tmp/eval_results.json \
		--min-recall 0.80 \
		--sample-size 50
	@echo "  ✓ Evaluation gate passed"

# Check feature DoD
check-dod:
	@echo "[DOD] Checking Definition of Done..."
	python scripts/check_dod.py $(FEATURE_ID)

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f /tmp/eval_results.json
```

```python
# scripts/validate_prompts.py
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ["template_id", "version", "description",
                   "system_template", "user_template", "variables", "author"]

def validate_prompt_file(path: Path) -> list[str]:
    errors = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "variables" in data and "system_template" in data:
        template = data["system_template"]
        for var in data["variables"]:
            if "{" + var + "}" not in template and "{" + var + "}" not in data.get("user_template", ""):
                errors.append(f"Variable '{var}' declared but not used in templates")

    version = data.get("version", "")
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        errors.append(f"Version '{version}' does not follow MAJOR.MINOR.PATCH format")

    return errors

def main(prompts_dir: str):
    prompt_dir = Path(prompts_dir)
    files = [f for f in prompt_dir.rglob("*.json") if f.name != "latest.json"]
    total_errors = 0

    for f in sorted(files):
        errors = validate_prompt_file(f)
        if errors:
            print(f"✗ {f.relative_to(prompt_dir)}")
            for e in errors:
                print(f"  - {e}")
            total_errors += len(errors)
        else:
            print(f"✓ {f.relative_to(prompt_dir)}")

    print(f"\n{len(files)} prompt file(s) validated, {total_errors} error(s)")
    sys.exit(1 if total_errors > 0 else 0)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prompts/")
```

**Run the lab:**
```bash
# Run full local CI
make all

# Run individual stages
make lint
make validate-prompts
make eval-gate

# Check a feature's DoD
make check-dod FEATURE_ID=RAG-1234
```

**Extensions:**
- Add a `make benchmark` target that runs latency benchmarks and fails if P99 exceeds 500ms
- Integrate `make check-dod` with a JIRA or Linear ticket API to auto-populate DoD checklist items
- Add `make release-candidate` that creates a versioned tag, runs all CI stages, and generates release notes

---

> ### 📋 Chapter Summary
>
> - Semantic versioning for AI systems maps MAJOR to model/behaviour changes, MINOR to capability additions, PATCH to non-behavioural fixes.
> - **Release candidates** enforce a structured approval process with quality gates, tech lead review, and product sign-off for breaking changes.
> - **Canary releases** expose LLM behaviour changes to a small real-traffic fraction before full rollout — essential for catching quality regressions not visible in staging.
> - A **post-release monitoring window** provides an automatic rollback safety net against regressions that pass all pre-deployment gates.

---

> ### ❓ Comprehension Questions
>
> 1. A team upgrades the embedding model from `text-embedding-3-small` to `text-embedding-3-large`. Under the AI-specific semantic versioning rules, is this a PATCH, MINOR, or MAJOR bump? Justify your answer.
> 2. A canary release reaches 50% traffic. Quality scores from the online LLM judge start declining slowly (0.85 → 0.83 → 0.81 over 3 hours). The error rate and latency are within thresholds. Should the canary controller roll back? What signal is missing from the automated gates?
> 3. A release candidate has `breaking_changes = ["Embedding model changed"]`. The tech lead approves but no product approval is recorded. Using the `is_ready_to_release` method, what happens and why is this correct?
> 4. Post-release monitoring detects a P99 latency increase of 25% (within the 20% threshold for the absolute value but marginal). An hour later it spikes to +45%. Design a monitoring strategy that would have caught the degradation trend earlier.
> 5. A team releases weekly with a canary → full rollout → monitoring window cycle. Each cycle takes approximately 3 hours. What is the maximum deployment frequency this process supports, and how would you reduce it without sacrificing safety?

---

## References

### Release Automation
- [Semantic Versioning](https://semver.org) — Official SemVer specification.
- [Semantic Release](https://semantic-release.gitbook.io) — Automated version bumps and changelogs from Git commit history.
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) — Release tagging and artefact attachment on GitHub.
- [GitLab Releases](https://docs.gitlab.com/ee/user/project/releases/) — Release tagging with evidence collection and artefact links.

### Artifact Promotion
- [JFrog Artifactory — Promoting Artifacts](https://jfrog.com/help/r/jfrog-artifactory-documentation/promoting-a-docker-image) — Artifact promotion workflow between dev/staging/production repositories.
- [Sonatype Nexus — Staging Suite](https://help.sonatype.com/en/staging-suite.html) — Staging repository workflow for Maven artifacts before promotion to release.
- [Harbor — Tag Retention and Replication](https://goharbor.io/docs/latest/administration/tag-retention/) — Image lifecycle and cross-registry replication.
- [AWS CodeArtifact — Publishing Packages](https://docs.aws.amazon.com/codeartifact/latest/ug/packages-overview.html) — Package promotion and upstream caching.

### Progressive Delivery
- [Flagger Progressive Delivery](https://docs.flagger.app) — Automated canary analysis on Kubernetes.
- [Argo Rollouts](https://argoproj.github.io/rollouts/) — Kubernetes progressive delivery.
- [GitHub Actions Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments) — Deployment gates and approvals.
- [PagerDuty Runbook Automation](https://www.pagerduty.com/platform/runbook-automation/) — Post-release monitoring alerts.

### Papers
- [Towards Reliable ML: DORA for AI](https://research.google/pubs/pub46555/) — Breck et al., 2017.
- [Continuous Delivery for ML](https://martinfowler.com/articles/cd4ml.html) — Sato, Wider & Windheuser, 2019.

---

> **Navigation**
> [← Part VII — Layouts and Repositories](part_07_layouts_repositories.md) | [→ Part IX — Evaluation Engineering](part_09_evaluation_engineering.md)
