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

---
[« Back to sdlc Index](index.md) | [🏠 Home](../index.md)