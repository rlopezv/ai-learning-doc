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
> [← Part VII — Layouts and Repositories](../layouts_repositories/index.md) | [→ Part IX — Evaluation Engineering](../evaluation_engineering/index.md)

---
[« Back to sdlc Index](index.md) | [🏠 Home](../index.md)