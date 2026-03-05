## Chapter 5 — Artifact Pipelines

### 5.1 End-to-End Artifact Pipeline

A complete artifact pipeline coordinates the build, test, validation, and promotion of all artifact types.

```python
from dataclasses import dataclass
from enum import Enum

class PipelineStage(str, Enum):
    BUILD = "build"
    TEST = "test"
    VALIDATE = "validate"
    PROMOTE = "promote"

@dataclass
class PipelineResult:
    stage: PipelineStage
    passed: bool
    message: str
    metrics: dict = None

class ArtifactPipeline:
    """
    Orchestrates: Build → Test → Validate → Promote
    Aborts on any failure.
    """
    def __init__(
        self,
        registry: PromptRegistry,
        runner: PromptTestRunner,
        index_manager: BlueGreenIndexManager
    ):
        self.registry = registry
        self.runner = runner
        self.index_manager = index_manager
        self.results: list[PipelineResult] = []

    def run(
        self,
        prompt: PromptTemplate,
        prompt_tests: list[PromptTestCase],
        documents: list[dict],
        eval_queries: list[dict],
        index_config: dict
    ) -> bool:
        print(f"\n{'─'*50}")
        print(f"Pipeline: {prompt.template_id} v{prompt.version}")
        print(f"{'─'*50}")

        # Stage 1: Build (register artifacts)
        self._stage(PipelineStage.BUILD, self._build(prompt))

        # Stage 2: Test (prompt regression tests)
        self._stage(PipelineStage.TEST, self._test_prompt(prompt, prompt_tests))
        if not self.results[-1].passed:
            print("✗ Aborted at TEST"); return False

        # Stage 3: Validate (index quality gate)
        self._stage(PipelineStage.VALIDATE, self._validate_index(documents, eval_queries, index_config))
        if not self.results[-1].passed:
            print("✗ Aborted at VALIDATE"); return False

        # Stage 4: Promote
        self._stage(PipelineStage.PROMOTE, PipelineResult(
            PipelineStage.PROMOTE, True,
            f"Promoted {prompt.template_id} v{prompt.version}"
        ))

        all_ok = all(r.passed for r in self.results)
        print(f"\n{"✓ PASSED" if all_ok else "✗ FAILED"}")
        return all_ok

    def _stage(self, stage: PipelineStage, result: PipelineResult):
        self.results.append(result)
        mark = "✓" if result.passed else "✗"
        print(f"[{stage.value.upper()}] {mark} {result.message}")

    def _build(self, prompt: PromptTemplate) -> PipelineResult:
        h = self.registry.save(prompt)
        return PipelineResult(PipelineStage.BUILD, True, f"Registered [{h}]")

    def _test_prompt(self, prompt, tests) -> PipelineResult:
        r = self.runner.run_suite(prompt, tests)
        return PipelineResult(
            PipelineStage.TEST, r["pass_rate"] == 1.0,
            f"Prompt tests {r['passed']}/{r['total']} passed",
            metrics={"pass_rate": r["pass_rate"]}
        )

    def _validate_index(self, documents, eval_queries, config) -> PipelineResult:
        version = self.index_manager.build_new_version(documents, config)
        passed = self.index_manager.validate(version, eval_queries, min_recall=0.80)
        return PipelineResult(
            PipelineStage.VALIDATE, passed,
            f"Index Recall@5={version.evaluation_score:.2%}",
            metrics={"recall_at_5": version.evaluation_score}
        )
```

---

### 5.2 Promotion Gates

```python
from dataclasses import dataclass

@dataclass
class PromotionGates:
    prompt_test_pass_rate: float = 1.0
    index_recall_at_5: float = 0.80
    max_latency_p99_ms: float = 500.0
    require_human_approval: bool = False

def evaluate_gates(
    test_results: dict,
    index_metrics: dict,
    latency_metrics: dict,
    gates: PromotionGates
) -> tuple[bool, list[str]]:
    failures = []
    if test_results.get("pass_rate", 0) < gates.prompt_test_pass_rate:
        failures.append(f"Prompt pass rate {test_results['pass_rate']:.0%} < required {gates.prompt_test_pass_rate:.0%}")
    if index_metrics.get("recall_at_5", 0) < gates.index_recall_at_5:
        failures.append(f"Recall@5 {index_metrics['recall_at_5']:.2%} < required {gates.index_recall_at_5:.2%}")
    if latency_metrics.get("p99_ms", 0) > gates.max_latency_p99_ms:
        failures.append(f"P99 {latency_metrics['p99_ms']}ms > max {gates.max_latency_p99_ms}ms")
    return len(failures) == 0, failures
```

---

### 5.3 Artifact Rollback

```python
class RollbackManager:
    def __init__(self, registry: PromptRegistry, index_manager: BlueGreenIndexManager):
        self.registry = registry
        self.index_manager = index_manager

    def rollback_prompt(self, template_id: str) -> bool:
        versions = self.registry.list_versions(template_id)
        if len(versions) < 2:
            print(f"Cannot rollback: only {len(versions)} version(s) available")
            return False
        # Reload second-most-recent version as production
        previous = versions[-2]
        print(f"Rolling back {template_id} to v{previous}")
        return True

    def rollback_index(self, previous_collection: str) -> bool:
        self.index_manager.active_collection = previous_collection
        print(f"Index rolled back to {previous_collection}")
        return True
```

---

### 5.4 Java Pipeline Integration

**Java — Artifact promotion with [Spring AI](https://docs.spring.io/spring-ai/reference):**
```java
@Service
public class ArtifactPromotionService {

    private final PromptRepository promptRepository;
    private final IndexVersionRepository indexRepository;
    private final EvaluationService evaluationService;

    @Transactional
    public PromotionResult promote(String promptId, String version, String indexVersion) {
        List<String> failures = new ArrayList<>();

        // Gate 1: Prompt regression tests
        double passRate = evaluationService.runPromptTests(promptId, version).passRate();
        if (passRate < 1.0) failures.add("Prompt pass rate: " + passRate);

        // Gate 2: Index recall
        double recall = evaluationService.evaluateIndex(indexVersion).recallAt5();
        if (recall < 0.80) failures.add("Index recall: " + recall);

        if (!failures.isEmpty()) return PromotionResult.failed(failures);

        // Atomic promotion
        promptRepository.setProduction(promptId, version);
        indexRepository.setActive(indexVersion);

        log.info("Promoted prompt={} v={}, index={}", promptId, version, indexVersion);
        return PromotionResult.success();
    }
}
```

---

> ### 📋 Chapter Summary
>
> - An artifact pipeline coordinates build → test → validate → promote for all artifact types.
> - **Promotion gates** enforce configurable quality thresholds at each stage transition.
> - Every production deployment must have a **rollback path** — the previous version must remain accessible.
> - Java pipelines integrate naturally with Spring's `@Transactional` for atomic multi-artifact promotion.

---

> ### ❓ Comprehension Questions
>
> 1. A pipeline promotes a prompt and index together. The prompt passes but the index fails. Should the prompt be promoted alone? What are the risks?
> 2. Design a rollback strategy that completes in under 30 seconds for both prompt and index artifacts.
> 3. `require_human_approval=True` blocks automated deployment. In what production contexts is human sign-off a compliance requirement?
> 4. The pipeline detects Recall@5 dropped from 0.87 to 0.79 overnight. What automated diagnostics would you run before human investigation?
> 5. Compare artifact promotion pipelines in LLM systems to CI/CD in traditional software. What is structurally the same, and what requires fundamentally different tooling?

---

## References

### Documentation
- [MLflow Pipelines](https://mlflow.org/docs/latest/pipelines.html)
- [GitHub Actions for ML](https://docs.github.com/en/actions)
- [LangSmith Automated Evaluation](https://docs.smith.langchain.com/evaluation)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/) — Kubernetes-native pipeline orchestration.
- [Prefect Documentation](https://docs.prefect.io) — Python-native pipeline orchestration.

### Papers
- [Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html) — Sato, Wider & Windheuser, 2019.
- [Challenges in Deploying Machine Learning](https://arxiv.org/abs/2011.09926) — Paleyes et al., 2020.

---

> **Navigation**
> [← Part V — Dataset Engineering](../dataset_engineering/index.md) | [→ Part VII — Layouts and Repositories](../layouts_repositories/index.md)

---
[« Back to artifact_engineering Index](index.md) | [🏠 Home](../index.md)