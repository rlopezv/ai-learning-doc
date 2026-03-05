# Part VI — Artifact Engineering

---

> **Navigation**
> [← Part V — Dataset Engineering](part_05_dataset_engineering.md) | [→ Part VII — Layouts and Repositories](part_07_layouts_repositories.md)

---

## Contents

- [Chapter 1 — What Is an Artifact in an LLM System](#chapter-1--what-is-an-artifact-in-an-llm-system)
- [Chapter 2 — Prompt Artifacts 🧪](#chapter-2--prompt-artifacts-)
- [Chapter 3 — Model Artifacts](#chapter-3--model-artifacts)
- [Chapter 4 — Index Artifacts](#chapter-4--index-artifacts)
- [Chapter 5 — Artifact Pipelines](#chapter-5--artifact-pipelines)

---

## Chapter 1 — What Is an Artifact in an LLM System

### 1.1 Artifacts Beyond Models

In traditional ML engineering, "artifact" typically refers to a trained model — a serialised set of weights. In LLM systems engineering, the concept is significantly broader. An LLM system produces and consumes multiple distinct artifacts, each changing independently, each requiring version control, and each capable of causing system regressions if mismanaged.

The critical insight is that **a prompt change is as consequential as a model weight change**. A poorly managed prompt update can silently degrade system quality just as a bad model update would — but it is far easier to make and far less likely to be tracked with the same rigour.

Artifact engineering applies consistent versioning, testing, promotion, and rollback discipline to every artifact a system depends on — not only models.

---

### 1.2 Artifact Taxonomy

| Artifact type | Examples | Change frequency | Risk if unversioned |
|---|---|---|---|
| **Prompt templates** | System prompt, RAG prompt, tool descriptions | Daily / weekly | Silent quality degradation |
| **Model weights** | Fine-tuned adapters (LoRA), base model versions | Per release | Regression, compliance breach |
| **Vector indexes** | Embedded knowledge base, semantic search index | Daily / per ingestion | Stale retrieval, inconsistency |
| **Evaluation datasets** | QA pairs, adversarial sets | Quarterly | Unreliable benchmarks |
| **Configuration** | Chunk size, top-K, embedding model name | Per deployment | Silent behaviour change |
| **Tool schemas** | Function definitions for tool-augmented LLMs | Per API change | Broken tool calls |
| **Pipeline DAGs** | Ingestion pipeline, evaluation pipeline | Per code release | Non-reproducible processing |

---

### 1.3 Artifact Lifecycle

Every artifact follows a defined promotion path:

```
Development → Staging → Production
     ↓              ↓          ↓
  Unit test    Integration   Canary
  Eval score   Eval gate     Full rollout
  Lint/format  A/B test      Monitoring
```

The promotion path enforces that artifacts only reach production after passing defined quality gates — evaluation score thresholds, regression checks, and where required, explicit human approval.

---

### 1.4 Artifact Registry Patterns

An artifact registry stores versioned artifacts with metadata, enabling promotion, rollback, and lineage queries.

```python
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ArtifactType(str, Enum):
    PROMPT = "prompt"
    MODEL = "model"
    INDEX = "index"
    DATASET = "dataset"
    CONFIG = "config"
    TOOL_SCHEMA = "tool_schema"

class ArtifactStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"

@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_type: ArtifactType
    version: str
    stage: ArtifactStage
    content_hash: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    promoted_at: Optional[str] = None
    created_by: str = "pipeline"
    parent_version: Optional[str] = None
    evaluation_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

class ArtifactRegistry:
    """
    File-based artifact registry.
    In production: back with a database or MLflow artifact store.
    """
    def __init__(self, registry_path: str = "./artifact_registry"):
        self.base = Path(registry_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base / "index.jsonl"

    def register(self, record: ArtifactRecord, content: bytes) -> str:
        artifact_dir = self.base / record.artifact_type.value / record.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{record.version}.bin").write_bytes(content)
        with open(self.index_file, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")
        return record.artifact_id

    def promote(self, artifact_id: str, version: str, target_stage: ArtifactStage):
        records = self._load_index()
        for r in records:
            if r["artifact_id"] == artifact_id and r["version"] == version:
                r["stage"] = target_stage.value
                r["promoted_at"] = datetime.utcnow().isoformat()
        self._save_index(records)

    def get_production(self, artifact_id: str) -> Optional[dict]:
        records = self._load_index()
        production = [
            r for r in records
            if r["artifact_id"] == artifact_id
            and r["stage"] == ArtifactStage.PRODUCTION.value
        ]
        return max(production, key=lambda r: r["created_at"]) if production else None

    def get_versions(self, artifact_id: str) -> list[dict]:
        return [r for r in self._load_index() if r["artifact_id"] == artifact_id]

    def _load_index(self) -> list[dict]:
        if not self.index_file.exists():
            return []
        with open(self.index_file) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _save_index(self, records: list[dict]):
        with open(self.index_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
```

---

> ### 📋 Chapter Summary
>
> - LLM systems depend on multiple artifact types beyond models: prompts, indexes, configurations, and tool schemas all require versioning.
> - A prompt change is as consequential as a model change — both can silently degrade production quality.
> - Every artifact follows a **promotion path** (development → staging → production) with quality gates at each transition.
> - An **artifact registry** stores versioned content with metadata enabling rollback, lineage queries, and promotion tracking.

---

> ### ❓ Comprehension Questions
>
> 1. A production RAG system degrades silently for two weeks. Investigation reveals a prompt template was updated without version control. What engineering controls would have caught this?
> 2. Explain why vector indexes should be treated as versioned artifacts with blue-green deployment rather than in-place updates.
> 3. An artifact registry records `parent_version`. What lineage queries does this field enable?
> 4. Configuration artifacts (chunk size, top-K, embedding model) change less often than prompts. Does this mean they require less rigorous versioning?
> 5. Design the promotion gates for a prompt artifact moving from staging to production. What must be true before promotion is allowed?

---

## References

### Documentation
- [MLflow Models](https://mlflow.org/docs/latest/models.html) — Model artifact packaging and registry.
- [Hugging Face Model Hub](https://huggingface.co/docs/hub/models-the-hub) — Model versioning and cards.
- [DVC Artifacts](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files) — Artifact tracking with DVC.
- [LangSmith](https://docs.smith.langchain.com) — Prompt versioning and evaluation.

### Papers
- [Hidden Technical Debt in ML Systems](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — Sculley et al., 2015.
- [Towards Reproducible ML](https://arxiv.org/abs/2012.04261) — Artifact versioning for reproducibility.

---

## Chapter 2 — Prompt Artifacts 🧪

### 2.1 Prompts as Versioned Code

A system prompt encodes the model's persona, task framing, output format, injected context placeholders, and safety constraints. Changes to any of these components can have non-obvious downstream effects:

- A changed output format instruction may break a downstream JSON parser
- A softened constraint may enable outputs previously prevented
- A new few-shot example may shift the model's style distribution across all queries

Treating prompts as versioned code — with the same review, testing, and rollback discipline applied to application code — prevents these failures from reaching production silently.

---

### 2.2 Prompt Template Engineering

Production prompts use template variables to inject dynamic content at runtime.

```python
from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class PromptTemplate:
    template_id: str
    version: str
    description: str
    system_template: str
    user_template: str
    variables: list[str]            # Required variables
    optional_variables: list[str]   # Optional with defaults
    defaults: dict                  # Default values
    examples: list[dict] = None     # Few-shot examples
    author: str = "unknown"
    tags: list[str] = None

    def render_system(self, **kwargs) -> str:
        self._validate_variables(kwargs)
        return self.system_template.format(**{**self.defaults, **kwargs})

    def render_user(self, **kwargs) -> str:
        return self.user_template.format(**{**self.defaults, **kwargs})

    def render_messages(self, **kwargs) -> list[dict]:
        messages = [{"role": "system", "content": self.render_system(**kwargs)}]
        if self.examples:
            for ex in self.examples:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": self.render_user(**kwargs)})
        return messages

    def _validate_variables(self, provided: dict):
        missing = [v for v in self.variables if v not in provided and v not in self.defaults]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "description": self.description,
            "system_template": self.system_template,
            "user_template": self.user_template,
            "variables": self.variables,
            "optional_variables": self.optional_variables,
            "defaults": self.defaults,
            "author": self.author,
            "tags": self.tags or [],
        }

# Production RAG prompt template
RAG_PROMPT = PromptTemplate(
    template_id="rag_qa",
    version="3.0.0",
    description="Standard RAG QA prompt with source attribution",
    system_template="""You are a {persona} for {organisation}.
Answer questions using only the provided context.
Cite sources with bracketed numbers: [1], [2].
If the answer is not in the context, respond: "I cannot find this information in the available documents."
Tone: {tone}. Language: {language}.""",
    user_template="""Context:
{context}

Question: {question}""",
    variables=["context", "question"],
    optional_variables=["persona", "organisation", "tone", "language"],
    defaults={
        "persona": "knowledge assistant",
        "organisation": "the company",
        "tone": "professional",
        "language": "English"
    },
    author="platform-team",
    tags=["rag", "qa", "production"]
)
```

**Java — Prompt template with [LangChain4j](https://docs.langchain4j.dev):**
```java
import dev.langchain4j.model.input.PromptTemplate;

PromptTemplate ragTemplate = PromptTemplate.from("""
    You are a {{persona}} for {{organisation}}.
    Answer using only the context below.
    Cite sources with [N] notation.
    If not found, say: "I cannot find this information."

    Context:
    {{context}}

    Question: {{question}}
    """);

Prompt rendered = ragTemplate.apply(Map.of(
    "persona", "support assistant",
    "organisation", "Acme Corp",
    "context", retrievedContext,
    "question", userQuery
));
```

---

### 2.3 Prompt Versioning and Registry

```python
import hashlib
from pathlib import Path

class PromptRegistry:
    """
    File-based prompt registry.
    For production: use LangSmith or a database-backed registry.
    """
    def __init__(self, registry_path: str = "./prompts"):
        self.base = Path(registry_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, template: PromptTemplate) -> str:
        """Save prompt version. Returns content hash."""
        template_dir = self.base / template.template_id
        template_dir.mkdir(exist_ok=True)

        content = json.dumps(template.to_dict(), indent=2, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

        (template_dir / f"{template.version}.json").write_text(content)
        (template_dir / "latest.json").write_text(content)  # Update latest pointer

        print(f"Saved {template.template_id} v{template.version} [{content_hash}]")
        return content_hash

    def load(self, template_id: str, version: str = "latest") -> PromptTemplate:
        version_file = self.base / template_id / f"{version}.json"
        if not version_file.exists():
            raise FileNotFoundError(f"Prompt {template_id} v{version} not found")
        data = json.loads(version_file.read_text())
        return PromptTemplate(**data)

    def list_versions(self, template_id: str) -> list[str]:
        template_dir = self.base / template_id
        if not template_dir.exists():
            return []
        return sorted([f.stem for f in template_dir.glob("*.json") if f.stem != "latest"])

    def diff(self, template_id: str, version_a: str, version_b: str) -> dict:
        """Return fields that changed between two versions."""
        a = self.load(template_id, version_a).to_dict()
        b = self.load(template_id, version_b).to_dict()
        return {
            k: {"from": a.get(k), "to": b.get(k)}
            for k in set(list(a) + list(b))
            if a.get(k) != b.get(k)
        }
```

---

### 2.4 Prompt Testing and Regression Detection

Every prompt change must pass automated regression tests before promotion.

```python
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()

@dataclass
class PromptTestCase:
    test_id: str
    description: str
    inputs: dict
    expected_contains: list[str]
    expected_not_contains: list[str]
    min_length: int = 20
    max_length: int = 2000

@dataclass
class PromptTestResult:
    test_id: str
    passed: bool
    output: str
    failures: list[str]

class PromptTestRunner:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def run_test(self, template: PromptTemplate, test_case: PromptTestCase) -> PromptTestResult:
        messages = template.render_messages(**test_case.inputs)
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=500
        )
        output = response.choices[0].message.content
        failures = []

        for s in test_case.expected_contains:
            if s.lower() not in output.lower():
                failures.append(f"Missing expected: '{s}'")
        for s in test_case.expected_not_contains:
            if s.lower() in output.lower():
                failures.append(f"Contains forbidden: '{s}'")
        if len(output) < test_case.min_length:
            failures.append(f"Too short: {len(output)} < {test_case.min_length}")

        return PromptTestResult(test_id=test_case.test_id, passed=not failures,
                                output=output, failures=failures)

    def run_suite(self, template: PromptTemplate, tests: list[PromptTestCase]) -> dict:
        results = [self.run_test(template, tc) for tc in tests]
        passed = sum(1 for r in results if r.passed)
        print(f"Prompt suite: {passed}/{len(results)} passed")
        for r in results:
            mark = "✓" if r.passed else "✗"
            print(f"  {mark} {r.test_id}")
            for f in r.failures:
                print(f"      → {f}")
        return {"pass_rate": passed / len(results), "passed": passed,
                "total": len(results), "results": results}

# Standard test suite for any RAG prompt
RAG_TEST_CASES = [
    PromptTestCase(
        test_id="cites_sources",
        description="Answer includes source citation",
        inputs={
            "context": "[1] Enterprise customers get 90-day returns.",
            "question": "How long do enterprise customers have to return items?"
        },
        expected_contains=["[1]", "90"],
        expected_not_contains=["I cannot find"],
    ),
    PromptTestCase(
        test_id="refuses_hallucination",
        description="Refuses when context lacks the information",
        inputs={
            "context": "[1] Standard plan: $50/month.",
            "question": "What is the refund policy for government accounts?"
        },
        expected_contains=["cannot find"],
        expected_not_contains=["government", "$"],
    ),
    PromptTestCase(
        test_id="respects_language",
        description="Responds in Spanish when language=Spanish",
        inputs={
            "context": "[1] El plazo de devolución es 30 días.",
            "question": "¿Cuál es el plazo de devolución?",
            "language": "Spanish"
        },
        expected_contains=["30"],
        expected_not_contains=[],
    ),
]
```

---

### 2.5 Prompt A/B Testing

When a new prompt version shows improvement in offline evaluation, A/B testing validates it under real traffic before full promotion.

```python
import random
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    experiment_id: str
    variant_a_version: str   # Control (current production)
    variant_b_version: str   # Treatment (new version)
    traffic_split: float     # Fraction routed to B (e.g. 0.1 = 10%)
    min_samples: int = 200

class PromptABRouter:
    def __init__(self, registry: PromptRegistry, config: ABTestConfig):
        self.registry = registry
        self.config = config
        self.results = {"a": [], "b": []}

    def get_prompt(self, template_id: str) -> tuple:
        """Returns (template, variant_label)."""
        if random.random() < self.config.traffic_split:
            return self.registry.load(template_id, self.config.variant_b_version), "b"
        return self.registry.load(template_id, self.config.variant_a_version), "a"

    def record_outcome(self, variant: str, score: float):
        self.results[variant].append(score)

    def summary(self) -> dict:
        a, b = self.results["a"], self.results["b"]
        if not a or not b:
            return {"status": "insufficient_data"}
        a_mean = sum(a) / len(a)
        b_mean = sum(b) / len(b)
        return {
            "variant_a": {"n": len(a), "mean": round(a_mean, 4)},
            "variant_b": {"n": len(b), "mean": round(b_mean, 4)},
            "delta": round(b_mean - a_mean, 4),
            "sufficient_data": len(a) >= self.config.min_samples,
            "recommendation": "promote_b" if b_mean > a_mean + 0.02 else "keep_a"
        }
```

---

### 🧪 Hands-on Lab: Prompt Version Control Pipeline

**Objective:** Build a complete prompt versioning, testing, and promotion pipeline.

**Prerequisites:** `openai`

```python
import json, hashlib
from pathlib import Path
from openai import OpenAI

client = OpenAI()
registry = PromptRegistry("./lab_prompts")
runner = PromptTestRunner(model="gpt-4o-mini")

# ── Version 1.0: Minimal prompt ───────────────────────────
v1 = PromptTemplate(
    template_id="support_rag", version="1.0.0",
    description="Support RAG v1 — minimal",
    system_template="Answer the question using only the context below.\n\nContext:\n{context}",
    user_template="{question}",
    variables=["context", "question"],
    optional_variables=[], defaults={}, author="platform-team"
)
registry.save(v1)

# ── Version 1.1: Added citation + refusal ─────────────────
v11 = PromptTemplate(
    template_id="support_rag", version="1.1.0",
    description="Support RAG v1.1 — citation and refusal",
    system_template="""Answer the question using only the context below.
Cite sources with [N] notation. If not found, say "I cannot find this information."

Context:
{context}""",
    user_template="{question}",
    variables=["context", "question"],
    optional_variables=[], defaults={}, author="platform-team"
)
registry.save(v11)

# ── Diff the two versions ─────────────────────────────────
diff = registry.diff("support_rag", "1.0.0", "1.1.0")
print("Changes in v1.1.0:")
for field, change in diff.items():
    print(f"  {field}:")
    print(f"    before: {str(change['from'])[:70]}")
    print(f"    after:  {str(change['to'])[:70]}")

# ── Test suite ────────────────────────────────────────────
tests = [
    PromptTestCase(
        test_id="basic_answer",
        description="Answers from context",
        inputs={"context": "[1] Standard refund window is 30 days.", "question": "How many days to return?"},
        expected_contains=["30"], expected_not_contains=[]
    ),
    PromptTestCase(
        test_id="refuses_out_of_scope",
        description="Refuses when not in context",
        inputs={"context": "[1] Standard plan: $50/month.", "question": "What is the cancellation fee?"},
        expected_contains=["cannot find"], expected_not_contains=["cancellation fee"]
    ),
]

print("\n── Testing v1.0.0 ───────────────────────────────")
r1 = runner.run_suite(v1, tests)

print("\n── Testing v1.1.0 ───────────────────────────────")
r11 = runner.run_suite(v11, tests)

# ── Promote if all tests pass ─────────────────────────────
if r11["pass_rate"] == 1.0:
    print("\n✓ All tests passed — v1.1.0 ready for promotion to staging")
else:
    print(f"\n✗ {r11['total'] - r11['passed']} test(s) failed — do not promote")
```

**Extensions:**
- Add a third version with an aggressive tone and verify it fails the test suite
- Implement the `PromptABRouter` and simulate 300 queries with random quality scores
- Add a lint check: warn if `{context}` is missing from the system template

---

> ### 📋 Chapter Summary
>
> - Prompts are **versioned code artifacts** requiring the same engineering rigour as application code.
> - `PromptTemplate` encapsulates content, required/optional variables, defaults, and metadata.
> - A **prompt registry** stores versioned templates with content hashes, enabling rollback and diff.
> - **Automated test suites** validate behaviour before promotion — checking required content, forbidden content, and output bounds.
> - **A/B testing** validates production impact before full rollout of a new prompt version.

---

> ### ❓ Comprehension Questions
>
> 1. A developer adds `{language}` to a prompt template without updating defaults. What happens to existing callers, and how does `_validate_variables` protect against this?
> 2. Why is `temperature=0` used in prompt regression tests? What does this sacrifice?
> 3. A prompt A/B test shows variant B improves score by 0.015 (1.5%) after 500 samples. The promotion threshold is 2%. What do you do?
> 4. The `diff` method shows `system_template` changed between versions. Design a quality impact assessment process for this change.
> 5. A financial advice assistant's prompt is updated to be "more helpful". How would you detect if this introduced responses that violate financial advice regulations?

---

## References

### Documentation
- [LangSmith Prompt Hub](https://docs.smith.langchain.com/prompt_engineering) — Prompt versioning and A/B testing.
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [LangChain4j AI Services](https://docs.langchain4j.dev/tutorials/ai-services) — Java prompt template integration.
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Helicone](https://docs.helicone.ai) — Prompt versioning with observability.

### Papers
- [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) — Zhou et al., 2022.
- [Prompt Programming for Large Language Models](https://arxiv.org/abs/2102.07350) — Reynolds & McDonell, 2021.

---

## Chapter 3 — Model Artifacts

### 3.1 Model Packaging Standards

Model artifacts in LLM systems fall into two categories: API-accessed models (OpenAI, Anthropic, Cohere) and locally deployed models (fine-tuned adapters, quantised weights). Both require artifact management.

For locally deployed models, [MLflow](https://mlflow.org/docs/latest/index.html) provides a standardised packaging format.

```python
import mlflow
import mlflow.pyfunc
from datetime import datetime

class LLMModelArtifact(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for a locally deployed LLM."""

    def load_context(self, context):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = context.artifacts["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def predict(self, context, model_input):
        prompts = model_input["prompt"].tolist()
        outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output = self.model.generate(**inputs, max_new_tokens=200)
            outputs.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
        return outputs

def log_model_to_mlflow(
    model_path: str,
    experiment_name: str,
    run_name: str,
    metrics: dict,
    params: dict
) -> str:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LLMModelArtifact(),
            artifacts={"model_path": model_path},
            registered_model_name=f"{experiment_name}_model"
        )
        return run.info.run_id
```

---

### 3.2 Fine-Tuned Adapter Management

LoRA adapters are significantly smaller than full model weights and are the primary artifact in enterprise fine-tuning workflows.

```python
import torch
import hashlib
import json
from pathlib import Path

def save_lora_adapter(base_model, adapter_path: str, metadata: dict) -> str:
    """Save LoRA adapter with version metadata. Returns content hash."""
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(adapter_path)

    # Metadata sidecar
    (Path(adapter_path) / "adapter_metadata.json").write_text(
        json.dumps({**metadata, "saved_at": datetime.utcnow().isoformat()}, indent=2)
    )

    # Integrity hash
    adapter_files = list(Path(adapter_path).glob("adapter_model*.bin"))
    if adapter_files:
        content_hash = hashlib.sha256(adapter_files[0].read_bytes()).hexdigest()
        (Path(adapter_path) / "adapter.sha256").write_text(content_hash)
        return content_hash
    return ""

def load_fine_tuned_model(base_model_name: str, adapter_path: str):
    """Load base model + LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tokenizer
```

🔓 **On-premise model serving with [Ollama](https://ollama.com):**
```dockerfile
# Modelfile — wraps a base model with a custom system prompt
FROM llama3

SYSTEM """
You are a customer support specialist for Acme Corp.
Answer questions only using information provided to you.
Be concise, professional, and helpful.
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
```

```bash
ollama create acme-support -f ./Modelfile
ollama serve
```

---

### 3.3 Model Cards

A model card documents what a model was trained on, what it can and cannot do, known limitations, and evaluation results. In regulated industries, model cards are a compliance requirement.

```python
from dataclasses import dataclass, field

@dataclass
class ModelCard:
    model_id: str
    model_version: str
    base_model: str
    description: str
    intended_use: str
    out_of_scope_use: str
    training_data_description: str
    training_data_versions: list[str]
    evaluation_results: dict
    known_limitations: list[str]
    bias_considerations: str
    authors: list[str]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    license: str = "proprietary"

    def to_markdown(self) -> str:
        evals = "\n".join([f"| {k} | {v} |" for k, v in self.evaluation_results.items()])
        limits = "\n".join([f"- {l}" for l in self.known_limitations])
        return f"""# Model Card: {self.model_id} v{self.model_version}

## Description
{self.description}

## Intended Use
{self.intended_use}

## Out of Scope
{self.out_of_scope_use}

## Training Data
{self.training_data_description}
Versions: {", ".join(self.training_data_versions)}

## Evaluation
| Metric | Score |
|---|---|
{evals}

## Known Limitations
{limits}

## Bias Considerations
{self.bias_considerations}

*Created: {self.created_at} | License: {self.license}*
"""
```

---

### 3.4 Model Serving Configuration

Beyond weights, production serving requires versioned serving configuration.

```python
@dataclass
class ModelServingConfig:
    model_id: str
    version: str
    base_model_name: str
    adapter_path: Optional[str] = None
    quantisation: str = "none"     # none | int8 | int4 | gptq
    max_tokens: int = 2048
    temperature: float = 0.0
    serving_framework: str = "vllm"  # vllm | ollama | triton
    endpoint_url: str = ""
    api_key_secret: str = ""         # Reference to secret manager key
    min_replicas: int = 1
    max_replicas: int = 4
    target_rps: int = 100
    health_check_path: str = "/health"
```

---

> ### 📋 Chapter Summary
>
> - Model artifacts include weights, LoRA adapters, Modelfiles, serving configs, and model cards.
> - [MLflow](https://mlflow.org/docs/latest/index.html) provides standardised packaging, experiment tracking, and a model registry.
> - **LoRA adapters** are the primary fine-tuning artifact — small, versioned, and composable with any compatible base model.
> - **Model cards** document training data, intended use, limitations, and evaluations — required for governance and compliance.

---

> ### ❓ Comprehension Questions
>
> 1. A LoRA adapter trained on `Llama-3-8B` is promoted to production. Three months later the base model is upgraded to `Llama-3-8B-Instruct`. What compatibility problem arises?
> 2. Why should model serving configuration be versioned as an artifact alongside model weights?
> 3. A model card states "intended for English-language support". A PM deploys it for Spanish queries. What governance failure has occurred?
> 4. Design an integrity check pipeline for LoRA adapter deployment that prevents corrupted files from being served.
> 5. When does the complexity of MLflow model registry justify its use over a simple file-based registry?

---

## References

### Documentation
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Hugging Face PEFT](https://huggingface.co/docs/peft) — LoRA and adapter fine-tuning.
- [vLLM Documentation](https://docs.vllm.ai) — High-throughput LLM inference.
- [Ollama Documentation](https://ollama.com)
- [Model Cards Toolkit](https://github.com/tensorflow/model-card-toolkit)

### Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021.
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Mitchell et al., 2019.

---

## Chapter 4 — Index Artifacts

### 4.1 Vector Index as a Deployable Artifact

A vector index is not a mutable database table — it is a versioned artifact that is built, validated, and deployed atomically. This distinction has significant engineering implications.

**Why atomic deployment matters:** An in-place update during active serving creates a window where the index contains a mix of old and new embeddings. If the embedding model changed between builds, queries compare vectors from two different semantic spaces — a silent, hard-to-debug failure.

Treating the index as an artifact that is built, tested, and swapped atomically eliminates this class of failure.

---

### 4.2 Index Versioning and Blue-Green Deployment

```python
import time
from enum import Enum
from dataclasses import dataclass

class IndexStatus(str, Enum):
    BUILDING = "building"
    STAGING = "staging"
    ACTIVE = "active"
    DEPRECATED = "deprecated"

@dataclass
class IndexVersion:
    index_id: str
    version: str
    embedding_model: str
    chunk_strategy: str
    chunk_size: int
    document_count: int
    collection_name: str
    status: IndexStatus
    created_at: str
    content_hash: str
    evaluation_score: float = 0.0

class BlueGreenIndexManager:
    """
    Build new index in a separate collection (green),
    validate it, then switch traffic atomically.
    """
    def __init__(self, vector_client, embedding_service):
        self.client = vector_client
        self.embedding_service = embedding_service
        self.active_collection: Optional[str] = None

    def build_new_version(
        self,
        documents: list[dict],
        config: dict
    ) -> IndexVersion:
        """Build a new index in an isolated collection."""
        version = f"v{int(time.time())}"
        collection_name = f"index_{version}"

        # Ingest into new collection
        self._build_collection(collection_name, documents, config)

        return IndexVersion(
            index_id="main_kb",
            version=version,
            embedding_model=config["embedding_model"],
            chunk_strategy=config["chunk_strategy"],
            chunk_size=config["chunk_size"],
            document_count=len(documents),
            collection_name=collection_name,
            status=IndexStatus.STAGING,
            created_at=datetime.utcnow().isoformat(),
            content_hash=self._hash_documents(documents)
        )

    def validate(
        self,
        index_version: IndexVersion,
        eval_queries: list[dict],
        min_recall: float = 0.80
    ) -> bool:
        """Evaluate retrieval quality on staging index before promotion."""
        hits = sum(
            1 for item in eval_queries
            if any(
                rid in [r["id"] for r in self._search(index_version.collection_name, item["query"])]
                for rid in item["relevant_ids"]
            )
        )
        recall = hits / len(eval_queries)
        index_version.evaluation_score = recall
        print(f"Index validation: Recall@5 = {recall:.2%} (min: {min_recall:.2%})")
        return recall >= min_recall

    def promote(self, index_version: IndexVersion):
        """Atomically switch active collection to new version."""
        old = self.active_collection
        self.active_collection = index_version.collection_name
        index_version.status = IndexStatus.ACTIVE
        print(f"Promoted to active: {index_version.collection_name}")
        if old:
            print(f"Deprecated: {old}")

    def _build_collection(self, collection_name, documents, config):
        pass  # Vector DB specific implementation

    def _search(self, collection_name, query, top_k=5):
        return []  # Vector DB specific implementation

    def _hash_documents(self, documents: list[dict]) -> str:
        content = json.dumps(sorted([d.get("id", "") for d in documents]))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

---

### 4.3 Index Lineage

```python
from dataclasses import dataclass

@dataclass
class IndexLineage:
    index_version: str
    source_corpus_id: str
    source_corpus_version: str
    embedding_model: str
    chunking_pipeline_version: str
    document_count: int
    built_at: str
    quality_metrics: dict

def record_index_lineage(
    index_version: IndexVersion,
    corpus_metadata: dict,
    pipeline_versions: dict
) -> IndexLineage:
    return IndexLineage(
        index_version=index_version.version,
        source_corpus_id=corpus_metadata["corpus_id"],
        source_corpus_version=corpus_metadata["version"],
        embedding_model=index_version.embedding_model,
        chunking_pipeline_version=pipeline_versions.get("chunking", "unknown"),
        document_count=index_version.document_count,
        built_at=index_version.created_at,
        quality_metrics={"recall_at_5": index_version.evaluation_score}
    )
```

---

> ### 📋 Chapter Summary
>
> - Vector indexes are **deployable artifacts**, not in-place mutable databases.
> - **Blue-green deployment** builds in a separate collection, validates, then switches traffic atomically.
> - Index validation enforces a minimum Recall@K gate before promotion to production.
> - **Index lineage** records source documents, embedding model, and pipeline versions for each index build.

---

> ### ❓ Comprehension Questions
>
> 1. An in-place update adds documents embedded with a new model into an index built with an older model. What retrieval failure occurs and why?
> 2. Blue-green promotion switches `active_collection` atomically. What happens to in-flight queries at the switch moment?
> 3. Index Recall@5 = 0.75, below the 0.80 threshold. What diagnostics would you run before deciding whether to lower the threshold?
> 4. A corpus is updated with 10% new documents. Should you rebuild the entire index or incrementally update? Discuss the trade-offs.
> 5. Index lineage records the embedding model version. How does this support a legal audit of "what documents underpinned responses last month"?

---

## References

### Documentation
- [Qdrant Collection Management](https://qdrant.tech/documentation/concepts/collections/)
- [Weaviate Multi-tenancy](https://weaviate.io/developers/weaviate/manage-data/multi-tenancy)
- [Milvus Collection Alias](https://milvus.io/docs/manage-collections.md) — Atomic collection switching.
- [Chroma Documentation](https://docs.trychroma.com/getting-started)

### Papers
- [BEIR: Zero-shot Evaluation of Information Retrieval](https://arxiv.org/abs/2104.08663) — Thakur et al., 2021.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020.

---

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
> [← Part V — Dataset Engineering](part_05_dataset_engineering.md) | [→ Part VII — Layouts and Repositories](part_07_layouts_repositories.md)
