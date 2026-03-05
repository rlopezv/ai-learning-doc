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

---
[« Back to artifact_engineering Index](index.md) | [🏠 Home](../index.md)