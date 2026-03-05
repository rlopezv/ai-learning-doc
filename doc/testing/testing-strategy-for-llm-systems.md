## Chapter 1 — Testing Strategy for LLM Systems

### 1.1 Why LLM Testing Is Different

Testing an LLM system requires rethinking every assumption from traditional software testing.

**No deterministic output.** A function that calls an LLM may return slightly different text on every invocation, even with `temperature=0`. Tests based on exact string matching are fragile or require deterministic mocking. Instead, tests must assert structural properties, semantic properties, or use probabilistic pass rates over multiple runs.

**The oracle problem.** In traditional testing, the expected output is known precisely. For an LLM generating a natural language answer, "correctness" is often a matter of degree. A test that checks whether an answer contains "30 days" is checking a necessary but not sufficient condition. A test that checks whether the answer is grounded, relevant, and complete requires evaluation machinery — not a simple assertion.

**External dependencies at the core.** LLM systems call external APIs as a primary operation. These calls are slow, costly, and occasionally unreliable. A test suite that makes real LLM API calls will be slow, expensive, and flaky. An effective test suite isolates the LLM behind a mock for unit and integration tests, and reserves real API calls for evaluation tests and smoke tests.

**Non-code artifacts require their own tests.** Prompt templates, vector index configurations, and dataset schemas require validation that does not fit the traditional test function paradigm. These validations must still be automated and integrated into the CI pipeline.

**Regressions are semantic, not syntactic.** A code regression typically produces an exception or wrong value. A prompt regression produces subtly worse answers — different in ways that require evaluation to detect, not assertion.

---

### 1.2 The LLM Testing Pyramid

```
           ┌────────────────────────────────┐
           │     Evaluation tests            │  Quality assessment
           │  (RAGAS, LLM-judge, human)      │  Part IX territory
           ├────────────────────────────────┤
           │     End-to-end smoke tests      │  Real API calls
           │  (5–20 golden queries)          │  Fast sanity check
           ├────────────────────────────────┤
           │     Integration tests           │  Containerised dependencies
           │  (RAG pipeline, vector DB)      │  Real embeddings, mocked LLM
           ├────────────────────────────────┤
           │     Contract tests              │  Provider API shape
           │  (OpenAI, Anthropic schemas)    │  Pact or schema validation
           ├────────────────────────────────┤
           │     Unit tests                  │  Mocked LLM, fast, cheap
           │  (parsers, chunkers, routers)   │  100s of tests, seconds
           ├────────────────────────────────┤
           │     Prompt validation tests     │  Schema, variables, format
           │  (JSON schema, lint rules)      │  Deterministic, instant
           └────────────────────────────────┘
              Fast, cheap ◄────────► Slow, expensive
              Many tests  ◄────────► Few tests
```

The key discipline: **the bottom three layers must never make real LLM API calls**. All LLM interactions at unit and integration test layers go through controlled test doubles. Real API calls are reserved for smoke tests and evaluation.

---

### 1.3 Test Classification and Scope

```python
from dataclasses import dataclass
from enum import Enum

class TestCategory(str, Enum):
    PROMPT_VALIDATION  = "prompt_validation"   # Schema, vars, format — instant
    UNIT               = "unit"                # Mocked LLM, < 1s each
    INTEGRATION        = "integration"         # Real vector DB, mocked LLM
    CONTRACT           = "contract"            # Provider API schema
    SMOKE              = "smoke"               # Real API, golden queries
    EVALUATION         = "evaluation"          # RAGAS / LLM-judge (Part IX)
    ADVERSARIAL        = "adversarial"         # Injection, jailbreak
    LOAD               = "load"                # Throughput, latency under load

@dataclass
class TestSuiteConfig:
    category: TestCategory
    # Run conditions
    run_on_pr: bool       = True
    run_on_merge: bool    = True
    run_on_release: bool  = True
    run_nightly: bool     = False
    # LLM calls
    uses_real_llm: bool   = False
    uses_real_vector_db: bool = False
    # Thresholds
    max_duration_seconds: int = 30
    min_pass_rate: float  = 1.0  # 100% for unit; lower for non-deterministic

# Define run conditions per suite
SUITE_CONFIGS = {
    TestCategory.PROMPT_VALIDATION: TestSuiteConfig(
        TestCategory.PROMPT_VALIDATION, run_on_pr=True, max_duration_seconds=5),
    TestCategory.UNIT: TestSuiteConfig(
        TestCategory.UNIT, run_on_pr=True, max_duration_seconds=30),
    TestCategory.INTEGRATION: TestSuiteConfig(
        TestCategory.INTEGRATION, run_on_pr=True,
        uses_real_vector_db=True, max_duration_seconds=120),
    TestCategory.CONTRACT: TestSuiteConfig(
        TestCategory.CONTRACT, run_on_pr=True, max_duration_seconds=10),
    TestCategory.SMOKE: TestSuiteConfig(
        TestCategory.SMOKE, run_on_pr=False, run_on_merge=True,
        uses_real_llm=True, max_duration_seconds=120, min_pass_rate=0.95),
    TestCategory.ADVERSARIAL: TestSuiteConfig(
        TestCategory.ADVERSARIAL, run_on_pr=False, run_on_release=True,
        uses_real_llm=True, max_duration_seconds=300),
    TestCategory.LOAD: TestSuiteConfig(
        TestCategory.LOAD, run_on_pr=False, run_nightly=True,
        uses_real_llm=True, max_duration_seconds=600),
}
```

---

### 1.4 Test Environment Design

```yaml
# docker-compose.test.yml — Local test environment
# Provides real vector DB + mock LLM server for integration tests

services:
  # Real vector database for integration tests
  chroma:
    image: chromadb/chroma:0.5.0
    ports: ["8000:8000"]
    environment:
      CHROMA_SERVER_AUTH_PROVIDER: ""
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 5s
      retries: 5

  # Mock LLM server — returns controlled responses
  mock-llm:
    image: wiremock/wiremock:3.9.1
    ports: ["9090:8080"]
    volumes:
      - ./tests/fixtures/wiremock:/home/wiremock
    command: ["--global-response-templating", "--verbose"]

  # Minimal test vector DB for Java integration tests
  qdrant-test:
    image: qdrant/qdrant:v1.11.0
    ports: ["6333:6333"]
    environment:
      QDRANT__SERVICE__GRPC_PORT: "6334"
```

```python
# tests/conftest.py — pytest fixtures for test environment
import pytest
import chromadb
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture(scope="session")
def chroma_client():
    """Real ChromaDB client for integration tests."""
    return chromadb.HttpClient(host="localhost", port=8000)

@pytest.fixture
def test_collection(chroma_client):
    """Fresh collection per test — no state leakage."""
    import uuid
    name = f"test_{uuid.uuid4().hex[:8]}"
    collection = chroma_client.create_collection(name)
    yield collection
    chroma_client.delete_collection(name)

@pytest.fixture
def mock_llm_response():
    """Factory: create controlled LLM responses for unit tests."""
    def _factory(content: str, model: str = "gpt-4o-mini"):
        mock = MagicMock()
        mock.choices[0].message.content = content
        mock.model = model
        mock.usage.prompt_tokens = 100
        mock.usage.completion_tokens = 50
        mock.usage.total_tokens = 150
        return mock
    return _factory

@pytest.fixture
def mock_openai_client(mock_llm_response):
    """Patched OpenAI client that returns controlled responses."""
    with patch("openai.OpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = mock_llm_response(
            "This is a test response from the mock LLM."
        )
        yield client

@pytest.fixture
def mock_embeddings():
    """Deterministic mock embeddings for retrieval tests."""
    def _embed(texts):
        # Return deterministic vectors based on text hash
        import hashlib
        results = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            # Normalised 384-dim vector seeded by hash
            import random
            rng = random.Random(h)
            vec = [rng.gauss(0, 1) for _ in range(384)]
            norm = sum(x**2 for x in vec) ** 0.5
            results.append([x / norm for x in vec])
        return results
    return _embed
```

---

> ### 📋 Chapter Summary
>
> - LLM testing differs in four fundamental ways: non-deterministic output, the oracle problem, external API dependencies, and semantic rather than syntactic regressions.
> - The **LLM testing pyramid** places prompt validation and unit tests at the bottom (fast, cheap, no API calls) and evaluation tests at the top (slow, expensive, real API calls).
> - **The bottom three layers must never make real LLM API calls** — all LLM interactions are mocked.
> - Test suite configuration specifies exactly which suites run on PR, on merge, on release, and nightly, with appropriate time budgets.

---

> ### ❓ Comprehension Questions
>
> 1. A developer writes a unit test that asserts the RAG pipeline output equals an exact expected string. The test is deterministic because `temperature=0`. Explain why this test is still fragile and what a better assertion strategy would be.
> 2. The test pyramid says integration tests should use a "real vector DB, mocked LLM". Why is mocking the vector DB in integration tests insufficient?
> 3. Smoke tests run on merge to main but not on every PR. A developer argues that any test worth running after merge should also run on the PR. Evaluate this argument.
> 4. An LLM system has 800 unit tests running in 12 seconds and 15 integration tests running in 4 minutes. A new developer proposes converting the integration tests to unit tests with mocked vector DBs to reduce CI time. What is the risk of this change?
> 5. A test uses `temperature=0` and runs the same query 5 times against a real GPT-4o endpoint, getting 5 different outputs. What does this reveal about the assumption that `temperature=0` guarantees determinism?

---

## References

### Documentation
- [pytest Documentation](https://docs.pytest.org) — Python testing framework.
- [WireMock](https://wiremock.org/docs/) — HTTP mock server for LLM API simulation.
- [Testcontainers](https://testcontainers.com/guides/) — Containerised test dependencies.
- [Pact](https://docs.pact.io) — Consumer-driven contract testing.

### Books
- *Growing Object-Oriented Software, Guided by Tests* — Freeman & Pryce (Addison-Wesley). Test double patterns applicable to LLM mocking.
- *The Art of Unit Testing* — Osherove (Manning). Unit test principles for non-deterministic systems.

---

---
[« Back to testing Index](index.md) | [🏠 Home](../index.md)