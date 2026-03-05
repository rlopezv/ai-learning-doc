## Chapter 3 — Integration and Contract Testing

### 3.1 Integration Test Architecture

Integration tests verify that multiple components work correctly together. For RAG systems, the primary integration scenarios are:

1. **Document ingestion:** chunker → embedder → vector store
2. **Query pipeline:** query → retriever → LLM (mocked) → output parser → response
3. **Index lifecycle:** build → validate → promote (blue-green)

Integration tests use real vector databases (via Docker or Testcontainers) but mock the LLM to avoid API costs and non-determinism.

```python
# tests/integration/test_ingestion_pipeline.py
import pytest
from libs.rag_core.ingestion import IngestionPipeline
from libs.rag_core.embeddings import SentenceTransformerEmbedder

@pytest.mark.integration
class TestIngestionPipeline:
    """Integration tests for the document ingestion pipeline.
    Requires: running ChromaDB on localhost:8000 (docker-compose.test.yml)
    """

    @pytest.fixture(scope="class")
    def embedder(self):
        return SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    @pytest.fixture
    def pipeline(self, embedder, test_collection):
        return IngestionPipeline(
            embedder=embedder,
            collection=test_collection,
            chunk_size=200,
            chunk_overlap=30
        )

    @pytest.fixture
    def sample_documents(self):
        return [
            {"id": "d1", "content": "Enterprise customers have a 90-day return window. "
             "Standard customers receive a 30-day return window with valid receipt.",
             "source": "policy.pdf", "title": "Return Policy"},
            {"id": "d2", "content": "Professional plans include 25 user seats and "
             "priority email support with 4-hour response SLA.",
             "source": "pricing.pdf", "title": "Pricing Guide"},
        ]

    def test_ingestion_creates_searchable_vectors(self, pipeline, sample_documents):
        pipeline.ingest(sample_documents)
        results = pipeline.search("enterprise return policy", top_k=3)
        assert len(results) > 0
        assert any("90-day" in r["content"] or "return" in r["content"].lower()
                   for r in results)

    def test_all_documents_ingested(self, pipeline, sample_documents):
        pipeline.ingest(sample_documents)
        count = pipeline.count()
        # 2 documents × estimated chunks each
        assert count >= 2

    def test_metadata_preserved_after_ingestion(self, pipeline, sample_documents):
        pipeline.ingest(sample_documents)
        results = pipeline.search("return policy", top_k=5)
        result_with_source = next(
            (r for r in results if r.get("metadata", {}).get("source")), None
        )
        assert result_with_source is not None
        assert result_with_source["metadata"]["source"] in ("policy.pdf", "pricing.pdf")

    def test_duplicate_document_ids_overwrite(self, pipeline, sample_documents):
        pipeline.ingest(sample_documents)
        count_before = pipeline.count()
        pipeline.ingest(sample_documents)  # Same IDs
        count_after = pipeline.count()
        assert count_after <= count_before + len(sample_documents)

    def test_empty_document_list_does_not_error(self, pipeline):
        pipeline.ingest([])  # Should not raise

    def test_search_returns_relevance_scores(self, pipeline, sample_documents):
        pipeline.ingest(sample_documents)
        results = pipeline.search("return window", top_k=3)
        for result in results:
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0
```

---

### 3.2 Testing the RAG Pipeline End-to-End

```python
# tests/integration/test_rag_pipeline.py
import pytest
from libs.rag_core.pipeline import RAGPipeline, RAGConfig
from unittest.mock import MagicMock

@pytest.mark.integration
class TestRAGPipelineIntegration:
    """
    End-to-end pipeline tests.
    Uses real ChromaDB, real embeddings, mocked LLM.
    """

    @pytest.fixture
    def rag_config(self):
        return RAGConfig(
            top_k=5,
            chunk_size=200,
            chunk_overlap=30,
            min_relevance_score=0.3
        )

    @pytest.fixture
    def mock_llm(self):
        """LLM mock that returns deterministic, citation-aware responses."""
        scenario_mock = ScenarioLLMMock()
        scenario_mock \
            .when_query_contains("return", "Enterprise customers have 90 days [1].") \
            .when_query_contains("pricing", "Professional plan costs $150/month [1].") \
            .when_query_contains("unknown_topic_xyz", "I cannot find this information.")
        return scenario_mock

    @pytest.fixture
    def pipeline(self, rag_config, test_collection, mock_llm):
        pipeline = RAGPipeline(
            config=rag_config,
            collection=test_collection,
            llm=mock_llm
        )
        # Ingest test documents
        pipeline.ingest([
            {"id": "d1", "content": "Enterprise customers get 90-day returns. Standard customers 30 days."},
            {"id": "d2", "content": "Professional plan: $150/month. 25 users. Priority support."},
        ])
        return pipeline

    def test_returns_answer_for_in_scope_query(self, pipeline):
        result = pipeline.query("What is the enterprise return policy?")
        assert result.answer
        assert not result.is_refusal

    def test_returns_refusal_for_out_of_scope_query(self, pipeline):
        result = pipeline.query("What is the unknown_topic_xyz policy?")
        assert result.is_refusal

    def test_answer_includes_source_citations(self, pipeline):
        result = pipeline.query("return policy")
        assert len(result.sources) > 0

    def test_sources_reference_ingested_documents(self, pipeline):
        result = pipeline.query("pricing plan")
        source_ids = {s["id"] for s in result.sources}
        assert "d1" in source_ids or "d2" in source_ids

    def test_pipeline_respects_top_k(self, pipeline, rag_config):
        result = pipeline.query("return policy")
        assert len(result.retrieved_documents) <= rag_config.top_k

    def test_pipeline_filters_low_relevance(self, pipeline):
        result = pipeline.query("completely unrelated topic basketball")
        # With min_relevance_score=0.3, very unrelated queries may return empty
        for doc in result.retrieved_documents:
            assert doc["score"] >= pipeline.config.min_relevance_score

    def test_latency_within_budget(self, pipeline):
        import time
        t0 = time.perf_counter()
        pipeline.query("return policy")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 2000, f"Pipeline took {elapsed_ms:.0f}ms (budget: 2000ms)"
```

---

### 3.3 Contract Tests for LLM Providers

Contract tests verify that the LLM provider's API responses match the schema your code depends on. They protect against provider-side breaking changes.

```python
# tests/contract/test_openai_contract.py
import pytest
import json
from openai import OpenAI

@pytest.mark.contract
@pytest.mark.requires_real_api
class TestOpenAIContract:
    """
    Contract tests for OpenAI API.
    Run infrequently (nightly) — validate provider schema hasn't changed.
    """

    @pytest.fixture(scope="class")
    def client(self):
        return OpenAI()

    def test_chat_completion_response_schema(self, client):
        """Verify response structure matches what our code depends on."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with: test"}],
            max_tokens=10,
            temperature=0
        )
        # Fields our code accesses
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert isinstance(response.choices[0].message.content, str)
        # Usage fields
        assert hasattr(response, "usage")
        assert hasattr(response.usage, "prompt_tokens")
        assert hasattr(response.usage, "completion_tokens")
        assert hasattr(response.usage, "total_tokens")

    def test_json_mode_response_is_valid_json(self, client):
        """Verify json_object response_format returns parseable JSON."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Return: {\"status\": \"ok\"}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=20,
            temperature=0
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_token_count_is_positive(self, client):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0

    def test_model_name_in_response(self, client):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        assert "gpt-4o-mini" in response.model

# Schema-based contract test (no API call — validates our mock matches real schema)
class TestMockConformsToContract:
    """Verify that mocks used in unit tests produce responses matching the real contract."""

    def test_mock_response_has_required_fields(self, mock_llm_response):
        mock = mock_llm_response("Test content")
        assert mock.choices[0].message.content == "Test content"
        assert mock.usage.total_tokens > 0

    def test_mock_response_content_is_string(self, mock_llm_response):
        mock = mock_llm_response("Hello world")
        assert isinstance(mock.choices[0].message.content, str)
```

---

### 3.4 Vector Database Integration Tests

```python
# tests/integration/test_vector_db_integration.py
import pytest
import uuid

@pytest.mark.integration
class TestVectorDBIntegration:
    """
    Integration tests for vector database operations.
    Requires: running ChromaDB (docker-compose.test.yml)
    """

    def test_add_and_retrieve_documents(self, test_collection, mock_embeddings):
        docs = ["Document about returns", "Document about pricing"]
        embeddings = mock_embeddings(docs)
        ids = [str(uuid.uuid4()) for _ in docs]

        test_collection.add(ids=ids, embeddings=embeddings, documents=docs)

        results = test_collection.query(
            query_embeddings=[mock_embeddings(["returns policy"])[0]],
            n_results=2
        )
        assert len(results["ids"][0]) == 2

    def test_metadata_filtering(self, test_collection, mock_embeddings):
        docs = ["Policy for enterprise", "Policy for standard"]
        embeddings = mock_embeddings(docs)
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        metadatas = [{"tier": "enterprise"}, {"tier": "standard"}]

        test_collection.add(ids=ids, embeddings=embeddings,
                           documents=docs, metadatas=metadatas)

        results = test_collection.query(
            query_embeddings=[mock_embeddings(["policy"])[0]],
            n_results=10,
            where={"tier": "enterprise"}
        )
        assert all(m["tier"] == "enterprise"
                   for m in results["metadatas"][0])

    def test_upsert_overwrites_existing(self, test_collection, mock_embeddings):
        doc_id = "fixed-id-001"
        embedding = mock_embeddings(["original content"])[0]
        test_collection.add(ids=[doc_id], embeddings=[embedding],
                           documents=["Original content"])

        new_embedding = mock_embeddings(["updated content"])[0]
        test_collection.upsert(ids=[doc_id], embeddings=[new_embedding],
                              documents=["Updated content"])

        result = test_collection.get(ids=[doc_id])
        assert result["documents"][0] == "Updated content"

    def test_delete_removes_document(self, test_collection, mock_embeddings):
        doc_id = str(uuid.uuid4())
        test_collection.add(
            ids=[doc_id],
            embeddings=mock_embeddings(["Temp document"])[0:1],
            documents=["Temp document"]
        )
        test_collection.delete(ids=[doc_id])
        result = test_collection.get(ids=[doc_id])
        assert result["documents"] == [None] or len(result["ids"]) == 0

    def test_collection_count_accurate(self, test_collection, mock_embeddings):
        n = 7
        test_collection.add(
            ids=[str(i) for i in range(n)],
            embeddings=mock_embeddings([f"doc{i}" for i in range(n)]),
            documents=[f"Document {i}" for i in range(n)]
        )
        assert test_collection.count() == n
```

---

> ### 📋 Chapter Summary
>
> - Integration tests use **real vector databases** (via Docker) but **mock the LLM** — validating real data persistence and retrieval without API costs.
> - The RAG pipeline integration test validates the complete query flow: ingest → retrieve → (mock) generate → structured response.
> - **Contract tests** verify the LLM provider's response schema has not changed — protecting against silent provider-side breaking changes.
> - Vector DB integration tests must cover: add, retrieve, metadata filtering, upsert, delete, and count — the full CRUD surface used by the application.

---

> ### ❓ Comprehension Questions
>
> 1. The ingestion pipeline integration test uses a `test_collection` fixture with `scope="function"` (a fresh collection per test). Why is this important, and what failure mode could occur if the collection were shared across tests?
> 2. The RAG pipeline integration test mocks the LLM with `ScenarioLLMMock`. A developer argues that this mock is too specific — the scenario matching hardcodes the expected LLM call pattern. When would a more general mock be preferable?
> 3. Contract tests for the OpenAI API are marked `@pytest.mark.requires_real_api` and run nightly. OpenAI releases a breaking change on a Tuesday afternoon. When is it first detected, and how would you reduce this detection window?
> 4. The `test_latency_within_budget` test asserts the pipeline runs in under 2000ms. A CI environment is slower than a developer's laptop. How would you make this test environment-aware?
> 5. Design a contract test for the embedding API that validates: (a) returned vectors have the expected dimension, (b) identical inputs produce identical vectors, (c) different inputs produce different vectors. What does each assertion protect against?

---

## References

### Documentation
- [Testcontainers Python](https://testcontainers-python.readthedocs.io/en/latest/) — Containerised test dependencies.
- [pytest-docker](https://github.com/avast/pytest-docker) — Docker Compose integration for pytest.
- [Pact Consumer-Driven Contracts](https://docs.pact.io) — Contract testing framework.
- [WireMock HTTP Mocking](https://wiremock.org/docs/) — LLM API simulation.

---

---
[« Back to testing Index](index.md) | [🏠 Home](../../index.md)