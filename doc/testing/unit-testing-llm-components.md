## Chapter 2 — Unit Testing LLM Components

### 2.1 Mocking LLM Calls

Effective LLM mocking has three patterns, each appropriate in different test scenarios.

```python
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from openai import OpenAI

# ── Pattern 1: Response factory ──────────────────────────────────────────
class MockLLMClient:
    """
    Full mock LLM client with configurable response scenarios.
    Use for testing logic that surrounds LLM calls.
    """
    def __init__(self):
        self.calls: list[dict] = []
        self._responses: list[str] = []
        self._call_index = 0

    def queue_response(self, content: str):
        """Queue a response to be returned on the next call."""
        self._responses.append(content)
        return self  # Fluent API for chaining

    def create_completion(self, messages: list, **kwargs) -> MagicMock:
        content = (self._responses[self._call_index]
                   if self._call_index < len(self._responses)
                   else "Default mock response.")
        self._call_index += 1
        self.calls.append({"messages": messages, **kwargs})
        mock = MagicMock()
        mock.choices[0].message.content = content
        mock.usage.total_tokens = 100
        return mock

    def assert_called_with_system_containing(self, substring: str):
        assert self.calls, "LLM was never called"
        last_call = self.calls[-1]
        system_msgs = [m for m in last_call["messages"] if m["role"] == "system"]
        assert any(substring in m["content"] for m in system_msgs), \
            f"System prompt did not contain '{substring}'. Got: {[m['content'] for m in system_msgs]}"

    def assert_temperature(self, expected: float):
        assert self.calls, "LLM was never called"
        actual = self.calls[-1].get("temperature")
        assert actual == expected, f"Expected temperature={expected}, got temperature={actual}"

# ── Pattern 2: Scenario-based mock ───────────────────────────────────────
class ScenarioLLMMock:
    """
    Returns different responses based on query content matching.
    Use for testing branching logic in the application layer.
    """
    def __init__(self):
        self.scenarios: list[tuple[str, str]] = []
        self.default = "I cannot find this information."

    def when_query_contains(self, substring: str, response: str) -> "ScenarioLLMMock":
        self.scenarios.append((substring.lower(), response))
        return self

    def create_completion(self, messages: list, **kwargs) -> MagicMock:
        user_content = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        ).lower()
        content = self.default
        for pattern, response in self.scenarios:
            if pattern in user_content:
                content = response
                break
        mock = MagicMock()
        mock.choices[0].message.content = content
        return mock

# ── Pattern 3: pytest.fixture with patch ─────────────────────────────────
@pytest.fixture
def patched_openai():
    """Patch OpenAI at import level. Use for tests of code that imports openai directly."""
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "Mocked answer with [1] citation."
        )
        yield mock_client

def _make_mock_response(content: str) -> MagicMock:
    r = MagicMock()
    r.choices[0].message.content = content
    r.usage.total_tokens = 80
    return r
```

---

### 2.2 Testing Prompt Templates

```python
import pytest
from libs.rag_core.prompts import PromptTemplate

class TestRAGPromptTemplate:
    """Tests for the RAG QA prompt template."""

    @pytest.fixture
    def template(self):
        return PromptTemplate(
            template_id="rag_qa",
            version="3.0.0",
            description="RAG QA prompt",
            system_template="You are a {persona}. Answer using only context. Tone: {tone}.",
            user_template="Context:\n{context}\n\nQuestion: {question}",
            variables=["context", "question"],
            optional_variables=["persona", "tone"],
            defaults={"persona": "assistant", "tone": "professional"},
            author="test"
        )

    # ── Variable rendering ────────────────────────────────────────────────
    def test_renders_required_variables(self, template):
        system = template.render_system(context="test", question="test")
        assert "assistant" in system
        assert "professional" in system

    def test_renders_optional_variables_when_provided(self, template):
        system = template.render_system(context="x", question="x", persona="expert")
        assert "expert" in system
        assert "assistant" not in system

    def test_defaults_applied_when_optional_not_provided(self, template):
        system = template.render_system(context="x", question="x")
        assert "assistant" in system

    # ── Variable validation ───────────────────────────────────────────────
    def test_raises_on_missing_required_variable(self, template):
        with pytest.raises(ValueError, match="Missing required variables"):
            template.render_system(question="test")  # context missing

    def test_raises_listing_all_missing_variables(self, template):
        with pytest.raises(ValueError) as exc_info:
            template.render_user()  # both context and question missing
        assert "context" in str(exc_info.value)
        assert "question" in str(exc_info.value)

    # ── Message structure ─────────────────────────────────────────────────
    def test_render_messages_produces_valid_structure(self, template):
        messages = template.render_messages(context="ctx", question="q")
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_context_in_user_turn(self, template):
        messages = template.render_messages(context="My context", question="My question")
        user_content = messages[-1]["content"]
        assert "My context" in user_content
        assert "My question" in user_content

    def test_system_not_empty(self, template):
        messages = template.render_messages(context="c", question="q")
        assert len(messages[0]["content"]) > 10

    # ── Versioning ────────────────────────────────────────────────────────
    def test_version_follows_semver(self, template):
        parts = template.version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    # ── Few-shot examples ─────────────────────────────────────────────────
    def test_few_shot_examples_inserted_between_system_and_user(self):
        template_with_examples = PromptTemplate(
            template_id="test", version="1.0.0", description="",
            system_template="System",
            user_template="{question}",
            variables=["question"], optional_variables=[], defaults={},
            examples=[{"user": "Example Q", "assistant": "Example A"}],
            author="test"
        )
        messages = template_with_examples.render_messages(question="Real Q")
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert messages[1]["content"] == "Example Q"
        assert messages[2]["content"] == "Example A"
```

---

### 2.3 Testing Retrieval Logic

```python
import pytest
from libs.rag_core.retrieval import ChunkingPipeline, HybridRetriever

class TestChunkingPipeline:
    """Unit tests for document chunking."""

    @pytest.fixture
    def pipeline(self):
        return ChunkingPipeline(chunk_size=100, chunk_overlap=20)

    def test_chunks_long_document(self, pipeline):
        text = "word " * 200  # 200 words
        chunks = pipeline.chunk(text)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, pipeline):
        text = "word " * 500
        chunks = pipeline.chunk(text)
        for chunk in chunks:
            # Allow 10% tolerance for word-boundary splitting
            assert len(chunk.split()) <= 110

    def test_overlap_present(self, pipeline):
        text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " * 5
        chunks = pipeline.chunk(text)
        if len(chunks) > 1:
            # Last words of chunk N should appear in chunk N+1
            first_chunk_words = set(chunks[0].split()[-5:])
            second_chunk_words = set(chunks[1].split()[:15])
            assert len(first_chunk_words & second_chunk_words) > 0

    def test_empty_document_returns_empty_list(self, pipeline):
        assert pipeline.chunk("") == []
        assert pipeline.chunk("   ") == []

    def test_short_document_returns_single_chunk(self, pipeline):
        text = "Short document."
        chunks = pipeline.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_preserves_content(self, pipeline):
        text = "The quick brown fox jumped over the lazy dog. " * 50
        chunks = pipeline.chunk(text)
        # All words should appear in some chunk
        all_chunked = " ".join(chunks)
        assert "quick" in all_chunked
        assert "lazy" in all_chunked

    def test_metadata_attached_to_chunks(self, pipeline):
        doc = {"id": "doc1", "content": "Sample " * 100, "source": "test.pdf"}
        chunks = pipeline.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk["source_doc_id"] == "doc1"
            assert chunk["chunk_index"] == i
            assert chunk["source"] == "test.pdf"

class TestHybridRetriever:
    """Unit tests for the hybrid dense+sparse retriever."""

    @pytest.fixture
    def retriever(self, mock_embeddings):
        return HybridRetriever(
            embedding_fn=mock_embeddings,
            alpha=0.7,           # 70% dense, 30% sparse
            top_k=5
        )

    def test_returns_top_k_results(self, retriever, test_collection):
        # Add 10 docs
        for i in range(10):
            test_collection.add(
                ids=[f"doc{i}"],
                documents=[f"Document {i} about topic {'A' if i < 5 else 'B'}"],
            )
        results = retriever.retrieve("topic A information", top_k=5)
        assert len(results) <= 5

    def test_results_have_required_fields(self, retriever, test_collection):
        test_collection.add(ids=["d1"], documents=["Test document content"])
        results = retriever.retrieve("test", top_k=1)
        if results:
            assert "id" in results[0]
            assert "content" in results[0]
            assert "score" in results[0]

    def test_alpha_zero_uses_sparse_only(self, mock_embeddings):
        sparse_retriever = HybridRetriever(embedding_fn=mock_embeddings, alpha=0.0)
        assert sparse_retriever.alpha == 0.0

    def test_empty_index_returns_empty_list(self, retriever, test_collection):
        results = retriever.retrieve("query against empty index", top_k=5)
        assert results == []
```

---

### 2.4 Testing Output Parsers

Output parsers transform raw LLM text into structured data. They must handle valid outputs, malformed outputs, and edge cases gracefully.

```python
import pytest
import json
from libs.rag_core.parsers import AnswerParser, SourceExtractor

class TestAnswerParser:
    """Tests for structured answer extraction from LLM output."""

    @pytest.fixture
    def parser(self):
        return AnswerParser()

    # ── Happy path ────────────────────────────────────────────────────────
    def test_parses_answer_with_citations(self, parser):
        raw = "Enterprise customers have a 90-day return window [1]. Standard customers get 30 days [2]."
        result = parser.parse(raw)
        assert result.answer_text == raw
        assert result.citation_indices == [1, 2]

    def test_parses_refusal_response(self, parser):
        raw = "I cannot find this information in the available documents."
        result = parser.parse(raw)
        assert result.is_refusal is True
        assert result.answer_text == raw

    def test_parses_answer_without_citations(self, parser):
        raw = "The system uses OAuth 2.0 for authentication."
        result = parser.parse(raw)
        assert result.citation_indices == []
        assert result.is_refusal is False

    # ── Citation extraction ───────────────────────────────────────────────
    def test_extracts_single_citation(self, parser):
        result = parser.parse("See document [3] for details.")
        assert result.citation_indices == [3]

    def test_extracts_multiple_citations(self, parser):
        result = parser.parse("Based on [1] and [2], the answer is [3].")
        assert set(result.citation_indices) == {1, 2, 3}

    def test_extracts_citation_zero(self, parser):
        result = parser.parse("According to [0] the answer is yes.")
        assert 0 in result.citation_indices

    def test_no_duplicate_citations(self, parser):
        result = parser.parse("Referenced in [1] and again in [1].")
        assert result.citation_indices.count(1) == 1

    # ── Refusal detection ─────────────────────────────────────────────────
    @pytest.mark.parametrize("refusal_phrase", [
        "I cannot find this information",
        "This information is not available in the documents",
        "I don't have information about",
        "Unable to find relevant information",
        "Not mentioned in the available documents",
    ])
    def test_detects_refusal_phrases(self, parser, refusal_phrase):
        result = parser.parse(refusal_phrase + " regarding the query.")
        assert result.is_refusal is True

    # ── Edge cases ────────────────────────────────────────────────────────
    def test_handles_empty_response(self, parser):
        result = parser.parse("")
        assert result.answer_text == ""
        assert result.citation_indices == []

    def test_handles_only_whitespace(self, parser):
        result = parser.parse("   \n  ")
        assert result.answer_text.strip() == ""

    def test_handles_square_brackets_not_citations(self, parser):
        result = parser.parse("The product [name] is available.")
        # Non-numeric brackets should not be treated as citations
        assert result.citation_indices == []


class TestSourceExtractor:
    """Tests for extracting source document references from retrieval results."""

    @pytest.fixture
    def extractor(self):
        return SourceExtractor()

    def test_maps_citation_index_to_source(self, extractor):
        sources = [
            {"id": "doc1", "content": "Content A", "title": "Document 1"},
            {"id": "doc2", "content": "Content B", "title": "Document 2"},
        ]
        cited = extractor.extract_cited_sources([1, 2], sources)
        assert len(cited) == 2
        assert cited[0]["title"] == "Document 1"

    def test_ignores_out_of_range_citation(self, extractor):
        sources = [{"id": "doc1", "content": "C", "title": "D1"}]
        cited = extractor.extract_cited_sources([1, 5], sources)
        # Citation [5] doesn't exist — should be silently ignored
        assert len(cited) == 1

    def test_citation_index_is_one_based(self, extractor):
        sources = [{"id": "doc1", "content": "A", "title": "T1"},
                   {"id": "doc2", "content": "B", "title": "T2"}]
        cited = extractor.extract_cited_sources([1], sources)
        assert cited[0]["id"] == "doc1"  # [1] maps to index 0
```

---

### 2.5 Java Unit Testing with LangChain4j

```java
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.data.message.AiMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RAGServiceTest {

    @Mock
    private ChatLanguageModel mockLlm;

    @Mock
    private DocumentRetriever mockRetriever;

    private RAGService ragService;

    @BeforeEach
    void setUp() {
        ragService = new RAGService(mockLlm, mockRetriever);
    }

    @Test
    void shouldReturnAnswerWithCitation() {
        // Arrange
        var docs = List.of(
            new Document("doc1", "Enterprise customers get 90-day returns.")
        );
        when(mockRetriever.retrieve(anyString(), eq(5))).thenReturn(docs);
        when(mockLlm.generate(anyList())).thenReturn(
            Response.from(AiMessage.from("90 days [1]."))
        );

        // Act
        var result = ragService.query("Enterprise return policy?");

        // Assert
        assertThat(result.answer()).contains("90 days");
        assertThat(result.citedSourceIds()).contains("doc1");
        verify(mockRetriever).retrieve("Enterprise return policy?", 5);
    }

    @Test
    void shouldReturnRefusalWhenContextEmpty() {
        when(mockRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of());
        when(mockLlm.generate(anyList())).thenReturn(
            Response.from(AiMessage.from("I cannot find this information."))
        );

        var result = ragService.query("Unknown topic?");

        assertThat(result.isRefusal()).isTrue();
    }

    @Test
    void shouldRespectTopKConfiguration() {
        ragService = new RAGService(mockLlm, mockRetriever, RAGConfig.of().topK(8));
        when(mockRetriever.retrieve(anyString(), eq(8))).thenReturn(List.of());
        when(mockLlm.generate(anyList())).thenReturn(
            Response.from(AiMessage.from("Answer"))
        );

        ragService.query("Test query");

        verify(mockRetriever).retrieve(anyString(), eq(8));
    }

    @Test
    void shouldHandleRetrieverException() {
        when(mockRetriever.retrieve(anyString(), anyInt()))
            .thenThrow(new RuntimeException("Vector DB unavailable"));

        assertThatThrownBy(() -> ragService.query("Test"))
            .isInstanceOf(RAGServiceException.class)
            .hasMessageContaining("retrieval failed");
    }

    @Test
    void shouldNotCallLlmWhenRetrievalReturnsNoDocuments() {
        when(mockRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of());
        // Configure to short-circuit on empty retrieval
        ragService = new RAGService(mockLlm, mockRetriever,
            RAGConfig.of().skipLlmOnEmptyRetrieval(true));

        ragService.query("Unknown query");

        verifyNoInteractions(mockLlm);
    }
}
```

---

> ### 📋 Chapter Summary
>
> - LLM mocking has three patterns: **response factory** (queue specific responses), **scenario-based** (match query patterns), and **pytest patch** (patch at import level).
> - Prompt template tests verify variable rendering, validation of missing variables, message structure, and few-shot example ordering — all without an LLM.
> - Chunking and retrieval unit tests validate size constraints, overlap, content preservation, and metadata attachment — all with mock or in-memory vector stores.
> - Output parser tests must cover happy path, malformed inputs, edge cases, and parameterised refusal phrase detection.

---

> ### ❓ Comprehension Questions
>
> 1. A developer mocks the LLM to always return "The answer is 42." A test passes. Three months later the test still passes but the system is broken in production. What assumption in the test design caused this?
> 2. The `TestChunkingPipeline.test_overlap_present` test checks that some words from the end of chunk 0 appear in the beginning of chunk 1. Why is this assertion weaker than checking the exact overlap size, and when would you need the stronger assertion?
> 3. `TestAnswerParser.test_detects_refusal_phrases` uses `@pytest.mark.parametrize` with 5 phrases. The LLM generates a novel refusal not in this list: "Based on available context, I'm unable to provide this information." How would your parser handle this, and how would you extend the test?
> 4. The Java test `shouldNotCallLlmWhenRetrievalReturnsNoDocuments` verifies that `mockLlm` is not called. What production behaviour does this test guard against, and why is that behaviour valuable?
> 5. A prompt template test runs in 2ms. An integration test for the same feature runs in 8 seconds. Describe three different failure modes that the integration test catches that the prompt template test cannot.

---

## References

### Documentation
- [pytest Documentation](https://docs.pytest.org)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html) — Python standard library mocking.
- [Mockito](https://site.mockito.org) — Java mocking framework.
- [AssertJ](https://assertj.github.io/doc/) — Fluent Java assertions.
- [LangChain4j Testing](https://docs.langchain4j.dev/tutorials/testing) — Mock models for Java tests.

---

---
[« Back to testing Index](index.md) | [🏠 Home](../../index.md)