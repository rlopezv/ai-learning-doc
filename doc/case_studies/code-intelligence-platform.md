## Chapter 4 — Code Intelligence Platform

### 4.1 Why Code RAG is Different

Code retrieval has fundamentally different requirements from document retrieval. Natural language and code differ in token distribution, semantic density, and query patterns.

```python
CODE_VS_PROSE_DIFFERENCES = {
    "token_density": {
        "prose":  "High semantic density per word; 100 words ≈ 1 concept",
        "code":   "Low semantic density; 100 tokens may be one function signature"
    },
    "query_patterns": {
        "prose":  "Natural language questions: 'What is the return policy?'",
        "code":   "Mix of NL + code: 'How to paginate with cursor?', 'show me usages of TokenBudget'"
    },
    "semantic_gap": {
        "prose":  "Query language matches document language",
        "code":   "NL query must bridge to code semantics — 'how to retry' → exponential_backoff_retry()"
    },
    "chunk_unit": {
        "prose":  "Paragraph or section",
        "code":   "Function, class, or file — syntax must not be broken"
    },
    "staleness": {
        "prose":  "Stale documents give outdated answers",
        "code":   "Stale code gives answers for deleted functions/APIs"
    },
    "cross_reference": {
        "prose":  "Documents reference each other by hyperlink",
        "code":   "Functions call other functions — call graph matters for context"
    }
}
```

---

### 4.2 Code-Aware Chunking and Indexing

```python
from dataclasses import dataclass
from typing import Optional
import ast, re

@dataclass
class CodeChunk:
    chunk_id: str
    content: str
    language: str
    chunk_type: str    # "function" | "class" | "file" | "docstring"
    name: str          # Function or class name
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    imports: list[str]
    calls: list[str]   # Functions/methods this chunk calls
    complexity: int    # Cyclomatic complexity (proxy for chunk difficulty)
    last_modified_commit: str

class PythonCodeChunker:
    """
    AST-based Python code chunker.
    Extracts functions and classes as semantic units rather than
    fixed-token windows — preserving syntactic integrity.
    """
    def chunk_file(self, file_path: str, content: str,
                   repo: str, commit_sha: str) -> list[CodeChunk]:
        chunks = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fall back to line-based chunking for unparseable files
            return self._line_chunk(file_path, content, repo, commit_sha)

        lines = content.split("\n")
        imports = self._extract_imports(tree)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunk_lines = lines[node.lineno - 1: node.end_lineno]
                chunk_text  = "\n".join(chunk_lines)
                docstring   = ast.get_docstring(node)
                calls       = self._extract_calls(node)

                # Enrich with docstring as searchable prefix
                searchable_text = ""
                if docstring:
                    searchable_text += f"# {docstring}\n"
                searchable_text += chunk_text

                chunks.append(CodeChunk(
                    chunk_id=f"{repo}/{file_path}:{node.name}",
                    content=searchable_text,
                    language="python",
                    chunk_type="class" if isinstance(node, ast.ClassDef)
                               else "function",
                    name=node.name,
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    docstring=docstring,
                    imports=imports,
                    calls=calls,
                    complexity=self._cyclomatic_complexity(node),
                    last_modified_commit=commit_sha
                ))
        return chunks

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _extract_calls(self, node: ast.AST) -> list[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
                elif isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
        return list(set(calls))

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """Count branches as proxy for complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For,
                                  ast.ExceptHandler, ast.With)):
                complexity += 1
        return complexity

    def _line_chunk(self, file_path: str, content: str,
                    repo: str, commit_sha: str) -> list[CodeChunk]:
        """Fallback: 60-line chunks with 10-line overlap."""
        lines = content.split("\n")
        chunks = []
        step = 50
        for i in range(0, len(lines), step):
            window = lines[i: i + 60]
            chunks.append(CodeChunk(
                chunk_id=f"{repo}/{file_path}:L{i+1}",
                content="\n".join(window),
                language="unknown",
                chunk_type="file",
                name=f"lines_{i+1}_{i+60}",
                file_path=file_path,
                start_line=i + 1, end_line=min(i + 60, len(lines)),
                docstring=None, imports=[], calls=[], complexity=0,
                last_modified_commit=commit_sha
            ))
        return chunks
```

---

### 4.3 Query Patterns and Retrieval Strategies

```python
from enum import Enum

class CodeQueryType(str, Enum):
    HOW_TO      = "how_to"        # "How do I implement X?"
    FIND_USAGE  = "find_usage"    # "Where is TokenBudget used?"
    EXPLAIN     = "explain"       # "What does this function do?"
    DEBUG       = "debug"         # "Why does X fail when Y?"
    REFACTOR    = "refactor"      # "How should I refactor X?"

class CodeRAGPipeline:
    """
    Code-aware RAG with query type detection and hybrid retrieval.
    Uses both semantic similarity AND keyword/identifier matching.
    """
    def __init__(self, dense_retriever, lexical_retriever, llm_gateway):
        self.dense   = dense_retriever
        self.lexical = lexical_retriever   # BM25 on identifiers + docstrings
        self.gateway = llm_gateway

    def query(self, question: str, language: Optional[str] = None,
              repo: Optional[str] = None) -> dict:
        query_type = self._classify_query(question)
        results    = self._retrieve(question, query_type,
                                    language=language, repo=repo)

        # System prompt varies by query type
        system_prompts = {
            CodeQueryType.HOW_TO:     "Provide a concise code example with explanation. "
                                      "Prefer patterns from the retrieved code.",
            CodeQueryType.FIND_USAGE: "List the locations and describe how the identifier is used.",
            CodeQueryType.EXPLAIN:    "Explain what the code does step by step. "
                                      "Include parameter descriptions.",
            CodeQueryType.DEBUG:      "Identify the likely root cause. Suggest a fix with code.",
            CodeQueryType.REFACTOR:   "Suggest refactoring approach with before/after example.",
        }

        context = "\n\n".join(
            f"// File: {r['metadata']['file_path']} "
            f"(function: {r['metadata'].get('name', '?')})\n{r['content']}"
            for r in results
        )
        messages = [
            {"role": "system",
             "content": system_prompts.get(query_type,
                                           "Answer the coding question using the provided context.")},
            {"role": "user",
             "content": f"Codebase context:\n```\n{context}\n```\n\nQuestion: {question}"}
        ]
        response = self.gateway.complete(messages, model_alias="default")
        return {
            "answer":     response["answer"],
            "query_type": query_type.value,
            "sources":    [{"id": r["id"], "file": r["metadata"]["file_path"],
                            "name": r["metadata"].get("name")} for r in results]
        }

    def _classify_query(self, question: str) -> CodeQueryType:
        q = question.lower()
        if any(w in q for w in ["how do", "how to", "example", "implement", "write"]):
            return CodeQueryType.HOW_TO
        if any(w in q for w in ["where", "used", "usages", "references", "calls"]):
            return CodeQueryType.FIND_USAGE
        if any(w in q for w in ["explain", "what does", "what is", "describe"]):
            return CodeQueryType.EXPLAIN
        if any(w in q for w in ["error", "bug", "fail", "exception", "why"]):
            return CodeQueryType.DEBUG
        if any(w in q for w in ["refactor", "improve", "better", "clean"]):
            return CodeQueryType.REFACTOR
        return CodeQueryType.HOW_TO

    def _retrieve(self, question: str, query_type: CodeQueryType,
                  language: Optional[str], repo: Optional[str]) -> list:
        filters = {}
        if language:
            filters["language"] = language
        if repo:
            filters["repo"] = repo

        dense_results  = self.dense.search(question, filter=filters, top_k=8)
        lexical_results = self.lexical.search(
            self._extract_identifiers(question), filter=filters, top_k=8
        )
        return self._rrf(dense_results, lexical_results)[:6]

    def _extract_identifiers(self, text: str) -> str:
        """Extract CamelCase and snake_case identifiers for lexical search."""
        identifiers = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b|\b[a-z]+_[a-z_]+\b', text)
        return " ".join(identifiers) if identifiers else text

    def _rrf(self, a: list, b: list, k: int = 60) -> list:
        scores: dict = {}
        for rank, doc in enumerate(a):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(b):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank + 1)
        all_docs = {d["id"]: d for d in a + b}
        return sorted([{"id": did, **all_docs[did]} for did in scores if did in all_docs],
                      key=lambda x: -scores[x["id"]])
```

---

### 4.4 IDE Integration and Response Formatting

```java
// VS Code extension: Language Server Protocol integration
// Sends code context alongside the natural language question

@RestController
@RequestMapping("/v1/code-intelligence")
public class CodeIntelligenceController {

    private final CodeRAGService ragService;
    private final AuditService auditService;

    @PostMapping("/query")
    public ResponseEntity<CodeQueryResponse> query(
            @RequestBody CodeQueryRequest request,
            @AuthenticationPrincipal JwtAuthenticationToken auth) {

        // Enrich question with IDE context
        String enrichedQuestion = enrichWithContext(
            request.getQuestion(),
            request.getActiveFileContent(),
            request.getSelectedText(),
            request.getCursorPosition()
        );

        CodeRAGResult result = ragService.query(
            enrichedQuestion,
            request.getLanguage(),
            request.getRepo()
        );

        auditService.logCodeQuery(auth.getName(), request.getQuestion(),
                                   result.getSources());

        return ResponseEntity.ok(CodeQueryResponse.builder()
            .answer(result.getAnswer())
            .sources(result.getSources())
            .queryType(result.getQueryType())
            .codeSnippets(extractCodeBlocks(result.getAnswer()))
            .build());
    }

    private String enrichWithContext(String question, String fileContent,
                                     String selectedText, int cursorPos) {
        StringBuilder ctx = new StringBuilder();
        if (selectedText != null && !selectedText.isBlank()) {
            ctx.append("Selected code:\n```\n").append(selectedText).append("\n```\n\n");
        }
        if (fileContent != null && fileContent.length() < 3000) {
            ctx.append("Current file context:\n```\n").append(fileContent).append("\n```\n\n");
        }
        return ctx + question;
    }

    private List<String> extractCodeBlocks(String markdown) {
        List<String> blocks = new ArrayList<>();
        Pattern p = Pattern.compile("```[a-z]*\\n([^`]+)```", Pattern.DOTALL);
        Matcher m = p.matcher(markdown);
        while (m.find()) blocks.add(m.group(1).trim());
        return blocks;
    }
}
```

---

> ### 📋 Chapter Summary
>
> - Code RAG requires **AST-based chunking** (function/class boundaries, not fixed token windows) to preserve syntactic integrity and enable identifier-level retrieval.
> - **Hybrid retrieval** is even more important for code: dense retrieval handles semantic queries ("how to retry"), while lexical retrieval handles identifier lookups ("where is `TokenBudget` used").
> - Query type classification (HOW_TO, FIND_USAGE, EXPLAIN, DEBUG, REFACTOR) allows system prompt specialisation — a debug query needs different framing than a how-to query.
> - IDE integration enriches questions with file context and selected text — reducing the semantic gap between the developer's intent and the retrieval query.

---

> ### ❓ Comprehension Questions
>
> 1. `PythonCodeChunker` uses AST parsing and falls back to line-based chunking for syntax errors. A codebase has 8% of files with syntax errors (generated code, partial migrations). How would you detect and handle each category differently?
> 2. A function has 0 lines of docstring and a highly generic name: `process()`. The AST chunker extracts it as a chunk, but dense retrieval never returns it because the embedding is semantically empty. How would you improve retrieval for undocumented functions?
> 3. `_extract_identifiers` extracts CamelCase and snake_case patterns. A developer asks: "How do I use the `v2_get_customer_records_by_date_range` function?" The function name has underscores and numbers. Does the regex `r'\b[a-z]+_[a-z_]+\b'` match this identifier? Fix the regex if not.
> 4. The IDE integration sends `fileContent` only if it is under 3,000 characters. A developer is working in a 500-line file (approximately 12,000 characters). The file context is dropped. How would you select a relevant window around the cursor position to include within the 3,000-character limit?
> 5. Code staleness is critical: a function retrieved from a 2-year-old commit may no longer exist. How would you implement a staleness signal in the retrieval pipeline that degrades the ranking of chunks from old commits without removing them entirely?

---

---
[« Back to case_studies Index](index.md) | [🏠 Home](../../index.md)