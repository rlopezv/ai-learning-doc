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

[« Back to sdlc Index](index.md) | [🏠 Home](../../index.md)
