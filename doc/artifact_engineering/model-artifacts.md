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

---
[« Back to artifact_engineering Index](index.md) | [🏠 Home](../../index.md)