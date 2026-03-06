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

---
[« Back to artifact-engineering Index](index.md) | [🏠 Home](../index.md)