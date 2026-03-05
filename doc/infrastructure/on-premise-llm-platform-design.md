## Chapter 4 — On-Premise LLM Platform Design 🔓

### 4.1 When to Go On-Premise

On-premise LLM deployment is warranted when one or more of the following hold:

**Data residency.** EU GDPR, HIPAA, banking regulations, or classified environments prohibit sending data to external providers. All inference must run on organisation-controlled infrastructure.

**Air-gapped environments.** Critical infrastructure and secure research environments have no internet connectivity. The entire model serving stack must run internally.

**Cost at scale.** For organisations running > 10M tokens/day continuously, on-premise hardware can be significantly cheaper than pay-per-token cloud APIs despite higher operational complexity.

**Latency requirements.** Some applications require sub-50ms P99 inference. A local serving stack eliminates the network round-trip to a cloud provider.

---

### 4.2 Ollama for Development and Small Workloads

[Ollama](https://ollama.com/docs) provides the simplest on-premise serving with an OpenAI-compatible API.

```bash
# Install and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text   # Embedding model

# Start server (default: localhost:11434)
ollama serve
```

```python
# Drop-in replacement for OpenAI client — same API surface
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # Required by SDK, not validated by Ollama
)

# LLM completion
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is RAG?"}],
    temperature=0
)
print(response.choices[0].message.content)

# Embeddings
embed = client.embeddings.create(
    model="nomic-embed-text",
    input="Document content"
)
print(f"Dimensions: {len(embed.data[0].embedding)}")
```

```
# Ollama resource requirements by model size
Model       Quantisation   RAM required   Notes
─────────   ────────────   ────────────   ─────────────────────────────
3B params   Q4_K_M         ~2 GB          Fast, limited capability
7B params   Q4_K_M         ~4 GB          Good quality, runs on laptop
13B params  Q4_K_M         ~8 GB          Better quality, 16GB system
70B params  Q4_K_M         ~40 GB         Near-GPT-4 quality, needs GPU
```

---

### 4.3 vLLM for High-Throughput Serving

[vLLM](https://docs.vllm.ai) provides production-grade serving with PagedAttention — GPU memory managed as virtual memory pages, dramatically increasing concurrent request throughput.

```bash
pip install vllm

# Serve Mistral 7B on GPU
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90
```

```yaml
# kubernetes/vllm/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-mistral-7b
  namespace: ai-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-mistral
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-a100
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.6.4
          args:
            - "--model"
            - "mistralai/Mistral-7B-Instruct-v0.3"
            - "--max-model-len"
            - "8192"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "40Gi"
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-weights-pvc
```

```python
# vLLM uses the same OpenAI client — change only base_url
from openai import OpenAI

vllm_client = OpenAI(
    base_url="http://vllm-mistral.ai-platform.svc:8000/v1",
    api_key="vllm-local"
)
response = vllm_client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Summarise the key points."}],
    temperature=0, max_tokens=500
)
```

---

### 4.4 On-Premise Infrastructure Stack

```
On-Premise AI Platform (single data centre)

┌───────────────────────────────────────────────────────┐
│  Load Balancer (Nginx / HAProxy)                       │
├───────────────────────────────────────────────────────┤
│  API Layer (FastAPI / Spring Boot)                     │
│  ├── LLM Gateway  → vLLM or Ollama                    │
│  ├── Prompt Service                                    │
│  └── Embedding Service → local SentenceTransformers   │
├───────────────────────────────────────────────────────┤
│  Compute                                               │
│  ├── CPU nodes: API, ingestion, embeddings             │
│  └── GPU nodes: vLLM serving (optional)               │
├───────────────────────────────────────────────────────┤
│  Data                                                  │
│  ├── Qdrant (vector DB) — 3-node StatefulSet           │
│  ├── Redis (Celery) — HA cluster                       │
│  ├── PostgreSQL (metadata, prompts) — primary+standby  │
│  └── MinIO (object store: models, datasets, snapshots) │
├───────────────────────────────────────────────────────┤
│  Secrets & Config                                      │
│  └── HashiCorp Vault + Consul                         │
└───────────────────────────────────────────────────────┘
```

```python
# On-premise gateway config: all traffic goes to local models
from part_11_platform_engineering import ModelAliasConfig, LLMProvider

ON_PREMISE_ALIASES = {
    "default":   ModelAliasConfig("default",  LLMProvider.OLLAMA, "llama3.2",
                                   cost_per_1m_input=0.0, cost_per_1m_output=0.0),
    "powerful":  ModelAliasConfig("powerful", LLMProvider.OLLAMA, "mistral",
                                   cost_per_1m_input=0.0, cost_per_1m_output=0.0),
    "fast":      ModelAliasConfig("fast",     LLMProvider.OLLAMA, "llama3.2:3b",
                                   cost_per_1m_input=0.0, cost_per_1m_output=0.0),
}
# Note: cost=0.0 is misleading — track GPU/CPU hours instead
```

---

> ### 📋 Chapter Summary
>
> - On-premise is warranted for data residency, air-gapped environments, cost at scale (>10M tokens/day), and strict latency requirements.
> - **Ollama** (development/small workloads) and **vLLM** (production GPU serving) both expose an OpenAI-compatible API — product code changes only `base_url`.
> - PagedAttention in vLLM manages GPU KV-cache as virtual memory pages, enabling far higher concurrent request throughput than naive memory management.
> - On-premise "cost" is not zero — GPU/CPU hours, hardware amortisation, and operational overhead must be tracked.

---

> ### ❓ Comprehension Questions
>
> 1. A financial services firm needs GPT-4o quality but cannot send data to OpenAI. What on-premise model family would you evaluate first, and what evaluation methodology would confirm quality parity?
> 2. vLLM's PagedAttention manages GPU KV-cache like virtual memory pages. Why does this increase throughput for concurrent requests compared to pre-allocating fixed-size KV buffers per request?
> 3. An Ollama server with 16GB RAM serves `llama3:8b` (Q4_K_M ~4.5GB). Three concurrent requests arrive simultaneously. What happens to memory and latency versus a single request?
> 4. The on-premise gateway sets `cost_per_1m_input=0.0`. What real costs should be tracked instead, and how would you modify `CostTracker` to capture compute hours?
> 5. A regulated firm requires model weights never leave their data centre. vLLM downloads from Hugging Face at startup. Design the model weight distribution workflow satisfying this requirement.

---

## References

### Documentation
- [Ollama Documentation](https://ollama.com/docs)
- [vLLM Documentation](https://docs.vllm.ai)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)
- [MinIO Documentation](https://min.io/docs/minio/container/index.html)
- [HashiCorp Vault](https://developer.hashicorp.com/vault/docs)

### Papers
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023.

---

---
[« Back to infrastructure Index](index.md) | [🏠 Home](../../index.md)