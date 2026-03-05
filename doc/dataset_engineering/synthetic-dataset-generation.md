## Chapter 3 — Synthetic Dataset Generation 🧪

### 3.1 Why Synthetic Data

Building high-quality datasets from human annotation is expensive, slow, and difficult to scale. Synthetic generation — using LLMs to create training and evaluation examples — addresses this constraint but introduces its own engineering challenges.

Synthetic data is valuable in four scenarios:

**Bootstrapping evaluation.** Before any production data exists, a synthetic evaluation set enables measurement from day one.

**Domain coverage expansion.** Human annotators cluster around common cases. LLMs can be instructed to generate rare, edge-case, and adversarial examples that humans tend to miss.

**Privacy preservation.** Synthetic data can replicate statistical properties of sensitive production data without containing real customer information.

**Scaling fine-tuning datasets.** High-quality fine-tuning data is scarce. Synthetic generation from a small set of expert-labelled examples can expand the dataset 10–100x.

The central risk is **distributional bias**: synthetic data reflects the generating LLM's biases, knowledge gaps, and stylistic tendencies. All synthetic datasets must be quality-filtered before use.

---

### 3.2 Generating QA Pairs from Documents

The most common synthetic generation task for RAG: given a corpus, generate `(question, answer, source_document)` triples.

```python
import json
import hashlib
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

client = OpenAI()

class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    source_document_id: str
    source_excerpt: str
    difficulty: str            # easy | medium | hard
    question_type: str         # factual | inferential | comparative | procedural
    requires_multi_hop: bool = False

def generate_qa_pairs(
    document: dict,
    n_pairs: int = 5,
    model: str = "gpt-4o"
) -> list[QAPair]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Generate exactly {n_pairs} question-answer pairs from the document.

Requirements:
- Questions must be answerable SOLELY from the provided text
- Include a mix of: factual, inferential, procedural, and comparative questions
- Vary difficulty: easy (direct lookup), medium (requires combining info), hard (inference)
- Identify the exact excerpt that supports each answer
- For hard questions, set requires_multi_hop: true when two sections must be combined

Return JSON:
{{
  "pairs": [
    {{
      "question": "...",
      "answer": "...",
      "source_excerpt": "exact quote from text",
      "difficulty": "easy|medium|hard",
      "question_type": "factual|inferential|comparative|procedural",
      "requires_multi_hop": false
    }}
  ]
}}"""
            },
            {"role": "user", "content": f"Document ID: {document['id']}\n\n{document['content']}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    data = json.loads(response.choices[0].message.content)
    pairs = []
    for pair in data.get("pairs", []):
        pair_id = hashlib.md5(f"{document['id']}_{pair['question']}".encode()).hexdigest()[:12]
        pairs.append(QAPair(
            id=pair_id,
            question=pair["question"],
            answer=pair["answer"],
            source_document_id=document["id"],
            source_excerpt=pair.get("source_excerpt", ""),
            difficulty=pair.get("difficulty", "medium"),
            question_type=pair.get("question_type", "factual"),
            requires_multi_hop=pair.get("requires_multi_hop", False)
        ))
    return pairs

def generate_eval_dataset(
    documents: list[dict],
    pairs_per_doc: int = 5,
    max_workers: int = 5
) -> list[QAPair]:
    all_pairs = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(generate_qa_pairs, doc, pairs_per_doc) for doc in documents]
        for future in futures:
            try:
                all_pairs.extend(future.result())
            except Exception as e:
                print(f"Generation failed: {e}")
    return all_pairs
```

**Java — QA generation with [LangChain4j](https://docs.langchain4j.dev):**
```java
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

interface QAPairGenerator {
    @SystemMessage("""
        Generate 5 question-answer pairs from the document.
        Mix difficulty (easy/medium/hard) and type (factual/inferential/procedural).
        Return JSON: {"pairs": [{"question":"...","answer":"...","difficulty":"...","question_type":"..."}]}
        """)
    @UserMessage("Document:\n{{document}}")
    String generate(String document);
}

QAPairGenerator generator = AiServices.builder(QAPairGenerator.class)
    .chatLanguageModel(model)
    .build();

String json = generator.generate(documentContent);
// Parse with Jackson or Gson
```

---

### 3.3 Persona-Driven Generation

Generating from a single prompt produces questions that reflect one implicit user type. Real query distributions are far more diverse. Persona-driven generation models different user types explicitly.

```python
PERSONAS = [
    {
        "name": "technical_expert",
        "description": "Senior engineer, precise technical terminology, asks about edge cases",
        "example_queries": [
            "What is the thread safety model for concurrent API calls?",
            "What happens to in-flight requests during rolling deployment?"
        ]
    },
    {
        "name": "business_user",
        "description": "Non-technical manager, plain language, focused on costs and outcomes",
        "example_queries": [
            "How much does it cost to process 10,000 documents?",
            "What happens if the service goes down?"
        ]
    },
    {
        "name": "new_employee",
        "description": "Recently joined, unfamiliar with domain, asks basic questions",
        "example_queries": ["What is a webhook?", "Where is the documentation?"]
    },
    {
        "name": "power_user",
        "description": "Advanced customer, knows the product deeply, asks about edge cases",
        "example_queries": [
            "Can I use custom embedding models?",
            "How do I override the default chunking strategy?"
        ]
    }
]

def generate_persona_qa(
    document: dict,
    persona: dict,
    n_pairs: int = 3,
    model: str = "gpt-4o-mini"
) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are simulating a {persona['name']}: {persona['description']}.

Example questions this persona would ask:
{chr(10).join('- ' + q for q in persona['example_queries'])}

Generate {n_pairs} questions this persona would ask about the document.
Questions must be answerable from the document.
Return JSON: {{"pairs": [{{"question": "...", "answer": "...", "persona": "{persona['name']}"}}]}}"""
            },
            {"role": "user", "content": document["content"]}
        ],
        response_format={"type": "json_object"},
        temperature=0.9
    )
    return json.loads(response.choices[0].message.content).get("pairs", [])

def generate_diverse_eval_dataset(
    documents: list[dict],
    pairs_per_persona: int = 2
) -> list[dict]:
    all_pairs = []
    for doc in documents:
        for persona in PERSONAS:
            pairs = generate_persona_qa(doc, persona, n_pairs=pairs_per_persona)
            all_pairs.extend(pairs)
    return all_pairs
```

---

### 3.4 Adversarial and Edge-Case Generation

A system that performs well on clean queries may fail on adversarial inputs. Deliberately generating difficult examples builds a more robust evaluation set.

```python
ADVERSARIAL_TYPES = {
    "unanswerable": "Generate a question that CANNOT be answered from the document.",
    "ambiguous": "Generate a question whose answer is ambiguous given only the document.",
    "contradictory_premise": "Generate a question containing a false premise about the content.",
    "multi_hop": "Generate a question requiring information from at least two sections.",
    "negation": "Generate a question phrased with negation ('what is NOT included...').",
    "numerical": "Generate a question requiring numerical calculation from the document.",
}

def generate_adversarial_pairs(
    document: dict,
    adversarial_types: list[str] = None,
    model: str = "gpt-4o"
) -> list[dict]:
    types_to_use = adversarial_types or list(ADVERSARIAL_TYPES.keys())
    pairs = []
    for adv_type in types_to_use:
        instruction = ADVERSARIAL_TYPES[adv_type]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"""{instruction}
Return JSON: {{"question": "...", "expected_behaviour": "...", "adversarial_type": "{adv_type}"}}
For unanswerable: expected_behaviour = "system should say it cannot find this information"."""
                },
                {"role": "user", "content": document["content"]}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        pair = json.loads(response.choices[0].message.content)
        pair["source_document_id"] = document["id"]
        pairs.append(pair)
    return pairs
```

---

### 3.5 Multi-Turn Conversation Datasets

For conversational systems, evaluation requires datasets that test context retention and follow-up handling.

```python
def generate_conversation(
    document: dict,
    n_turns: int = 4,
    model: str = "gpt-4o"
) -> dict:
    """Generate a realistic multi-turn conversation grounded in a document."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Generate a realistic {n_turns}-turn conversation between
a user and a knowledge assistant about the provided document.

Requirements:
- Turn 1: User asks a general opening question
- Turns 2–{n_turns}: Follow-up questions referencing prior context
  (use pronouns, "what about the other option?", "how does that compare?")
- Answers grounded in the document
- Include at least one clarification request

Return JSON:
{{
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "document_id": "{document['id']}"
}}"""
            },
            {"role": "user", "content": document["content"]}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    return json.loads(response.choices[0].message.content)
```

---

### 3.6 Quality Filtering for Synthetic Data

All synthetic data must be filtered before use. Common issues: unanswerable questions, hallucinated answers, duplicates, and malformed output.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SyntheticDataQualityFilter:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(embed_model_name)

    def filter(self, pairs: list[dict], documents_by_id: dict[str, str]) -> list[dict]:
        filtered = self._remove_malformed(pairs)
        filtered = self._remove_ungrounded(filtered, documents_by_id)
        filtered = self._remove_near_duplicates(filtered, threshold=0.92)
        return filtered

    def _remove_malformed(self, pairs: list[dict]) -> list[dict]:
        return [
            p for p in pairs
            if p.get("question") and p.get("answer")
            and len(p["question"]) >= 20
            and len(p["answer"]) >= 10
        ]

    def _remove_ungrounded(
        self, pairs: list[dict], documents_by_id: dict[str, str]
    ) -> list[dict]:
        """
        Filter answers with low embedding similarity to their source document.
        Low similarity indicates the answer was not grounded in the document text.
        """
        passing = []
        for pair in pairs:
            doc_text = documents_by_id.get(pair.get("source_document_id", ""), "")
            if not doc_text:
                continue
            a_emb = self.embed_model.encode(pair["answer"])
            d_emb = self.embed_model.encode(doc_text[:1000])
            sim = float(cosine_similarity([a_emb], [d_emb])[0][0])
            if sim >= 0.4:
                pair["grounding_score"] = sim
                passing.append(pair)
        return passing

    def _remove_near_duplicates(
        self, pairs: list[dict], threshold: float = 0.92
    ) -> list[dict]:
        if len(pairs) < 2:
            return pairs
        questions = [p["question"] for p in pairs]
        embeddings = self.embed_model.encode(questions)
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        keep = np.ones(len(pairs), dtype=bool)
        for i in range(len(pairs)):
            if keep[i]:
                keep[np.where((sim_matrix[i] >= threshold) & (np.arange(len(pairs)) > i))] = False
        return [p for p, k in zip(pairs, keep) if k]
```

---

### 🧪 Hands-on Lab: Build a RAG Evaluation Dataset

**Objective:** Generate, filter, and validate a synthetic evaluation dataset from a small document corpus.

**Prerequisites:** `openai`, `sentence-transformers`, `scikit-learn`, `pandas`

```python
import json, hashlib, pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

DOCUMENTS = [
    {
        "id": "doc_001", "title": "Refund Policy",
        "content": """Our standard refund policy allows returns within 30 days of purchase.
Items must be in original, unopened condition. Digital products are non-refundable once downloaded.
Enterprise customers have an extended 90-day return window.
Refunds are processed in 5–7 business days for card payments, 10–14 days for bank transfers."""
    },
    {
        "id": "doc_002", "title": "Subscription Plans",
        "content": """We offer Free, Professional ($150/mo), and Enterprise ($500/mo) tiers.
Free: 5 users, 10GB, community support. Professional: 25 users, 100GB, 24h email SLA.
Enterprise: unlimited users, 1TB, dedicated account manager, 4-hour SLA for critical issues.
Annual billing provides a 20% discount on all paid tiers."""
    },
    {
        "id": "doc_003", "title": "API Reference",
        "content": """The REST API uses OAuth 2.0. Tokens expire after 3600 seconds.
Rate limits: Free 100 req/min, Professional 1000 req/min, Enterprise 10000 req/min.
Current stable API version is v2. Version v1 is deprecated and retires December 31, 2025."""
    }
]

docs_by_id = {d["id"]: d["content"] for d in DOCUMENTS}

# ── Step 1: Generate ──────────────────────────────────────
print("Generating QA pairs...")
all_pairs = []
for doc in DOCUMENTS:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Generate 6 diverse QA pairs from the document.
Include: 2 easy factual, 2 medium inferential, 1 hard multi-step, 1 unanswerable.
For unanswerable: set answer to "NOT_IN_DOCUMENT" and answerable: false.
Return JSON: {"pairs": [{"question":"...","answer":"...","difficulty":"...","answerable":true}]}"""
            },
            {"role": "user", "content": f"Document {doc['id']}:\n{doc['content']}"}
        ],
        response_format={"type": "json_object"}, temperature=0.8
    )
    pairs = json.loads(response.choices[0].message.content).get("pairs", [])
    for p in pairs:
        p["source_document_id"] = doc["id"]
        p["id"] = hashlib.md5(p["question"].encode()).hexdigest()[:10]
    all_pairs.extend(pairs)
    print(f"  {doc['id']}: {len(pairs)} pairs generated")

print(f"\nTotal generated: {len(all_pairs)}")

# ── Step 2: Quality filtering ─────────────────────────────
valid = [p for p in all_pairs if len(p.get("question","")) >= 20 and len(p.get("answer","")) >= 5]
print(f"After malformed filter: {len(valid)}")

grounded = []
for p in valid:
    if not p.get("answerable", True):
        p["grounding_score"] = 0.0
        grounded.append(p)
        continue
    doc_text = docs_by_id.get(p["source_document_id"], "")
    a_emb = embed_model.encode(p["answer"])
    d_emb = embed_model.encode(doc_text[:800])
    sim = float(cosine_similarity([a_emb], [d_emb])[0][0])
    p["grounding_score"] = sim
    if sim >= 0.35:
        grounded.append(p)
print(f"After grounding filter: {len(grounded)}")

import numpy as np
questions = [p["question"] for p in grounded]
q_embs = embed_model.encode(questions)
sim_matrix = cosine_similarity(q_embs)
np.fill_diagonal(sim_matrix, 0)
keep = np.ones(len(grounded), dtype=bool)
for i in range(len(grounded)):
    if keep[i]:
        keep[(sim_matrix[i] >= 0.90) & (np.arange(len(grounded)) > i)] = False
final = [p for p, k in zip(grounded, keep) if k]
print(f"After dedup filter: {len(final)}")

# ── Step 3: Statistics ────────────────────────────────────
df = pd.DataFrame(final)
print(f"\n── Dataset Statistics ───────────────────────")
print(f"Total: {len(final)}")
print(f"Difficulty:\n{df['difficulty'].value_counts()}")
print(f"Avg grounding score: {df['grounding_score'].mean():.3f}")

# ── Step 4: Save and seal ─────────────────────────────────
output = "/tmp/eval_dataset_v1.jsonl"
with open(output, "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")

checksum = hashlib.sha256(open(output, "rb").read()).hexdigest()
open(output + ".sha256", "w").write(checksum)
print(f"\nSaved: {output}")
print(f"SHA256: {checksum[:16]}...")
```

**Extensions:**
- Add persona-driven generation for 4 personas × 3 questions × 3 documents
- Add adversarial examples (unanswerable, contradictory premises)
- Compare grounding score threshold 0.3 vs 0.5 and measure impact on dataset size
- Generate multi-turn conversations and verify context retention

---

> ### 📋 Chapter Summary
>
> - Synthetic data solves the bootstrapping problem — you can measure quality before any production data exists.
> - **QA pair generation** from documents is the core technique; diversity requires explicit difficulty and question type variation.
> - **Persona-driven generation** captures the real diversity of user query styles.
> - **Adversarial generation** exposes system weaknesses: unanswerable questions, false premises, multi-hop reasoning.
> - All synthetic data requires **quality filtering**: malformed removal, grounding checks, and deduplication.

---

> ### ❓ Comprehension Questions
>
> 1. A synthetic evaluation dataset has 90% easy factual questions. How would you modify the generation prompt to produce a balanced difficulty distribution?
> 2. Explain the grounding score filter. What does a low grounding score indicate about a generated answer?
> 3. A team uses GPT-4o to generate a test set and GPT-4o as the system under test. What evaluation bias does this introduce?
> 4. Persona-driven generation produces diverse queries. What is the risk of over-representing one persona?
> 5. Your system must handle unanswerable queries gracefully. How do you include them in an evaluation set and what metric measures this capability?

---

## References

### Papers
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Es et al., 2023. Includes synthetic dataset generation methodology.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al., 2022.
- [Generating Training Data with LLMs for Information Extraction](https://arxiv.org/abs/2205.09712) — Møller et al., 2022.

### Documentation
- [OpenAI Fine-tuning Data Preparation](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [RAGAS TestsetGenerator](https://docs.ragas.io/en/latest/getstarted/testset_generation.html)
- [LangSmith Dataset Management](https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application)

---

---
[« Back to dataset_engineering Index](index.md) | [🏠 Home](../../index.md)