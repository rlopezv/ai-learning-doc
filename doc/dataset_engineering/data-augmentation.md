## Chapter 5 — Data Augmentation

### 5.1 When Augmentation Is Needed

Data augmentation artificially expands a dataset by generating variants of existing examples. In LLM systems engineering, augmentation addresses three problems:

**Class imbalance in evaluation sets.** A generated evaluation set may have 90% factual questions and 10% inferential. Augmenting underrepresented categories creates a balanced distribution.

**Vocabulary coverage gaps.** A system trained on formal English may fail on informal queries. Augmenting the test set with informal paraphrases exposes this gap before production.

**Robustness testing.** A model may perform well on clean inputs but fail with typos, mixed languages, or noisy formatting. Augmenting with noisy variants tests this robustness.

---

### 5.2 Paraphrase Augmentation

Generate semantically equivalent variants of existing examples to expand coverage and test consistency.

```python
def paraphrase_question(
    question: str,
    n_variants: int = 3,
    model: str = "gpt-4o-mini"
) -> list[str]:
    """Generate paraphrase variants in different registers."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Generate {n_variants} paraphrase variants of the question.
Vary the register: formal, informal, and technical.
Each variant must have the same information need as the original.
Return JSON: {{"variants": ["variant1", "variant2", ...]}}"""
            },
            {"role": "user", "content": f"Original: {question}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    return json.loads(response.choices[0].message.content).get("variants", [])

def augment_with_paraphrases(
    records: list[dict],
    n_variants: int = 2
) -> list[dict]:
    """Variants inherit the original answer — valid only for factual answers."""
    augmented = []
    for record in records:
        variants = paraphrase_question(record["question"], n_variants=n_variants)
        for i, variant in enumerate(variants):
            augmented.append({
                **record,
                "id": f"{record['id']}_aug{i}",
                "question": variant,
                "is_augmented": True,
                "original_id": record["id"],
                "augmentation_type": "paraphrase"
            })
    return augmented
```

---

### 5.3 Back-Translation

Back-translation generates paraphrases by translating to an intermediate language and back. It produces naturally varied phrasing via translation divergence.

```python
def back_translate(
    text: str,
    intermediate_language: str = "Spanish",
    model: str = "gpt-4o-mini"
) -> str:
    """Translate to intermediate language, then back to English."""
    # Step 1: to intermediate
    intermediate = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Translate to {intermediate_language}. Return only the translation."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()

    # Step 2: back to English
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Translate to English. Return only the translation."},
            {"role": "user", "content": intermediate}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()

def augment_with_back_translation(
    records: list[dict],
    languages: list[str] = ("Spanish", "French", "German"),
    field: str = "question"
) -> list[dict]:
    augmented = []
    for record in records:
        for lang in languages:
            bt = back_translate(record[field], intermediate_language=lang)
            augmented.append({
                **record,
                "id": f"{record['id']}_bt_{lang[:2].lower()}",
                field: bt,
                "is_augmented": True,
                "original_id": record["id"],
                "augmentation_type": f"back_translation_{lang}"
            })
    return augmented
```

---

### 5.4 Noise Injection for Robustness

Deliberately introduce realistic noise to test system robustness.

```python
import random, string, re

class NoiseInjector:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def add_typos(self, text: str, error_rate: float = 0.05) -> str:
        result = list(text)
        for i in range(len(result)):
            if random.random() < error_rate and result[i].isalpha():
                op = random.choice(["substitute", "delete", "insert", "swap"])
                if op == "substitute":
                    result[i] = random.choice(string.ascii_lowercase)
                elif op == "delete":
                    result[i] = ""
                elif op == "insert":
                    result[i] = result[i] + random.choice(string.ascii_lowercase)
                elif op == "swap" and i < len(result) - 1:
                    result[i], result[i+1] = result[i+1], result[i]
        return "".join(result)

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

    def add_whitespace_noise(self, text: str) -> str:
        return re.sub(r' ', lambda m: '  ' if random.random() < 0.1 else ' ', text)

def create_robustness_test_set(
    records: list[dict],
    noise_types: list[str] = None
) -> list[dict]:
    noise_types = noise_types or ["typos", "lowercase", "no_punctuation"]
    injector = NoiseInjector()
    fns = {
        "typos":          lambda t: injector.add_typos(t, 0.08),
        "lowercase":      injector.lowercase,
        "no_punctuation": injector.remove_punctuation,
        "whitespace":     injector.add_whitespace_noise,
    }
    augmented = []
    for record in records:
        for noise_type in noise_types:
            fn = fns.get(noise_type)
            if fn:
                augmented.append({
                    **record,
                    "id": f"{record['id']}_noise_{noise_type}",
                    "question": fn(record["question"]),
                    "is_augmented": True,
                    "augmentation_type": f"noise_{noise_type}"
                })
    return augmented
```

---

### 5.5 Augmentation Pipelines in Production

Augmentation is a pipeline stage — reproducible, configurable, and bounded.

```python
from dataclasses import dataclass
from enum import Enum

class AugmentationType(str, Enum):
    PARAPHRASE = "paraphrase"
    BACK_TRANSLATION = "back_translation"
    NOISE = "noise"

@dataclass
class AugmentationConfig:
    augmentation_types: list[AugmentationType]
    paraphrase_n_variants: int = 2
    back_translation_languages: list[str] = None
    noise_types: list[str] = None
    max_augmentation_ratio: float = 3.0  # Never grow beyond 3x original

    def __post_init__(self):
        if self.back_translation_languages is None:
            self.back_translation_languages = ["Spanish", "French"]
        if self.noise_types is None:
            self.noise_types = ["typos", "lowercase"]

class AugmentationPipeline:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def run(self, records: list[dict]) -> dict:
        original_count = len(records)
        all_records = list(records)
        max_additions = int(original_count * (self.config.max_augmentation_ratio - 1))
        total_added = 0
        report = {"original": original_count, "counts": {}}

        if AugmentationType.PARAPHRASE in self.config.augmentation_types:
            added = augment_with_paraphrases(records, self.config.paraphrase_n_variants)
            added = added[:max_additions - total_added]
            all_records.extend(added)
            total_added += len(added)
            report["counts"]["paraphrase"] = len(added)

        if AugmentationType.NOISE in self.config.augmentation_types:
            added = create_robustness_test_set(records, self.config.noise_types)
            added = added[:max_additions - total_added]
            all_records.extend(added)
            total_added += len(added)
            report["counts"]["noise"] = len(added)

        report["final_count"] = len(all_records)
        report["augmentation_ratio"] = round(len(all_records) / original_count, 2)
        return {"records": all_records, "report": report}
```

---

> ### 📋 Chapter Summary
>
> - Augmentation expands datasets to cover vocabulary gaps, class imbalance, and robustness test cases.
> - **Paraphrase augmentation** generates stylistically varied versions of existing queries.
> - **Back-translation** produces naturally varied paraphrases via translation divergence.
> - **Noise injection** tests robustness to typos, casing, and punctuation.
> - Augmentation pipelines must be **configurable, reproducible, and bounded** via `max_augmentation_ratio`.

---

> ### ❓ Comprehension Questions
>
> 1. A RAG system performs well on clean test queries but poorly on mobile queries with typos. What augmentation strategy would you apply?
> 2. Back-translation through Spanish produces different paraphrases than through French. Why, and what does this imply about using multiple intermediate languages?
> 3. An augmentation pipeline produces a 10x expansion. What problems arise from excessive augmentation?
> 4. Paraphrase augmentation reuses the original answer. Under what conditions is this assumption incorrect?
> 5. Design an augmentation pipeline for a multilingual RAG system supporting English, Spanish, and French.

---

## References

### Papers
- [EDA: Easy Data Augmentation for Text Classification](https://arxiv.org/abs/1901.11196) — Wei & Zou, 2019.
- [Back-Translation as Data Augmentation for NMT](https://arxiv.org/abs/1511.06709) — Sennrich et al., 2016.
- [Data Augmentation Approaches in NLP: A Survey](https://arxiv.org/abs/2110.01852) — Feng et al., 2021.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al., 2022.

### Documentation
- [Hugging Face Datasets Processing](https://huggingface.co/docs/datasets/process) — Dataset transformation.
- [nlpaug](https://github.com/makcedward/nlpaug) — Text augmentation library.
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning) — Data preparation including augmentation.
- [LangChain4j Documentation](https://docs.langchain4j.dev) — Java dataset handling.

---

> **Navigation**
> [← Part IV — Advanced RAG](../advanced_rag/index.md) | [→ Part VI — Artifact Engineering](../artifact_engineering/index.md)

---
[« Back to dataset_engineering Index](index.md) | [🏠 Home](../index.md)