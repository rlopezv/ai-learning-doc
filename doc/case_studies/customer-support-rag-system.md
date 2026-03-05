## Chapter 1 — Customer Support RAG System

### 1.1 Requirements and Constraints

A B2B SaaS company with 4,000 enterprise customers deployed a RAG-powered support assistant to handle Tier-1 questions before human escalation. The system had to satisfy conflicting requirements: high accuracy (wrong answers escalate to senior support engineers), low latency (<3s P99), and cost efficiency ($50/day budget for LLM calls).

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SupportSystemRequirements:
    # Traffic
    peak_qps: float = 8.0              # Monday morning surge
    daily_queries: int = 15_000
    concurrent_sessions: int = 200

    # Quality
    accuracy_target: float = 0.88      # Correct answers (human-evaluated)
    refusal_rate_max: float = 0.15     # Max 15% "I don't know" responses
    hallucination_rate_max: float = 0.02  # Hard limit on factual errors

    # Latency
    p50_ms: float = 1200
    p99_ms: float = 3000

    # Cost
    daily_llm_budget_usd: float = 50.0
    cost_per_query_max_usd: float = 0.005  # $5/1000 queries

    # Corpus
    product_articles: int = 8_400
    api_docs_pages: int = 2_100
    changelog_entries: int = 640
    update_frequency: str = "daily"

    # Compliance
    pii_in_tickets: bool = True        # Customer queries may contain PII
    data_residency: str = "EU"         # GDPR requirement

    def validate(self):
        max_cost_at_peak = self.daily_queries * self.cost_per_query_max_usd
        assert max_cost_at_peak <= self.daily_llm_budget_usd, (
            f"Budget conflict: {max_cost_at_peak} > {self.daily_llm_budget_usd}"
        )
        print(f"  Max daily cost at target: ${max_cost_at_peak:.2f} (budget: ${self.daily_llm_budget_usd:.2f}) ✓")

req = SupportSystemRequirements()
req.validate()
```

---

### 1.2 Architecture Decisions

Key decisions made during design, with the rationale that would not be obvious from code alone:

**Decision 1: GPT-4o-mini over GPT-4o.** At $0.003/query (budget was $0.005), GPT-4o-mini delivers 89% accuracy vs GPT-4o's 93% on the support domain evaluation set. The 4-point accuracy delta was acceptable given the 5× cost difference. Human escalation handles the remaining hard cases.

**Decision 2: Hybrid retrieval (BM25 + dense) for product terminology.** Product feature names ("WorkflowAutomator", "DataSync Pro") are exact-match tokens that dense retrieval scores poorly. BM25 component boosts recall for exact product names by 18 percentage points.

**Decision 3: Semantic cache with 0.88 threshold.** Support questions are highly repetitive (top 100 questions account for 65% of volume). A lower threshold (0.88 vs standard 0.92) was chosen after A/B testing showed negligible quality degradation and 32% cache hit rate.

**Decision 4: EU-only data residency via Azure West Europe.** OpenAI's EU data residency offering routes processing within the EU, satisfying GDPR Art. 46. Corpus and logs stored in Azure West Europe.

```python
SUPPORT_ADR_SUMMARY = {
    "llm":         ("gpt-4o-mini",         "5× cheaper, 4-point quality gap acceptable"),
    "retrieval":   ("hybrid BM25+dense",   "+18% recall on product terminology"),
    "cache_thr":   (0.88,                  "32% hit rate, negligible quality degradation"),
    "residency":   ("Azure West Europe",   "GDPR Art. 46 compliance"),
    "chunking":    ("512 tokens, 50 overlap", "Preserves procedure steps in single chunk"),
    "reranker":    ("cohere-rerank-v3",    "+6% faithfulness at +15ms latency"),
}
```

---

### 1.3 Implementation Highlights

```python
from dataclasses import dataclass, field
from typing import Optional

class SupportRAGPipeline:
    """
    Customer support RAG with hybrid retrieval, semantic cache,
    PII redaction, and escalation detection.
    """
    def __init__(self, dense_retriever, bm25_retriever, reranker,
                 cache, pii_detector, llm_gateway, escalation_classifier):
        self.dense      = dense_retriever
        self.bm25       = bm25_retriever
        self.reranker   = reranker
        self.cache      = cache
        self.pii        = pii_detector
        self.gateway    = llm_gateway
        self.escalation = escalation_classifier

    def query(self, question: str, customer_tier: str = "standard") -> dict:
        # 1. PII redaction before any logging or caching
        pii_result = self.pii.scan(question)
        clean_q    = pii_result.redacted_text

        # 2. Semantic cache lookup (on redacted query)
        cached = self.cache.lookup(clean_q, threshold=0.88)
        if cached:
            return {**cached, "from_cache": True}

        # 3. Hybrid retrieval: dense + BM25, RRF fusion
        dense_docs = self.dense.search(clean_q, top_k=10)
        bm25_docs  = self.bm25.search(clean_q, top_k=10)
        fused      = self._rrf_fusion(dense_docs, bm25_docs)

        # 4. Rerank top-20 to top-5
        reranked = self.reranker.rerank(clean_q, fused[:20])[:5]

        # 5. Check if escalation is needed before generation
        needs_escalation = self.escalation.should_escalate(
            question=clean_q,
            retrieved_scores=[r["score"] for r in reranked],
            customer_tier=customer_tier
        )
        if needs_escalation:
            return {
                "answer": "This question requires specialist support. "
                          "I'm connecting you to a senior engineer.",
                "escalated": True,
                "from_cache": False
            }

        # 6. Generate with citation requirement
        model = "gpt-4o" if customer_tier == "enterprise" else "gpt-4o-mini"
        context = "\n".join(f"[{i+1}] {r['content']}" for i, r in enumerate(reranked))
        messages = [
            {"role": "system", "content":
             "You are a support assistant. Answer using ONLY the provided context. "
             "Cite every claim with [N]. If you cannot answer, say: "
             "'I need to escalate this to our support team.' Do NOT guess."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {clean_q}"}
        ]
        response = self.gateway.complete(messages, model_alias=model)

        result = {
            "answer":    response["answer"],
            "sources":   [r["id"] for r in reranked],
            "model":     model,
            "escalated": False,
            "from_cache": False
        }
        # Store in cache for future similar questions
        self.cache.store(clean_q, result)
        return result

    def _rrf_fusion(self, dense: list, bm25: list, k: int = 60) -> list:
        """Reciprocal Rank Fusion: combines dense and BM25 rankings."""
        scores: dict = {}
        for rank, doc in enumerate(dense):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(bm25):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank + 1)
        all_docs = {d["id"]: d for d in dense + bm25}
        return sorted(
            [{"id": did, **all_docs[did], "rrf_score": score}
             for did, score in scores.items() if did in all_docs],
            key=lambda x: -x["rrf_score"]
        )

class EscalationClassifier:
    """Determines whether a query should bypass the LLM and go to human support."""
    ESCALATION_TRIGGERS = [
        "billing dispute", "account suspension", "data breach",
        "legal", "GDPR", "refund", "contract termination",
        "outage", "data loss", "security incident"
    ]
    def should_escalate(self, question: str, retrieved_scores: list,
                        customer_tier: str) -> bool:
        q_lower = question.lower()
        for trigger in self.ESCALATION_TRIGGERS:
            if trigger in q_lower:
                return True
        # Low retrieval confidence → escalate
        if retrieved_scores and max(retrieved_scores) < 0.55:
            return True
        # Enterprise customers get more aggressive escalation
        if customer_tier == "enterprise" and max(retrieved_scores, default=1) < 0.70:
            return True
        return False
```

---

### 1.4 Production Results and Lessons Learned

```python
PRODUCTION_RESULTS = {
    "timeline": {
        "design_to_mvp":     "6 weeks",
        "mvp_to_production": "4 weeks",
        "stabilisation":     "3 weeks",
    },
    "quality_metrics_at_30_days": {
        "accuracy":          0.87,   # Target: 0.88 (1pt below — acceptable)
        "refusal_rate":      0.12,   # Target: ≤0.15 ✓
        "hallucination_rate": 0.009, # Target: ≤0.02 ✓
        "cache_hit_rate":    0.31,   # Target: >0.20 ✓
        "escalation_rate":   0.18,   # 18% of queries escalated to humans
    },
    "performance_metrics": {
        "p50_latency_ms": 1050,   # Target: 1200 ✓
        "p99_latency_ms": 2800,   # Target: 3000 ✓
        "availability":   0.9992,
    },
    "cost_metrics": {
        "daily_llm_cost_usd":    38.20,  # Under $50 budget ✓
        "cost_per_query_usd":    0.00255,
        "cache_savings_usd_day": 12.50,  # Semantic cache saves 32% of LLM calls
    },
    "business_impact": {
        "tier1_deflection_rate": 0.68,   # 68% of Tier-1 questions answered without human
        "avg_resolution_time_reduction": "4.2 minutes per ticket",
        "support_team_capacity_freed":   "2.1 FTE equivalent",
    }
}

LESSONS_LEARNED = [
    "Hybrid retrieval is non-negotiable for product terminology — pure dense search "
    "misses exact product names by large margins.",

    "Escalation logic should be conservative: a wrong escalation costs 5 minutes of "
    "engineer time; a wrong answer costs customer trust.",

    "PII redaction before caching prevents customer email addresses from appearing "
    "as cache keys visible to other tenants.",

    "The reranker improved faithfulness by 6 points but added 15ms P99 latency. "
    "Worth it: the accuracy gain reduced escalation rate from 23% to 18%.",

    "Semantic cache threshold 0.88 (not 0.92) is safe for FAQ-style questions. "
    "We A/B tested this on 500 query pairs — false hit rate was <0.5%.",

    "Monitor refusal rate by product area: 'DataSync Pro' had 28% refusal rate "
    "because documentation was 3 months out of date.",
]
```

---

> ### 📋 Chapter Summary
>
> - The support RAG system chose GPT-4o-mini over GPT-4o based on domain-specific evaluation (4-point quality gap, 5× cost difference) — model selection must be evidence-driven, not default-driven.
> - **Hybrid retrieval** (BM25 + dense, RRF fusion) is essential for product terminology; pure dense retrieval undersells exact-match product names by 18 percentage points.
> - **Escalation logic** is a first-class feature: low retrieval confidence, high-stakes topics, and enterprise tier all trigger human escalation before the LLM is called.
> - Production hit a 68% Tier-1 deflection rate and 32% cache hit rate, delivering $12.50/day in savings while staying within the $50/day budget.

---

> ### ❓ Comprehension Questions
>
> 1. The system targets 0.88 accuracy but achieves 0.87. The team proposes switching to GPT-4o to close the gap. The daily cost would increase from $38 to ~$190. Is this justified? What analysis would you perform before making this decision?
> 2. `EscalationClassifier.should_escalate` uses a hardcoded list of trigger phrases. A customer asks: "My account was hacked" — this should escalate (security incident) but the phrase "hacked" is not in the trigger list. How would you make the escalation classifier more robust without adding every possible phrase?
> 3. PII is redacted before caching. A query "What is the return policy for order #12345?" has the order number redacted to "[ID]". The cache stores this. A future query "What is the return policy for order #67890?" hits the same cache entry. Is this correct behaviour, and what edge cases could it cause?
> 4. The cache threshold was lowered from 0.92 to 0.88. An A/B test showed a false hit rate of <0.5% on 500 query pairs. Is this sample sufficient to be confident at 1M daily queries? Calculate the expected number of false hits per day.
> 5. The system deflects 68% of Tier-1 questions. The remaining 32% escalate to humans. A product manager asks: "Can we increase deflection to 85%?" What would need to change in the system, and what are the quality trade-offs?

---

## References

### Documentation
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)
- [BM25 in Python (rank-bm25)](https://github.com/dorianbrown/rank_bm25)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Cormack et al., 2009.

---

---
[« Back to case_studies Index](index.md) | [🏠 Home](../index.md)