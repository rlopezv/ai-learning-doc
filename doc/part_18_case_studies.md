# Part XVIII — Practical Case Studies

---

> **Navigation**
> [← Part XVII — Reference Architectures](part_17_reference_architectures.md) | [→ Part XIX — The Future](part_19_future.md)

---

## Contents

- [Chapter 1 — Customer Support RAG System](#chapter-1--customer-support-rag-system)
  - [1.1 Requirements and Constraints](#11-requirements-and-constraints)
  - [1.2 Architecture Decisions](#12-architecture-decisions)
  - [1.3 Implementation Highlights](#13-implementation-highlights)
  - [1.4 Production Results and Lessons Learned](#14-production-results-and-lessons-learned)
- [Chapter 2 — Internal Knowledge Base for Engineering Teams](#chapter-2--internal-knowledge-base-for-engineering-teams)
  - [2.1 Problem Statement](#21-problem-statement)
  - [2.2 Multi-Source Corpus Design](#22-multi-source-corpus-design)
  - [2.3 Developer-Facing API Design](#23-developer-facing-api-design)
  - [2.4 Measuring Adoption and Impact](#24-measuring-adoption-and-impact)
- [Chapter 3 — Financial Document Analysis](#chapter-3--financial-document-analysis)
  - [3.1 Regulatory Context and Constraints](#31-regulatory-context-and-constraints)
  - [3.2 Document Processing Pipeline](#32-document-processing-pipeline)
  - [3.3 High-Precision Retrieval for Financial Data](#33-high-precision-retrieval-for-financial-data)
  - [3.4 Compliance and Audit Trail Design](#34-compliance-and-audit-trail-design)
- [Chapter 4 — Code Intelligence Platform](#chapter-4--code-intelligence-platform)
  - [4.1 Why Code RAG is Different](#41-why-code-rag-is-different)
  - [4.2 Code-Aware Chunking and Indexing](#42-code-aware-chunking-and-indexing)
  - [4.3 Query Patterns and Retrieval Strategies](#43-query-patterns-and-retrieval-strategies)
  - [4.4 IDE Integration and Response Formatting](#44-ide-integration-and-response-formatting)
- [Chapter 5 — Healthcare Knowledge Assistant 🧪](#chapter-5--healthcare-knowledge-assistant-)
  - [5.1 Clinical Use Case Constraints](#51-clinical-use-case-constraints)
  - [5.2 Medical Corpus Management](#52-medical-corpus-management)
  - [5.3 Safety-First Generation Design](#53-safety-first-generation-design)
  - [5.4 Evaluation for Clinical Settings](#54-evaluation-for-clinical-settings)
  - [🧪 Hands-on Lab: End-to-End Case Study Simulator](#-hands-on-lab-end-to-end-case-study-simulator)

---

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

## Chapter 2 — Internal Knowledge Base for Engineering Teams

### 2.1 Problem Statement

A 400-engineer organisation had knowledge scattered across Confluence (12,000 pages), GitHub wikis (3,200 pages), internal Slack (not indexed), Jira (not indexed), and 8 years of runbook PDFs. New engineers took 6–8 weeks to become productive. Senior engineers spent 20% of their time answering the same questions repeatedly.

```python
KNOWLEDGE_FRAGMENTATION_AUDIT = {
    "confluence_pages":      12_000,
    "github_wiki_pages":     3_200,
    "runbook_pdfs":          840,
    "architecture_docs":     320,
    "postmortem_reports":    180,
    "onboarding_guides":     45,
    "stale_content_pct":     0.38,    # 38% of docs not updated in >12 months
    "orphan_content_pct":    0.22,    # 22% of docs not linked from anywhere
    "avg_search_time_min":   8.3,     # Minutes per knowledge search
    "questions_to_humans_weekly": 1_200,
}
```

---

### 2.2 Multi-Source Corpus Design

```python
from dataclasses import dataclass
from enum import Enum

class SourceType(str, Enum):
    CONFLUENCE    = "confluence"
    GITHUB_WIKI   = "github_wiki"
    PDF_RUNBOOK   = "pdf_runbook"
    POSTMORTEM    = "postmortem"
    API_DOCS      = "api_docs"

@dataclass
class CorpusSourceConfig:
    source_type: SourceType
    connector_class: str
    refresh_interval_hours: int
    chunking_strategy: str
    chunk_size_tokens: int
    overlap_tokens: int
    metadata_fields: list[str]
    quality_filter: str    # How to detect stale/low-quality content
    priority_weight: float  # For result ranking across sources

INTERNAL_KB_SOURCES = [
    CorpusSourceConfig(
        SourceType.CONFLUENCE,
        connector_class="ConfluenceConnector",
        refresh_interval_hours=4,
        chunking_strategy="heading_aware",      # Split on H2/H3 boundaries
        chunk_size_tokens=400,
        overlap_tokens=40,
        metadata_fields=["space_key", "page_id", "last_modified",
                         "author", "labels", "url"],
        quality_filter="exclude pages not modified in >24 months",
        priority_weight=1.0
    ),
    CorpusSourceConfig(
        SourceType.GITHUB_WIKI,
        connector_class="GitHubWikiConnector",
        refresh_interval_hours=1,
        chunking_strategy="markdown_section",   # Split on ## / ### headings
        chunk_size_tokens=512,
        overlap_tokens=50,
        metadata_fields=["repo", "path", "last_commit_sha",
                         "last_modified", "url"],
        quality_filter="exclude files not committed in >18 months",
        priority_weight=1.2    # Higher: engineers trust own wiki more
    ),
    CorpusSourceConfig(
        SourceType.PDF_RUNBOOK,
        connector_class="PDFRunbookConnector",
        refresh_interval_hours=24,
        chunking_strategy="fixed_token",
        chunk_size_tokens=600,
        overlap_tokens=80,
        metadata_fields=["filename", "version", "team", "upload_date"],
        quality_filter="flag PDFs older than 12 months for review",
        priority_weight=1.5    # Highest: runbooks are authoritative
    ),
    CorpusSourceConfig(
        SourceType.POSTMORTEM,
        connector_class="PostmortemConnector",
        refresh_interval_hours=12,
        chunking_strategy="section_aware",      # Root cause, action items as separate chunks
        chunk_size_tokens=300,
        overlap_tokens=30,
        metadata_fields=["incident_id", "severity", "date", "services_affected"],
        quality_filter="no staleness filter — historical record",
        priority_weight=0.8
    ),
]

class MultiSourceIngestionOrchestrator:
    """Ingests from multiple sources, deduplicates, and maintains a unified index."""

    def __init__(self, connectors: dict, pipeline, deduplicator):
        self.connectors    = connectors
        self.pipeline      = pipeline
        self.deduplicator  = deduplicator

    def run_full_refresh(self) -> dict:
        all_docs = []
        source_stats = {}
        for config in INTERNAL_KB_SOURCES:
            connector = self.connectors[config.source_type]
            docs      = connector.fetch_all()
            # Apply quality filter
            filtered  = [d for d in docs if self._passes_quality(d, config)]
            # Enrich with source metadata
            for doc in filtered:
                doc["metadata"]["source_type"]    = config.source_type.value
                doc["metadata"]["priority_weight"] = config.priority_weight
            all_docs.extend(filtered)
            source_stats[config.source_type.value] = {
                "fetched": len(docs), "after_filter": len(filtered)
            }
        # Deduplication: remove near-duplicate chunks across sources
        deduped = self.deduplicator.deduplicate(all_docs, threshold=0.97)
        result  = self.pipeline.ingest_batch(deduped)
        return {"source_stats": source_stats, "total_indexed": len(deduped)}

    def _passes_quality(self, doc: dict, config: CorpusSourceConfig) -> bool:
        content = doc.get("content", "")
        if len(content) < 100:
            return False   # Too short to be useful
        return True
```

---

### 2.3 Developer-Facing API Design

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class DevQueryRequest(BaseModel):
    question: str
    context: Optional[str] = None     # e.g. "kubernetes", "payments-service"
    source_filter: Optional[list[str]] = None  # Limit to specific sources
    include_postmortems: bool = False  # Opt-in: postmortems are verbose

class DevQueryResponse(BaseModel):
    answer: str
    sources: list[dict]     # [{title, url, source_type, last_modified, excerpt}]
    related_incidents: list[dict]   # Relevant postmortems if opted-in
    suggested_owners: list[str]     # Team/person to contact for more detail
    confidence: str         # "high" | "medium" | "low"
    feedback_url: str       # Deep link to rate this answer

# Slack bot integration
SLACK_COMMAND_HANDLERS = {
    "/ask": "Query the knowledge base",
    "/incident-history": "Find past incidents for a service",
    "/who-owns": "Find the team responsible for a service or component",
    "/runbook": "Retrieve the runbook for a service or incident type",
}

class SlackRAGBot:
    """Slack slash command handler for the internal knowledge base."""
    def __init__(self, rag_pipeline, slack_client):
        self.rag    = rag_pipeline
        self.slack  = slack_client

    def handle_ask(self, user_id: str, channel_id: str, text: str) -> dict:
        result = self.rag.query(
            question=text,
            source_filter=None,
            include_postmortems="/incident" in text.lower()
        )
        # Format for Slack Block Kit
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn",
             "text": f"*Answer:*\n{result['answer']}"}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn",
             "text": "*Sources:*\n" + "\n".join(
                 f"• <{s['url']}|{s['title']}> ({s['source_type']})"
                 for s in result['sources'][:3]
             )}},
            {"type": "actions", "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "👍 Helpful"},
                 "action_id": f"feedback_positive_{result.get('request_id','')}"},
                {"type": "button", "text": {"type": "plain_text", "text": "👎 Not helpful"},
                 "action_id": f"feedback_negative_{result.get('request_id','')}"},
            ]}
        ]
        return {"blocks": blocks, "response_type": "in_channel"}
```

---

### 2.4 Measuring Adoption and Impact

```python
IMPACT_METRICS_AT_90_DAYS = {
    "adoption": {
        "daily_active_users":   180,   # out of 400 engineers
        "queries_per_day":      420,
        "slack_bot_queries":    310,   # 74% via Slack
        "api_queries":          110,   # 26% via API/IDE
    },
    "quality": {
        "thumbs_up_rate":        0.73,
        "avg_answer_rating":     3.8,   # out of 5
        "answer_used_as_is_pct": 0.61,  # Engineer used answer without further search
    },
    "productivity": {
        "avg_search_time_reduction_min": 5.1,  # From 8.3 to 3.2 min per search
        "senior_eng_interruptions_reduction_pct": 0.38,
        "onboarding_time_reduction_weeks": 1.5,  # From 6–8 to 5–6.5 weeks
    },
    "corpus_health": {
        "stale_content_flagged_for_review": 4_200,  # Out of 12,000 Confluence pages
        "orphan_pages_identified":          2_600,
        "runbooks_updated_after_flag":      47,
    }
}

# Unexpected side effect: corpus quality improved because the system
# surfaced stale and orphaned documentation that no one knew existed.
UNEXPECTED_OUTCOMES = [
    "Corpus health scan identified 4,200 stale pages — prompted a documentation "
    "cleanup sprint that reduced corpus size by 30% and improved retrieval quality.",
    "Engineers started linking to knowledge base answers in PR comments, "
    "organically spreading adoption.",
    "Postmortem retrieval revealed 3 recurring infrastructure issues that "
    "had been independently worked around 7+ times without root cause fix.",
]
```

---

> ### 📋 Chapter Summary
>
> - Multi-source corpora require per-source connector configuration: different chunking strategies (heading-aware for Confluence, markdown sections for GitHub wiki), refresh intervals, and priority weights.
> - **Postmortems** are first-class corpus content: they surface recurring failure patterns that would otherwise require searching across 8 years of incident history.
> - Developer-facing APIs must expose source metadata (URL, author, last modified) and confidence signals — engineers need to evaluate sources, not just accept answers.
> - A corpus health scan surfaced 4,200 stale pages as an unexpected benefit — the RAG system revealed documentation debt that had been invisible.

---

> ### ❓ Comprehension Questions
>
> 1. `priority_weight` is stored as metadata and used for result ranking. A PDF runbook has `priority_weight=1.5` but was last updated 18 months ago. A GitHub wiki page has `priority_weight=1.2` and was updated yesterday. How would you combine recency and priority weight into a single ranking signal?
> 2. The Slack bot handles 310 of 420 daily queries. A user asks a question in a public Slack channel and the bot responds in-channel. If the question contains a partial service name that is also a customer name, what privacy risk does an in-channel response create?
> 3. `MultiSourceIngestionOrchestrator._passes_quality` only checks content length. A Confluence page has 500 characters of content but was last modified 3 years ago. It passes the length check. How would you implement the time-based quality filter described in `CorpusSourceConfig.quality_filter`?
> 4. `IMPACT_METRICS_AT_90_DAYS` shows 180 daily active users out of 400 engineers (45%). What are the most likely reasons the other 55% are not using the system, and how would you investigate and address each barrier?
> 5. The system reduced senior engineer interruptions by 38%. A senior engineer argues this metric is misleading because "the easy questions were deflected, but hard questions still escalate." How would you measure the quality of escalated questions before and after the system was deployed?

---

## References

### Documentation
- [Confluence REST API](https://developer.atlassian.com/cloud/confluence/rest/v2/intro/)
- [GitHub REST API — Repository Contents](https://docs.github.com/en/rest/repos/contents)
- [Slack Block Kit](https://api.slack.com/block-kit)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)

---

## Chapter 3 — Financial Document Analysis

### 3.1 Regulatory Context and Constraints

A mid-sized asset manager deployed a RAG system for analysts to query earnings reports, regulatory filings (10-K, 10-Q, 8-K), and internal research notes. The system operates under MiFID II (EU), FCA rules, and internal compliance mandates that constrain what the system can and cannot do.

```python
FINANCIAL_RAG_CONSTRAINTS = {
    "cannot_do": [
        "Provide investment recommendations ('Buy', 'Sell', 'Hold')",
        "Generate price targets or valuations",
        "Answer questions about non-public material information (MNPI)",
        "Provide advice that could constitute regulated financial advice",
    ],
    "can_do": [
        "Summarise disclosed financial figures from filings",
        "Extract specific metrics from earnings reports",
        "Compare disclosed figures across periods or companies",
        "Surface relevant risk factors from regulatory filings",
        "Answer questions about disclosed company policies",
    ],
    "mandatory": [
        "Every answer must cite the source document and filing date",
        "Disclaim that output is for research purposes, not investment advice",
        "Log all queries and responses for compliance review",
        "MNPI detection: flag and refuse queries about undisclosed information",
        "Maintain a 7-year audit trail (MiFID II Art. 25)",
    ]
}
```

---

### 3.2 Document Processing Pipeline

```python
from dataclasses import dataclass
from typing import Optional
import re

class FinancialDocumentProcessor:
    """
    Specialised processing for financial documents (10-K, 10-Q, 8-K, earnings).
    Handles: table extraction, number normalisation, period tagging, XBRL data.
    """

    def chunk_financial_doc(self, doc: dict) -> list[dict]:
        """
        Financial document chunking strategy:
        - Preserve table integrity (tables are NOT split across chunks)
        - Tag each chunk with fiscal period
        - Separate risk factors section (often very long, needs own chunking)
        - Preserve MD&A (Management Discussion & Analysis) as coherent units
        """
        content  = doc.get("content", "")
        doc_type = doc.get("metadata", {}).get("doc_type", "10-K")
        ticker   = doc.get("metadata", {}).get("ticker", "UNKNOWN")
        period   = doc.get("metadata", {}).get("fiscal_period", "")

        chunks = []

        # Split on major SEC filing sections
        sections = self._split_by_sections(content, doc_type)
        for section_name, section_text in sections.items():
            sub_chunks = self._chunk_section(section_text, section_name)
            for i, chunk_text in enumerate(sub_chunks):
                chunks.append({
                    "id": f"{doc['id']}_{section_name}_{i}",
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "section":       section_name,
                        "ticker":        ticker,
                        "fiscal_period": period,
                        "has_tables":    self._has_tables(chunk_text),
                        "contains_numbers": self._has_financial_numbers(chunk_text),
                    }
                })
        return chunks

    def _split_by_sections(self, content: str, doc_type: str) -> dict:
        """Split SEC filing by Item numbers (Item 1, Item 1A, Item 7, etc.)"""
        section_patterns = {
            "business":      r"Item\s+1[^A].*?(?=Item\s+1A|Item\s+2|\Z)",
            "risk_factors":  r"Item\s+1A.*?(?=Item\s+1B|Item\s+2|\Z)",
            "mda":           r"Item\s+7[^A].*?(?=Item\s+7A|Item\s+8|\Z)",
            "financials":    r"Item\s+8.*?(?=Item\s+9|\Z)",
        }
        sections = {}
        for name, pattern in section_patterns.items():
            match = re.search(pattern, content,
                              re.IGNORECASE | re.DOTALL)
            if match:
                sections[name] = match.group(0)
        if not sections:
            sections["full"] = content
        return sections

    def _chunk_section(self, text: str, section: str,
                       max_tokens: int = 600) -> list[str]:
        """Chunk a section, preserving table boundaries."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        in_table = False

        for line in text.split("\n"):
            is_table_line = "|" in line or line.strip().startswith("+-")
            if is_table_line and not in_table:
                in_table = True
            elif not is_table_line and in_table:
                in_table = False
                # Flush current chunk including complete table
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

            line_tokens = len(line.split())
            if current_tokens + line_tokens > max_tokens and not in_table:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return [c for c in chunks if len(c.strip()) > 50]

    def _has_tables(self, text: str) -> bool:
        return bool(re.search(r'\|.*\|', text) or
                    re.search(r'\$[\d,]+', text))

    def _has_financial_numbers(self, text: str) -> bool:
        return bool(re.search(r'\$[\d,]+|\d+\s*million|\d+\s*billion', text,
                               re.IGNORECASE))

    def normalise_financial_number(self, text: str) -> str:
        """Normalise financial shorthand: '$1.2B' → '$1,200,000,000'."""
        def replace_b(m):
            return f"${int(float(m.group(1)) * 1e9):,}"
        def replace_m(m):
            return f"${int(float(m.group(1)) * 1e6):,}"
        text = re.sub(r'\$([0-9.]+)\s*[Bb](?:illion)?', replace_b, text)
        text = re.sub(r'\$([0-9.]+)\s*[Mm](?:illion)?', replace_m, text)
        return text
```

---

### 3.3 High-Precision Retrieval for Financial Data

```python
class FinancialRAGPipeline:
    """
    Financial document RAG with compliance guardrails:
    - MNPI detection (refuse questions about non-public information)
    - Mandatory citation with document + filing date
    - Investment advice refusal
    - 7-year query audit log
    """
    INVESTMENT_ADVICE_PATTERNS = [
        r'\b(buy|sell|hold|purchase|invest|recommend)\b.*\bstock\b',
        r'\bprice\s+target\b',
        r'\bshould\s+i\s+(invest|buy|sell)\b',
        r'\boverweight|underweight|outperform|underperform\b',
    ]
    MNPI_PATTERNS = [
        r'\bunpublished|undisclosed|non-?public\b',
        r'\binside\s+information\b',
        r'\bbefore\s+(the\s+)?announcement\b',
    ]

    def __init__(self, retriever, llm_gateway, audit_logger):
        self.retriever     = retriever
        self.gateway       = llm_gateway
        self.audit         = audit_logger
        self._advice_re    = [re.compile(p, re.IGNORECASE)
                              for p in self.INVESTMENT_ADVICE_PATTERNS]
        self._mnpi_re      = [re.compile(p, re.IGNORECASE)
                              for p in self.MNPI_PATTERNS]

    def query(self, analyst_id: str, question: str,
              ticker: Optional[str] = None,
              period: Optional[str] = None) -> dict:
        # Compliance check 1: investment advice
        if any(p.search(question) for p in self._advice_re):
            self.audit.log_refusal(analyst_id, question, "INVESTMENT_ADVICE_REQUEST")
            return {
                "answer": ("This system provides factual summaries of disclosed "
                           "financial information only. For investment recommendations "
                           "please consult a licensed financial adviser. "
                           "This output does not constitute investment advice."),
                "compliance_flag": "INVESTMENT_ADVICE_REFUSED",
                "sources": []
            }

        # Compliance check 2: MNPI
        if any(p.search(question) for p in self._mnpi_re):
            self.audit.log_refusal(analyst_id, question, "MNPI_QUERY")
            return {
                "answer": "Questions about non-public material information cannot be answered.",
                "compliance_flag": "MNPI_REFUSED",
                "sources": []
            }

        # Retrieval with ticker/period filter
        metadata_filter = {}
        if ticker:
            metadata_filter["ticker"] = ticker
        if period:
            metadata_filter["fiscal_period"] = period

        docs = self.retriever.search(question, filter=metadata_filter, top_k=6)

        # Generate with strict citation and disclaimer requirement
        context = "\n".join(
            f"[{i+1}] Source: {d['metadata'].get('ticker','?')} "
            f"{d['metadata'].get('doc_type','?')} "
            f"({d['metadata'].get('fiscal_period','?')}) — {d['content']}"
            for i, d in enumerate(docs)
        )
        messages = [
            {"role": "system", "content":
             "You are a financial research assistant. "
             "Answer ONLY using disclosed information from the provided context. "
             "Every numerical claim MUST cite its source [N] with document and period. "
             "Never provide investment recommendations or price targets. "
             "End every response with: "
             "'[DISCLAIMER: This is a research summary only and does not constitute "
             "investment advice.]'"},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        response = self.gateway.complete(messages, model_alias="default")
        self.audit.log_query(analyst_id, question, response["answer"],
                             [d["id"] for d in docs])
        return {
            "answer":     response["answer"],
            "sources":    [{"id": d["id"],
                            "ticker": d["metadata"].get("ticker"),
                            "doc_type": d["metadata"].get("doc_type"),
                            "period":  d["metadata"].get("fiscal_period")}
                           for d in docs],
            "compliance_flag": None
        }
```

---

### 3.4 Compliance and Audit Trail Design

```python
from datetime import datetime, timedelta
import json

class FinancialAuditLogger:
    """
    MiFID II-compliant audit logger.
    Retains records for 7 years (Art. 25 requirement).
    Records: analyst ID, query, retrieved sources, generated answer, timestamp.
    """
    RETENTION_YEARS = 7

    def __init__(self, store):
        self.store = store

    def log_query(self, analyst_id: str, query: str, answer: str,
                  source_ids: list[str]):
        record = {
            "event_type":    "financial_query",
            "analyst_id":    analyst_id,
            "timestamp":     datetime.utcnow().isoformat(),
            "query":         query,
            "answer_hash":   self._hash(answer),   # Don't store full answer for privacy
            "answer_length": len(answer),
            "source_ids":    source_ids,
            "source_count":  len(source_ids),
            "retention_until": (datetime.utcnow() +
                                timedelta(days=365 * self.RETENTION_YEARS)).isoformat(),
        }
        self.store.append(record)

    def log_refusal(self, analyst_id: str, query: str, reason: str):
        record = {
            "event_type":  "compliance_refusal",
            "analyst_id":  analyst_id,
            "timestamp":   datetime.utcnow().isoformat(),
            "query_hash":  self._hash(query),   # Hash for privacy; not stored plain
            "reason":      reason,
            "retention_until": (datetime.utcnow() +
                                timedelta(days=365 * self.RETENTION_YEARS)).isoformat(),
        }
        self.store.append(record)

    def _hash(self, text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def compliance_report(self, analyst_id: str, period_start: str,
                          period_end: str) -> dict:
        records = self.store.query(
            analyst_id=analyst_id,
            start=period_start, end=period_end
        )
        queries  = [r for r in records if r["event_type"] == "financial_query"]
        refusals = [r for r in records if r["event_type"] == "compliance_refusal"]
        return {
            "analyst_id":    analyst_id,
            "period":        f"{period_start} to {period_end}",
            "total_queries": len(queries),
            "refusals":      len(refusals),
            "refusal_reasons": [r["reason"] for r in refusals],
            "report_date":   datetime.utcnow().isoformat()
        }
```

---

> ### 📋 Chapter Summary
>
> - Financial RAG operates under strict compliance constraints: investment advice refusal, MNPI detection, and 7-year MiFID II audit retention are non-negotiable design requirements.
> - **Financial document chunking** preserves table integrity and tags every chunk with ticker, document type, and fiscal period — enabling filtered retrieval for specific company/period combinations.
> - **Mandatory citations** in financial answers include source document type and fiscal period, not just document ID — enabling analysts to verify against original filings.
> - The audit logger stores query hashes (not plain text) for privacy while maintaining the evidentiary record required by regulation.

---

> ### ❓ Comprehension Questions
>
> 1. `FinancialRAGPipeline` refuses questions matching investment advice patterns. An analyst asks: "What was AAPL's EPS growth compared to analyst consensus?" This is factual research but contains an implicit comparison that could inform a buy/sell decision. Should this be refused? Where is the line?
> 2. The audit log stores `answer_hash` but not the full answer. A compliance officer needs to verify the exact answer given to an analyst 3 years ago in response to a regulatory inquiry. Is the hash sufficient? What additional information should be retained?
> 3. Financial tables are preserved intact in chunks. A table comparing revenue across 8 quarters occupies 800 tokens — exceeding the 600-token chunk limit. The processor uses `in_table = True` to prevent splitting. What happens if the table is genuinely too large for the context window at generation time?
> 4. The MNPI regex pattern `r'\bunpublished|undisclosed|non-?public\b'` uses `\b` word boundaries. An analyst asks: "What undisclosed risks does the company mention in their risk factors?" — the word "undisclosed" matches even though the question is about disclosed risk factors. How would you reduce false positives in MNPI detection?
> 5. MiFID II requires 7-year retention. The system processes 500 analyst queries/day. Estimate the audit log storage at 1KB per record over 7 years, and design a tiered hot/warm/cold storage strategy that keeps recent records fast to query while minimising cost for older records.

---

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

## Chapter 5 — Healthcare Knowledge Assistant 🧪

### 5.1 Clinical Use Case Constraints

A hospital network deployed an internal RAG system to help clinical staff query treatment protocols, drug interaction databases, and discharge planning guidelines. The system is NOT patient-facing and is designed to assist clinicians who retain full professional responsibility.

```python
CLINICAL_CONSTRAINTS = {
    "system_classification": "Clinical Decision Support Tool (CDST)",
    "eu_ai_act_tier":        "high_risk (medical device adjacent)",
    "intended_users":        ["registered nurses", "junior doctors", "pharmacists"],
    "NOT_intended_for":      ["patients", "self-diagnosis", "replacing clinical judgment"],
    "mandatory_disclaimer":  (
        "This system provides reference information from approved clinical guidelines only. "
        "All clinical decisions remain the responsibility of the licensed clinician. "
        "Never substitute for direct patient assessment or specialist consultation."
    ),
    "prohibited_outputs": [
        "Specific drug dosing for individual patients",
        "Diagnosis of specific patient conditions",
        "Recommendations that contradict the retrieved guideline",
        "Answers about patients by name or ID (HIPAA/GDPR protection)",
    ],
    "required_outputs": [
        "Explicit citation of source guideline with version and date",
        "Explicit uncertainty acknowledgement when guideline is ambiguous",
        "Escalation prompt when question is outside guideline scope",
    ],
    "data_requirements": {
        "corpus_retention": "Current version + 1 prior version",
        "audit_retention_years": 7,
        "pii_in_corpus":    False,    # Clinical guidelines only — no patient data
        "access_control":   "Staff ID + ward/role verification",
    }
}
```

---

### 5.2 Medical Corpus Management

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class ClinicalGuideline:
    guideline_id: str
    title: str
    issuing_body: str          # "NICE" | "WHO" | "NHS" | "hospital_internal"
    version: str
    published_date: str
    effective_date: str
    expiry_date: Optional[str]  # Some guidelines have explicit expiry
    supersedes: Optional[str]   # ID of previous version this replaces
    specialty: str
    evidence_grade: str         # "A" | "B" | "C" | "D" | "GPP" (Good Practice Point)
    status: str                 # "active" | "archived" | "under_review"
    content_hash: str           # Detect unauthorised modifications

class ClinicalCorpusManager:
    """
    Manages clinical guideline corpus with strict version control.
    Only ACTIVE guidelines are indexed; ARCHIVED versions are retained
    for audit but excluded from retrieval.
    """
    def __init__(self, vector_db, metadata_store, audit_logger):
        self.vdb    = vector_db
        self.meta   = metadata_store
        self.audit  = audit_logger

    def publish_guideline(self, guideline: ClinicalGuideline,
                          content: str, approved_by: str):
        """Publish a new guideline and supersede previous version."""
        # Verify content integrity
        import hashlib
        computed_hash = hashlib.sha256(content.encode()).hexdigest()
        assert computed_hash == guideline.content_hash, "Content hash mismatch"

        # Archive previous version if superseding
        if guideline.supersedes:
            self._archive_version(guideline.supersedes)

        # Index new version
        chunks = self._chunk_guideline(content, guideline)
        self.vdb.upsert_batch(
            ids=[c["id"] for c in chunks],
            embeddings=self._embed([c["content"] for c in chunks]),
            payloads=[{
                **c["metadata"],
                "guideline_id":   guideline.guideline_id,
                "version":        guideline.version,
                "issuing_body":   guideline.issuing_body,
                "evidence_grade": guideline.evidence_grade,
                "status":         "active",
                "effective_date": guideline.effective_date,
            } for c in chunks]
        )
        self.audit.log({
            "event": "guideline_published",
            "guideline_id": guideline.guideline_id,
            "version": guideline.version,
            "approved_by": approved_by,
            "timestamp": datetime.utcnow().isoformat()
        })

    def _archive_version(self, guideline_id: str):
        """Mark all chunks for this guideline as archived — removed from retrieval."""
        self.vdb.update_payload(
            filter={"guideline_id": guideline_id},
            payload={"status": "archived"}
        )
        print(f"  Archived guideline: {guideline_id}")

    def check_expired(self) -> list[str]:
        """Return guidelines past their expiry date still marked active."""
        today = datetime.utcnow().isoformat()[:10]
        return [g.guideline_id for g in self.meta.get_active_guidelines()
                if g.expiry_date and g.expiry_date < today]

    def _chunk_guideline(self, content: str, guideline: ClinicalGuideline) -> list[dict]:
        # Clinical guidelines: chunk at section level, preserve recommendation boxes
        import re
        sections = re.split(r'\n(?=\d+\.\s+[A-Z])', content)
        return [{"id": f"{guideline.guideline_id}_v{guideline.version}_{i}",
                 "content": sec, "metadata": {"specialty": guideline.specialty}}
                for i, sec in enumerate(sections) if len(sec.strip()) > 100]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]  # Placeholder
```

---

### 5.3 Safety-First Generation Design

```python
import re

class ClinicalRAGPipeline:
    """
    Safety-first clinical RAG with:
    - Patient PII detection (block queries about specific patients)
    - Dosing question detection (require specialist escalation)
    - Mandatory evidence grade and disclaimer in every answer
    - Active-only guideline retrieval
    """
    DOSING_PATTERNS = [
        r'\b(\d+\s*mg|\d+\s*ml|\d+\s*mcg)\b',
        r'\b(dose|dosing|dosage)\s+for\b',
        r'\bhow\s+much\s+to\s+(give|administer|prescribe)\b',
    ]
    PATIENT_PATTERNS = [
        r'\bpatient\s+[A-Z][a-z]+\b',
        r'\bMr\.\s*[A-Z]\b',
        r'\bNHS\s+number\b',
        r'\bDOB\b|\bdate\s+of\s+birth\b',
    ]

    def __init__(self, retriever, llm_gateway, audit_logger):
        self.retriever    = retriever
        self.gateway      = llm_gateway
        self.audit        = audit_logger
        self._dosing_re   = [re.compile(p, re.IGNORECASE) for p in self.DOSING_PATTERNS]
        self._patient_re  = [re.compile(p, re.IGNORECASE) for p in self.PATIENT_PATTERNS]

    def query(self, staff_id: str, question: str, specialty: str = None) -> dict:
        # Safety check 1: specific patient data
        if any(p.search(question) for p in self._patient_re):
            return self._safety_refusal("PATIENT_PII_QUERY",
                "Questions about individual patients cannot be processed. "
                "Please consult the patient's record system directly.")

        # Safety check 2: specific dosing request
        is_dosing = any(p.search(question) for p in self._dosing_re)

        # Retrieve from ACTIVE guidelines only
        results = self.retriever.search(
            question,
            filter={"status": "active",
                    **({"specialty": specialty} if specialty else {})},
            top_k=5
        )

        if not results or max(r.get("score", 0) for r in results) < 0.60:
            return self._safety_refusal("LOW_CONFIDENCE",
                "The question falls outside available clinical guidelines. "
                "Please consult a senior clinician or specialist.")

        # Build context with evidence grades visible
        context = "\n".join(
            f"[{i+1}] {r['metadata'].get('issuing_body','?')} Guideline "
            f"v{r['metadata'].get('version','?')} "
            f"(Evidence Grade: {r['metadata'].get('evidence_grade','?')}) — "
            f"{r['content']}"
            for i, r in enumerate(results)
        )

        dosing_instruction = (
            "\n\nIMPORTANT: This question relates to dosing. "
            "Do NOT provide specific doses. State that dosing must be confirmed "
            "with a pharmacist or prescriber per local protocol."
        ) if is_dosing else ""

        messages = [
            {"role": "system", "content":
             "You are a clinical reference assistant. "
             "Answer ONLY using information from the provided clinical guidelines. "
             "Every answer MUST include: (1) source guideline and version cited as [N], "
             "(2) evidence grade from the source, "
             "(3) the mandatory disclaimer at the end. "
             "If the guideline is ambiguous or the answer is unclear, say so explicitly."
             + dosing_instruction},
            {"role": "user",
             "content": (f"Clinical guidelines context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Required ending: {CLINICAL_CONSTRAINTS['mandatory_disclaimer']}")}
        ]
        response = self.gateway.complete(messages, model_alias="default")
        self.audit.log_query(staff_id, question, [r["id"] for r in results])
        return {
            "answer":     response["answer"],
            "sources":    results[:3],
            "is_dosing":  is_dosing,
            "confidence": "high" if results[0].get("score", 0) > 0.80 else "medium"
        }

    def _safety_refusal(self, reason: str, message: str) -> dict:
        return {"answer": message, "safety_refusal": reason,
                "sources": [], "confidence": "refused"}
```

---

### 5.4 Evaluation for Clinical Settings

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClinicalEvalCase:
    case_id: str
    question: str
    expected_guideline_id: str   # Which guideline should be retrieved
    expected_evidence_grade: str
    is_in_scope: bool            # Should system answer or refuse?
    is_dosing_question: bool
    contains_patient_pii: bool
    clinical_reviewer_approved: bool   # Human expert approved this test case
    gold_standard_answer: Optional[str] = None

CLINICAL_EVAL_METRICS = {
    "guideline_recall":         "Correct guideline retrieved in top-3 (%)",
    "evidence_grade_cited":     "Response cites correct evidence grade (%)",
    "disclaimer_present":       "Mandatory disclaimer present in response (%)",
    "refusal_precision":        "Out-of-scope questions correctly refused (%)",
    "refusal_recall":           "All out-of-scope questions refused (%) — critical",
    "dosing_refusal_rate":      "Dosing questions refused or escalated (%)",
    "patient_pii_refusal":      "Patient PII queries refused (%) — must be 100%",
    "factual_accuracy":         "Clinical expert agreement with answer (%)",
    "hallucination_rate":       "Claims not in source guideline (%) — must be near 0",
}

class ClinicalEvaluator:
    def __init__(self, pipeline, eval_cases: list[ClinicalEvalCase]):
        self.pipeline = pipeline
        self.cases    = eval_cases

    def run(self) -> dict:
        results = {m: [] for m in CLINICAL_EVAL_METRICS}
        for case in self.cases:
            resp = self.pipeline.query("eval_runner", case.question)

            # Patient PII: must refuse (100% target)
            if case.contains_patient_pii:
                results["patient_pii_refusal"].append(
                    1 if resp.get("safety_refusal") else 0
                )
                continue

            # Out-of-scope: must refuse
            if not case.is_in_scope:
                results["refusal_recall"].append(
                    1 if resp.get("safety_refusal") else 0
                )
                continue

            # Dosing: must escalate or refuse dosing
            if case.is_dosing_question:
                results["dosing_refusal_rate"].append(
                    1 if resp.get("is_dosing") else 0
                )

            # Guideline retrieval
            source_ids = [s.get("id", "") for s in resp.get("sources", [])]
            guideline_retrieved = any(
                case.expected_guideline_id in sid for sid in source_ids[:3]
            )
            results["guideline_recall"].append(1 if guideline_retrieved else 0)

            # Disclaimer
            disclaimer_present = (
                CLINICAL_CONSTRAINTS["mandatory_disclaimer"][:50].lower()
                in resp.get("answer", "").lower()
            )
            results["disclaimer_present"].append(1 if disclaimer_present else 0)

        return {
            metric: round(sum(vals) / len(vals), 4) if vals else None
            for metric, vals in results.items()
        }
```

---

### 🧪 Hands-on Lab: End-to-End Case Study Simulator

**Objective:** Simulate key patterns from all five case studies in a unified runnable script, demonstrating: escalation logic, hybrid RRF fusion, compliance refusal, code query classification, and clinical safety checks.

```python
#!/usr/bin/env python3
"""
case_study_simulator.py — Patterns from all 5 case studies.
No external dependencies required.
"""
import re, math
from typing import Optional

print("=" * 60)
print("  Case Study Pattern Simulator")
print("=" * 60)

# ── Case 1: Support escalation classifier ─────────────────────
print("\n[CS-1] Customer Support — Escalation Classifier")
ESCALATION_TRIGGERS = ["billing dispute", "data breach", "legal", "refund",
                       "outage", "account suspension", "data loss"]
def should_escalate(question: str, top_score: float,
                    tier: str = "standard") -> tuple[bool, str]:
    for t in ESCALATION_TRIGGERS:
        if t in question.lower():
            return True, f"trigger:{t}"
    if top_score < 0.55:
        return True, f"low_confidence:{top_score:.2f}"
    if tier == "enterprise" and top_score < 0.70:
        return True, f"enterprise_low_confidence:{top_score:.2f}"
    return False, "ok"

support_cases = [
    ("How do I reset my password?",             0.91, "standard"),
    ("There is a data breach in my account",    0.88, "standard"),
    ("I want a refund for my subscription",     0.72, "professional"),
    ("How do I export data?",                   0.50, "standard"),
    ("What is the SLA for enterprise tier?",    0.65, "enterprise"),
]
for q, score, tier in support_cases:
    escalate, reason = should_escalate(q, score, tier)
    icon = "↑ ESCALATE" if escalate else "✓ ANSWER"
    print(f"  {icon}  [{tier}] {q[:45]} ({reason})")

# ── Case 2: Hybrid RRF Fusion ──────────────────────────────────
print("\n[CS-2] Internal KB — RRF Fusion Demo")
def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank + 1)

# Simulated dense and BM25 results
dense_results = [
    {"id": "doc_kubernetes_001", "score": 0.91},
    {"id": "doc_kubernetes_002", "score": 0.85},
    {"id": "doc_helm_003",       "score": 0.80},
]
bm25_results = [
    {"id": "doc_helm_003",       "score": 12.4},   # BM25 boosts exact match
    {"id": "doc_kubernetes_001", "score": 9.1},
    {"id": "doc_runbook_k8s",    "score": 8.7},
]
rrf: dict = {}
for rank, doc in enumerate(dense_results):
    rrf[doc["id"]] = rrf.get(doc["id"], 0) + rrf_score(rank)
for rank, doc in enumerate(bm25_results):
    rrf[doc["id"]] = rrf.get(doc["id"], 0) + rrf_score(rank)
fused = sorted(rrf.items(), key=lambda x: -x[1])
print(f"  {'Doc ID':<30} {'RRF Score'}")
for doc_id, score in fused:
    print(f"  {doc_id:<30} {score:.5f}")

# ── Case 3: Financial compliance guardrails ────────────────────
print("\n[CS-3] Financial RAG — Compliance Guardrails")
ADVICE_PATTERNS = [r'\b(buy|sell|recommend)\b.*\bstock\b',
                   r'\bprice\s+target\b', r'\bshould\s+i\s+(invest|buy)\b']
MNPI_PATTERNS   = [r'\bunpublished\b', r'\bnon-?public\b', r'\binside\s+info\b']

def financial_compliance_check(q: str) -> tuple[str, Optional[str]]:
    for p in ADVICE_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "REFUSED", "INVESTMENT_ADVICE"
    for p in MNPI_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "REFUSED", "MNPI_QUERY"
    return "ALLOWED", None

fin_queries = [
    "What was AAPL revenue in Q3 2024?",
    "Should I buy MSFT stock given their earnings?",
    "What are the disclosed risk factors for Tesla?",
    "What is the unpublished forecast for Q4?",
    "Compare Amazon and Alphabet operating margins in 2023",
]
for q in fin_queries:
    status, reason = financial_compliance_check(q)
    icon = "✗" if status == "REFUSED" else "✓"
    print(f"  {icon} {status:<8} {reason or '':<22} {q[:50]}")

# ── Case 4: Code query classification ─────────────────────────
print("\n[CS-4] Code Intelligence — Query Type Classification")
def classify_code_query(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["how do", "how to", "implement", "example"]):
        return "HOW_TO"
    if any(w in q for w in ["where", "used", "usages", "references"]):
        return "FIND_USAGE"
    if any(w in q for w in ["explain", "what does", "describe"]):
        return "EXPLAIN"
    if any(w in q for w in ["error", "bug", "fail", "exception"]):
        return "DEBUG"
    if any(w in q for w in ["refactor", "improve", "better"]):
        return "REFACTOR"
    return "HOW_TO"

code_queries = [
    "How do I implement pagination with cursor?",
    "Where is TokenBudget class used in the codebase?",
    "Explain what the retry_with_backoff function does",
    "Why does the CircuitBreaker fail after 5 errors?",
    "How should I refactor the FallbackChain class?",
]
for q in code_queries:
    qtype = classify_code_query(q)
    print(f"  [{qtype:<12}] {q}")

# ── Case 5: Clinical safety checks ────────────────────────────
print("\n[CS-5] Healthcare RAG — Clinical Safety Checks")
DOSING_PATTERNS  = [r'\b(\d+\s*mg|\d+\s*ml)', r'\b(dose|dosing)\s+for', r'how\s+much\s+to\s+give']
PATIENT_PATTERNS = [r'\bpatient\s+[A-Z][a-z]+', r'\bNHS\s+number\b', r'\bDOB\b']

def clinical_safety_check(q: str, top_score: float = 0.75) -> tuple[str, str]:
    for p in PATIENT_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "REFUSED", "PATIENT_PII"
    for p in DOSING_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return "ESCALATE", "DOSING_QUESTION"
    if top_score < 0.60:
        return "REFUSED", "LOW_CONFIDENCE"
    return "ANSWER", "OK"

clinical_cases = [
    ("What is the NICE guideline for sepsis management?",    0.88),
    ("What dose of amoxicillin 500mg should I give?",        0.82),
    ("Patient John Smith has a rash — what is the cause?",   0.70),
    ("What are the contraindications for metformin?",         0.85),
    ("What is the evidence for CPAP in sleep apnoea?",        0.45),
]
for q, score in clinical_cases:
    status, reason = clinical_safety_check(q, score)
    icon = {"ANSWER": "✓", "ESCALATE": "⚠", "REFUSED": "✗"}[status]
    print(f"  {icon} {status:<8} [{reason:<18}] {q[:55]}")

print("\n" + "=" * 60)
print("  All 5 case study patterns demonstrated")
```

**Run the lab:**
```bash
python case_study_simulator.py
```

**Extensions:**
- Add a **Case 6**: multi-language RAG for a global product where the same question arrives in EN, ES, and FR — implement a language detection + query translation step before retrieval
- Extend the clinical safety check to also flag questions referencing drug names and output a "PHARMACIST_REVIEW" status instead of a hard refusal
- Add a metrics summary at the end of the simulator showing: escalation rate (CS-1), RRF rank improvement (CS-2), compliance refusal rate (CS-3)

---

> ### 📋 Chapter Summary
>
> - Clinical RAG requires three safety layers unique to healthcare: patient PII detection (100% refusal required), dosing question escalation, and active-only guideline retrieval with evidence grade citation.
> - **Clinical evaluation** treats `patient_pii_refusal` and `refusal_recall` as must-be-100% metrics — a single incorrect answer to an out-of-scope question is an unacceptable safety failure.
> - Version-controlled corpus management is critical in clinical settings: archived guidelines must be excluded from retrieval but retained for audit, and expired guidelines must be detected automatically.
> - The case study simulator validates all five domain-specific patterns — support escalation, hybrid fusion, financial compliance, code query routing, and clinical safety — in a single runnable lab.

---

> ### ❓ Comprehension Questions
>
> 1. `ClinicalRAGPipeline` refuses queries with `top_score < 0.60`. A nurse asks about a treatment not covered by the indexed NICE guidelines but covered by a WHO guideline not yet ingested. The system refuses. What process should exist for nurses to report gaps in corpus coverage, and how quickly should they be addressed?
> 2. `ClinicalEvaluator` tests `patient_pii_refusal` — but the test cases must be created by someone. Generating test cases with fake patient names requires careful handling to avoid them leaking into the knowledge base. What controls would you put around the clinical test case generation process?
> 3. The `mandatory_disclaimer` is appended to every answer. A clinician complains that it appears even on straightforward factual questions (e.g. "What is the definition of sepsis?"), reducing trust in the system. How would you preserve safety while reducing disclaimer fatigue?
> 4. `ClinicalCorpusManager.check_expired()` identifies guidelines past their expiry date. What should happen automatically versus requiring manual review, and who must approve removal of an expired guideline from the active index?
> 5. The financial RAG system stores `query_hash` instead of the full query for privacy. The clinical system stores `staff_id` alongside query content (for audit purposes). Justify this difference: why is it acceptable to retain clinical query content but not financial query content in plaintext?

---

## References

### Documentation
- [NICE Evidence Standards](https://www.nice.org.uk/standards-and-indicators) — Clinical guideline quality standards.
- [MiFID II Article 25](https://www.esma.europa.eu/regulation/post-trading/mifid-ii-and-mifir) — Investment suitability requirements.
- [Python `ast` module](https://docs.python.org/3/library/ast.html) — AST parsing for code chunking.
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection for healthcare data.

### Papers
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Cormack et al., 2009.
- [RAG for Clinical Decision Support](https://arxiv.org/abs/2402.01030) — Clinical RAG evaluation challenges, 2024.

---

> **Navigation**
> [← Part XVII — Reference Architectures](part_17_reference_architectures.md) | [→ Part XIX — The Future](part_19_future.md)
