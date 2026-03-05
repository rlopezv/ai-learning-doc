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

---
[« Back to case_studies Index](index.md) | [🏠 Home](../../index.md)