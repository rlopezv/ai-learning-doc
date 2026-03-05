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

---
[« Back to case_studies Index](index.md) | [🏠 Home](../../index.md)