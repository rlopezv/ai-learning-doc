## Chapter 3 — Data Security and Privacy

### 3.1 PII Detection and Redaction

```python
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PIIEntityType(str, Enum):
    EMAIL          = "email"
    PHONE          = "phone"
    CREDIT_CARD    = "credit_card"
    SSN            = "ssn"               # US Social Security Number
    UK_NIN         = "uk_nin"            # UK National Insurance Number
    PASSPORT       = "passport"
    IP_ADDRESS     = "ip_address"
    IBAN           = "iban"
    NHS_NUMBER     = "nhs_number"        # UK NHS
    DATE_OF_BIRTH  = "date_of_birth"
    PERSON_NAME    = "person_name"       # Requires NLP model

@dataclass
class PIIMatch:
    entity_type: PIIEntityType
    start: int
    end: int
    raw_value: str
    replacement: str

@dataclass
class PIIScanResult:
    has_pii: bool
    matches: list[PIIMatch]
    redacted_text: str
    entity_types: list[str]

class RegexPIIDetector:
    """
    Regex-based PII detector for structured PII types.
    For unstructured PII (names, addresses), complement with
    a Presidio or spaCy NER model.
    """
    PATTERNS = {
        PIIEntityType.EMAIL: (
            r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b',
            "[EMAIL]"
        ),
        PIIEntityType.PHONE: (
            r'(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}',
            "[PHONE]"
        ),
        PIIEntityType.CREDIT_CARD: (
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}'
            r'|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            "[CREDIT_CARD]"
        ),
        PIIEntityType.SSN: (
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            "[SSN]"
        ),
        PIIEntityType.UK_NIN: (
            r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[ABCD]\b',
            "[NIN]"
        ),
        PIIEntityType.IBAN: (
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
            "[IBAN]"
        ),
        PIIEntityType.NHS_NUMBER: (
            r'\b\d{3}\s?\d{3}\s?\d{4}\b',
            "[NHS]"
        ),
        PIIEntityType.IP_ADDRESS: (
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            "[IP_ADDRESS]"
        ),
    }

    def __init__(self):
        self._compiled = {
            entity_type: (re.compile(pattern, re.IGNORECASE), replacement)
            for entity_type, (pattern, replacement) in self.PATTERNS.items()
        }

    def scan(self, text: str) -> PIIScanResult:
        matches: list[PIIMatch] = []
        for entity_type, (pattern, replacement) in self._compiled.items():
            for m in pattern.finditer(text):
                matches.append(PIIMatch(
                    entity_type=entity_type,
                    start=m.start(),
                    end=m.end(),
                    raw_value=m.group(),
                    replacement=replacement
                ))

        # Sort by position and redact (right-to-left to preserve offsets)
        matches.sort(key=lambda m: m.start)
        redacted = text
        for match in reversed(matches):
            redacted = redacted[:match.start] + match.replacement + redacted[match.end:]

        return PIIScanResult(
            has_pii=len(matches) > 0,
            matches=matches,
            redacted_text=redacted,
            entity_types=list({m.entity_type.value for m in matches})
        )

    def redact(self, text: str) -> str:
        return self.scan(text).redacted_text


class PresidioPIIDetector:
    """
    🔓 On-premise alternative: Microsoft Presidio for comprehensive PII detection.
    Handles unstructured PII (names, addresses) using NLP.
    Install: pip install presidio-analyzer presidio-anonymizer
    """
    def __init__(self):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self._available = True
        except ImportError:
            self._available = False

    def redact(self, text: str, language: str = "en") -> str:
        if not self._available:
            raise RuntimeError("Presidio not installed: pip install presidio-analyzer")
        results = self.analyzer.analyze(text=text, language=language)
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
```

---

### 3.2 Data Classification for RAG Systems

```python
from dataclasses import dataclass
from enum import Enum

class DataClassification(str, Enum):
    PUBLIC        = "public"         # Freely shareable
    INTERNAL      = "internal"       # Company internal, not public
    CONFIDENTIAL  = "confidential"   # Restricted to specific teams
    RESTRICTED    = "restricted"     # PII, financial, legal — tightly controlled
    SECRET        = "secret"         # Classified — requires special handling

@dataclass
class DocumentClassification:
    document_id: str
    classification: DataClassification
    contains_pii: bool
    pii_types: list[str]
    allowed_team_ids: list[str]      # Empty = all internal teams
    requires_redaction: bool
    data_residency: str              # "EU" | "US" | "any"
    retention_days: int              # 0 = no automatic deletion

CLASSIFICATION_POLICIES = {
    DataClassification.PUBLIC: {
        "can_be_indexed": True,
        "requires_pii_scan": False,
        "allowed_in_context": True,
        "log_access": False,
    },
    DataClassification.INTERNAL: {
        "can_be_indexed": True,
        "requires_pii_scan": True,
        "allowed_in_context": True,
        "log_access": False,
    },
    DataClassification.CONFIDENTIAL: {
        "can_be_indexed": True,
        "requires_pii_scan": True,
        "allowed_in_context": True,
        "log_access": True,
    },
    DataClassification.RESTRICTED: {
        "can_be_indexed": True,
        "requires_pii_scan": True,
        "requires_redaction_before_indexing": True,
        "allowed_in_context": True,
        "log_access": True,
        "requires_explicit_user_consent": True,
    },
    DataClassification.SECRET: {
        "can_be_indexed": False,   # Never index SECRET documents
        "allowed_in_context": False,
        "log_access": True,
    },
}

class DocumentClassifier:
    """
    Classifies documents before ingestion based on content analysis.
    Determines whether and how a document can be indexed.
    """
    def __init__(self, pii_detector: RegexPIIDetector):
        self.pii_detector = pii_detector

    def classify(self, document: dict) -> DocumentClassification:
        content = document.get("content", "")
        metadata = document.get("metadata", {})

        # Explicit classification in metadata takes precedence
        if "classification" in metadata:
            classification = DataClassification(
                metadata["classification"].lower()
            )
        else:
            classification = DataClassification.INTERNAL

        # PII scan always runs for INTERNAL and above
        pii_result = self.pii_detector.scan(content)

        # Upgrade to RESTRICTED if PII detected and not already higher
        if pii_result.has_pii and classification in (
            DataClassification.PUBLIC, DataClassification.INTERNAL
        ):
            classification = DataClassification.CONFIDENTIAL

        return DocumentClassification(
            document_id=document.get("id", ""),
            classification=classification,
            contains_pii=pii_result.has_pii,
            pii_types=pii_result.entity_types,
            allowed_team_ids=metadata.get("allowed_teams", []),
            requires_redaction=pii_result.has_pii,
            data_residency=metadata.get("data_residency", "any"),
            retention_days=metadata.get("retention_days", 0)
        )

    def can_ingest(self, classification: DocumentClassification) -> tuple[bool, str]:
        policy = CLASSIFICATION_POLICIES.get(classification.classification, {})
        if not policy.get("can_be_indexed", False):
            return False, f"Classification {classification.classification} prohibits indexing"
        return True, "approved"
```

---

### 3.3 Secrets Detection in Documents

```python
import re
from dataclasses import dataclass

@dataclass
class SecretMatch:
    secret_type: str
    location: str           # "content" | "metadata" | "filename"
    severity: str           # "critical" | "high" | "medium"
    masked_value: str       # Show first/last chars only

class SecretsDetector:
    """
    Detects secrets (credentials, API keys, tokens) in documents
    before ingestion. Prevents credentials from entering the vector index
    and appearing in RAG responses.
    """
    PATTERNS = {
        "openai_api_key": (
            r'sk-[A-Za-z0-9]{48}',
            "critical"
        ),
        "anthropic_api_key": (
            r'sk-ant-[A-Za-z0-9\-]{95,}',
            "critical"
        ),
        "aws_access_key": (
            r'AKIA[0-9A-Z]{16}',
            "critical"
        ),
        "aws_secret_key": (
            r'[0-9a-zA-Z/+]{40}',  # Combined with context
            "high"
        ),
        "github_token": (
            r'(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}',
            "critical"
        ),
        "google_api_key": (
            r'AIza[0-9A-Za-z\-_]{35}',
            "critical"
        ),
        "jwt_token": (
            r'eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+',
            "high"
        ),
        "private_key_pem": (
            r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
            "critical"
        ),
        "password_in_url": (
            r'[a-zA-Z]+://[^:@\s]+:[^@\s]+@',
            "critical"
        ),
        "generic_api_key": (
            r'(api[_\-]?key|apikey|api[_\-]?token)\s*[=:]\s*["\']?[A-Za-z0-9\-_]{20,}',
            "high"
        ),
        "connection_string": (
            r'(Server|Host|Data Source)\s*=\s*[^\s;]+.*'
            r'(Password|Pwd)\s*=\s*[^\s;]+',
            "critical"
        ),
    }

    def __init__(self):
        self._compiled = {
            name: (re.compile(pattern, re.IGNORECASE), severity)
            for name, (pattern, severity) in self.PATTERNS.items()
        }

    def scan(self, document: dict) -> list[SecretMatch]:
        matches = []
        check_targets = {
            "content":  document.get("content", ""),
            "metadata": str(document.get("metadata", {})),
        }
        if "filename" in document:
            check_targets["filename"] = document["filename"]

        for location, text in check_targets.items():
            for secret_type, (pattern, severity) in self._compiled.items():
                for m in pattern.finditer(text):
                    raw = m.group()
                    # Mask: show first 4 and last 4 chars only
                    masked = (raw[:4] + "****" + raw[-4:]) if len(raw) > 8 else "****"
                    matches.append(SecretMatch(
                        secret_type=secret_type,
                        location=location,
                        severity=severity,
                        masked_value=masked
                    ))
        return matches

    def is_clean(self, document: dict) -> tuple[bool, list[SecretMatch]]:
        matches = self.scan(document)
        critical = [m for m in matches if m.severity == "critical"]
        return len(critical) == 0, matches
```

---

### 3.4 Tenant Isolation and Data Boundary Enforcement

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TenantContext:
    tenant_id: str
    user_id: str
    team_id: str
    classification_clearance: DataClassification
    allowed_corpus_ids: list[str]

class TenantIsolationEnforcer:
    """
    Enforces tenant data boundaries at every access point:
    - Vector search: mandatory tenant_id filter
    - Document access: classification clearance check
    - Response assembly: strip disallowed content before returning
    """
    def __init__(self, vector_db_client):
        self.vdb = vector_db_client

    def build_tenant_filter(self, ctx: TenantContext) -> dict:
        """
        Build a mandatory vector DB filter that cannot be overridden by client.
        Applied server-side by the isolation layer before every query.
        """
        return {
            "must": [
                {"key": "tenant_id", "match": {"value": ctx.tenant_id}},
                {"key": "corpus_id", "match": {"any": ctx.allowed_corpus_ids}},
            ]
        }

    def search(
        self,
        ctx: TenantContext,
        query_vector: list[float],
        top_k: int = 5,
        client_filter: Optional[dict] = None
    ) -> list[dict]:
        """
        Tenant-isolated search. Client-supplied filters are additive only —
        they cannot remove the mandatory tenant_id filter.
        """
        # Start with mandatory tenant isolation filter
        mandatory_filter = self.build_tenant_filter(ctx)

        # Merge client filter as additional constraint (not replacement)
        if client_filter:
            mandatory_filter["must"].append(client_filter)

        results = self.vdb.search(
            query_vector=query_vector,
            filter=mandatory_filter,
            top_k=top_k
        )

        # Post-retrieval classification check
        return [
            r for r in results
            if self._has_clearance(ctx, r.get("metadata", {}))
        ]

    def _has_clearance(self, ctx: TenantContext, metadata: dict) -> bool:
        doc_classification = DataClassification(
            metadata.get("classification", "internal")
        )
        clearance_order = list(DataClassification)
        return (clearance_order.index(ctx.classification_clearance) >=
                clearance_order.index(doc_classification))
```

---

> ### 📋 Chapter Summary
>
> - **PII detection** combines regex patterns for structured PII (email, phone, SSN, IBAN) with optional NLP-based detection (Presidio) for unstructured PII.
> - **Document classification** (PUBLIC → SECRET) is determined at ingestion time; SECRET documents are never indexed.
> - **Secrets detection** scans documents for API keys, tokens, and credentials before ingestion — preventing credentials from appearing in RAG responses.
> - **Tenant isolation** applies mandatory server-side filters to every vector search; client-supplied filters are additive only and cannot remove the tenant constraint.

---

> ### ❓ Comprehension Questions
>
> 1. `RegexPIIDetector` matches UK NHS numbers as `\b\d{3}\s?\d{3}\s?\d{4}\b`. A product price "£123 456 7890" also matches this pattern. How would you reduce false positives for this pattern while retaining sensitivity for genuine NHS numbers?
> 2. `DocumentClassifier.classify()` upgrades a document to CONFIDENTIAL if PII is detected. A legal contract contains names and emails but also represents strategic business intelligence. Should the classification be RESTRICTED rather than CONFIDENTIAL? How would you implement context-aware classification upgrade logic?
> 3. The `TenantIsolationEnforcer` applies a `tenant_id` filter server-side. A developer bypasses the enforcer and calls the vector DB directly from the application. What access control at the infrastructure level would prevent this bypass?
> 4. `SecretsDetector` finds an OpenAI API key in a document during ingestion. The document should not be rejected outright — it may contain valuable information. Design a remediation workflow: what happens to the document, who is notified, and what audit trail is created?
> 5. PII redaction replaces email addresses with `[EMAIL]`. A user queries "What documents mention j.smith@company.com?" The redacted index cannot answer accurately. How would you design a system that satisfies both privacy (no PII in index) and searchability (query by email)?

---

## References

### Documentation
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection and anonymisation.
- [AWS Macie](https://docs.aws.amazon.com/macie/latest/user/what-is-macie.html) — Managed PII detection in S3.
- [detect-secrets](https://github.com/Yelp/detect-secrets) — Secrets scanning library.
- [truffleHog](https://github.com/trufflesecurity/trufflehog) — Secrets scanning in git history.
- [Qdrant Payload Filtering](https://qdrant.tech/documentation/concepts/filtering/) — Server-side filter syntax.

---

---
[« Back to security Index](index.md) | [🏠 Home](../../index.md)