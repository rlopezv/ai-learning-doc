## Rate Limits
- Standard: 100 req/min
- Enterprise: 1000 req/min
""",
    contact={"name": "Platform Team", "email": "platform@company.com"},
    openapi_tags=[
        {"name": "query", "description": "Knowledge base query operations"},
        {"name": "admin", "description": "Administrative operations"},
    ]
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="The question to answer from the knowledge base",
                          example="What is the enterprise refund policy?")
    language: Optional[str] = Field("en", description="ISO 639-1 language code for response",
                                    example="en")
    top_k: Optional[int] = Field(5, ge=1, le=20,
                                 description="Number of source documents to retrieve")

class Source(BaseModel):
    document_id: str
    title: str
    excerpt: str = Field(..., description="Relevant excerpt from the source document")
    relevance_score: float = Field(..., ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    query_id: str = Field(..., description="Unique identifier for this query, for feedback")
    latency_ms: int

@app.post(
    "/v2/query",
    response_model=QueryResponse,
    tags=["query"],
    summary="Query the knowledge base",
    response_description="Answer with supporting source documents"
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    Submit a natural language question and receive an answer grounded in
    the internal knowledge base with source citations.

    The response includes the answer text and up to `top_k` source documents
    that were used to generate the answer.
    """
    # Implementation here
    raise HTTPException(status_code=501, detail="Not implemented in example")
```

---

### 4.5 Automated Documentation Pipelines

```yaml
# .github/workflows/docs.yml
name: Documentation Pipeline

on:
  push:
    branches: [main]
  pull_request:
    paths:
      - 'docs/**'
      - 'apps/**/*.py'
      - 'prompts/**'

jobs:
  validate-adrs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate ADR format
        run: python scripts/adr_manager.py --validate
      - name: Check ADRs have status
        run: |
          python3 -c "
          import re
          from pathlib import Path
          errors = []
          for f in Path('docs/adr').glob('ADR-*.md'):
              if not re.search(r'\*\*Status:\*\*\s*(Accepted|Proposed|Deprecated|Superseded)', f.read_text()):
                  errors.append(str(f))
          if errors:
              print('ADRs missing valid status:', errors)
              exit(1)
          print(f'All {len(list(Path(\"docs/adr\").glob(\"ADR-*.md\")))} ADRs have valid status')
          "

  generate-api-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install fastapi[all]
      - name: Export OpenAPI spec
        run: |
          python3 -c "
          import json
          from apps.api.src.main import app
          spec = app.openapi()
          open('docs/api/openapi.json', 'w').write(json.dumps(spec, indent=2))
          "
      - name: Check for breaking API changes
        run: |
          # Compare current spec with last released spec
          diff docs/api/openapi.json docs/api/openapi.last-release.json || \
            echo "::warning::API spec changed — review for breaking changes"
```

---

### 🧪 Hands-on Lab: ADR Pipeline

**Objective:** Build an ADR creation and validation CLI. Demonstrate the complete documentation-as-code workflow.

**Prerequisites:** Python standard library only.

```python
#!/usr/bin/env python3
# scripts/adr.py — ADR management CLI

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

ADR_DIR = Path("docs/adr")

TEMPLATE = """# ADR-{number:03d}: {title}

**Date:** {date}
**Status:** Proposed
**Deciders:** [List decision makers]

---
[« Back to layouts_repositories Index](index.md) | [🏠 Home](../../index.md)