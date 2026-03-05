## Links
- [Related ADR or documentation]
"""

VALID_STATUSES = {"Proposed", "Accepted", "Deprecated", "Superseded"}

def cmd_new(title: str) -> Path:
    ADR_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(ADR_DIR.glob("ADR-*.md"))
    number = 1
    if existing:
        nums = [int(re.search(r'ADR-(\d+)', f.stem).group(1)) for f in existing
                if re.search(r'ADR-(\d+)', f.stem)]
        number = max(nums) + 1 if nums else 1
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    filepath = ADR_DIR / f"ADR-{number:03d}-{slug}.md"
    filepath.write_text(TEMPLATE.format(
        number=number, title=title,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    ))
    print(f"Created: {filepath}")
    return filepath

def cmd_list() -> list[dict]:
    adrs = []
    for f in sorted(ADR_DIR.glob("ADR-*.md")):
        text = f.read_text()
        title = re.search(r'# ADR-\d+: (.+)', text)
        status = re.search(r'\*\*Status:\*\*\s*(\w+)', text)
        date = re.search(r'\*\*Date:\*\*\s*([\d-]+)', text)
        adrs.append({
            "file": f.name,
            "title": title.group(1) if title else "—",
            "status": status.group(1) if status else "MISSING",
            "date": date.group(1) if date else "—",
        })
    if adrs:
        print(f"{'File':<35} {'Status':<12} {'Date':<12} {'Title'}")
        print("─" * 90)
        for a in adrs:
            print(f"{a['file']:<35} {a['status']:<12} {a['date']:<12} {a['title']}")
    else:
        print("No ADRs found in docs/adr/")
    return adrs

def cmd_validate() -> int:
    errors = []
    warnings = []
    files = list(ADR_DIR.glob("ADR-*.md"))
    if not files:
        print("No ADRs to validate.")
        return 0

    for f in sorted(files):
        text = f.read_text()
        # Required sections
        required = ["## Context", "## Decision", "## Consequences"]
        for section in required:
            if section not in text:
                errors.append(f"{f.name}: missing section '{section}'")
        # Valid status
        status_match = re.search(r'\*\*Status:\*\*\s*(\w+)', text)
        if not status_match:
            errors.append(f"{f.name}: missing **Status:**")
        elif status_match.group(1) not in VALID_STATUSES:
            errors.append(f"{f.name}: invalid status '{status_match.group(1)}' — must be one of {VALID_STATUSES}")
        # Placeholder detection
        if "[List decision makers]" in text or "[What is the problem?" in text:
            warnings.append(f"{f.name}: contains unfilled template placeholders")

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR:   {e}", file=sys.stderr)

    total = len(files)
    print(f"\nValidated {total} ADR(s): {len(errors)} error(s), {len(warnings)} warning(s)")
    return len(errors)

def cmd_accept(adr_file: str):
    filepath = ADR_DIR / adr_file
    if not filepath.exists():
        print(f"Not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    text = filepath.read_text()
    updated = re.sub(r'\*\*Status:\*\*\s*\w+', '**Status:** Accepted', text)
    filepath.write_text(updated)
    print(f"Updated status to Accepted: {filepath.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADR management CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_new = sub.add_parser("new", help="Create a new ADR")
    p_new.add_argument("title", help="ADR title")

    sub.add_parser("list", help="List all ADRs")
    sub.add_parser("validate", help="Validate ADR format (for CI)")

    p_accept = sub.add_parser("accept", help="Mark ADR as Accepted")
    p_accept.add_argument("file", help="ADR filename (e.g. ADR-007-vector-database.md)")

    args = parser.parse_args()

    if args.cmd == "new":
        cmd_new(args.title)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "validate":
        sys.exit(cmd_validate())
    elif args.cmd == "accept":
        cmd_accept(args.file)
    else:
        parser.print_help()
```

**Run the lab:**
```bash
# Create a new ADR
python scripts/adr.py new "Embedding Model Selection"

# List ADRs
python scripts/adr.py list

# Validate format (CI gate)
python scripts/adr.py validate

# Accept a decision
python scripts/adr.py accept ADR-001-embedding-model-selection.md
```

**Extensions:**
- Add a `supersede` command that marks an existing ADR as Superseded and links to the new one
- Add a `--check-links` flag to `validate` that verifies all `[text](url)` links in ADRs are reachable
- Integrate `validate` into the CI workflow so that invalid ADRs block merges

---

> ### 📋 Chapter Summary
>
> - **Documentation as code** applies version control, review, and quality discipline to architectural documentation.
> - **ADRs** record significant decisions with context, options considered, rationale, and consequences — answering "why" when the code only shows "what".
> - **Runbooks as code** are executable, testable operational procedures versioned alongside the systems they describe.
> - **OpenAPI specs** auto-generated from code remain always current and enable breaking-change detection in CI.
> - **Automated documentation pipelines** enforce ADR format, detect API breaking changes, and publish documentation on every merge.

---

> ### ❓ Comprehension Questions
>
> 1. Six months after deployment, the team debates changing the chunk size from 512 to 1024 tokens. With an ADR for the original decision, what information is immediately available that would otherwise be lost?
> 2. A runbook is written as a Markdown document with manual steps. Compare this with the Python runbook approach in section 4.3. What are the operational advantages of executable runbooks?
> 3. The API documentation pipeline detects that a field was removed from `QueryResponse`. How should the CI pipeline handle this, and what process should follow?
> 4. An ADR status is "Proposed" for six months with no decision made. What organisational anti-pattern does this indicate, and how would you address it?
> 5. Describe a strategy for keeping ADRs current as the system evolves. When should an ADR be superseded vs updated in place?

---

## References

### Documentation
- [ADR GitHub Organisation](https://adr.github.io) — Architecture Decision Records tooling and examples.
- [Michael Nygard's ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — Original ADR blog post.
- [FastAPI Documentation](https://fastapi.tiangolo.com) — Auto-generated OpenAPI docs.
- [Swagger/OpenAPI Specification](https://swagger.io/specification/) — OpenAPI 3.1 reference.
- [adr-tools](https://github.com/npryce/adr-tools) — Shell-based ADR management.

### Books
- *Documenting Architecture Decisions* — Michael Keeling (O'Reilly). ADR patterns and practices.
- *The DevOps Handbook* — Kim, Humble, Debois, Willis (IT Revolution). Documentation in DevOps culture.

---

> **Navigation**
> [← Part VI — Artifact Engineering](part_06_artifact_engineering.md) | [→ Part VIII — AI Systems SDLC](part_08_sdlc.md)

---
[« Back to layouts_repositories Index](index.md) | [🏠 Home](../../index.md)