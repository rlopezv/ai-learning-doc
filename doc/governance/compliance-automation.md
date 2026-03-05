## Chapter 4 — Compliance Automation 🧪

### 4.1 Policy as Code

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

class PolicyOutcome(str, Enum):
    PASS    = "pass"
    FAIL    = "fail"
    WARNING = "warning"
    SKIP    = "skip"   # Policy not applicable to this system

@dataclass
class PolicyCheck:
    policy_id: str
    name: str
    description: str
    category: str       # "data" | "model" | "operations" | "legal"
    severity: str       # "critical" | "high" | "medium" | "low"
    check_fn: Callable[[dict], tuple[PolicyOutcome, str]]
    remediation: str

def check_model_card_exists(system_context: dict) -> tuple[PolicyOutcome, str]:
    from pathlib import Path
    system_id = system_context.get("system_id", "")
    card_path = Path(f"governance/artefacts/{system_id}/model_card")
    if not card_path.exists() or not list(card_path.glob("*.json")):
        return PolicyOutcome.FAIL, "No model card found"
    return PolicyOutcome.PASS, "Model card present"

def check_pii_scan_on_corpus(system_context: dict) -> tuple[PolicyOutcome, str]:
    last_scan = system_context.get("last_pii_scan_date")
    if not last_scan:
        return PolicyOutcome.FAIL, "No PII scan recorded"
    from datetime import datetime, timedelta
    last = datetime.fromisoformat(last_scan)
    if (datetime.utcnow() - last).days > 90:
        return PolicyOutcome.WARNING, f"PII scan is {(datetime.utcnow()-last).days} days old"
    return PolicyOutcome.PASS, "PII scan recent"

def check_audit_logging_enabled(system_context: dict) -> tuple[PolicyOutcome, str]:
    if not system_context.get("audit_logging_enabled", False):
        return PolicyOutcome.FAIL, "Audit logging not enabled"
    retention = system_context.get("audit_log_retention_days", 0)
    if retention < 180:
        return PolicyOutcome.FAIL, f"Retention {retention}d < required 180d (EU AI Act)"
    return PolicyOutcome.PASS, f"Audit logging enabled, retention={retention}d"

def check_human_oversight_mechanism(system_context: dict) -> tuple[PolicyOutcome, str]:
    mechanisms = system_context.get("human_oversight_mechanisms", [])
    if not mechanisms:
        return PolicyOutcome.FAIL, "No human oversight mechanism documented"
    return PolicyOutcome.PASS, f"Oversight mechanisms: {mechanisms}"

def check_evaluation_report_current(system_context: dict) -> tuple[PolicyOutcome, str]:
    last_eval = system_context.get("last_evaluation_date")
    if not last_eval:
        return PolicyOutcome.FAIL, "No evaluation report found"
    from datetime import datetime
    last = datetime.fromisoformat(last_eval)
    age_days = (datetime.utcnow() - last).days
    if age_days > 90:
        return PolicyOutcome.WARNING, f"Evaluation report is {age_days} days old"
    return PolicyOutcome.PASS, "Evaluation report current"

COMPLIANCE_POLICIES = [
    PolicyCheck("POL-001", "Model Card Exists",
        "Every AI system must have a current model card",
        "model", "critical", check_model_card_exists,
        "Create model card using ModelCard dataclass and store via GovernanceArtefactStore"),

    PolicyCheck("POL-002", "PII Scan on Corpus",
        "Corpus must be PII-scanned within last 90 days",
        "data", "high", check_pii_scan_on_corpus,
        "Run PII detection pipeline on full corpus and record scan date"),

    PolicyCheck("POL-003", "Audit Logging Enabled",
        "Audit logging must be active with ≥180-day retention",
        "operations", "critical", check_audit_logging_enabled,
        "Enable AuditLogger with store retention configured to 180+ days"),

    PolicyCheck("POL-004", "Human Oversight Mechanism",
        "System must have documented human oversight controls",
        "legal", "high", check_human_oversight_mechanism,
        "Implement and document HumanInTheLoopGateway triggers"),

    PolicyCheck("POL-005", "Evaluation Report Current",
        "Evaluation report must be ≤90 days old",
        "model", "medium", check_evaluation_report_current,
        "Run evaluation pipeline and store report via GovernanceArtefactStore"),
]
```

---

### 4.2 Automated Compliance Checks in CI/CD

```python
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ComplianceRunResult:
    system_id: str
    run_at: str
    policies_checked: int
    passed: int
    failed: int
    warnings: int
    critical_failures: list[str]
    overall_status: str   # "compliant" | "non_compliant" | "warnings"
    details: list[dict]

class ComplianceRunner:
    def __init__(self, policies: list[PolicyCheck]):
        self.policies = policies

    def run(self, system_context: dict) -> ComplianceRunResult:
        from datetime import datetime
        results = []
        critical_failures = []

        for policy in self.policies:
            try:
                outcome, message = policy.check_fn(system_context)
            except Exception as e:
                outcome = PolicyOutcome.FAIL
                message = f"Check errored: {e}"

            results.append({
                "policy_id":  policy.policy_id,
                "name":       policy.name,
                "category":   policy.category,
                "severity":   policy.severity,
                "outcome":    outcome.value,
                "message":    message,
                "remediation": policy.remediation if outcome == PolicyOutcome.FAIL else None
            })
            if outcome == PolicyOutcome.FAIL and policy.severity == "critical":
                critical_failures.append(policy.policy_id)

        passed   = sum(1 for r in results if r["outcome"] == "pass")
        failed   = sum(1 for r in results if r["outcome"] == "fail")
        warnings = sum(1 for r in results if r["outcome"] == "warning")

        if critical_failures:
            status = "non_compliant"
        elif failed > 0:
            status = "non_compliant"
        elif warnings > 0:
            status = "warnings"
        else:
            status = "compliant"

        return ComplianceRunResult(
            system_id=system_context.get("system_id", "unknown"),
            run_at=datetime.utcnow().isoformat(),
            policies_checked=len(self.policies),
            passed=passed, failed=failed, warnings=warnings,
            critical_failures=critical_failures,
            overall_status=status,
            details=results
        )

    def run_and_exit(self, system_context: dict, fail_on: str = "non_compliant"):
        """CI/CD integration: exit non-zero on policy failure."""
        import sys
        result = self.run(system_context)
        print(f"\nCompliance Check: {result.overall_status.upper()}")
        print(f"  Passed:   {result.passed}/{result.policies_checked}")
        print(f"  Failed:   {result.failed}")
        print(f"  Warnings: {result.warnings}")
        if result.critical_failures:
            print(f"\nCRITICAL FAILURES: {result.critical_failures}")
            for detail in result.details:
                if detail["outcome"] == "fail":
                    print(f"  [{detail['policy_id']}] {detail['message']}")
                    print(f"    → {detail['remediation']}")
        if result.overall_status == fail_on or (
            fail_on == "non_compliant" and result.overall_status == "non_compliant"
        ):
            sys.exit(1)
        return result
```

```yaml
# .github/workflows/compliance.yml
name: AI Governance Compliance Check
on:
  push:
    branches: [main, release/*]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"   # Weekly Monday 06:00 UTC

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.lock

      - name: Run compliance checks
        run: python scripts/run_compliance.py --system-id rag-support-api --fail-on non_compliant
        env:
          GOVERNANCE_STORE_PATH: ${{ github.workspace }}/governance

      - name: Upload compliance report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: compliance-report
          path: governance/compliance_reports/
```

---

### 4.3 Evidence Collection and Audit Readiness

```python
from pathlib import Path
import json, zipfile
from datetime import datetime

class AuditEvidencePackage:
    """
    Assembles a complete evidence package for a compliance audit.
    Collects all governance artefacts, compliance reports, and
    incident logs into a structured, signed ZIP archive.
    """
    def __init__(self, system_id: str,
                 governance_store: "GovernanceArtefactStore",
                 audit_logger: "AuditLogger"):
        self.system_id = system_id
        self.gov_store = governance_store
        self.audit_logger = audit_logger

    def assemble(self, output_path: str, period_start: str, period_end: str) -> str:
        """
        Assemble audit package for a specified period.
        Returns path to the ZIP archive.
        """
        package_id = f"{self.system_id}_{period_start}_{period_end}"
        zip_path = Path(output_path) / f"{package_id}_audit_evidence.zip"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        manifest = {
            "system_id":     self.system_id,
            "period_start":  period_start,
            "period_end":    period_end,
            "assembled_at":  datetime.utcnow().isoformat(),
            "contents":      []
        }

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Model card (current version)
            model_card = self.gov_store.get_current(self.system_id, "model_card")
            if model_card:
                zf.writestr("model_card.json", json.dumps(model_card, indent=2))
                manifest["contents"].append("model_card.json")

            # 2. All compliance check reports in period
            reports_dir = Path(f"governance/compliance_reports/{self.system_id}")
            if reports_dir.exists():
                for report in sorted(reports_dir.glob("*.json")):
                    data = json.loads(report.read_text())
                    if period_start <= data.get("run_at", "") <= period_end:
                        zf.write(report, f"compliance_reports/{report.name}")
                        manifest["contents"].append(f"compliance_reports/{report.name}")

            # 3. Incident log
            incidents_dir = Path("governance/incidents")
            if incidents_dir.exists():
                for inc_file in sorted(incidents_dir.glob("*.json")):
                    data = json.loads(inc_file.read_text())
                    if (period_start <= data.get("detected_at", "") <= period_end
                            and data.get("affected_system_id") == self.system_id):
                        zf.write(inc_file, f"incidents/{inc_file.name}")
                        manifest["contents"].append(f"incidents/{inc_file.name}")

            # 4. Audit log chain verification report
            valid, count, error = self.audit_logger.verify_chain()
            chain_report = {
                "verified_at": datetime.utcnow().isoformat(),
                "records_verified": count,
                "chain_intact": valid,
                "error": error
            }
            zf.writestr("audit_chain_verification.json",
                        json.dumps(chain_report, indent=2))
            manifest["contents"].append("audit_chain_verification.json")

            # 5. Manifest
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

        print(f"  ✓ Audit package assembled: {zip_path}")
        print(f"    {len(manifest['contents'])} documents included")
        return str(zip_path)
```

---

### 🧪 Hands-on Lab: Governance Scorecard

**Objective:** Run a governance compliance check on a simulated AI system context and produce a human-readable scorecard.

```python
#!/usr/bin/env python3
"""
governance_scorecard.py — Governance compliance check demo.
No external dependencies required.
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

# ── Inline minimal policy framework ───────────────────────────────
class PolicyOutcome(str, Enum):
    PASS    = "pass"
    FAIL    = "fail"
    WARNING = "warning"

@dataclass
class Policy:
    id: str; name: str; severity: str; check: Callable; remediation: str

def _days_ago(d: str) -> int:
    return (datetime.utcnow() - datetime.fromisoformat(d)).days

POLICIES = [
    Policy("P01", "Model card exists",     "critical",
           lambda ctx: (PolicyOutcome.PASS, "present")
           if ctx.get("model_card_version") else
           (PolicyOutcome.FAIL, "No model card found"),
           "Create and store a ModelCard document"),

    Policy("P02", "Corpus PII scan",       "high",
           lambda ctx: (PolicyOutcome.PASS, "recent")
           if ctx.get("pii_scan_date") and _days_ago(ctx["pii_scan_date"]) <= 90
           else (PolicyOutcome.WARNING, f"Scan is {_days_ago(ctx.get('pii_scan_date','2000-01-01'))}d old")
           if ctx.get("pii_scan_date") else
           (PolicyOutcome.FAIL, "No scan recorded"),
           "Run PII scan on full corpus"),

    Policy("P03", "Audit log retention",   "critical",
           lambda ctx: (PolicyOutcome.PASS, f"{ctx.get('audit_retention_days',0)}d")
           if ctx.get("audit_logging_enabled") and ctx.get("audit_retention_days", 0) >= 180
           else (PolicyOutcome.FAIL, "Logging disabled or retention < 180d"),
           "Enable audit logging with 180+ day retention"),

    Policy("P04", "Evaluation current",    "medium",
           lambda ctx: (PolicyOutcome.PASS, "recent")
           if ctx.get("last_eval_date") and _days_ago(ctx["last_eval_date"]) <= 90
           else (PolicyOutcome.WARNING, "Evaluation > 90 days old")
           if ctx.get("last_eval_date") else
           (PolicyOutcome.FAIL, "No evaluation report"),
           "Run evaluation pipeline and store report"),

    Policy("P05", "Human oversight",       "high",
           lambda ctx: (PolicyOutcome.PASS, str(ctx.get("oversight_mechanisms", [])))
           if ctx.get("oversight_mechanisms") else
           (PolicyOutcome.FAIL, "No oversight mechanism documented"),
           "Implement HumanInTheLoopGateway"),

    Policy("P06", "EU AI Act classification","critical",
           lambda ctx: (PolicyOutcome.PASS, ctx["eu_ai_act_tier"])
           if ctx.get("eu_ai_act_tier") else
           (PolicyOutcome.FAIL, "No EU AI Act classification"),
           "Classify system under EU AI Act risk tiers"),

    Policy("P07", "Incident response plan","medium",
           lambda ctx: (PolicyOutcome.PASS, "documented")
           if ctx.get("incident_response_plan") else
           (PolicyOutcome.WARNING, "Incident response plan missing"),
           "Document incident response plan for AI-specific incident types"),
]

# ── System context: simulated AI system registry ───────────────────
SYSTEMS = [
    {
        "system_id": "rag-support-v2",
        "name": "Customer Support RAG",
        "eu_ai_act_tier": "limited_risk",
        "model_card_version": "2.1.0",
        "pii_scan_date": (datetime.utcnow() - timedelta(days=45)).isoformat(),
        "audit_logging_enabled": True,
        "audit_retention_days": 180,
        "last_eval_date": (datetime.utcnow() - timedelta(days=20)).isoformat(),
        "oversight_mechanisms": ["human_review_queue", "refusal_fallback"],
        "incident_response_plan": True,
    },
    {
        "system_id": "hiring-screener-v1",
        "name": "Resume Screening RAG",
        "eu_ai_act_tier": "high_risk",
        "model_card_version": None,    # Missing
        "pii_scan_date": None,          # Missing
        "audit_logging_enabled": True,
        "audit_retention_days": 90,     # Too short
        "last_eval_date": (datetime.utcnow() - timedelta(days=120)).isoformat(),
        "oversight_mechanisms": [],     # Missing
        "incident_response_plan": False,
    },
]

# ── Run and render scorecard ───────────────────────────────────────
ICONS = {PolicyOutcome.PASS: "✓", PolicyOutcome.FAIL: "✗", PolicyOutcome.WARNING: "⚠"}
COLORS = {PolicyOutcome.PASS: "", PolicyOutcome.FAIL: "FAIL ", PolicyOutcome.WARNING: "WARN "}

for system in SYSTEMS:
    print(f"\n{'='*58}")
    print(f"  Governance Scorecard: {system['name']}")
    print(f"  System ID: {system['system_id']}")
    print(f"  EU AI Act tier: {system.get('eu_ai_act_tier','unknown').upper()}")
    print(f"{'='*58}")

    results = []
    for policy in POLICIES:
        outcome, message = policy.check(system)
        results.append((policy, outcome, message))

    passed   = sum(1 for _,o,_ in results if o == PolicyOutcome.PASS)
    failed   = sum(1 for _,o,_ in results if o == PolicyOutcome.FAIL)
    warnings = sum(1 for _,o,_ in results if o == PolicyOutcome.WARNING)
    critical_fails = [p.id for p,o,_ in results if o == PolicyOutcome.FAIL and p.severity == "critical"]

    for policy, outcome, message in results:
        icon = ICONS[outcome]
        print(f"  {icon} [{policy.id}] {policy.name:<30} {message}")
        if outcome == PolicyOutcome.FAIL:
            print(f"       ↳ {policy.remediation}")

    print(f"\n  Score: {passed}/{len(POLICIES)} passed  "
          f"| {failed} failed | {warnings} warnings")

    if critical_fails:
        print(f"  CRITICAL FAILURES: {critical_fails}")
        print(f"  STATUS: NON-COMPLIANT ← blocks production deployment")
    elif failed:
        print(f"  STATUS: NON-COMPLIANT")
    elif warnings:
        print(f"  STATUS: COMPLIANT WITH WARNINGS")
    else:
        print(f"  STATUS: FULLY COMPLIANT ✓")
```

**Run the lab:**
```bash
python governance_scorecard.py
```

**Extensions:**
- Add a `--fix` flag that automatically generates skeleton artefacts for failing policies (empty model card, PII scan trigger)
- Export the scorecard to `governance/compliance_reports/{system_id}_{date}.json` for audit trail
- Add policy `P08`: check that `last_eval_date` faithfulness score ≥ 0.80 by reading the evaluation JSON report

---

> ### 📋 Chapter Summary
>
> - **Policy as code** encodes governance requirements as executable `check_fn` callables, making compliance checks runnable in CI/CD pipelines.
> - `ComplianceRunner.run_and_exit` provides a standard CI/CD integration that exits non-zero on critical policy failures — blocking deploys of non-compliant systems.
> - **Audit evidence packages** assemble all governance artefacts (model card, compliance reports, incident logs, audit chain verification) into a signed ZIP — ready for regulatory inspection.
> - The governance scorecard lab demonstrates a complete compliance gate: two systems, 7 policies, clear COMPLIANT vs NON-COMPLIANT verdict.

---

> ### ❓ Comprehension Questions
>
> 1. `check_pii_scan_on_corpus` returns `WARNING` if the scan is more than 90 days old. A corpus that updates daily could contain newly ingested PII within 24 hours of a scan. Should the threshold be time-based or event-based (scan triggered on each ingestion)? Argue both approaches.
> 2. `ComplianceRunner.run_and_exit` blocks a production deploy when `status == "non_compliant"`. A compliance failure is detected on a Friday afternoon. The fix requires a model card to be written — a 2-hour task. Is blocking the deploy the correct control, or should a time-bounded waiver process exist?
> 3. `AuditEvidencePackage.assemble` includes the audit chain verification result. An auditor asks: "How do you know the audit log was not tampered with before the evidence package was assembled?" What additional mechanism would strengthen this assurance?
> 4. The CI/CD compliance workflow runs on `push` to `main` and on a weekly schedule. A governance artefact becomes stale (evaluation > 90 days old) between pushes. How long can this go undetected, and how would you reduce the detection window?
> 5. `POLICIES` contains 7 checks. A senior engineer proposes adding `P08: LLM provider contract reviewed annually`. This requires checking an external document management system. What are the implications for the `check_fn` interface, CI/CD execution time, and failure handling?

---

## References

### Documentation
- [Open Policy Agent (OPA)](https://www.openpolicyagent.org/docs/latest/) — Policy as code engine.
- [EU AI Act Conformity Assessment](https://artificialintelligenceact.eu/assessment/)
- [NIST SP 800-53](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final) — Security and privacy controls.
- [SOC 2 Compliance Guide](https://www.aicpa.org/resources/landing/system-and-organization-controls-soc-suite-of-services)

### Papers
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Mitchell et al., 2019.
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) — Gebru et al., 2021.

---

> **Navigation**
> [← Part XIV — Security](part_14_security.md) | [→ Part XVI — Operations](part_16_operations.md)

---
[« Back to governance Index](index.md) | [🏠 Home](../../index.md)