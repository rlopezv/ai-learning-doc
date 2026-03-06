## Chapter 1 — Repository Structure for AI Systems

### 1.1 Why Repository Layout Matters

Repository layout is an architectural decision, not a cosmetic preference. It determines:

- **Where developers look** for specific components, reducing navigation overhead
- **Where CI/CD pipelines look** for build targets, test suites, and deployment scripts
- **Where security scanners look** for secrets, dependencies, and configuration
- **How change blast radius is contained** — a prompt change in a well-structured repo triggers only prompt tests; in a poorly structured one, it triggers a full system rebuild

AI systems have unique layout requirements compared to traditional applications. They carry multiple artifact types (prompts, datasets, models, indexes), multiple runtime environments (cloud API, on-premise Ollama, GPU serving), and a mixed language reality (Python for ML workloads, Java for enterprise services). A layout that works for a pure Python ML project fails when enterprise Java services are added, and vice versa.

The canonical layout described in this chapter accommodates this reality from the start.

---

### 1.2 Monorepo vs Polyrepo

The first structural decision is whether AI system components live in a single repository (monorepo) or separate repositories (polyrepo).

**Monorepo advantages** for AI systems:
- Atomic commits across prompt + code + evaluation changes
- Shared tooling (linters, formatters, CI templates) defined once
- Cross-component refactors in a single PR
- Dataset and prompt versions co-located with the code that uses them
- Simpler dependency management between internal components

**Polyrepo advantages**:
- Independent release cadences per component
- Cleaner access control boundaries (e.g., separate repo for fine-tuning dataset containing PII)
- Smaller clone sizes for components that do not need the full system

**Recommendation for enterprises:** Start with a monorepo. Extract to polyrepo only when a component has demonstrably different release cadence, team ownership, or access control requirements. The cost of a premature polyrepo split is high; the cost of a late extraction is manageable.

```
Monorepo extraction triggers:
✓ Component has a separate team with independent release schedule
✓ Component contains data requiring stricter access control than the rest of the repo
✓ Component build time dominates CI and cannot be optimised within the monorepo
✗ "It feels cleaner" — not sufficient justification
```

---

### 1.3 Canonical Monorepo Layout

```
ai-system/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # Test + lint on every PR
│   │   ├── eval.yml             # Evaluation pipeline on merge to main
│   │   └── deploy.yml           # Deployment pipeline with promotion gates
│   └── CODEOWNERS
│
├── apps/                        # Deployable services
│   ├── api/                     # Main REST API (Java/Spring Boot)
│   │   ├── src/
│   │   ├── pom.xml
│   │   └── Dockerfile
│   ├── ingestion/               # Document ingestion pipeline (Python)
│   │   ├── src/
│   │   ├── pyproject.toml
│   │   └── Dockerfile
│   └── eval-worker/             # Evaluation pipeline worker (Python)
│       ├── src/
│       └── pyproject.toml
│
├── libs/                        # Shared internal libraries
│   ├── rag-core/                # Core retrieval abstractions (Python)
│   ├── llm-client/              # Unified LLM client (Python + Java)
│   ├── evaluation/              # Evaluation metrics library (Python)
│   └── java-rag/                # Java RAG utilities (Maven module)
│
├── prompts/                     # Versioned prompt templates
│   ├── rag_qa/
│   │   ├── 1.0.0.json
│   │   ├── 1.1.0.json
│   │   └── latest.json          # Symlink to current production version
│   ├── summarisation/
│   └── tool_use/
│
├── datasets/                    # Dataset manifests (content in DVC remote)
│   ├── eval/
│   │   ├── eval_v1.jsonl.dvc    # DVC pointer
│   │   └── eval_v1.meta.json    # Metadata sidecar
│   └── finetune/
│
├── models/                      # Model cards and serving configs
│   ├── cards/
│   └── serving/
│
├── infrastructure/              # Infrastructure as Code
│   ├── terraform/
│   ├── kubernetes/
│   └── docker-compose.yml       # Local development stack
│
├── docs/                        # Documentation as code
│   ├── adr/                     # Architecture Decision Records
│   ├── runbooks/
│   └── api/                     # OpenAPI specs
│
├── scripts/                     # Developer and CI utility scripts
│   ├── setup_dev.sh
│   ├── run_evals.sh
│   └── promote_prompt.py
│
├── .dvc/                        # DVC configuration
├── .env.example                 # Environment variable template
├── pyproject.toml               # Root Python tooling config
└── Makefile                     # Top-level developer commands
```

This layout enforces three structural rules:

1. **`apps/` depends on `libs/`, never the reverse.** Services consume shared libraries; libraries do not import from services.
2. **`prompts/` and `datasets/` are first-class directories**, not buried inside `apps/`. They are system-level artifacts consumed by multiple components.
3. **`infrastructure/` is code**, not a separate infrastructure team's concern. The team that builds the system owns the infrastructure that runs it.

---

### 1.4 Module Boundaries and Dependency Rules

Explicit dependency rules prevent the accretion of implicit coupling between components.

```python
# scripts/check_imports.py
# Enforce architectural dependency rules via import scanning

import ast
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DependencyRule:
    source_pattern: str     # Glob pattern for source module
    forbidden_imports: list[str]  # Modules this source must NOT import
    reason: str

RULES = [
    DependencyRule(
        source_pattern="libs/**/*.py",
        forbidden_imports=["apps."],
        reason="Libraries must not import from applications"
    ),
    DependencyRule(
        source_pattern="libs/evaluation/**/*.py",
        forbidden_imports=["libs.rag_core.", "libs.llm_client."],
        reason="Evaluation library must be independent of RAG implementation"
    ),
    DependencyRule(
        source_pattern="apps/api/**/*.py",
        forbidden_imports=["apps.ingestion.", "apps.eval_worker."],
        reason="API service must not directly import from other services"
    ),
]

def check_file(path: Path, rule: DependencyRule) -> list[str]:
    """Return list of violations for a single file."""
    import fnmatch
    if not fnmatch.fnmatch(str(path), rule.source_pattern):
        return []

    violations = []
    try:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name if node.names else ""
                for forbidden in rule.forbidden_imports:
                    if module.startswith(forbidden):
                        violations.append(
                            f"{path}: imports '{module}' — forbidden by rule: {rule.reason}"
                        )
    except SyntaxError:
        pass
    return violations

def run_checks(repo_root: str = ".") -> int:
    """Run all dependency rules. Returns number of violations."""
    total_violations = []
    for rule in RULES:
        for path in Path(repo_root).glob(rule.source_pattern):
            total_violations.extend(check_file(path, rule))

    for v in total_violations:
        print(f"VIOLATION: {v}")
    print(f"\n{len(total_violations)} dependency violation(s) found")
    return len(total_violations)
```

---

### 1.5 Java Project Structure with Maven/Gradle

Java components in the monorepo use a multi-module Maven or Gradle structure that mirrors the top-level layout.

```xml
<!-- Root pom.xml — parent for all Java modules -->
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.company.ai</groupId>
  <artifactId>ai-system-parent</artifactId>
  <version>1.0.0-SNAPSHOT</version>
  <packaging>pom</packaging>

  <modules>
    <module>apps/api</module>
    <module>libs/java-rag</module>
    <module>libs/llm-client-java</module>
  </modules>

  <!-- Dependency management — all versions defined here -->
  <dependencyManagement>
    <dependencies>
      <!-- LangChain4j BOM — pins all langchain4j module versions -->
      <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-bom</artifactId>
        <version>0.36.2</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
      <!-- Spring Boot BOM -->
      <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-dependencies</artifactId>
        <version>3.3.4</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
    </dependencies>
  </dependencyManagement>

  <build>
    <plugins>
      <!-- Enforcer: forbid duplicate dependencies, enforce Java version -->
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-enforcer-plugin</artifactId>
        <version>3.5.0</version>
        <executions>
          <execution>
            <goals><goal>enforce</goal></goals>
            <configuration>
              <rules>
                <requireJavaVersion>
                  <version>[21,)</version>
                </requireJavaVersion>
                <banDuplicatePomDependencyVersions/>
              </rules>
            </configuration>
          </execution>
        </executions>
      </plugin>
    </plugins>
  </build>
</project>
```

**Gradle equivalent with version catalogue:**
```toml
# gradle/libs.versions.toml
[versions]
langchain4j = "0.36.2"
spring-boot = "3.3.4"
junit = "5.11.0"

[libraries]
langchain4j-core = { module = "dev.langchain4j:langchain4j", version.ref = "langchain4j" }
langchain4j-openai = { module = "dev.langchain4j:langchain4j-open-ai", version.ref = "langchain4j" }
spring-boot-starter = { module = "org.springframework.boot:spring-boot-starter", version.ref = "spring-boot" }
junit-jupiter = { module = "org.junit.jupiter:junit-jupiter", version.ref = "junit" }

[plugins]
spring-boot = { id = "org.springframework.boot", version.ref = "spring-boot" }
```

```groovy
// apps/api/build.gradle.kts
dependencies {
    implementation(libs.langchain4j.core)
    implementation(libs.langchain4j.openai)
    implementation(libs.spring.boot.starter)
    implementation(project(":libs:java-rag"))
    testImplementation(libs.junit.jupiter)
}
```

---

> ### 📋 Chapter Summary
>
> - Repository layout is an **architectural decision** that determines navigation overhead, CI/CD efficiency, and change blast radius.
> - **Monorepo** is recommended for AI systems: atomic cross-artifact commits, shared tooling, simpler dependency management.
> - The canonical layout segregates `apps/`, `libs/`, `prompts/`, `datasets/`, `infrastructure/`, and `docs/` as first-class directories.
> - **Explicit dependency rules** (checked in CI) prevent architectural coupling from accumulating silently.
> - Java multi-module Maven/Gradle structure mirrors the top-level layout with version management centralised in a parent POM or version catalogue.

---

> ### ❓ Comprehension Questions
>
> 1. A developer stores prompt templates inside `apps/api/src/resources/prompts/`. What problems does this create when the evaluation worker also needs to use those prompts?
> 2. Explain the "libraries must not import from applications" rule. What architectural pattern does this enforce and why does violating it cause problems at scale?
> 3. Your team proposes extracting the ingestion pipeline to a separate repository for independent deployments. Using the extraction triggers listed in section 1.2, evaluate whether this is justified.
> 4. The Maven enforcer plugin requires Java 21+. A platform team wants to deploy on Java 17. Where in the repository structure would you manage this constraint, and how?
> 5. A CI pipeline builds and tests the entire monorepo on every PR. Build time is 45 minutes. Describe how you would implement selective CI (only build affected modules) using the repository layout described.

---

## References

### Documentation
- [Nx Monorepo Tooling](https://nx.dev/concepts/more-concepts/why-monorepos) — Monorepo tooling with affected module detection.
- [Turborepo](https://turbo.build/repo/docs) — High-performance build system for monorepos.
- [Maven Multi-Module Projects](https://maven.apache.org/guides/mini/guide-multiple-modules.html)
- [Gradle Version Catalogues](https://docs.gradle.org/current/userguide/version_catalogs.html)
- [DVC Project Structure](https://dvc.org/doc/user-guide/project-structure)

### Books
- *Building Microservices* — Sam Newman (O'Reilly). Chapter on team topology and repository structure.
- *Fundamentals of Software Architecture* — Richards & Ford (O'Reilly). Module coupling and cohesion.

---

---
[« Back to layouts-repositories Index](index.md) | [🏠 Home](../index.md)