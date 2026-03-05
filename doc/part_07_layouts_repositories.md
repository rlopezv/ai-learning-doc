# Part VII — Layouts and Repositories

---

> **Navigation**
> [← Part VI — Artifact Engineering](part_06_artifact_engineering.md) | [→ Part VIII — AI Systems SDLC](part_08_sdlc.md)

---

## Contents

- [Chapter 1 — Repository Structure for AI Systems](#chapter-1--repository-structure-for-ai-systems)
  - [1.1 Why Repository Layout Matters](#11-why-repository-layout-matters)
  - [1.2 Monorepo vs Polyrepo](#12-monorepo-vs-polyrepo)
  - [1.3 Canonical Monorepo Layout](#13-canonical-monorepo-layout)
  - [1.4 Module Boundaries and Dependency Rules](#14-module-boundaries-and-dependency-rules)
  - [1.5 Java Project Structure with Maven/Gradle](#15-java-project-structure-with-mavengradle)
- [Chapter 2 — Configuration Management](#chapter-2--configuration-management)
  - [2.1 The Configuration Problem in LLM Systems](#21-the-configuration-problem-in-llm-systems)
  - [2.2 Layered Configuration Architecture](#22-layered-configuration-architecture)
  - [2.3 Environment-Specific Configuration](#23-environment-specific-configuration)
  - [2.4 Secret Management](#24-secret-management)
  - [2.5 Configuration Validation at Startup](#25-configuration-validation-at-startup)
- [Chapter 3 — Dependency Management](#chapter-3--dependency-management)
  - [3.1 LLM SDK Dependency Risks](#31-llm-sdk-dependency-risks)
  - [3.2 Python Dependency Pinning](#32-python-dependency-pinning)
  - [3.3 Java Dependency Management with Maven BOM](#33-java-dependency-management-with-maven-bom)
  - [3.4 Dependency Scanning and Vulnerability Management](#34-dependency-scanning-and-vulnerability-management)
  - [3.5 Vendoring and Air-Gapped Deployments](#35-vendoring-and-air-gapped-deployments)
- [Chapter 4 — Documentation as Code 🧪](#chapter-4--documentation-as-code-)
  - [4.1 Living Documentation for AI Systems](#41-living-documentation-for-ai-systems)
  - [4.2 Architecture Decision Records](#42-architecture-decision-records)
  - [4.3 Runbooks as Code](#43-runbooks-as-code)
  - [4.4 API Documentation with OpenAPI](#44-api-documentation-with-openapi)
  - [4.5 Automated Documentation Pipelines](#45-automated-documentation-pipelines)
  - [🧪 Hands-on Lab: ADR Pipeline](#-hands-on-lab-adr-pipeline)

---

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

## Chapter 2 — Configuration Management

### 2.1 The Configuration Problem in LLM Systems

LLM systems have an unusually complex configuration surface. A single production deployment may be configured by:

- **LLM provider credentials** (API keys, endpoints) — environment-specific secrets
- **Model selection** (gpt-4o vs gpt-4o-mini vs local Ollama) — environment-specific non-secret
- **RAG parameters** (chunk size, top-K, embedding model) — tuned per environment
- **Prompt template versions** (which version is active) — deployment artifact reference
- **Feature flags** (enable reranking, enable GraphRAG) — runtime toggles
- **Infrastructure** (database URLs, queue endpoints) — environment-specific infrastructure

Managing this diversity without a deliberate strategy leads to configuration scattered across environment variables, hardcoded values, YAML files, and database tables — with no single source of truth and no validation that the configuration is internally consistent.

---

### 2.2 Layered Configuration Architecture

A layered architecture resolves configuration from multiple sources in a defined priority order.

```
Priority (highest → lowest):
1. Environment variables        — deployment-specific overrides, secrets
2. .env.{environment} files    — environment-specific defaults
3. config.{environment}.yaml   — environment-specific structured config
4. config.base.yaml            — shared defaults across all environments
5. Code defaults                — hardcoded fallbacks of last resort
```

```python
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field, validator

class LLMConfig(BaseModel):
    provider: str = "openai"            # openai | anthropic | ollama | azure_openai
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None      # For Ollama or Azure OpenAI
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1000, ge=1, le=128000)
    timeout_seconds: int = Field(30, ge=1, le=300)
    max_retries: int = Field(3, ge=0, le=10)

class RAGConfig(BaseModel):
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(1536, ge=64, le=3072)
    chunk_size: int = Field(512, ge=64, le=8192)
    chunk_overlap: int = Field(64, ge=0)
    top_k: int = Field(5, ge=1, le=50)
    reranking_enabled: bool = False
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_search_enabled: bool = True
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0)

class VectorDBConfig(BaseModel):
    provider: str = "chroma"           # chroma | qdrant | weaviate | pinecone
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "knowledge_base"
    api_key: Optional[str] = None

class AppConfig(BaseModel):
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)

    @validator("environment")
    def validate_environment(cls, v):
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

class ConfigLoader:
    def __init__(self, config_dir: str = "./config", env: str = None):
        self.config_dir = Path(config_dir)
        self.env = env or os.getenv("APP_ENV", "development")

    def load(self) -> AppConfig:
        config = {}

        # Layer 1: Base config
        base_file = self.config_dir / "config.base.yaml"
        if base_file.exists():
            config.update(yaml.safe_load(base_file.read_text()) or {})

        # Layer 2: Environment-specific config
        env_file = self.config_dir / f"config.{self.env}.yaml"
        if env_file.exists():
            config = self._deep_merge(config, yaml.safe_load(env_file.read_text()) or {})

        # Layer 3: Environment variables (highest priority for non-secrets)
        env_overrides = self._read_env_overrides()
        config = self._deep_merge(config, env_overrides)

        # Layer 4: Secrets from environment variables
        config = self._inject_secrets(config)

        return AppConfig(**config)

    def _read_env_overrides(self) -> dict:
        """Read APP_ prefixed environment variables as config overrides."""
        overrides = {}
        for key, value in os.environ.items():
            if key.startswith("APP_"):
                path = key[4:].lower().split("__")
                self._set_nested(overrides, path, self._coerce(value))
        return overrides

    def _inject_secrets(self, config: dict) -> dict:
        """Inject secrets from environment variables."""
        secret_mappings = {
            "OPENAI_API_KEY":   ["llm", "api_key"],
            "ANTHROPIC_API_KEY":["llm", "api_key"],
            "VECTOR_DB_API_KEY":["vector_db", "api_key"],
        }
        for env_var, config_path in secret_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested(config, config_path, value)
        return config

    def _deep_merge(self, base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _set_nested(self, d: dict, path: list[str], value):
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _coerce(self, value: str):
        """Coerce string env var values to appropriate Python types."""
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
```

---

### 2.3 Environment-Specific Configuration

```yaml
# config/config.base.yaml
environment: development
debug: false
log_level: INFO
llm:
  provider: openai
  model_name: gpt-4o-mini
  temperature: 0.0
  max_tokens: 1000
  timeout_seconds: 30
  max_retries: 3
rag:
  embedding_model: text-embedding-3-small
  chunk_size: 512
  chunk_overlap: 64
  top_k: 5
  hybrid_search_enabled: true
  hybrid_alpha: 0.7
vector_db:
  provider: chroma
  host: localhost
  port: 8000
  collection_name: knowledge_base
```

```yaml
# config/config.production.yaml
environment: production
debug: false
log_level: WARNING
llm:
  model_name: gpt-4o          # Upgrade to full model in production
  timeout_seconds: 60
  max_retries: 5
rag:
  top_k: 8
  reranking_enabled: true     # Enable reranking in production only
vector_db:
  provider: qdrant
  host: qdrant.internal       # Internal DNS for production cluster
  port: 6333
```

```yaml
# config/config.development.yaml — 🔓 On-premise development
environment: development
debug: true
log_level: DEBUG
llm:
  provider: ollama
  model_name: llama3
  base_url: http://localhost:11434
vector_db:
  provider: chroma
  host: localhost
  port: 8000
```

**Java — Spring Boot layered configuration:**
```yaml
# src/main/resources/application.yml
spring:
  config:
    import: "optional:configserver:"
  profiles:
    active: ${APP_ENV:development}

app:
  llm:
    provider: ${LLM_PROVIDER:openai}
    model-name: ${LLM_MODEL:gpt-4o-mini}
    temperature: 0.0
    max-tokens: 1000
  rag:
    chunk-size: 512
    top-k: 5
    hybrid-search-enabled: true
```

```yaml
# src/main/resources/application-production.yml
app:
  llm:
    model-name: gpt-4o
  rag:
    top-k: 8
    reranking-enabled: true
```

---

### 2.4 Secret Management

API keys and credentials must never appear in configuration files committed to version control.

```python
import os
from typing import Optional
from functools import lru_cache

class SecretManager:
    """
    Abstract secret manager. Implementations backed by
    HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, or env vars.
    """
    def get(self, secret_name: str) -> Optional[str]:
        raise NotImplementedError

class EnvSecretManager(SecretManager):
    """Development: secrets from environment variables."""
    def get(self, secret_name: str) -> Optional[str]:
        return os.getenv(secret_name)

class HashiCorpVaultSecretManager(SecretManager):
    """🔓 Production on-premise: secrets from HashiCorp Vault."""
    def __init__(self, vault_url: str, token: str, mount_path: str = "secret"):
        import hvac
        self.client = hvac.Client(url=vault_url, token=token)
        self.mount_path = mount_path

    def get(self, secret_name: str) -> Optional[str]:
        try:
            result = self.client.secrets.kv.v2.read_secret_version(
                path=secret_name,
                mount_point=self.mount_path
            )
            return result["data"]["data"].get("value")
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve secret '{secret_name}': {e}")

class AWSSecretManager(SecretManager):
    """Cloud: secrets from AWS Secrets Manager."""
    def __init__(self, region: str = "eu-west-1"):
        import boto3
        self.client = boto3.client("secretsmanager", region_name=region)

    @lru_cache(maxsize=32)
    def get(self, secret_name: str) -> Optional[str]:
        import json
        response = self.client.get_secret_value(SecretId=secret_name)
        secret = response.get("SecretString", "{}")
        return json.loads(secret).get("value")

def get_secret_manager() -> SecretManager:
    """Factory: choose implementation based on environment."""
    env = os.getenv("APP_ENV", "development")
    if env == "production":
        vault_url = os.getenv("VAULT_URL")
        vault_token = os.getenv("VAULT_TOKEN")
        if vault_url and vault_token:
            return HashiCorpVaultSecretManager(vault_url, vault_token)
        return AWSSecretManager()
    return EnvSecretManager()
```

**`.env.example` — committed to repository:**
```bash
# Copy to .env and fill in your values. Never commit .env.
APP_ENV=development

# LLM Provider credentials
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# On-premise alternatives (uncomment to use Ollama)
# APP_LLM__PROVIDER=ollama
# APP_LLM__BASE_URL=http://localhost:11434
# APP_LLM__MODEL_NAME=llama3

# Vector DB
VECTOR_DB_API_KEY=
APP_VECTOR_DB__HOST=localhost
```

---

### 2.5 Configuration Validation at Startup

Fail fast with clear error messages rather than failing at runtime with cryptic errors.

```python
import sys

class StartupValidator:
    def __init__(self, config: AppConfig, secret_manager: SecretManager):
        self.config = config
        self.secrets = secret_manager
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> bool:
        """Run all startup checks. Returns False if any critical check fails."""
        self._check_llm_credentials()
        self._check_vector_db_connectivity()
        self._check_rag_parameter_consistency()
        self._check_production_safety()
        self._report()
        return len(self.errors) == 0

    def _check_llm_credentials(self):
        provider = self.config.llm.provider
        if provider == "openai" and not self.config.llm.api_key:
            if not self.secrets.get("OPENAI_API_KEY"):
                self.errors.append("OPENAI_API_KEY is required for provider=openai")
        if provider == "anthropic" and not self.config.llm.api_key:
            if not self.secrets.get("ANTHROPIC_API_KEY"):
                self.errors.append("ANTHROPIC_API_KEY is required for provider=anthropic")
        if provider == "ollama" and not self.config.llm.base_url:
            self.errors.append("llm.base_url is required for provider=ollama")

    def _check_vector_db_connectivity(self):
        """Verify vector DB is reachable at startup."""
        import socket
        host = self.config.vector_db.host
        port = self.config.vector_db.port
        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
        except (socket.timeout, ConnectionRefusedError, OSError):
            self.errors.append(
                f"Vector DB unreachable at {host}:{port}. "
                f"Is {self.config.vector_db.provider} running?"
            )

    def _check_rag_parameter_consistency(self):
        rag = self.config.rag
        if rag.chunk_overlap >= rag.chunk_size:
            self.errors.append(
                f"chunk_overlap ({rag.chunk_overlap}) must be less than chunk_size ({rag.chunk_size})"
            )
        if rag.reranking_enabled and not rag.reranking_model:
            self.errors.append("rag.reranking_model is required when reranking_enabled=true")

    def _check_production_safety(self):
        if self.config.environment == "production":
            if self.config.debug:
                self.errors.append("debug=true is not allowed in production")
            if self.config.log_level == "DEBUG":
                self.warnings.append("log_level=DEBUG in production may expose sensitive data")
            if self.config.llm.model_name == "gpt-4o-mini":
                self.warnings.append(
                    "Using gpt-4o-mini in production — confirm this is intentional"
                )

    def _report(self):
        for warning in self.warnings:
            print(f"[CONFIG WARNING] {warning}")
        for error in self.errors:
            print(f"[CONFIG ERROR] {error}", file=sys.stderr)
        if self.errors:
            print(f"\n{len(self.errors)} configuration error(s). Refusing to start.", file=sys.stderr)

# Usage at application startup
def create_app():
    config = ConfigLoader().load()
    secrets = get_secret_manager()
    validator = StartupValidator(config, secrets)
    if not validator.validate():
        sys.exit(1)
    return config
```

---

> ### 📋 Chapter Summary
>
> - LLM systems have an unusually complex configuration surface spanning credentials, model selection, RAG parameters, prompt versions, and feature flags.
> - **Layered configuration** (env vars > env-specific files > base files > code defaults) provides a clear priority order with a single source of truth per layer.
> - Secrets must be managed separately from configuration — never in committed files. Use [HashiCorp Vault](https://developer.hashicorp.com/vault/docs) on-premise or AWS/Azure Secrets Manager in cloud.
> - **Startup validation** fails fast with clear error messages, preventing cryptic runtime failures from misconfiguration.

---

> ### ❓ Comprehension Questions
>
> 1. A developer commits `config.production.yaml` containing an API key. What are the immediate and long-term risks, and what process should prevent this?
> 2. The `ConfigLoader` merges config layers with `APP_` prefixed environment variables taking highest priority. Why is this ordering correct for container deployments?
> 3. A production deployment fails at startup with "Vector DB unreachable at qdrant.internal:6333". What does the startup validator tell you that a runtime error at first query would not?
> 4. Design a feature flag system layered on top of `AppConfig` that allows toggling `reranking_enabled` per request without a deployment. What configuration changes are required?
> 5. Your Spring Boot application uses `@ConfigurationProperties` bound to `AppConfig`. How would you validate that all required fields are populated before the application context starts?

---

## References

### Documentation
- [HashiCorp Vault](https://developer.hashicorp.com/vault/docs) — Secret management for on-premise deployments.
- [AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html)
- [Spring Boot Externalized Configuration](https://docs.spring.io/spring-boot/reference/features/external-config.html)
- [Pydantic Settings Management](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — Type-safe configuration from env vars.
- [python-dotenv](https://saurabh-kumar.com/python-dotenv/) — `.env` file support for Python.

### Books
- *The Twelve-Factor App* — Heroku (https://12factor.net). Factors III (config) and IV (backing services) directly applicable.

---

## Chapter 3 — Dependency Management

### 3.1 LLM SDK Dependency Risks

LLM SDK dependencies carry specific risks that traditional library dependencies do not:

**Rapid version churn.** `openai`, `anthropic`, `langchain`, and `llama-index` release breaking changes frequently. A project pinned to a two-month-old version may be incompatible with current API features.

**Transitive dependency conflicts.** LLM frameworks have large, overlapping dependency trees. Adding `langchain` and `llama-index` to the same project frequently triggers version conflicts for shared transitive dependencies (`httpx`, `pydantic`, `tiktoken`).

**Model deprecations.** Model name strings embedded in configuration (`gpt-3.5-turbo-16k`, `text-davinci-003`) silently break when the provider deprecates them. Dependency management must extend to model version strings, not only library versions.

**Security vulnerabilities.** LLM SDKs process untrusted input (user queries) and make external network calls. A security vulnerability in an HTTP client transitively used by an LLM SDK is a real attack surface.

---

### 3.2 Python Dependency Pinning

```toml
# pyproject.toml — root Python tooling config
[project]
name = "ai-system"
version = "1.0.0"
requires-python = ">=3.11"

# Direct dependencies with version constraints
dependencies = [
    "openai>=1.40.0,<2.0.0",
    "anthropic>=0.34.0,<1.0.0",
    "langchain>=0.3.0,<0.4.0",
    "langchain-openai>=0.2.0,<0.3.0",
    "sentence-transformers>=3.0.0,<4.0.0",
    "chromadb>=0.5.0,<0.6.0",
    "pydantic>=2.8.0,<3.0.0",
    "httpx>=0.27.0,<1.0.0",
    "tiktoken>=0.7.0,<1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
]

[tool.uv]
# Lock file generated by uv — pins exact transitive dependency versions
# Commit uv.lock to repository for reproducible installs

[tool.ruff]
target-version = "py311"
line-length = 100
select = ["E", "W", "F", "I", "N", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
```

**Lock file management with [uv](https://docs.astral.sh/uv/) (recommended) or pip-tools:**

```bash
# uv: fast Python package manager with lockfile support
uv sync                    # Install from lock file (reproducible)
uv add openai              # Add dependency, update lock file
uv lock --upgrade-package openai  # Upgrade a single dependency

# pip-tools: compile requirements with full pin
pip-compile pyproject.toml --output-file=requirements.lock
pip-sync requirements.lock  # Install exactly what's in the lock file
```

**Dependency update strategy:**
```yaml
# .github/workflows/dependency-update.yml
name: Weekly Dependency Update
on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9am

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          pip install uv
          uv lock --upgrade
          # Run full test suite
          uv run pytest
          # If tests pass, open a PR with the updated lock file
```

---

### 3.3 Java Dependency Management with Maven BOM

A Bill of Materials (BOM) pins all related library versions in one place.

```xml
<!-- libs/java-rag/pom.xml -->
<project>
  <parent>
    <groupId>com.company.ai</groupId>
    <artifactId>ai-system-parent</artifactId>
    <version>1.0.0-SNAPSHOT</version>
  </parent>
  <artifactId>java-rag</artifactId>

  <dependencies>
    <!-- LangChain4j — version from parent BOM, no version needed here -->
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j</artifactId>
    </dependency>
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j-open-ai</artifactId>
    </dependency>
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j-embeddings-all-minilm-l6-v2</artifactId>
    </dependency>

    <!-- 🔓 On-premise: local embedding model, no API calls -->
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j-ollama</artifactId>
    </dependency>
  </dependencies>

  <!-- OWASP dependency vulnerability check -->
  <build>
    <plugins>
      <plugin>
        <groupId>org.owasp</groupId>
        <artifactId>dependency-check-maven</artifactId>
        <version>10.0.4</version>
        <configuration>
          <failBuildOnCVSS>7</failBuildOnCVSS>  <!-- Fail on HIGH+ vulnerabilities -->
          <formats>HTML,JSON</formats>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
```

**Detecting conflicting dependency versions:**
```bash
# Maven: show dependency tree to find conflicts
mvn dependency:tree -Dverbose -Dincludes=com.fasterxml.jackson

# Gradle: show dependency insight
./gradlew dependencyInsight --dependency jackson-databind --configuration runtimeClasspath
```

---

### 3.4 Dependency Scanning and Vulnerability Management

```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  python-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install pip-audit
      - run: pip-audit --require-hashes -r requirements.lock

  java-owasp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: mvn org.owasp:dependency-check-maven:check
      - uses: actions/upload-artifact@v4
        with:
          name: dependency-check-report
          path: target/dependency-check-report.html
```

**Python — Programmatic vulnerability scanning:**
```python
import subprocess
import json
from dataclasses import dataclass

@dataclass
class Vulnerability:
    package: str
    installed_version: str
    vuln_id: str
    severity: str
    description: str
    fix_version: str

def scan_python_dependencies(requirements_file: str = "requirements.lock") -> list[Vulnerability]:
    """Run pip-audit and return structured vulnerability report."""
    result = subprocess.run(
        ["pip-audit", "--format", "json", "-r", requirements_file],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return []  # No vulnerabilities
    try:
        report = json.loads(result.stdout)
        vulns = []
        for item in report.get("dependencies", []):
            for vuln in item.get("vulns", []):
                vulns.append(Vulnerability(
                    package=item["name"],
                    installed_version=item["version"],
                    vuln_id=vuln["id"],
                    severity=vuln.get("aliases", ["UNKNOWN"])[0],
                    description=vuln.get("description", "")[:200],
                    fix_version=vuln.get("fix_versions", ["none"])[0]
                ))
        return vulns
    except json.JSONDecodeError:
        return []
```

---

### 3.5 Vendoring and Air-Gapped Deployments

Enterprise deployments in regulated industries often cannot reach public package registries at build time. Vendoring provides a solution.

```bash
# Python: vendor all dependencies into the repository
pip install --target=vendor -r requirements.lock --no-index

# Or: use a private PyPI mirror (Artifactory, Nexus, AWS CodeArtifact)
pip install --extra-index-url https://pypi.internal.company.com/simple/ openai
```

```xml
<!-- Maven: proxy repository configuration for air-gapped environments -->
<!-- settings.xml -->
<settings>
  <mirrors>
    <mirror>
      <id>internal-nexus</id>
      <url>https://nexus.internal.company.com/repository/maven-public/</url>
      <mirrorOf>*</mirrorOf>
    </mirror>
  </mirrors>
</settings>
```

🔓 **On-premise model weights** — for fully air-gapped LLM deployments, model weights must also be vendored:

```bash
# Download model weights to local registry
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir ./vendor/models/all-MiniLM-L6-v2

# Load from local path at runtime
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("./vendor/models/all-MiniLM-L6-v2")
```

---

> ### 📋 Chapter Summary
>
> - LLM SDK dependencies carry unique risks: rapid version churn, transitive conflicts, model deprecations, and security vulnerabilities.
> - **Exact pinning via lock files** (`uv.lock`, `pip-compile`, Maven BOM) ensures reproducible builds across environments.
> - **Automated vulnerability scanning** (pip-audit, OWASP Dependency-Check) runs in CI on every PR.
> - **Vendoring and private mirrors** enable deployment in air-gapped or restricted environments.

---

> ### ❓ Comprehension Questions
>
> 1. Two LLM frameworks in the same Python project require `pydantic>=1.9` and `pydantic>=2.0` respectively. How would you detect this conflict before it causes a runtime failure, and what are your resolution options?
> 2. A weekly dependency update PR fails the test suite due to a breaking change in `langchain 0.4.0`. Describe the triage and resolution process.
> 3. OWASP Dependency-Check reports a CVSS 8.5 vulnerability in a transitive dependency of `langchain4j`. The fix has not yet been released. What interim mitigation options do you have?
> 4. Explain why model name strings (e.g., `"gpt-4o-mini"`) should be treated as a versioned dependency. What monitoring would detect when a model is deprecated?
> 5. A financial services firm requires that no package leaves or enters the build environment. Describe the complete dependency management architecture for their Python and Java AI systems.

---

## References

### Documentation
- [uv — Python Package Manager](https://docs.astral.sh/uv/) — Fast lock-file based Python dependency management.
- [pip-audit](https://pypi.org/project/pip-audit/) — Python dependency vulnerability scanner.
- [OWASP Dependency-Check](https://owasp.org/www-project-dependency-check/) — Java/Maven vulnerability scanning.
- [Maven BOM](https://maven.apache.org/guides/introduction/introduction-to-dependency-mechanism.html#bill-of-materials-bom-poms) — Version management with Bill of Materials.
- [Sonatype Nexus](https://help.sonatype.com/en/nexus-repository.html) — Private package mirror for air-gapped environments.
- [JFrog Artifactory](https://jfrog.com/help/r/jfrog-artifactory-documentation) — Enterprise artifact repository.

---

## Chapter 4 — Documentation as Code 🧪

### 4.1 Living Documentation for AI Systems

AI systems suffer from a documentation anti-pattern: decisions are made verbally in meetings, committed as code without explanation, and the reasoning is lost within weeks. Six months later, no one can answer "why is the chunk size 512?" or "why did we choose Qdrant over Weaviate?"

Documentation as code treats documentation with the same version control, review, and quality discipline applied to source code. It has three practical components:

1. **Architecture Decision Records (ADRs)** — document significant decisions and their rationale
2. **Runbooks** — executable, testable operational procedures
3. **API documentation** — auto-generated from code, always current

The key discipline: documentation lives in the repository alongside the code it describes. It is updated in the same PR as the code change. It is reviewed by the same reviewers. It becomes outdated for the same reason code becomes outdated — and is visible in the same git blame.

---

### 4.2 Architecture Decision Records

An ADR records a significant architectural decision: its context, the options considered, the decision made, and the consequences.

```markdown
<!-- docs/adr/ADR-007-vector-database-selection.md -->
# ADR-007: Vector Database Selection

**Date:** 2024-11-15
**Status:** Accepted
**Deciders:** Platform Team (Alice, Bob, Carlos)
**Supersedes:** ADR-003 (FAISS in-process index)

## Context

The RAG system needs a vector database for semantic search over 500K+ documents.
Current FAISS in-process solution cannot support:
- Multi-node horizontal scaling
- Real-time document updates without full re-index
- Metadata filtering alongside vector similarity
- Production-grade operational tooling (monitoring, backup, access control)

## Decision Drivers

- Must support hybrid search (dense + sparse) natively
- Must support metadata filtering at query time
- Must have a managed cloud offering AND a self-hosted option (EU data residency requirement)
- Must support Java client SDK for main API service
- Must be stable with > 5,000 GitHub stars and active maintenance

## Options Considered

| Criterion | Qdrant | Weaviate | Pinecone | Milvus |
|---|---|---|---|---|
| Hybrid search native | ✓ | ✓ | ✗ | ✓ |
| Metadata filtering | ✓ | ✓ | ✓ | ✓ |
| Self-hosted option | ✓ | ✓ | ✗ | ✓ |
| Java SDK | ✓ | ✓ | ✓ | ✓ |
| Operational maturity | High | High | High | Medium |
| Benchmark (1M vectors, top-5 latency p99) | 8ms | 12ms | 15ms | 10ms |

Pinecone was eliminated due to no self-hosted option (EU data residency requirement).
Milvus was eliminated due to higher operational complexity for the team's Kubernetes experience level.

## Decision

**Qdrant** — best combination of hybrid search support, performance, operational maturity,
and alignment with our self-hosted requirement.

## Consequences

### Positive
- Native hybrid search eliminates need for separate BM25 index
- Rust-based core provides low-latency P99 performance
- Docker Compose deployment for development, Kubernetes Operator for production

### Negative
- Team must learn Qdrant-specific APIs (Java SDK is less mature than Python)
- Migration from FAISS requires re-embedding entire corpus (~2 hours)

### Risks
- Qdrant Java SDK lags Python SDK in feature parity — mitigated by wrapping in internal `java-rag` library

## Links
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [Benchmark Script](../../scripts/vector_db_benchmark.py)
- [Migration Plan](ADR-007-migration-plan.md)
```

**ADR management tooling:**
```python
# scripts/adr_manager.py — Create and list ADRs

import re
from pathlib import Path
from datetime import datetime

ADR_DIR = Path("docs/adr")
ADR_TEMPLATE = """# ADR-{number:03d}: {title}

**Date:** {date}
**Status:** Proposed
**Deciders:** [Add decision makers]

## Context

[Describe the situation and problem that requires a decision]

## Decision Drivers

- [Driver 1]
- [Driver 2]

## Options Considered

| Criterion | Option A | Option B |
|---|---|---|
| [Criterion 1] | | |

## Decision

[State the decision]

## Consequences

### Positive
- [Positive consequence]

### Negative
- [Negative consequence]

## Links
- [Relevant documentation or tickets]
"""

def next_adr_number() -> int:
    existing = list(ADR_DIR.glob("ADR-*.md"))
    if not existing:
        return 1
    numbers = [int(re.search(r'ADR-(\d+)', f.stem).group(1)) for f in existing
               if re.search(r'ADR-(\d+)', f.stem)]
    return max(numbers) + 1 if numbers else 1

def create_adr(title: str) -> Path:
    ADR_DIR.mkdir(parents=True, exist_ok=True)
    number = next_adr_number()
    slug = title.lower().replace(" ", "-").replace("/", "-")
    filename = ADR_DIR / f"ADR-{number:03d}-{slug}.md"
    content = ADR_TEMPLATE.format(
        number=number,
        title=title,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    )
    filename.write_text(content)
    print(f"Created: {filename}")
    return filename

def list_adrs() -> list[dict]:
    adrs = []
    for f in sorted(ADR_DIR.glob("ADR-*.md")):
        content = f.read_text()
        status_match = re.search(r'\*\*Status:\*\*\s*(\w+)', content)
        title_match = re.search(r'# ADR-\d+: (.+)', content)
        adrs.append({
            "file": f.name,
            "title": title_match.group(1) if title_match else "Unknown",
            "status": status_match.group(1) if status_match else "Unknown"
        })
    return adrs
```

---

### 4.3 Runbooks as Code

A runbook describes how to perform an operational procedure. As code, it is versioned, testable, and automatically validated.

```python
# docs/runbooks/reindex_knowledge_base.py
"""
Runbook: Re-index Knowledge Base
=================================
Use when: Embedding model changed, corpus updated, index corrupted.
Owner: Platform Team
Last tested: 2024-11-01
SLA: Complete within 4 hours for corpus < 100K documents.

Steps:
1. Build new index in staging collection
2. Validate recall against eval dataset
3. Switch active collection (blue-green)
4. Deprecate old collection
5. Notify #platform-alerts
"""

import argparse
import sys
from pathlib import Path

def step_1_build_staging_index(corpus_path: str, config: dict) -> str:
    """Build new index in an isolated staging collection. Returns collection name."""
    print("[STEP 1] Building staging index...")
    # Implementation calls BlueGreenIndexManager.build_new_version()
    collection_name = f"kb_staging_{int(__import__('time').time())}"
    print(f"  ✓ Staging collection created: {collection_name}")
    return collection_name

def step_2_validate_recall(collection_name: str, eval_dataset_path: str) -> float:
    """Validate retrieval quality meets minimum threshold."""
    print("[STEP 2] Validating recall on staging index...")
    # Load eval dataset, run queries, compute Recall@5
    recall = 0.87  # Example
    print(f"  ✓ Recall@5 = {recall:.2%}")
    if recall < 0.80:
        print(f"  ✗ Recall below threshold (0.80). Aborting.", file=sys.stderr)
        sys.exit(1)
    return recall

def step_3_promote_index(collection_name: str):
    """Switch active collection atomically."""
    print(f"[STEP 3] Promoting {collection_name} to active...")
    print("  ✓ Active collection updated")

def step_4_deprecate_old(old_collection: str):
    """Mark old collection as deprecated (retain for 7 days for rollback)."""
    print(f"[STEP 4] Deprecating {old_collection} (7-day retention for rollback)...")
    print("  ✓ Old collection deprecated")

def step_5_notify(recall: float, collection_name: str, slack_webhook: str = None):
    """Post completion notification."""
    message = f"Knowledge base re-indexed. New collection: {collection_name}. Recall@5: {recall:.2%}"
    print(f"[STEP 5] Notification: {message}")
    if slack_webhook:
        import urllib.request, json
        data = json.dumps({"text": message}).encode()
        urllib.request.urlopen(urllib.request.Request(slack_webhook, data=data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index knowledge base")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--old-collection", required=True)
    parser.add_argument("--slack-webhook")
    args = parser.parse_args()

    config = {"embedding_model": "text-embedding-3-small", "chunk_size": 512}

    new_collection = step_1_build_staging_index(args.corpus, config)
    recall = step_2_validate_recall(new_collection, args.eval_dataset)
    step_3_promote_index(new_collection)
    step_4_deprecate_old(args.old_collection)
    step_5_notify(recall, new_collection, args.slack_webhook)

    print("\n✓ Re-indexing complete")
```

---

### 4.4 API Documentation with OpenAPI

```python
# apps/api/src/main.py — FastAPI with auto-generated OpenAPI docs
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="AI Knowledge Assistant API",
    version="2.0.0",
    description="""
Enterprise RAG API for querying the internal knowledge base.

## Authentication
All endpoints require an `Authorization: Bearer <token>` header.

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

## Context

[What is the problem? What forces are at play?]

## Decision Drivers

- [Driver 1]
- [Driver 2]

## Options Considered

| Criterion | Option A | Option B | Option C |
|---|---|---|---|
| [Performance] | | | |
| [Cost] | | | |
| [Operational complexity] | | | |

## Decision

**[Chosen Option]** because [primary reason].

## Consequences

### Positive
- [Benefit 1]

### Negative
- [Trade-off 1]

### Neutral
- [Neutral consequence]

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
