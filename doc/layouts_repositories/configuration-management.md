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

---
[« Back to layouts_repositories Index](index.md) | [🏠 Home](../index.md)