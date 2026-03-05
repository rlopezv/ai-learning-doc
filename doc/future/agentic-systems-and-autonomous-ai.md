## Chapter 2 — Agentic Systems and Autonomous AI

### 2.1 From RAG to Agentic Pipelines

RAG is a one-shot pattern: receive question, retrieve, generate, return. Agentic systems introduce iteration, tool use, and decision-making. The transition is not cosmetic — it changes the failure modes, the evaluation surface, and the operational requirements fundamentally.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

class AgentStepType(str, Enum):
    THINK    = "think"     # Internal reasoning — no external side effect
    RETRIEVE = "retrieve"  # Query a knowledge store
    TOOL_USE = "tool_use"  # Call an external API or service
    GENERATE = "generate"  # Produce a final response
    DECIDE   = "decide"    # Choose between branches
    VERIFY   = "verify"    # Check an output against a criterion

@dataclass
class AgentStep:
    step_type: AgentStepType
    description: str
    has_side_effects: bool     # Tool calls can create, modify, or delete
    reversible: bool           # Can this step be undone?
    requires_human_approval: bool

RAG_VS_AGENT = {
    "rag": [
        AgentStep(AgentStepType.RETRIEVE, "Embed query, search vector DB",
                  has_side_effects=False, reversible=True, requires_human_approval=False),
        AgentStep(AgentStepType.GENERATE, "Generate answer from context",
                  has_side_effects=False, reversible=True, requires_human_approval=False),
    ],
    "agent": [
        AgentStep(AgentStepType.THINK, "Decompose task into sub-tasks",
                  has_side_effects=False, reversible=True, requires_human_approval=False),
        AgentStep(AgentStepType.RETRIEVE, "Search knowledge base",
                  has_side_effects=False, reversible=True, requires_human_approval=False),
        AgentStep(AgentStepType.TOOL_USE, "Call external API (create calendar event)",
                  has_side_effects=True, reversible=False, requires_human_approval=True),
        AgentStep(AgentStepType.VERIFY, "Verify API response is correct",
                  has_side_effects=False, reversible=True, requires_human_approval=False),
        AgentStep(AgentStepType.GENERATE, "Summarise completed actions",
                  has_side_effects=False, reversible=True, requires_human_approval=False),
    ]
}

AGENTIC_NEW_FAILURE_MODES = [
    "Irreversible tool calls (send email, delete record) with wrong parameters",
    "Infinite loops: agent calls tool, result triggers another tool call, repeat",
    "Context accumulation: long agent runs fill context window with intermediate steps",
    "Error amplification: early wrong step invalidates all subsequent steps",
    "Non-determinism: same task produces different action sequences on re-run",
    "Scope creep: agent interprets task more broadly than intended",
    "Security: adversarial inputs in tool results that redirect agent behaviour",
]
```

---

### 2.2 Tool Use and Function Calling at Scale

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict      # JSON Schema for parameters
    returns: str          # Description of return value
    has_side_effects: bool
    idempotent: bool      # Safe to call multiple times with same params?
    rate_limit_per_min: int
    requires_human_approval: bool

PRODUCTION_TOOL_CATALOG = [
    ToolDefinition(
        "search_knowledge_base",
        "Search the internal knowledge base for relevant documents",
        parameters={"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}},
        returns="List of document excerpts with relevance scores",
        has_side_effects=False, idempotent=True,
        rate_limit_per_min=60, requires_human_approval=False
    ),
    ToolDefinition(
        "get_ticket_details",
        "Retrieve details of a support ticket by ID",
        parameters={"ticket_id": {"type": "string"}},
        returns="Ticket status, priority, assignee, history",
        has_side_effects=False, idempotent=True,
        rate_limit_per_min=120, requires_human_approval=False
    ),
    ToolDefinition(
        "update_ticket_status",
        "Update the status of a support ticket",
        parameters={"ticket_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "pending", "resolved"]},
                    "resolution_note": {"type": "string"}},
        returns="Updated ticket object",
        has_side_effects=True, idempotent=False,
        rate_limit_per_min=30, requires_human_approval=True
    ),
    ToolDefinition(
        "send_customer_email",
        "Send an email to a customer",
        parameters={"to": {"type": "string"}, "subject": {"type": "string"},
                    "body": {"type": "string"}},
        returns="Message ID and delivery status",
        has_side_effects=True, idempotent=False,
        rate_limit_per_min=10, requires_human_approval=True
    ),
    ToolDefinition(
        "run_sql_query",
        "Execute a read-only SQL query against the analytics database",
        parameters={"query": {"type": "string"},
                    "max_rows": {"type": "integer", "default": 100}},
        returns="Query results as JSON",
        has_side_effects=False, idempotent=True,
        rate_limit_per_min=20, requires_human_approval=False
    ),
]

class ToolCallGuard:
    """
    Safety layer for agentic tool calls.
    Enforces rate limits, requires human approval for side-effecting tools,
    and logs every tool invocation for audit.
    """
    def __init__(self, catalog: list[ToolDefinition],
                 approval_queue, audit_logger, rate_limiter):
        self.catalog  = {t.name: t for t in catalog}
        self.approvals = approval_queue
        self.audit    = audit_logger
        self.limiter  = rate_limiter

    def execute(self, tool_name: str, params: dict,
                agent_id: str, task_id: str) -> dict:
        tool = self.catalog.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Rate limit check
        if not self.limiter.allow(tool_name, tool.rate_limit_per_min):
            raise RuntimeError(f"Rate limit exceeded for {tool_name}")

        # Human approval gate for side-effecting tools
        if tool.requires_human_approval:
            approval = self.approvals.request(
                agent_id=agent_id, task_id=task_id,
                tool_name=tool_name, params=params
            )
            if not approval.approved:
                self.audit.log_tool_denial(agent_id, tool_name, params)
                raise PermissionError(f"Tool {tool_name} denied by human reviewer")

        # Execute (delegated to actual tool implementations)
        result = self._dispatch(tool_name, params)
        self.audit.log_tool_call(agent_id, task_id, tool_name, params, result)
        return result

    def _dispatch(self, tool_name: str, params: dict) -> dict:
        # In production: routes to actual API clients
        return {"status": "ok", "tool": tool_name, "params": params}
```

---

### 2.3 Multi-Agent Architectures

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentRole:
    role_id: str
    name: str
    responsibility: str
    tools_available: list[str]
    can_delegate_to: list[str]
    trust_level: str    # "high" | "medium" | "low"

MULTI_AGENT_ROLES = [
    AgentRole(
        "orchestrator",
        "Task Orchestrator",
        "Receives user task, decomposes into sub-tasks, delegates to specialist agents",
        tools_available=["search_knowledge_base"],
        can_delegate_to=["researcher", "writer", "critic"],
        trust_level="high"
    ),
    AgentRole(
        "researcher",
        "Research Agent",
        "Searches internal and external knowledge sources to gather relevant information",
        tools_available=["search_knowledge_base", "run_sql_query"],
        can_delegate_to=[],   # Leaf agent — no further delegation
        trust_level="medium"
    ),
    AgentRole(
        "writer",
        "Content Agent",
        "Generates structured outputs (reports, emails, summaries) from research",
        tools_available=["search_knowledge_base"],
        can_delegate_to=[],
        trust_level="medium"
    ),
    AgentRole(
        "critic",
        "Quality Critic",
        "Reviews outputs from other agents against quality criteria before delivery",
        tools_available=["search_knowledge_base"],
        can_delegate_to=[],
        trust_level="high"    # Critic results gate final output
    ),
]

MULTI_AGENT_PATTERNS = {
    "sequential": (
        "Agents run one after another in a fixed pipeline. "
        "Simple, predictable, debuggable. "
        "Use case: document drafting (research → write → review)."
    ),
    "parallel": (
        "Multiple agents run simultaneously on different sub-tasks. "
        "Use case: parallel research on multiple sources before synthesis."
    ),
    "hierarchical": (
        "Orchestrator delegates to specialist agents. "
        "Each specialist may have sub-agents. "
        "Use case: complex enterprise workflows with clear role boundaries."
    ),
    "debate": (
        "Two agents generate competing answers; a judge picks the better one. "
        "Use case: high-stakes decisions requiring adversarial validation."
    ),
}

MULTI_AGENT_ENGINEERING_CHALLENGES = [
    "Message passing schemas: agents must agree on structured message formats",
    "Context propagation: each agent needs sufficient context without exceeding window",
    "Error handling: one agent failure should not silently corrupt the entire pipeline",
    "Cost control: each agent call is a LLM call — multi-agent runs can be 10–50× "
    "more expensive than single-turn RAG",
    "Determinism: multi-agent runs are inherently non-deterministic — testing is hard",
    "Observability: tracing must capture each agent's inputs, outputs, and tool calls",
]
```

---

### 2.4 Safety and Control in Agentic Systems

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class SafetyLevel(str, Enum):
    READ_ONLY    = "read_only"      # Agent can only read; no writes
    SUPERVISED   = "supervised"     # Writes allowed with human approval per action
    SEMI_AUTO    = "semi_auto"      # Writes allowed; human reviews after N actions
    AUTONOMOUS   = "autonomous"     # Full autonomy within defined scope

@dataclass
class AgentSafetyConfig:
    max_iterations: int = 10          # Prevent infinite loops
    max_tool_calls: int = 20          # Budget cap on tool invocations
    max_cost_usd: float = 0.50        # Cost cap per agent run
    safety_level: SafetyLevel = SafetyLevel.SUPERVISED
    allowed_tool_names: list[str] = None   # Allowlist — None means all
    forbidden_tool_names: list[str] = None  # Blocklist — None means none
    require_final_human_review: bool = True
    timeout_seconds: int = 300

AGENT_SAFETY_PRINCIPLES = [
    "Minimal footprint: request only the permissions needed for this task; "
    "don't accumulate access across sessions.",

    "Prefer reversible actions: if two approaches achieve the same result "
    "and one is reversible, choose it.",

    "Human in the loop for irreversible operations: email send, "
    "record delete, payment initiation — always require human confirmation.",

    "Explicit scope definition: the task prompt must define what the agent "
    "is authorised to do and what is explicitly out of scope.",

    "Audit every tool call: each tool invocation is logged with agent_id, "
    "task_id, parameters, and result — not just the final output.",

    "Cost budget as a safety control: a runaway agent that loops can "
    "generate large LLM costs; cost caps are part of the safety layer.",

    "Output validation before delivery: agent outputs should pass "
    "the same output scanner as RAG responses (PII, injection, schema).",
]
```

---

> ### 📋 Chapter Summary
>
> - Agentic systems introduce irreversible side effects, non-determinism, and error amplification that do not exist in one-shot RAG — these require new safety and observability approaches.
> - A **tool catalog** with explicit `has_side_effects`, `idempotent`, and `requires_human_approval` flags makes safety policy machine-readable and enforceable at the gateway layer.
> - Multi-agent architectures (sequential, parallel, hierarchical, debate) address task decomposition, but introduce cost multiplication and determinism challenges that RAG does not have.
> - Agent safety is governed by five controls: max iterations, max tool calls, cost cap, explicit scope, and human-in-the-loop for irreversible operations.

---

> ### ❓ Comprehension Questions
>
> 1. `AgentSafetyConfig.max_iterations=10` prevents infinite loops. An agent solving a complex research task legitimately requires 12 iterations. Should the hard limit be raised, or should the agent architecture be redesigned? Argue both positions.
> 2. The `ToolCallGuard` requires human approval for `send_customer_email`. A customer service agent handles 500 queries per day and 30% require a follow-up email. At 1 minute per approval, this is 2.5 hours of human review daily. How would you reduce the approval burden while maintaining safety for high-risk emails?
> 3. Multi-agent runs are described as "10–50× more expensive than single-turn RAG." A product manager proposes using a 5-agent pipeline for every customer query. At a current RAG cost of $0.003/query and 15,000 queries/day, calculate the daily cost at 10× and at 50× multiplication. Is this economically viable?
> 4. The "debate" pattern uses two agents producing competing answers with a judge. If both agents use the same base model (GPT-4o), will they reliably produce different answers? What would make the debate pattern genuinely adversarial rather than two correlated outputs?
> 5. `AGENT_SAFETY_PRINCIPLES` states "minimal footprint: don't accumulate access across sessions." An agent that handles multi-step tasks across multiple days (e.g., a project management agent) needs persistent context and permissions. How do you reconcile minimal footprint with the need for persistent agentic workflows?

---

## References

### Documentation
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Stateful multi-agent orchestration.
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

### Papers
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al., 2022.
- [AutoGen: Enabling Next-Generation LLM Applications](https://arxiv.org/abs/2308.08155) — Wu et al., 2023.

---

---
[« Back to future Index](index.md) | [🏠 Home](../../index.md)