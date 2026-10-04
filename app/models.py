"""Domain-independent HTTP contracts. Add your own decision/state models below."""
from typing import Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ExternalContext(Contract):
    source: str = Field(min_length=1, max_length=100)
    content: str = Field(max_length=10000)
    trust: Literal['untrusted'] = 'untrusted'


class Fault(Contract):
    type: Literal['none', 'tool_timeout', 'malformed_tool_output',
                  'invalid_agent_decision'] = 'none'
    trigger: Literal['first_matching_operation'] = 'first_matching_operation'


class ArenaConfig(Contract):
    max_steps: int = Field(default=6, ge=1, le=6)
    fault: Fault = Field(default_factory=Fault)

    @field_validator('fault', mode='before')
    @classmethod
    def accept_assignment_shorthand(cls, value):
        return {'type': value} if isinstance(value, str) else value


class ArenaRequest(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str = Field(default_factory=lambda: str(uuid4()),
                            min_length=1, max_length=80)
    task: str = Field(min_length=1, max_length=10000)
    external_context: list[ExternalContext] = Field(default_factory=list,
                                                    max_length=20)
    arena_config: ArenaConfig = Field(default_factory=ArenaConfig)

    @field_validator('task')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Task must not be blank')
        return value


class ToolTrace(Contract):
    step: int = Field(ge=1, le=6)
    tool: str
    attempt: int = Field(default=1, ge=1, le=3)
    outcome: Literal['success', 'timeout', 'malformed_output',
                     'rejected', 'exception']
    latency_ms: float = Field(default=0, ge=0)


class Metrics(Contract):
    latency_ms: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0, le=6)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class ArenaResponse(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str
    status: Literal['completed', 'needs_clarification', 'blocked',
                    'approval_required', 'tool_error', 'contract_error',
                    'budget_exceeded', 'failed']
    final_response: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=0, ge=0, le=6)
    stop_reason: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)


class ChatRequest(ArenaRequest):
    session_id: str = Field(min_length=16, max_length=80)
    model: str = Field(default='unconfigured', max_length=120)


# ---------------------------------------------------------------------------
# Domain models for Job-Application Tracker
# ---------------------------------------------------------------------------

VALID_STATUSES = [
    'applied', 'screening', 'interview_scheduled', 'interviewed',
    'offer_received', 'accepted', 'rejected', 'withdrawn', 'saved'
]


class JobApplication(BaseModel):
    """A single job application record in the sandbox board."""
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    company: str = Field(min_length=1, max_length=200)
    role: str = Field(min_length=1, max_length=200)
    date_applied: str = Field(default='', max_length=20)
    status: str = Field(default='applied')
    url: str = Field(default='', max_length=500)
    notes: str = Field(default='', max_length=1000)

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        v_lower = v.strip().lower()
        if v_lower not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{v}'. Must be one of: {VALID_STATUSES}")
        return v_lower


class AgentDecision(BaseModel):
    """Typed schema for the LLM's decision at each step.

    The model MUST return JSON matching this schema. The system validates
    it before any tool execution."""
    model_config = ConfigDict(extra='forbid')

    status: Literal['continue', 'needs_clarification', 'completed',
                    'blocked', 'failed', 'approval_required']
    action: Optional[str] = None
    arguments: dict = Field(default_factory=dict)
    user_message: str = Field(default='', max_length=1500)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        allowed = [
            'search_applications', 'add_application', 'update_status',
            'detect_duplicates', 'get_summary', None
        ]
        if v is not None and v not in allowed:
            raise ValueError(
                f"Unknown action '{v}'. Allowed: {[a for a in allowed if a]}")
        return v


class AgentState(BaseModel):
    """Per-run execution state maintained across loop iterations."""
    step_count: int = 0
    max_steps: int = 6
    events: list[dict] = Field(default_factory=list)
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    model_calls: int = 0
    observations: list[str] = Field(default_factory=list)
    final_status: str = 'failed'
    stop_reason: str = ''
    final_response: str = ''
