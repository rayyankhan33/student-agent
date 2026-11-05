"""Typed contracts.

Part 1 — Arena HTTP contracts (kept identical to the starter scaffold so the
evaluator's request/response models match).
Part 2 — Agent contracts: the AgentDecision the model must produce, typed tool
arguments, the tool-output contract, and domain records.
"""
from datetime import date
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)


# ===========================================================================
# Part 1 — Arena HTTP contracts
# ===========================================================================
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


RunStatus = Literal['completed', 'needs_clarification', 'blocked',
                    'approval_required', 'tool_error', 'contract_error',
                    'budget_exceeded', 'failed']


class ArenaResponse(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str
    status: RunStatus
    final_response: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=0, ge=0, le=6)
    stop_reason: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)


class ChatRequest(ArenaRequest):
    session_id: str = Field(min_length=16, max_length=80)
    model: str = Field(default='', max_length=120)


class ChatResponse(ArenaResponse):
    """/chat only (not the Arena contract): adds session bookkeeping for the UI."""
    session_id: str
    model: str
    history_messages: int = 0
    pending_goal: Optional[str] = None


# ===========================================================================
# Part 2 — Agent contracts
# ===========================================================================
ApplicationStatus = Literal['saved', 'applied', 'screening',
                            'interview_scheduled', 'interviewed',
                            'offer_received', 'accepted', 'rejected',
                            'withdrawn']
VALID_STATUSES: list[str] = list(ApplicationStatus.__args__)

DecisionStatus = Literal['continue', 'needs_clarification', 'completed',
                         'blocked', 'approval_required', 'failed']


class AgentDecision(BaseModel):
    """The ONLY shape in which the model may influence execution.

    Schema-level consistency rules live here; rules that need runtime state
    (is the tool registered? are the arguments valid for it?) are checked by
    the agent's semantic validator before anything executes.
    """
    model_config = ConfigDict(extra='forbid')

    status: DecisionStatus
    action: Optional[str] = Field(default=None, max_length=60)
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_message: str = Field(default='', max_length=1500)
    missing_fields: list[str] = Field(default_factory=list, max_length=10)
    reasoning: str = Field(default='', max_length=600)

    @model_validator(mode='after')
    def consistent(self):
        if self.status == 'continue':
            if not self.action:
                raise ValueError('status "continue" requires an action')
        else:
            if self.action:
                raise ValueError(f'status "{self.status}" must have action=null')
            if self.arguments:
                raise ValueError(f'status "{self.status}" must have empty arguments')
            if not self.user_message.strip():
                raise ValueError(f'status "{self.status}" requires a user_message')
        if self.status == 'needs_clarification' and not self.missing_fields:
            raise ValueError('needs_clarification requires missing_fields')
        return self


# --- Domain record ---------------------------------------------------------
class JobApplication(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    id: str = Field(min_length=1, max_length=20)
    company: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=120)
    date_applied: str = Field(default='', max_length=10)
    status: ApplicationStatus = 'applied'
    url: str = Field(default='', max_length=500)
    notes: str = Field(default='', max_length=1000)


# --- Typed tool arguments (extra='forbid' rejects hallucinated parameters) --
class ToolArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class SearchArgs(ToolArgs):
    query: str = Field(default='', max_length=200)
    status: Optional[ApplicationStatus] = None
    company: str = Field(default='', max_length=120)


class AddArgs(ToolArgs):
    company: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=120)
    date_applied: str = Field(default='', max_length=10)
    status: ApplicationStatus = 'applied'
    url: str = Field(default='', max_length=500)
    notes: str = Field(default='', max_length=1000)
    allow_duplicate: bool = False

    @field_validator('date_applied')
    @classmethod
    def iso_date_not_future(cls, value):
        if not value:
            return value
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError('date_applied must be YYYY-MM-DD') from None
        if parsed > date.today():
            raise ValueError('date_applied cannot be in the future')
        return value

    @field_validator('url')
    @classmethod
    def http_url(cls, value):
        if value and not value.lower().startswith(('http://', 'https://')):
            raise ValueError('url must start with http:// or https://')
        return value


class UpdateStatusArgs(ToolArgs):
    application_id: str = Field(min_length=1, max_length=20)
    new_status: ApplicationStatus
    note: str = Field(default='', max_length=300)


class EmptyArgs(ToolArgs):
    pass


class DraftFollowUpArgs(ToolArgs):
    application_id: str = Field(min_length=1, max_length=20)
    subject: str = Field(min_length=3, max_length=150)
    body: str = Field(min_length=20, max_length=1500)


class ApplicationIdArgs(ToolArgs):
    application_id: str = Field(default='', max_length=20)


class EmailArgs(ToolArgs):
    to: str = Field(default='', max_length=200)
    subject: str = Field(default='', max_length=150)
    body: str = Field(default='', max_length=1500)
    application_id: str = Field(default='', max_length=20)


class SubmitArgs(ToolArgs):
    company: str = Field(default='', max_length=120)
    role: str = Field(default='', max_length=120)
    url: str = Field(default='', max_length=500)


# --- Tool-output contract ---------------------------------------------------
class ToolResult(BaseModel):
    """Every tool must return this shape; anything else is 'malformed output'."""
    model_config = ConfigDict(extra='forbid')

    ok: bool
    summary: str = Field(min_length=1, max_length=600)
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[str] = None   # set when ok is False
