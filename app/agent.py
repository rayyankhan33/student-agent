"""The bounded agent loop.

    observe -> assemble context -> model decides -> validate (schema + semantics)
      -> policy gate -> execute tool via gateway -> observation -> repeat

The model only *proposes* an AgentDecision. The system owns:
  * validation and bounded contract repair (max_repair_attempts)
  * the autonomy gate (approval_required tools never execute)
  * tool timeouts and retries (inside the gateway)
  * no-progress detection (repeated identical calls)
  * every stopping rule: steps, tokens, wall-clock time, repair budget, dependency failure
"""
from __future__ import annotations

import json
import logging
import re
import zlib
from dataclasses import dataclass, field
from time import monotonic, perf_counter
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from app.config import settings
from app.models import AgentDecision, ArenaResponse, Metrics, ToolTrace
from app.prompts import build_messages, new_boundary
from app.tools import (TOOL_REGISTRY, Board, FaultInjector, ToolSpec,
                       execute_tool)

log = logging.getLogger('agent')


# ===========================================================================
# Run state
# ===========================================================================
@dataclass
class RunState:
    request_id: str
    task: str
    model: str
    max_steps: int
    started: float = field(default_factory=monotonic)
    step: int = 0
    model_calls: int = 0
    llm_failures: int = 0
    repairs_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    observations: list[dict] = field(default_factory=list)
    tool_calls: list[ToolTrace] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    done_summaries: list[str] = field(default_factory=list)
    executed: dict[str, int] = field(default_factory=dict)   # call signature -> count
    failed_tools: set[str] = field(default_factory=set)      # retries exhausted
    unresolved_failure: bool = False                          # no success since a failure
    grounding_refusals: int = 0
    cost_usd: float = 0.0
    models_used: list[str] = field(default_factory=list)
    last_note: str = ''

    def event(self, kind: str, **data):
        entry = {'step': self.step, 'event': kind, **data}
        self.events.append(entry)
        log.info(json.dumps({'request_id': self.request_id, **entry}, default=str)[:1500])

    def error(self, kind: str, detail: str, **data):
        self.errors.append({'step': self.step, 'type': kind, 'detail': detail[:300], **data})


# ===========================================================================
# Model access
# ===========================================================================
_LLM_CACHE: dict[str, Any] = {}


def get_llm(model: str):
    if model in _LLM_CACHE:
        return _LLM_CACHE[model]
    provider = settings.model_provider.lower()
    if provider not in ('gemini', 'google'):
        raise RuntimeError(f'Unsupported or unconfigured MODEL_PROVIDER "{provider}"')
    if not settings.gemini_api_key:
        raise RuntimeError('GEMINI_API_KEY is not set')
    from langchain_google_genai import ChatGoogleGenerativeAI
    kwargs: dict[str, Any] = dict(
        model=model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,                         # one transport retry; loop adds no more
        response_mime_type='application/json', # provider-native JSON mode
    )
    if model.startswith('gemini-2.5'):
        kwargs['thinking_budget'] = 0          # hidden reasoning would eat the output budget
    llm = ChatGoogleGenerativeAI(**kwargs)
    _LLM_CACHE[model] = llm
    return llm


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ===========================================================================
# Contract validation
# ===========================================================================
class ContractViolation(Exception):
    pass


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith('{'):
        start, end = text.find('{'), text.rfind('}')
        if start == -1 or end <= start:
            raise ContractViolation('reply is not a JSON object')
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f'invalid JSON ({exc.msg} at char {exc.pos})') from None
    if not isinstance(data, dict):
        raise ContractViolation('top-level JSON value must be an object')
    return data


def _fmt(exc: ValidationError) -> str:
    return '; '.join(f"{'.'.join(map(str, e['loc'])) or 'decision'}: {e['msg']}"
                     for e in exc.errors()[:4])


def validate_decision(raw: str) -> tuple[AgentDecision, Optional[ToolSpec], Optional[BaseModel]]:
    """Schema validation, then semantic validation against the tool registry."""
    data = _extract_json(raw)
    try:
        decision = AgentDecision.model_validate(data)
    except ValidationError as exc:
        raise ContractViolation(f'schema: {_fmt(exc)}') from None
    if decision.status != 'continue':
        return decision, None, None
    spec = TOOL_REGISTRY.get(decision.action or '')
    if spec is None:
        raise ContractViolation(f'unknown action "{decision.action}". Allowed: '
                                f'{", ".join(TOOL_REGISTRY)}')
    if spec.permission != 'auto':
        return decision, spec, None                # policy gate handles it; never executed
    try:
        args = spec.args_model.model_validate(decision.arguments)
    except ValidationError as exc:
        raise ContractViolation(f'arguments for {spec.name}: {_fmt(exc)}') from None
    return decision, spec, args


_CORRUPTIONS = [
    'Sure! I will look into that for you right away.',                       # not JSON
    '{"status": "proceed", "action": 42, "arguments": "all"}',               # wrong enum/types
    '{"status": "continue", "action": null, "arguments": {}, "user_message": ""}',  # inconsistent
    '{"status": "continue", "action": "search_applications", '
    '"arguments": {"limit": "ten", "sort": true}}',                         # bad arguments
]


def corrupt_decision(request_id: str) -> str:
    """Arena fault invalid_agent_decision: replace the first model reply."""
    return _CORRUPTIONS[zlib.crc32(request_id.encode()) % len(_CORRUPTIONS)]


# ===========================================================================
# Response construction
# ===========================================================================
def _add_cost(state: RunState, model: str, tokens_in: int, tokens_out: int) -> None:
    price = settings.price(model)
    if price:
        state.cost_usd += (tokens_in * price[0] + tokens_out * price[1]) / 1e6


def _cost(state: RunState) -> Optional[float]:
    return round(state.cost_usd, 6) if state.cost_usd else None


def finish(state: RunState, status: str, stop_reason: str, message: str) -> ArenaResponse:
    message = (message or '').strip() or f'Run stopped with status {status}.'
    if len(message) > 1990:
        message = message[:1987] + '...'
    state.event('stop', status=status, stop_reason=stop_reason)
    return ArenaResponse(
        request_id=state.request_id, status=status, final_response=message,
        steps=state.step, stop_reason=stop_reason, tool_calls=state.tool_calls,
        errors=state.errors, events=state.events,
        metrics=Metrics(latency_ms=round((monotonic() - state.started) * 1000, 1),
                        model_calls=state.model_calls,
                        input_tokens=state.input_tokens or None,
                        output_tokens=state.output_tokens or None,
                        estimated_cost_usd=_cost(state)))


def budget_message(state: RunState, what: str) -> str:
    done = '; '.join(s.rstrip('.') for s in state.done_summaries[-4:]) or 'no tool actions completed'
    msg = f'Stopped: the {what} ran out before the request was finished. Completed so far: {done}.'
    if state.last_note:
        msg += f' Last plan: {state.last_note}'
    return msg + ' You can ask again with a smaller request to continue.'


# ===========================================================================
# The loop
# ===========================================================================
TERMINAL = {
    'completed': ('completed', 'goal_completed'),
    'needs_clarification': ('needs_clarification', 'clarification_required'),
    'blocked': ('blocked', 'out_of_scope_or_unsafe'),
    'approval_required': ('approval_required', 'approval_required'),
    'failed': ('failed', 'agent_reported_failure'),
}


def _ids_in(value) -> set[str]:
    """Every application id mentioned in a (nested) observation payload."""
    found: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ('id', 'application_id') and isinstance(v, str):
                found.add(v.lower())
            else:
                found |= _ids_in(v)
    elif isinstance(value, list):
        for v in value:
            found |= _ids_in(v)
    return found


def check_target(app_id: str, user_texts: list[str], observations: list[dict]) -> Optional[tuple[str, str, list]]:
    """Semantic validation for actions on an EXISTING record.

    The id must be grounded: named by the user (task, history, pending goal), or seen
    in this run's observations. If the most recent observation that mentions it is a
    search that matched several records, the target is ambiguous. Returns
    (error_code, message, candidate_ids) or None when the target is acceptable.
    """
    key = app_id.strip().lower()
    pattern = re.compile(rf'(?<![\w-]){re.escape(key)}(?![\w-])')
    if any(pattern.search(t.lower()) for t in user_texts):
        return None
    for obs in reversed(observations):
        if not obs.get('ok') or key not in _ids_in(obs.get('data', {})):
            continue
        if obs.get('tool') == 'search_applications' and obs['data'].get('count', 0) > 1:
            ids = [a['id'] for a in obs['data'].get('applications', [])]
            return ('ambiguous_target',
                    f'Not executed: the search that found {app_id} matched {len(ids)} records '
                    f'({", ".join(ids)}). Narrow the search with details the user gave, or ask '
                    f'the user which one.', ids)
        return None
    return ('ungrounded_target',
            f'Not executed: {app_id} was not named by the user and has not been looked up in '
            f'this run. Use search_applications to identify the record first.', [])


async def run_agent(request, *, state: RunState, board: Board, history: list | None = None,
                    pending: dict | None = None, llm=None, llm_factory=None) -> ArenaResponse:
    faults = FaultInjector(request.arena_config.fault.type, seed=request.request_id)
    boundary = new_boundary()
    history = history or []
    feedback = ''
    user_texts = [request.task] + [str(m.content) for m in history]
    if pending:
        user_texts += [pending.get('goal', ''), pending.get('question', '')]
    state.event('run_start', model=state.model, max_steps=state.max_steps,
                fault=faults.fault_type, external_items=len(request.external_context),
                history_messages=len(history), pending_goal=bool(pending))

    factory = llm_factory or (get_llm if llm is None else None)
    fallbacks = ([m for m in settings.models() if m != state.model]
                 if factory and settings.model_fallback else [])
    switched = False
    if llm is None:
        try:
            llm = factory(state.model)
        except Exception as exc:
            state.error('model_unavailable', str(exc))
            return finish(state, 'failed', 'model_not_configured',
                          'The language model is not configured on this server.')
    state.models_used.append(state.model)

    while True:
        # ---------------- stopping rules owned by the system ----------------
        if state.step >= state.max_steps:
            return finish(state, 'budget_exceeded', 'step_budget_reached',
                          budget_message(state, f'{state.max_steps}-step budget'))
        if state.input_tokens + state.output_tokens >= settings.max_run_tokens:
            return finish(state, 'budget_exceeded', 'token_budget_reached',
                          budget_message(state, f'{settings.max_run_tokens}-token budget'))
        remaining_time = settings.run_timeout_seconds - (monotonic() - state.started)
        if remaining_time < 3:
            return finish(state, 'budget_exceeded', 'time_budget_reached',
                          budget_message(state, f'{settings.run_timeout_seconds:.0f}-second time budget'))

        state.step += 1
        messages = build_messages(
            task=request.task, history=history, external_context=request.external_context,
            observations=state.observations, board_snapshot=board.overview(),
            step=state.step, max_steps=state.max_steps,
            repairs_left=settings.max_repair_attempts - state.repairs_used,
            boundary=boundary, pending=pending, feedback=feedback)

        # ---------------- model call (a dependency that can fail) ----------
        started = perf_counter()
        try:
            reply = await llm.ainvoke(messages)
        except Exception as exc:
            state.llm_failures += 1
            state.model_calls += 1
            state.error('llm_call_failed', f'{type(exc).__name__}: {exc}', model=state.model)
            state.event('llm_error', error=type(exc).__name__, model=state.model)
            if fallbacks and not switched:
                try:
                    nxt = fallbacks[0]
                    llm = factory(nxt)
                    state.event('model_fallback', from_model=state.model, to_model=nxt)
                    state.model, switched = nxt, True
                    state.models_used.append(nxt)
                    continue
                except Exception as fexc:
                    state.error('model_fallback_failed', str(fexc))
            if state.llm_failures >= (3 if switched else 2):
                return finish(state, 'failed', 'llm_unavailable',
                              'The language model is unavailable right now, so the request '
                              'could not be processed. No changes were made after the failure.')
            continue
        state.model_calls += 1
        usage = getattr(reply, 'usage_metadata', None) or {}
        t_in, t_out = int(usage.get('input_tokens') or 0), int(usage.get('output_tokens') or 0)
        state.input_tokens += t_in
        state.output_tokens += t_out
        _add_cost(state, state.model, t_in, t_out)
        raw = _text(reply.content)
        state.event('llm_reply', model=state.model,
                    latency_ms=round((perf_counter() - started) * 1000, 1),
                    chars=len(raw), input_tokens=usage.get('input_tokens'),
                    output_tokens=usage.get('output_tokens'))

        if faults.take('invalid_agent_decision'):
            raw = corrupt_decision(request.request_id)
            state.event('fault_injected', fault='invalid_agent_decision')

        # ---------------- contract gate --------------------------------------
        try:
            decision, spec, args = validate_decision(raw)
        except ContractViolation as exc:
            state.repairs_used += 1
            state.error('contract_violation', str(exc))
            state.event('validation', passed=False, detail=str(exc)[:200],
                        repairs_used=state.repairs_used)
            if state.repairs_used > settings.max_repair_attempts:
                return finish(state, 'contract_error', 'contract_repair_exhausted',
                              'The model kept returning decisions that violate the agent '
                              'contract, so nothing further was executed. Please try again.')
            feedback = str(exc)
            continue
        feedback = ''
        if decision.reasoning or decision.user_message:
            state.last_note = (decision.user_message or decision.reasoning)[:300]
        state.event('validation', passed=True)
        state.event('decision', status=decision.status, action=decision.action,
                    arguments=decision.arguments, reasoning=decision.reasoning[:200])

        # ---------------- terminal decisions ---------------------------------
        if decision.status != 'continue':
            status, reason = TERMINAL[decision.status]
            if decision.status in ('completed', 'failed') and state.unresolved_failure:
                status, reason = 'tool_error', 'tool_retries_exhausted'
            return finish(state, status, reason, decision.user_message)

        # ---------------- autonomy gate --------------------------------------
        if spec.permission != 'auto':
            state.tool_calls.append(ToolTrace(step=state.step, tool=spec.name,
                                              outcome='rejected'))
            state.event('policy_gate', tool=spec.name, verdict='approval_required')
            alt = (f' I can do the safe alternative instead ({spec.safe_alternative}) '
                   f'if you ask.' if spec.safe_alternative else '')
            note = f' {decision.user_message}' if decision.user_message else ''
            return finish(state, 'approval_required', 'approval_required',
                          f'"{spec.name}" is a consequential action outside my authority, so I '
                          f'did not perform it. It needs explicit human approval outside this '
                          f'agent.{alt}{note}')

        # ---------------- no-progress guard ----------------------------------
        signature = spec.name + json.dumps(args.model_dump(), sort_keys=True, default=str)
        if spec.name in state.failed_tools:
            return finish(state, 'tool_error', 'tool_retries_exhausted',
                          f'The {spec.name} tool is failing repeatedly, so I stopped instead '
                          f'of guessing. Completed so far: '
                          f'{"; ".join(state.done_summaries) or "nothing"}.')
        if state.executed.get(signature):
            state.executed[signature] += 1
            state.error('repeated_action', f'{spec.name} called again with identical arguments')
            if state.executed[signature] > 2:
                return finish(state, 'failed', 'no_progress_loop',
                              'Stopped: the agent kept repeating the same action without '
                              'making progress.')
            state.observations.append({'step': state.step, 'tool': spec.name, 'ok': False,
                                       'error_code': 'duplicate_call',
                                       'summary': 'Not executed: identical call already made '
                                                  'in this run. Use the earlier observation.'})
            continue

        # ---------------- semantic check: is the target record grounded? ----
        target = getattr(args, 'application_id', None) if spec.mutates else None
        if target:
            problem = check_target(target, user_texts, state.observations)
            if problem:
                code, message, candidates = problem
                state.grounding_refusals += 1
                state.tool_calls.append(ToolTrace(step=state.step, tool=spec.name,
                                                  outcome='rejected'))
                state.error(code, message, tool=spec.name)
                state.event('policy_gate', tool=spec.name, verdict=code, target=target)
                if state.grounding_refusals >= 2:
                    # The model keeps picking an unconfirmed record: the system asks instead.
                    options = ', '.join(_describe(board, i) for i in candidates) or \
                        'the application id (see "Show my applications")'
                    return finish(state, 'needs_clarification', 'clarification_required',
                                  f'I need to be sure which application you mean before changing '
                                  f'anything. Which one: {options}?')
                state.observations.append({'step': state.step, 'tool': spec.name, 'ok': False,
                                           'error_code': code, 'summary': message})
                continue

        # ---------------- execute through the gateway ------------------------
        outcome = await execute_tool(spec, args, board, state.step, faults,
                                     settings.max_tool_retries, settings.tool_timeout_seconds)
        state.tool_calls.extend(outcome.traces)
        for t in outcome.traces:
            if t.outcome != 'success':
                state.error(f'tool_{t.outcome}', f'{t.tool} attempt {t.attempt}: {t.outcome}',
                            tool=t.tool, attempt=t.attempt)
        if outcome.result is None:
            state.failed_tools.add(spec.name)
            state.unresolved_failure = True
            obs = {'step': state.step, 'tool': spec.name, 'ok': False,
                   'error_code': 'dependency_failure',
                   'summary': f'{spec.name} failed {len(outcome.traces)} time(s) '
                              f'({outcome.failure}). It is unavailable for the rest of this run.'}
        else:
            r = outcome.result
            state.executed[signature] = 1
            obs = {'step': state.step, 'tool': spec.name, 'arguments': args.model_dump(
                exclude_defaults=True), 'ok': r.ok, 'summary': r.summary, 'data': r.data}
            if r.error_code:
                obs['error_code'] = r.error_code
            state.unresolved_failure = False
            if r.ok:
                state.done_summaries.append(r.summary)
        state.observations.append(obs)
        data_preview = json.dumps(obs.get('data', {}), default=str)
        state.event('observation', tool=spec.name, ok=obs['ok'],
                    attempts=len(outcome.traces), summary=obs['summary'][:300],
                    error_code=obs.get('error_code'),
                    data=data_preview if len(data_preview) <= 700 else data_preview[:700] + '…')


def _describe(board: Board, app_id: str) -> str:
    r = board.find(app_id)
    return f'{r.id} ({r.company}, {r.role}, {r.status})' if r else app_id
