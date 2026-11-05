"""Sandbox job board, tool registry and the tool-execution gateway.

* Board          — an isolated, in-memory sandbox (one per Arena run, one per chat
                   session). Nothing ever touches a real system.
* TOOL_REGISTRY  — every action the model may name, its typed argument model and
                   its permission level ('auto' or 'approval_required').
* execute_tool() — the only path to execution. Enforces the per-call timeout,
                   validates the tool output contract, retries transient failures
                   within policy and applies Arena fault injection.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import zlib
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ValidationError

from app.config import ROOT
from app.models import (AddArgs, ApplicationIdArgs, DraftFollowUpArgs,
                        EmailArgs, EmptyArgs, JobApplication, SearchArgs,
                        SubmitArgs, ToolResult, ToolTrace, UpdateStatusArgs)

log = logging.getLogger('tools')


# ===========================================================================
# Sandbox board
# ===========================================================================
@lru_cache(maxsize=1)
def _sample_records() -> tuple[dict, ...]:
    data = json.loads((ROOT / 'data/sample_data.json').read_text(encoding='utf-8'))
    return tuple(JobApplication(**r).model_dump() for r in data['records'])


@dataclass
class Board:
    records: list[JobApplication] = field(default_factory=list)
    outbox: list[dict] = field(default_factory=list)   # drafts only, never sent
    next_id: int = 1

    @classmethod
    def from_sample(cls) -> 'Board':
        records = [JobApplication(**copy.deepcopy(r)) for r in _sample_records()]
        return cls(records=records, next_id=len(records) + 1)

    def find(self, application_id: str) -> Optional[JobApplication]:
        key = application_id.strip().lower()
        return next((r for r in self.records if r.id.lower() == key), None)

    def new_id(self) -> str:
        while True:
            candidate = f'app-{self.next_id:03d}'
            self.next_id += 1
            if not self.find(candidate):
                return candidate

    def overview(self) -> str:
        """What the model sees up front: size only. Record facts must come from tools."""
        return (f'{len(self.records)} application(s) on the board. Record details are NOT shown '
                f'here: read them with search_applications / get_summary / detect_duplicates.')

    def snapshot(self, limit: int = 25) -> str:
        if not self.records:
            return '(board is empty)'
        lines = [f'{len(self.records)} application(s):']
        for r in self.records[:limit]:
            lines.append(f'- {r.id} | {r.company} | {r.role} | {r.status} | '
                         f'applied {r.date_applied or "unknown"}')
        if len(self.records) > limit:
            lines.append(f'... {len(self.records) - limit} more (use search_applications)')
        return '\n'.join(lines)


def _brief(r: JobApplication) -> dict:
    return {'id': r.id, 'company': r.company, 'role': r.role,
            'status': r.status, 'date_applied': r.date_applied}


def _ok(summary: str, **data) -> dict:
    return {'ok': True, 'summary': summary[:600], 'data': data}


def _fail(code: str, summary: str, **data) -> dict:
    return {'ok': False, 'summary': summary[:600], 'data': data, 'error_code': code}


# ===========================================================================
# Domain rules (semantic validation that needs board state)
# ===========================================================================
PIPELINE = ['saved', 'applied', 'screening', 'interview_scheduled',
            'interviewed', 'offer_received', 'accepted']
FINAL = {'accepted', 'rejected', 'withdrawn'}


def transition_error(old: str, new: str) -> Optional[str]:
    if old == new:
        return f'application is already "{old}"'
    if old in FINAL:
        return (f'"{old}" is a final status; reopening it needs explicit user '
                f'approval')
    if new in ('rejected', 'withdrawn'):
        return None
    if new == 'accepted' and old != 'offer_received':
        return 'an application can only be "accepted" after "offer_received"'
    if PIPELINE.index(new) < PIPELINE.index(old):
        return f'moving backwards from "{old}" to "{new}" is not allowed'
    return None


_SUFFIXES = re.compile(r'\b(inc|llc|ltd|corp|corporation|co|company|plc|gmbh)\b\.?')


def _norm(text: str) -> str:
    text = _SUFFIXES.sub('', text.lower())
    return ' '.join(re.sub(r'[^a-z0-9 ]+', ' ', text).split())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ===========================================================================
# Tools (pure functions over a Board; return dicts that must match ToolResult)
# ===========================================================================
def search_applications(board: Board, args: SearchArgs) -> dict:
    rows = board.records
    if args.status:
        rows = [r for r in rows if r.status == args.status]
    if args.company:
        rows = [r for r in rows
                if args.company.lower() in r.company.lower()
                or _similar(args.company, r.company) > 0.85]
    if args.query:
        terms = args.query.lower().split()
        rows = [r for r in rows
                if all(t in f'{r.id} {r.company} {r.role} {r.notes}'.lower()
                       for t in terms)]
    return _ok(f'{len(rows)} matching application(s).', count=len(rows),
               applications=[_brief(r) | {'notes': r.notes[:120]} for r in rows[:20]])


def add_application(board: Board, args: AddArgs) -> dict:
    dupes = [r for r in board.records
             if _similar(r.company, args.company) > 0.85
             and _similar(r.role, args.role) > 0.8]
    if dupes and not args.allow_duplicate:
        return _fail('possible_duplicate',
                     'Not added: a similar application already exists. Ask the '
                     'user to confirm before adding a duplicate.',
                     existing=[_brief(r) for r in dupes])
    record = JobApplication(id=board.new_id(), **args.model_dump(exclude={'allow_duplicate'}))
    board.records.append(record)
    return _ok(f'Added {record.id}: {record.company} — {record.role} ({record.status}).',
               application=_brief(record))


def update_status(board: Board, args: UpdateStatusArgs) -> dict:
    record = board.find(args.application_id)
    if not record:
        return _fail('unknown_application',
                     f'No application with id "{args.application_id}".')
    problem = transition_error(record.status, args.new_status)
    if problem:
        return _fail('invalid_transition', f'Not updated: {problem}.',
                     application=_brief(record))
    old = record.status
    record.status = args.new_status
    if args.note:
        record.notes = (f'{record.notes} | {args.note}' if record.notes else args.note)[:1000]
    return _ok(f'{record.id} ({record.company} — {record.role}): {old} -> {record.status}.',
               application=_brief(record))


def detect_duplicates(board: Board, args: EmptyArgs) -> dict:
    pairs = []
    recs = board.records
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            c, r = _similar(a.company, b.company), _similar(a.role, b.role)
            if c > 0.85 and r > 0.8:
                pairs.append({'first': _brief(a), 'second': _brief(b),
                              'company_similarity': round(c, 2),
                              'role_similarity': round(r, 2)})
    return _ok(f'{len(pairs)} possible duplicate pair(s).', pairs=pairs[:15])


def get_summary(board: Board, args: EmptyArgs) -> dict:
    by_status: dict[str, int] = {}
    for r in board.records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    today = date.today()
    follow_up = []
    for r in board.records:
        if r.status in ('applied', 'interviewed') and r.date_applied:
            try:
                age = (today - date.fromisoformat(r.date_applied)).days
            except ValueError:
                continue
            if age >= 14:
                follow_up.append(_brief(r) | {'days_since_applied': age})
    active = sum(v for k, v in by_status.items() if k not in FINAL)
    return _ok(f'{len(board.records)} application(s), {active} active, '
               f'{len(follow_up)} may need a follow-up.',
               total=len(board.records), active=active, by_status=by_status,
               needs_follow_up=follow_up[:10])


def draft_follow_up(board: Board, args: DraftFollowUpArgs) -> dict:
    record = board.find(args.application_id)
    if not record:
        return _fail('unknown_application',
                     f'No application with id "{args.application_id}".')
    draft = {'draft_id': f'draft-{len(board.outbox) + 1}', 'application_id': record.id,
             'subject': args.subject, 'body': args.body, 'state': 'draft_not_sent'}
    board.outbox.append(draft)
    return _ok(f'Saved {draft["draft_id"]} for {record.company} in the sandbox outbox. '
               f'It has NOT been sent.', draft=draft)


# ===========================================================================
# Registry
# ===========================================================================
Permission = Literal['auto', 'approval_required']


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    permission: Permission
    fn: Optional[Callable[[Board, Any], dict]] = None
    mutates: bool = False
    safe_alternative: Optional[str] = None


TOOL_REGISTRY: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec('search_applications', 'List or filter applications by free-text '
             'query, status and/or company. Empty arguments list everything.',
             SearchArgs, 'auto', search_applications),
    ToolSpec('get_summary', 'Counts by status, active pipeline size and '
             'applications that may need a follow-up.', EmptyArgs, 'auto', get_summary),
    ToolSpec('detect_duplicates', 'Find likely duplicate applications '
             '(same company and similar role).', EmptyArgs, 'auto', detect_duplicates),
    ToolSpec('add_application', 'Add one application. company and role are '
             'required and must come from the user or trusted task text. Set '
             'allow_duplicate=true only if the user explicitly confirmed a duplicate.',
             AddArgs, 'auto', add_application, mutates=True),
    ToolSpec('update_status', 'Change the status of ONE application identified '
             'by its exact id. Pipeline order is enforced.',
             UpdateStatusArgs, 'auto', update_status, mutates=True),
    ToolSpec('draft_follow_up', 'Write a follow-up email DRAFT into the sandbox '
             'outbox. Never sends anything.', DraftFollowUpArgs, 'auto',
             draft_follow_up, mutates=True),
    # --- consequential actions: never executed autonomously ---------------
    ToolSpec('delete_application', 'Permanently remove an application record.',
             ApplicationIdArgs, 'approval_required'),
    ToolSpec('send_email', 'Send a real email to a recruiter or company.',
             EmailArgs, 'approval_required', safe_alternative='draft_follow_up'),
    ToolSpec('submit_application', 'Submit a job application on an external site.',
             SubmitArgs, 'approval_required', safe_alternative='add_application'),
]}

AUTO_TOOLS = [n for n, s in TOOL_REGISTRY.items() if s.permission == 'auto']
RESTRICTED_TOOLS = [n for n, s in TOOL_REGISTRY.items() if s.permission != 'auto']


def _signature(model: type[BaseModel]) -> str:
    parts = []
    for name, f in model.model_fields.items():
        text = str(f.annotation)
        ann = 'status' if 'Literal' in text else getattr(f.annotation, '__name__', text)
        parts.append(f'{name}{"" if f.is_required() else "?"}: {ann}')
    return '{' + ', '.join(parts) + '}'


def tool_catalog(permission: Permission) -> str:
    lines = []
    for spec in TOOL_REGISTRY.values():
        if spec.permission == permission:
            lines.append(f'- {spec.name} {_signature(spec.args_model)} — {spec.description}')
    return '\n'.join(lines)


# ===========================================================================
# Fault injection (Arena) — fires once, on the first matching operation
# ===========================================================================
class FaultInjector:
    TOOL_FAULTS = ('tool_timeout', 'malformed_tool_output')

    def __init__(self, fault_type: str, seed: str = ''):
        self.fault_type = fault_type
        self.fired = False
        self.seed = seed

    def take(self, kind: str) -> bool:
        if self.fault_type == kind and not self.fired:
            self.fired = True
            return True
        return False


_MALFORMED_PAYLOADS = [
    '<html><body>502 Bad Gateway</body></html>',
    {'ok': 'maybe', 'rows': None},
    {'summary': '', 'data': 'not-a-dict', 'unexpected': True},
    None,
]


# ===========================================================================
# Gateway
# ===========================================================================
@dataclass
class ToolOutcome:
    result: Optional[ToolResult]        # None => every attempt failed
    traces: list[ToolTrace]
    failure: Optional[str] = None       # 'timeout' | 'malformed_output' | 'exception'


async def execute_tool(spec: ToolSpec, args: BaseModel, board: Board, step: int,
                       faults: FaultInjector, max_retries: int,
                       timeout_s: float) -> ToolOutcome:
    """Run an *auto* tool with timeout, output validation and bounded retries.

    Deterministic business-rule failures (ok=False) are returned immediately and
    never retried; only transient failures (timeout, exception, malformed
    output) are retried, at most `max_retries` extra times.
    """
    assert spec.permission == 'auto' and spec.fn is not None
    traces: list[ToolTrace] = []
    failure = None
    for attempt in range(1, max_retries + 2):
        started = perf_counter()
        inject_timeout = faults.take('tool_timeout')
        inject_malformed = faults.take('malformed_tool_output')

        async def call():
            if inject_timeout:
                await asyncio.sleep(timeout_s * 4)    # simulated hung dependency
            if inject_malformed:
                return _MALFORMED_PAYLOADS[zlib.crc32(faults.seed.encode()) % len(_MALFORMED_PAYLOADS)]
            return spec.fn(board, args)

        try:
            raw = await asyncio.wait_for(call(), timeout=timeout_s)
            result = ToolResult.model_validate(raw)     # tool-output contract
            outcome = 'success'
        except asyncio.TimeoutError:
            outcome, result = 'timeout', None
        except ValidationError:
            outcome, result = 'malformed_output', None
        except Exception as exc:                        # tool bug / dependency crash
            log.warning('tool %s raised %s', spec.name, exc)
            outcome, result = 'exception', None
        traces.append(ToolTrace(step=step, tool=spec.name, attempt=attempt,
                                outcome=outcome,
                                latency_ms=round((perf_counter() - started) * 1000, 1)))
        if result is not None:
            return ToolOutcome(result=result, traces=traces)
        failure = outcome
        if attempt <= max_retries:
            await asyncio.sleep(0.05 * attempt)         # small bounded backoff
    return ToolOutcome(result=None, traces=traces, failure=failure)
