"""Five sandbox tools for the Job-Application Tracker agent.

Each tool:
  - Has typed inputs and outputs
  - Validates business rules before mutation
  - Returns a structured result dict

All tool execution is routed through `execute_tool()` which acts as a gateway
for Arena fault injection (tool_timeout, malformed_tool_output).
"""
import asyncio
import json
import logging
from time import perf_counter
from typing import Optional
from difflib import SequenceMatcher

from app.models import JobApplication, ToolTrace, VALID_STATUSES

log = logging.getLogger('tools')

# ---------------------------------------------------------------------------
# In-memory sandbox job board (per-process; reset on restart)
# ---------------------------------------------------------------------------
_board: list[JobApplication] = []
_board_initialized = False


def _load_sample_data():
    """Load sample data into the board on first use."""
    global _board, _board_initialized
    if _board_initialized:
        return
    _board_initialized = True
    try:
        from app.config import ROOT
        data = json.loads((ROOT / 'data/sample_data.json').read_text())
        for record in data.get('records', []):
            try:
                app = JobApplication(**record)
                _board.append(app)
            except Exception as e:
                log.warning('Skipped invalid sample record: %s', e)
    except Exception as e:
        log.warning('Could not load sample data: %s', e)


def get_board_snapshot() -> str:
    """Return a compact text summary of the current board for the LLM."""
    _load_sample_data()
    if not _board:
        return "(Board is empty — no applications tracked yet)"
    lines = [f"Total applications: {len(_board)}"]
    for app in _board:
        lines.append(
            f"  [{app.id}] {app.company} — {app.role} | "
            f"Status: {app.status} | Applied: {app.date_applied or 'N/A'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_applications(
    query: str = '',
    status_filter: str = '',
    company_filter: str = ''
) -> dict:
    """Search and filter job applications on the board."""
    _load_sample_data()
    results = list(_board)

    if status_filter:
        status_filter = status_filter.strip().lower()
        if status_filter not in VALID_STATUSES:
            return {
                'success': False,
                'error': f"Invalid status filter '{status_filter}'. "
                         f"Valid: {VALID_STATUSES}"
            }
        results = [a for a in results if a.status == status_filter]

    if company_filter:
        company_lower = company_filter.strip().lower()
        results = [a for a in results
                   if company_lower in a.company.lower()]

    if query:
        query_lower = query.strip().lower()
        results = [
            a for a in results
            if (query_lower in a.company.lower()
                or query_lower in a.role.lower()
                or query_lower in a.notes.lower()
                or query_lower in a.id.lower())
        ]

    return {
        'success': True,
        'count': len(results),
        'applications': [a.model_dump() for a in results]
    }


def add_application(
    company: str,
    role: str,
    date_applied: str = '',
    status: str = 'applied',
    url: str = '',
    notes: str = ''
) -> dict:
    """Add a new job application to the sandbox board."""
    _load_sample_data()

    # Validate required fields
    if not company or not company.strip():
        return {'success': False, 'error': 'Company name is required.'}
    if not role or not role.strip():
        return {'success': False, 'error': 'Role/position is required.'}

    # Check for potential duplicates before adding
    company_lower = company.strip().lower()
    role_lower = role.strip().lower()
    duplicates = [
        a for a in _board
        if a.company.lower() == company_lower and a.role.lower() == role_lower
    ]

    try:
        app = JobApplication(
            company=company.strip(),
            role=role.strip(),
            date_applied=date_applied.strip() if date_applied else '',
            status=status.strip().lower() if status else 'applied',
            url=url.strip() if url else '',
            notes=notes.strip() if notes else ''
        )
    except Exception as e:
        return {'success': False, 'error': str(e)}

    _board.append(app)

    result = {
        'success': True,
        'application': app.model_dump(),
        'message': f"Added application [{app.id}]: {app.company} — {app.role}"
    }
    if duplicates:
        result['duplicate_warning'] = (
            f"Warning: {len(duplicates)} existing application(s) found for "
            f"the same company and role: "
            f"{[d.id for d in duplicates]}"
        )
    return result


def update_status(
    application_id: str,
    new_status: str,
    notes: str = ''
) -> dict:
    """Update the status of an existing application."""
    _load_sample_data()

    if not application_id or not application_id.strip():
        return {'success': False,
                'error': 'Application ID is required.'}
    if not new_status or not new_status.strip():
        return {'success': False,
                'error': 'New status is required.'}

    new_status_lower = new_status.strip().lower()
    if new_status_lower not in VALID_STATUSES:
        return {
            'success': False,
            'error': f"Invalid status '{new_status}'. "
                     f"Valid: {VALID_STATUSES}"
        }

    app_id = application_id.strip().lower()
    found = [a for a in _board if a.id.lower() == app_id]

    if not found:
        return {
            'success': False,
            'error': f"No application found with ID '{application_id}'."
        }

    app = found[0]
    old_status = app.status
    app.status = new_status_lower
    if notes:
        app.notes = (app.notes + ' | ' + notes.strip()) if app.notes else notes.strip()

    return {
        'success': True,
        'application': app.model_dump(),
        'message': (f"Updated [{app.id}] {app.company} — {app.role}: "
                    f"{old_status} → {new_status_lower}")
    }


def detect_duplicates() -> dict:
    """Find potential duplicate applications on the board."""
    _load_sample_data()

    if len(_board) < 2:
        return {
            'success': True,
            'duplicates': [],
            'message': 'Not enough applications to check for duplicates.'
        }

    duplicates = []
    for i in range(len(_board)):
        for j in range(i + 1, len(_board)):
            a, b = _board[i], _board[j]
            company_sim = SequenceMatcher(
                None, a.company.lower(), b.company.lower()
            ).ratio()
            role_sim = SequenceMatcher(
                None, a.role.lower(), b.role.lower()
            ).ratio()

            if company_sim > 0.8 and role_sim > 0.7:
                duplicates.append({
                    'app1': {'id': a.id, 'company': a.company,
                             'role': a.role, 'status': a.status},
                    'app2': {'id': b.id, 'company': b.company,
                             'role': b.role, 'status': b.status},
                    'company_similarity': round(company_sim, 2),
                    'role_similarity': round(role_sim, 2)
                })

    return {
        'success': True,
        'duplicates': duplicates,
        'message': (f"Found {len(duplicates)} potential duplicate pair(s)."
                    if duplicates
                    else "No duplicates detected.")
    }


def get_summary() -> dict:
    """Get analytics and statistics about all tracked applications."""
    _load_sample_data()

    if not _board:
        return {
            'success': True,
            'total': 0,
            'by_status': {},
            'by_company': {},
            'message': 'No applications on the board yet.'
        }

    by_status: dict[str, int] = {}
    by_company: dict[str, int] = {}

    for app in _board:
        by_status[app.status] = by_status.get(app.status, 0) + 1
        by_company[app.company] = by_company.get(app.company, 0) + 1

    return {
        'success': True,
        'total': len(_board),
        'by_status': by_status,
        'by_company': by_company,
        'recent': [
            {'id': a.id, 'company': a.company, 'role': a.role,
             'status': a.status}
            for a in _board[-5:]
        ],
        'message': f"Board has {len(_board)} application(s) across "
                   f"{len(by_company)} companies."
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
TOOLS = {
    'search_applications': search_applications,
    'add_application': add_application,
    'update_status': update_status,
    'detect_duplicates': detect_duplicates,
    'get_summary': get_summary,
}

# Actions that are blocked (autonomy boundary)
BLOCKED_ACTIONS = {
    'delete_all', 'delete_all_applications', 'clear_board',
    'send_email', 'send_notification', 'export_data',
    'access_external', 'scrape_website', 'apply_to_job',
}


# ---------------------------------------------------------------------------
# Fault injection + tool execution gateway
# ---------------------------------------------------------------------------

async def execute_tool(
    tool_name: str,
    arguments: dict,
    step: int,
    fault_type: str = 'none',
    fault_triggered: bool = False,
    max_retries: int = 2,
) -> tuple[dict, ToolTrace, bool]:
    """Execute a tool with optional Arena fault injection.

    Returns (result_dict, tool_trace, fault_was_triggered_now).
    Handles retries internally up to max_retries.
    """
    # Check autonomy boundary
    if tool_name in BLOCKED_ACTIONS:
        trace = ToolTrace(
            step=step, tool=tool_name, attempt=1, outcome='rejected'
        )
        return (
            {'success': False,
             'error': f"Action '{tool_name}' is blocked. This action is "
                      f"outside the agent's permitted authority."},
            trace,
            False
        )

    if tool_name not in TOOLS:
        trace = ToolTrace(
            step=step, tool=tool_name, attempt=1, outcome='rejected'
        )
        return (
            {'success': False,
             'error': f"Unknown tool '{tool_name}'. Available: "
                      f"{list(TOOLS.keys())}"},
            trace,
            False
        )

    tool_fn = TOOLS[tool_name]
    should_inject = (fault_type != 'none' and not fault_triggered)

    for attempt in range(1, max_retries + 2):  # 1-indexed, up to max_retries+1
        started = perf_counter()
        try:
            # Inject fault on first matching operation
            if should_inject and attempt == 1:
                if fault_type == 'tool_timeout':
                    # Simulate a tool timeout
                    await asyncio.sleep(6)
                    raise TimeoutError("Tool execution timed out")
                elif fault_type == 'malformed_tool_output':
                    # Return garbage instead of real result
                    latency = (perf_counter() - started) * 1000
                    trace = ToolTrace(
                        step=step, tool=tool_name, attempt=attempt,
                        outcome='malformed_output', latency_ms=latency
                    )
                    return (
                        {'__malformed__': True,
                         'garbage': 'xK#!@not-json{{{'},
                        trace,
                        True
                    )

            # Normal execution
            result = tool_fn(**arguments)
            latency = (perf_counter() - started) * 1000
            trace = ToolTrace(
                step=step, tool=tool_name, attempt=attempt,
                outcome='success', latency_ms=latency
            )
            return result, trace, (should_inject and attempt == 1)

        except TimeoutError:
            latency = (perf_counter() - started) * 1000
            trace = ToolTrace(
                step=step, tool=tool_name, attempt=attempt,
                outcome='timeout', latency_ms=latency
            )
            if attempt <= max_retries:
                log.warning('Tool %s attempt %d timed out, retrying...',
                            tool_name, attempt)
                should_inject = False  # Don't re-inject on retry
                continue
            return (
                {'success': False,
                 'error': f"Tool '{tool_name}' timed out after "
                          f"{attempt} attempts."},
                trace,
                True
            )

        except Exception as e:
            latency = (perf_counter() - started) * 1000
            trace = ToolTrace(
                step=step, tool=tool_name, attempt=attempt,
                outcome='exception', latency_ms=latency
            )
            if attempt <= max_retries:
                log.warning('Tool %s attempt %d failed: %s, retrying...',
                            tool_name, attempt, e)
                continue
            return (
                {'success': False,
                 'error': f"Tool '{tool_name}' failed after {attempt} "
                          f"attempts: {str(e)}"},
                trace,
                (should_inject and attempt == 1)
            )

    # Should not reach here, but safety fallback
    return (
        {'success': False, 'error': 'Unexpected tool execution state'},
        ToolTrace(step=step, tool=tool_name, attempt=1, outcome='exception'),
        False
    )
