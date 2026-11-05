"""Common run boundary shared by POST /arena/run and POST /chat.

* clamps the requested step budget to the server maximum
* enforces the hard wall-clock timeout while keeping the partial trace
* converts any unexpected exception into a typed 'failed' response (never a 500)
* guarantees the response matches the Arena contract and size limit
"""
import asyncio
import logging

from app.agent import RunState, finish, run_agent
from app.config import settings
from app.models import ArenaResponse
from app.tools import Board

log = logging.getLogger('arena')
MAX_RESPONSE_BYTES = 50_000


async def execute(request, *, model: str, board: Board | None = None,
                  history: list | None = None, pending: dict | None = None,
                  llm=None, llm_factory=None) -> ArenaResponse:
    request = request.model_copy(deep=True)
    request.arena_config.max_steps = min(request.arena_config.max_steps, settings.max_steps)
    state = RunState(request_id=request.request_id, task=request.task, model=model,
                     max_steps=request.arena_config.max_steps)
    board = board if board is not None else Board.from_sample()   # Arena runs are isolated
    try:
        async with asyncio.timeout(settings.run_timeout_seconds):
            result = await run_agent(request, state=state, board=board, history=history,
                                     pending=pending, llm=llm, llm_factory=llm_factory)
        result = ArenaResponse.model_validate(result.model_dump())
    except TimeoutError:
        result = finish(state, 'budget_exceeded', 'time_budget_reached',
                        f'Stopped: the {settings.run_timeout_seconds:.0f}-second time budget '
                        f'ran out before the request was finished.')
    except Exception as exc:                      # last line of defence
        log.exception('run %s crashed', request.request_id)
        state.error('internal_error', type(exc).__name__)
        result = finish(state, 'failed', 'internal_error',
                        'The run stopped because of an internal error. No further actions were taken.')

    if len(result.model_dump_json().encode()) > MAX_RESPONSE_BYTES:
        result.events = result.events[-20:]
        result.errors = result.errors[-10:]
        if len(result.model_dump_json().encode()) > MAX_RESPONSE_BYTES:
            result.events, result.errors = [], []
    log.info('request=%s model=%s status=%s stop=%s steps=%s calls=%s tokens=%s/%s',
             request.request_id, model, result.status, result.stop_reason, result.steps,
             result.metrics.model_calls, result.metrics.input_tokens,
             result.metrics.output_tokens)
    return result
