import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.arena import execute
from app.config import ROOT, settings
from app.models import ArenaRequest, ArenaResponse, ChatRequest, ChatResponse

router = APIRouter()


def default_model() -> str:
    return settings.model_name or 'unconfigured'


@router.get('/', include_in_schema=False)
def index():
    return FileResponse(ROOT / 'app/static/index.html')


@router.get('/health')
def health():
    return {'status': 'ok',
            'implementation': 'job-application-tracker',
            'model_provider': settings.model_provider,
            'model_name': default_model(),
            'model_configured': bool(settings.model_name and settings.gemini_api_key)}


@router.get('/arena/manifest')
def manifest():
    return json.loads((ROOT / 'arena_manifest.json').read_text(encoding='utf-8'))


@router.get('/models')
def models():
    return {'default': default_model(), 'models': settings.models() or ['unconfigured']}


@router.get('/demo/scenarios')
def demo_scenarios():
    """Public evaluation cases plus a two-turn clarification script, for the demo UI."""
    cases = json.loads((ROOT / 'evaluation/public_cases.json').read_text(encoding='utf-8'))
    scenarios = [{'name': c['name'], 'category': c['category'], 'turns': [c['request']['task']],
                  'external_context': c['request'].get('external_context', []),
                  'arena_config': c['request'].get('arena_config', {}),
                  'expected_status': c['expected_status']} for c in cases]
    scenarios.append({'name': 'Multi-turn: clarification then bare id', 'category': 'multi_turn',
                      'turns': ['Move my Google software engineer application to interviewed',
                                'app-001'],
                      'external_context': [], 'arena_config': {'max_steps': 6, 'fault': 'none'},
                      'expected_status': ['needs_clarification', 'completed']})
    return {'scenarios': scenarios}


@router.post('/arena/run', response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest):
    """Stateless evaluation endpoint: fresh sandbox board, no chat memory."""
    return await execute(payload, model=default_model())


@router.post('/chat', response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    model = payload.model or default_model()
    if model not in (settings.models() or ['unconfigured']):
        raise HTTPException(400, f'Model "{model}" is not enabled.')
    busy = request.app.state.busy
    if payload.session_id in busy:
        raise HTTPException(409, 'This chat is already running')
    busy.add(payload.session_id)
    memory = request.app.state.memory
    try:
        session = memory.session(payload.session_id)
        arena_request = ArenaRequest(**payload.model_dump(exclude={'session_id', 'model'}))
        result = await execute(arena_request, model=model, board=session.board,
                               history=memory.get(payload.session_id),
                               pending=session.pending)
        memory.add(payload.session_id, payload.task, result.final_response, result.status)
        return ChatResponse(**result.model_dump(), session_id=payload.session_id, model=model,
                            history_messages=len(session.messages),
                            pending_goal=(session.pending or {}).get('goal'))
    finally:
        busy.discard(payload.session_id)


@router.get('/chat/{session_id}/board')
def board(session_id: str, request: Request):
    s = request.app.state.memory.sessions.get(session_id)
    if not s:
        return {'records': [], 'outbox': []}
    return {'records': [r.model_dump() for r in s.board.records], 'outbox': s.board.outbox}


@router.delete('/chat/{session_id}')
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy:
        raise HTTPException(409, 'Chat is running')
    request.app.state.memory.clear(session_id)
    return {'status': 'cleared'}
