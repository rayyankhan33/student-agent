import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from app.config import ROOT, settings
from app.models import ArenaRequest, ArenaResponse, ChatRequest
from app.arena import execute

router = APIRouter()

# ---------------------------------------------------------------------------
# Determine available models from configuration
# ---------------------------------------------------------------------------
def _get_available_models() -> list[str]:
    """Return list of configured model identifiers."""
    models = []
    if settings.model_name and settings.model_provider != 'unconfigured':
        models.append(settings.model_name)
    # Provide a second model option for comparison (same provider)
    if settings.model_provider in ('gemini', 'google') and settings.gemini_api_key:
        if 'gemini-2.5-flash' not in models:
            models.append('gemini-2.5-flash')
    if not models:
        models.append('unconfigured')
    return models


@router.get('/')
def index():
    return FileResponse(ROOT / 'app/static/index.html')


@router.get('/health')
def health():
    return {
        'status': 'ok',
        'implementation': 'job-application-tracker',
        'model_provider': settings.model_provider,
        'model_name': settings.model_name or 'unconfigured',
    }


@router.get('/arena/manifest')
def manifest():
    return json.loads((ROOT / 'arena_manifest.json').read_text())


@router.get('/models')
def models():
    return {'models': _get_available_models()}


@router.post('/arena/run', response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest):
    return await execute(payload, model=settings.model_name or 'unconfigured')


@router.post('/chat', response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    available = _get_available_models()
    model = payload.model
    if model == 'unconfigured' and settings.model_name:
        model = settings.model_name
    if model not in available:
        raise HTTPException(400, f'Model is not enabled. Available: {available}')
    busy = request.app.state.busy
    if payload.session_id in busy:
        raise HTTPException(409, 'This chat is already running')
    busy.add(payload.session_id)
    memory = request.app.state.memory
    try:
        result = await execute(payload, memory.get(payload.session_id), model)
        memory.add(payload.session_id, payload.task, result.final_response)
        return result
    finally:
        busy.discard(payload.session_id)


@router.delete('/chat/{session_id}')
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy:
        raise HTTPException(409, 'Chat is running')
    request.app.state.memory.clear(session_id)
    return {'status': 'cleared'}
