import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import ROOT, settings
from app.memory import Memory

logging.basicConfig(level=settings.log_level.upper(),
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
# Keep provider client logs quiet; they can include request metadata.
for noisy in ('httpx', 'httpcore', 'google', 'urllib3'):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app):
    app.state.memory = Memory()   # in-process: requires a single worker
    app.state.busy = set()
    yield


app = FastAPI(title='JobTracker — Agent Arena',
              description='Bounded job-application tracking agent with the common Arena interface.',
              version='2.0.0', lifespan=lifespan)
app.include_router(router)
app.mount('/static', StaticFiles(directory=ROOT / 'app/static'), name='static')
