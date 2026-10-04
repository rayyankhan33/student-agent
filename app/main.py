from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import router
from app.config import ROOT
from app.memory import Memory

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)


@asynccontextmanager
async def lifespan(app):
    app.state.memory = Memory()
    app.state.busy = set()
    yield


app = FastAPI(
    title='JobTracker — Agent Arena',
    description='Bounded job-application tracking agent with Arena interface',
    lifespan=lifespan
)
app.include_router(router)
app.mount('/static', StaticFiles(directory=ROOT / 'app/static'), name='static')
