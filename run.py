"""Local entry point: python run.py (reads HOST/PORT from the environment or .env)."""
import uvicorn

from app.config import settings

if __name__ == '__main__':
    uvicorn.run('app.main:app', host=settings.host, port=settings.port, workers=1)
