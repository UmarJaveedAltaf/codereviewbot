import time

from fastapi import FastAPI

from config import settings
from webhook.handler import router as webhook_router

app = FastAPI(
    title="CodeReviewBot",
    version="0.1.0",
    description="AI-powered GitHub PR reviewer with AST parsing and vector memory",
)

app.include_router(webhook_router, prefix="/webhook", tags=["webhook"])

START_TIME = time.time()


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.1.0",
        "env": settings.APP_ENV,
        "uptime_seconds": round(time.time() - START_TIME),
    }
