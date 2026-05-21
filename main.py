import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from webhook.handler import router as webhook_router

app = FastAPI(
    title="CodeReviewBot",
    version="0.1.0",
    description="AI-powered GitHub PR reviewer with AST parsing and vector memory",
)

app.include_router(webhook_router, prefix="/webhook", tags=["webhook"])

# Serve static assets (CSS, JS, images) from the frontend/ directory.
# FileResponse for "/" needs aiofiles installed; StaticFiles handles the rest.
if os.path.isdir("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

START_TIME = time.time()


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.1.0",
        "env": settings.APP_ENV,
        "uptime_seconds": round(time.time() - START_TIME),
    }
