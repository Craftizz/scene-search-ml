from dataclasses import dataclass
from fastapi import FastAPI
import logging
from contextlib import asynccontextmanager

from app.api.v1.routes.caption import router as caption_router
from app.api.v1.routes.embed import router as embedding_router
from app.api.v1.routes.similar import router as similar_router
from app.manager.model_manager import ModelManager

logger = logging.getLogger("scene_search")


@dataclass
class Settings():
    version: str = "1.0.0"
    enable_metrics: bool = True
    max_batch_size: int = 32
    
    class Config:
        env_file = ".env"


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("Starting up application")
    await ModelManager.initialize()

    app.include_router(caption_router)
    app.include_router(embedding_router)
    app.include_router(similar_router)
    yield

    logger.info("Shutting down application")
    await ModelManager.cleanup()


settings = Settings()

app = FastAPI(
    title="ML Model API",
    version=settings.version,
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Scene Search API",
        "version": settings.version,
        "docs": "/docs"
    }
