 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.api.v1.routes.caption import router as caption_router
from app.api.v1.routes.embed import router as embedding_router
from app.api.v1.routes.similar import router as similar_router
from app.manager.model_manager import ModelManager
from app.core.config import settings

logger = logging.getLogger("scene_search")

logger.info(
    "Application configured: env=%s, cors=%s, api_key_set=%s, docs_enabled=%s",
    settings.environment,
    settings.cors_origins,
    bool(settings.api_key),
    settings.enable_docs,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Handles model initialization and cleanup.
    """

    logger.info("Starting up application")
    await ModelManager.initialize()

    app.include_router(caption_router)
    app.include_router(embedding_router)
    app.include_router(similar_router)
    yield

    logger.info("Shutting down application")
    await ModelManager.cleanup()


app = FastAPI(
    title="ML Model API",
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "documentation": "/docs" if settings.enable_docs else None,
        "endpoints": {
            "health": "/health",
            "caption": "/api/v1/caption",
            "embed": "/api/v1/embed",
            "similar": "/api/v1/similar",
        },
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
    )

