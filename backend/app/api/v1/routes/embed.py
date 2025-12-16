from io import BytesIO
from typing import Annotated, List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from pydantic import BaseModel
from PIL import Image
from app.manager.model_manager import ModelManager
from app.security.auth import verify_api_key
from app.services.embedder_service import Embedder
from app.core.config import settings
from app.core.rate_limiter import limiter
from app.core.image_limits import validate_image_dimensions


async def _rate_limit_dependency(request: Request, api_key: str = Depends(verify_api_key)):
    key = api_key if api_key else (request.client.host if request.client else "anon")
    await limiter.check_or_raise(str(key))


router = APIRouter(prefix="/v1/embedding", tags=["embedding"])


class EmbeddedFrames(BaseModel):
    vector: list[float]


class EmbeddingBatchResponse(BaseModel):
    frames: List[EmbeddedFrames]


async def get_model() -> Embedder:
    """Dependency to inject model into endpoints"""

    try:
        return ModelManager.get_embedder()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/", response_model=EmbeddingBatchResponse)
async def embed_images(
    embedder: Annotated[Embedder, Depends(get_model)],
    files: List[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limit_dependency),
):
    """Accept one or more uploaded images and return generated embeddings.

    Upload multiple files with form field name `files`. Returns a list of
    embedding vectors preserving the order of uploaded files. This endpoint
    only returns embeddings; scene boundary detection and similarity
    comparisons are handled by other services.
    """

    # Enforce batch size limits per request
    if hasattr(settings, "max_batch_size") and len(files) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail=f"Too many files in batch (max={settings.max_batch_size})")

    images: List[Image.Image] = []

    for f in files:
        content_type = (f.content_type or "").lower()
        if not content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="All uploaded files must be images")

        data = await f.read()

        # Enforce upload size limits
        if hasattr(settings, "max_upload_size_bytes") and len(data) > settings.max_upload_size_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file too large")
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to parse one of the image files")

        # Validate dimensions using shared utility
        try:
            validate_image_dimensions(img)
        except ValueError:
            raise HTTPException(status_code=413, detail="Image dimensions exceed allowed limits")

        images.append(img)
    try:
        results = await embedder.embed_images(images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    frames_out = [EmbeddedFrames(vector=r.vector) for r in results]

    return EmbeddingBatchResponse(frames=frames_out)