from io import BytesIO
from typing import Annotated, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image
from app.manager.model_manager import ModelManager
from app.security.auth import verify_api_key
from app.services.embedder_service import Embedder


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
):
    """Accept one or more uploaded images and return generated embeddings.

    Upload multiple files with form field name `files`. Returns a list of
    embedding vectors preserving the order of uploaded files. This endpoint
    only returns embeddings; scene boundary detection and similarity
    comparisons are handled by other services.
    """

    images: List[Image.Image] = []

    for f in files:
        content_type = (f.content_type or "").lower()
        if not content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="All uploaded files must be images")

        data = await f.read()
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to parse one of the image files")

        images.append(img)
    try:
        results = await embedder.embed_images(images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    frames_out = [EmbeddedFrames(vector=r.vector) for r in results]

    return EmbeddingBatchResponse(frames=frames_out)