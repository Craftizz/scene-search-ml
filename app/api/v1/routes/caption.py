from io import BytesIO
from typing import Annotated, List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image

from app.manager.model_manager import ModelManager
from app.services.captioner_service import Captioner


router = APIRouter(prefix="/v1/caption", tags=["caption"])


class CaptionResponse(BaseModel):
    caption: str


async def get_model() -> Captioner:
    """Dependency to inject model into endpoints"""

    try:
        return ModelManager.get_captioner()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/", response_model=List[CaptionResponse])
async def caption_images(
    captioner: Annotated[Captioner, Depends(get_model)],
    files: List[UploadFile] = File(...),
):
    """Accept one or more uploaded images and return generated captions."""

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
        # captioner.caption expects a list and returns a list of CaptionResult
        results = await captioner.caption(images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {e}")

    return [CaptionResponse(caption=r.text) for r in results]
