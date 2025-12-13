from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import numpy as np

from app.manager.model_manager import ModelManager
from app.security.auth import verify_api_key
from app.services.embedder_service import Embedder
from app.services.similarity_service import Similarity


router = APIRouter(prefix="/v1/similar", tags=["similar"])


class SimilarRequest(BaseModel):
    string: str
    frames: List[dict[str, Any]]
    top_percent: Optional[float] = None
    threshold_raw: Optional[float] = None


class SimilarResponse(BaseModel):
    results: List[dict[str, Any]]


async def get_model() -> Embedder:
    """Dependency to inject model into endpoints"""

    try:
        return ModelManager.get_embedder()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# API Endpoint: accept single JSON body matching SimilarRequest
@router.post("/", response_model=SimilarResponse)
async def similar(
    embedder: Annotated[Embedder, Depends(get_model)],
    payload: SimilarRequest,
    api_key: str = Depends(verify_api_key),
):
    """Accept a text string and return similar frames based on embeddings.

    The request body should be JSON: { "string": "...", "frames": [ {embedding: [...], ...}, ... ] }
    """

    try:
        query_embedding_result = await embedder.embed_text(payload.string)
        query_vector = query_embedding_result.vector

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    try:
        similarity = Similarity()

        # determine top_k from payload.top_percent (relative). If not provided,
        # default to 10% of frames.
        num_frames = len(payload.frames or [])
        p = float(payload.top_percent) if payload.top_percent is not None else 0.1
        p = max(0.0, min(1.0, p))
        # round to nearest integer, but ensure at least 1 result when frames exist
        top_k_val = max(1, int(round(num_frames * p))) if num_frames > 0 else 0

        results = similarity.search(
            query_vector=np.array(query_vector, dtype=np.float32),
            frames=payload.frames,
            alpha=0.85,
            top_k=top_k_val,
        )

        # Option A (recommended): take the top_k results, then apply a minimum threshold.
        # If `threshold_raw` is provided, filter by raw cosine (range -1..1) first.
        if payload.threshold_raw is not None:
            try:
                thr_raw = float(payload.threshold_raw)
            except Exception:
                thr_raw = -1.0
            thr_raw = max(-1.0, min(1.0, thr_raw))

            filtered = [r for r in results if r.get("raw_cosine", 0.0) >= thr_raw]

            # If filtering removes everything, return the single best match (if available)
            if not filtered and results:
                filtered = [results[0]]

            results = filtered
        # probability-based threshold removed; we only support `threshold_raw` now

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")

    # debug logging removed

    return SimilarResponse(results=results)


