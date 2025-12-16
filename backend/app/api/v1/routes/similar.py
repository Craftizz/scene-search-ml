from typing import Annotated, Any, List, Optional
import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import numpy as np

from app.manager.model_manager import ModelManager
from app.security.auth import verify_api_key
from app.services.embedder_service import Embedder
from app.services.similarity_service import Similarity


router = APIRouter(prefix="/v1/similar", tags=["similar"])
logger = logging.getLogger("scene_search")


class SimilarRequest(BaseModel):
    string: str
    frames: List[dict[str, Any]]
    min_similarity: Optional[float] = 0.030
    top_k: Optional[int] = 50


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

    logger.info(f"[Similar] Query: '{payload.string[:50]}...' with {len(payload.frames)} frames")

    try:
        query_embedding_result = await embedder.embed_text(payload.string)
        query_vector = query_embedding_result.vector
        query_arr = np.array(query_vector, dtype=np.float32)
        logger.info(f"[Similar] Query embedding dim: {len(query_vector)}")
        logger.info(f"[Similar] Query embedding norm: {np.linalg.norm(query_arr):.4f}")
        logger.info(f"[Similar] Query embedding sample: {query_arr[:5]}")
        
        # Check first frame embedding
        if payload.frames and 'embedding' in payload.frames[0]:
            frame_emb = np.array(payload.frames[0]['embedding'], dtype=np.float32)
            logger.info(f"[Similar] Frame[0] embedding dim: {len(payload.frames[0]['embedding'])}")
            logger.info(f"[Similar] Frame[0] embedding norm: {np.linalg.norm(frame_emb):.4f}")
            logger.info(f"[Similar] Frame[0] embedding sample: {frame_emb[:5]}")

    except Exception as e:
        logger.error(f"[Similar] Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    try:
        similarity = Similarity()

        # Search with configurable threshold and top_k
        min_sim = payload.min_similarity if payload.min_similarity is not None else 0.60
        k = payload.top_k if payload.top_k is not None else 50
        
        results = similarity.search(
            query_vector=query_arr,
            frames=payload.frames,
            min_similarity=min_sim,
            top_k=k)
        logger.info(f"[Similar] Found {len(results)} results (min_similarity={min_sim:.2f}, top_k={k})")
        if results:
            logger.info(f"[Similar] Top result similarity: {results[0].get('similarity', 0):.4f}")
            logger.info(f"[Similar] Bottom result similarity: {results[-1].get('similarity', 0):.4f}")

    except Exception as e:
        logger.error(f"[Similar] Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")

    # debug logging removed

    return SimilarResponse(results=results)


