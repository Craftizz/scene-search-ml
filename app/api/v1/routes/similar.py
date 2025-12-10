from dataclasses import dataclass
from typing import Annotated, Any
 
from fastapi import APIRouter, HTTPException, Depends
import numpy as np

from app.manager.model_manager import ModelManager
from app.services.embedder_service import Embedder
from app.services.similarity_service import Similarity


router = APIRouter(prefix="/v1/similar", tags=["similar"])


@dataclass
class SimilarResponse():
    results: list[dict[str, Any]]
    # TODO


async def get_model() -> Embedder:
    """Dependency to inject model into endpoints"""

    try:
        return ModelManager.get_embedder()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    

# # API Endpoints
@router.post("/", response_model=SimilarResponse)
async def similar(
    embedder: Annotated[Embedder, Depends(get_model)],
    string: str,
    frames: list[dict[str, Any]]
):
    """Accept a text string and return similar frames based on embeddings."""

    try:
        query_embedding_result = await embedder.embed_text(string)
        query_vector = query_embedding_result.vector

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    try:
        similarity = Similarity()
        results = similarity.search(
            query_vector=np.array(query_vector, dtype=np.float32),
            frames=frames,
            alpha=0.85,
            top_k=50
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")


    return SimilarResponse(results=results)
    
    
