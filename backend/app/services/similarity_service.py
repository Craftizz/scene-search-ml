from typing import Any
import numpy as np
from numpy.linalg import norm


class Similarity:


    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (norm(a) * norm(b)))
    

    def search(self,
               query_vector: np.ndarray,
               frames: list[dict[str, Any]],
               top_k: int = 50,
               min_similarity: float = 0.60,
    ) -> list[dict[str, Any]]:
        """Search frames based on cosine similarity to the query vector.

        Args:
            query_vector: The embedding vector for the query image.
            frames: A list of frames, each containing an 'embedding' key with its vector.
            top_k: Maximum number of results to return.
            min_similarity: Minimum similarity threshold (0.0 to 1.0). Only frames 
                          with similarity >= this value will be returned.

        Returns:
            A list of frames sorted by similarity score (cosine similarity), filtered
            by min_similarity and limited to top_k results.
        """

        query_vector = np.array(query_vector, dtype=np.float32)
        
        # Defensive normalization to ensure unit vectors
        query_norm = norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
    
        results = []
        for frame in frames:
            frame_vector = np.array(frame['embedding'], dtype=np.float32)
            
            # Defensive normalization for frame vector
            frame_norm = norm(frame_vector)
            if frame_norm > 0:
                frame_vector = frame_vector / frame_norm
            
            # Dot product of normalized vectors = cosine similarity
            similarity_score = float(np.dot(query_vector, frame_vector))
            
            results.append({
                'frame': frame,
                'similarity': similarity_score,
                'raw_cosine': similarity_score,
                'probability': max(0.0, min(1.0, similarity_score)),
            })
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Filter by minimum similarity threshold
        filtered_results = [r for r in results if r['similarity'] >= min_similarity]
        
        # Return top_k results
        return filtered_results[:top_k]