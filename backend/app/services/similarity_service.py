from typing import Any, List, Dict
import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray


class Similarity:
    """Similarity utilities for computing cosine similarity and searching.

    Methods accept and return plain Python types (lists/dicts) for easy
    serialization across API boundaries.
    """

    def cosine_similarity(self, a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        """Compute cosine similarity between two 1-D numpy arrays.

        Args:
            a: 1-D array-like numeric vector.
            b: 1-D array-like numeric vector.

        Returns:
            Cosine similarity as a float in [-1.0, 1.0].
        """

        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        a_norm = norm(a_arr)
        b_norm = norm(b_arr)
        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(np.dot(a_arr, b_arr) / (a_norm * b_norm))


    def search(
        self,
        query_vector: NDArray[np.floating],
        frames: List[Dict[str, Any]],
        top_k: int = 50,
        min_similarity: float = 0.60,
    ) -> List[Dict[str, Any]]:
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