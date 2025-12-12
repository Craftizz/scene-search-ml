from typing import Any
import numpy as np
from numpy.linalg import norm


class Similarity:


    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (norm(a) * norm(b)))
    

    def search(self,
               query_vector: np.ndarray,
               frames: list[dict[str, Any]],
               alpha: float = 0.85,
               top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Search frames based on cosine similarity to the query vector.

        Args:
            query_vector: The embedding vector for the query image.
            frames: A list of frames, each containing an 'embedding' key with its vector.
            alpha: Weight for CLIP similarity (semantic)

        Returns:
            A list of top_k frames sorted by similarity score.
        """
        results = []

        for frame in frames:
            frame_vector = np.array(frame['embedding'], dtype=np.float32)
            vector_score = self.cosine_similarity(query_vector, frame_vector)


            adjusted_score = alpha * vector_score + (1 - alpha) * (1 - vector_score)
            results.append({
                'frame': frame,
                'similarity': adjusted_score
            })

        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:top_k]