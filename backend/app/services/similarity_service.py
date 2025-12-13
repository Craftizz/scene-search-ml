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

            # adjusted score mixes the raw cosine with a complementary signal
            adjusted_score = alpha * vector_score + (1 - alpha) * (1 - vector_score)

            # Normalize adjusted_score to a probability in [0,1]. The theoretical
            # min and max of adjusted_score over vector_score in [-1,1] are:
            #   min = 2 - 3*alpha
            #   max = alpha
            # We map adjusted_score -> [0,1] via linear scaling and clamp.
            min_possible = 2.0 - 3.0 * alpha
            max_possible = float(alpha)
            denom = max_possible - min_possible if max_possible != min_possible else 1.0
            probability = float((adjusted_score - min_possible) / denom)
            probability = max(0.0, min(1.0, probability))

            # add result (ranking/thresholding will use `probability`)
            results.append({
                'frame': frame,
                'similarity': adjusted_score,
                'raw_cosine': vector_score,
                'probability': probability,
            })

        # Sort results by probability (higher = more similar)
        results.sort(key=lambda x: x.get('probability', 0.0), reverse=True)

        return results[:top_k]