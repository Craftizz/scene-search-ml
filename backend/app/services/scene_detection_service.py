from typing import List, Optional, Dict, Any
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def detect_boundaries(
    embeddings: List[np.ndarray],
    prev_embedding: Optional[np.ndarray] = None,
    threshold: float = 0.85,
    min_scene_length: int = 3,
) -> List[Dict[str, Any]]:
    """
    Detect scene boundaries from a stream of embeddings (1 fps).
    
    Args:
        embeddings: List of frame embeddings for current chunk
        prev_embedding: Embedding of the last accepted scene (for continuity)
        threshold: Similarity threshold below which a cut is detected (0-1)
        min_scene_length: Minimum frames between scene cuts (prevents flickering)

    Returns:
        List of scene boundaries with index and embedding
    """
    scenes: List[Dict[str, Any]] = []
    n = len(embeddings)
    # detection entry - silent in production
    if n == 0:
        return scenes
    
    # No blur detection: consider all frames valid
    valid_frames = [True] * n
    
    # Baseline embedding for the last accepted scene. Do NOT allow
    # immediate detection at the start of a chunk; require at least
    # `min_scene_length` frames before declaring the next cut.
    last_scene_embedding = prev_embedding if prev_embedding is not None else embeddings[0]
    last_scene_index = 0
    
    # If this is a new stream (no prev_embedding), add first frame
    if prev_embedding is None:
        scenes.append({
            "index": 0,
            "embedding": embeddings[0].copy()
        })
        last_scene_embedding = embeddings[0].copy()
        last_scene_index = 0
    
    # Scan for scene boundaries
    for i in range(n):
        # Skip invalid (blurry) frames
        if not valid_frames[i]:
            continue
        
        # Enforce minimum scene length. Allow detection at the start of a
        # chunk (i == 0) when a `prev_embedding` was provided so we can
        # detect true cuts that occur exactly on chunk boundaries, while
        # still preventing immediate cuts elsewhere.
        if i - last_scene_index < min_scene_length:
            if not (i == 0 and prev_embedding is not None):
                continue
        
        # Compare current frame to last accepted scene
        similarity = cosine_similarity(last_scene_embedding, embeddings[i])

        if similarity < threshold:
            # Potential scene cut detected
            # Verify it's stable (next few frames are also dissimilar to old scene)
            stable = is_stable_cut(embeddings, i, last_scene_embedding, threshold, valid_frames)
            if stable:
                scenes.append({
                    "index": i,
                    "embedding": embeddings[i].copy(),
                    "similarity": similarity
                })
                last_scene_embedding = embeddings[i].copy()
                last_scene_index = i
    
    return scenes


def is_stable_cut(
    embeddings: List[np.ndarray],
    cut_index: int,
    old_scene_embedding: np.ndarray,
    threshold: float,
    valid_frames: List[bool],
    lookforward: int = 2
) -> bool:
    """
    Verify a scene cut is stable by checking if subsequent frames
    are also dissimilar to the old scene.
    """
    n = len(embeddings)
    confirmed = 0
    checked = 0
    
    for offset in range(1, lookforward + 1):
        idx = cut_index + offset
        if idx >= n:
            break
        if not valid_frames[idx]:
            continue
        sim = cosine_similarity(old_scene_embedding, embeddings[idx])
        checked += 1
        if sim < threshold:
            confirmed += 1
    
    # If we couldn't check any frames, do NOT accept the cut. This avoids
    # accepting cuts at chunk ends where no lookahead context exists.
    if checked == 0:
        return False
    
    # Require at least 50% of lookahead frames to confirm
    ratio = confirmed / checked
    return ratio >= 0.5


def detect_boundaries_adaptive(
    embeddings: List[np.ndarray],
    prev_embedding: Optional[np.ndarray] = None,
    base_threshold: float = 0.85,
    min_scene_length: int = 3,
    adaptive_window: int = 10
) -> List[Dict[str, Any]]:
    """
    Advanced scene detection with adaptive thresholding.
    
    This version computes local statistics to adapt the threshold,
    reducing false positives from gradual changes (fades, lighting).
    """
    scenes: List[Dict[str, Any]] = []
    n = len(embeddings)
    if n == 0:
        return scenes
    
    # No blur detection: consider all frames valid
    valid_frames = [True] * n
    
    # Compute all pairwise similarities for adaptive thresholding
    similarities = []
    for i in range(n - 1):
        if valid_frames[i] and valid_frames[i + 1]:
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
    
    if not similarities:
        # No valid frame pairs
        if prev_embedding is None and n > 0:
            return [{"index": 0, "embedding": embeddings[0].copy()}]
        return []
    
    # Adaptive threshold: mean - 2*std (catches outliers)
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    adaptive_threshold = min(base_threshold, mean_sim - 2 * std_sim)
    
    # Initialize
    last_scene_embedding = prev_embedding if prev_embedding is not None else embeddings[0]
    last_scene_index = -min_scene_length
    
    if prev_embedding is None:
        scenes.append({
            "index": 0,
            "embedding": embeddings[0].copy(),
            "threshold": adaptive_threshold
        })
        last_scene_embedding = embeddings[0].copy()
        last_scene_index = 0
    
    # Detect boundaries
    for i in range(n):
        if not valid_frames[i]:
            continue
        
        if i - last_scene_index < min_scene_length:
            continue
        
        similarity = cosine_similarity(last_scene_embedding, embeddings[i])
        
        # Use adaptive threshold
        if similarity < adaptive_threshold:
            # Compute local context for this potential cut
            local_sims = []
            for j in range(max(0, i - adaptive_window), min(n - 1, i + adaptive_window)):
                if valid_frames[j] and valid_frames[j + 1]:
                    local_sims.append(cosine_similarity(embeddings[j], embeddings[j + 1]))
            
            if local_sims:
                local_mean = np.mean(local_sims)
                # Confirm if this is a significant drop from local baseline
                if similarity < local_mean - std_sim:
                    scenes.append({
                        "index": i,
                        "embedding": embeddings[i].copy(),
                        "similarity": similarity,
                        "threshold": adaptive_threshold,
                        "local_mean": local_mean
                    })
                    last_scene_embedding = embeddings[i].copy()
                    last_scene_index = i
    
    return scenes




# Utility function to merge scenes from chunked processing
def merge_scene_chunks(
    all_scenes: List[List[Dict[str, Any]]],
    chunk_size: int
) -> List[Dict[str, Any]]:
    """
    Merge scene detections from multiple chunks into a single timeline.
    
    Args:
        all_scenes: List of scene lists, one per chunk
        chunk_size: Number of frames per chunk
        
    Returns:
        Merged list with global indices
    """
    merged = []
    for chunk_idx, chunk_scenes in enumerate(all_scenes):
        offset = chunk_idx * chunk_size
        for scene in chunk_scenes:
            merged_scene = scene.copy()
            merged_scene["index"] = scene["index"] + offset
            merged.append(merged_scene)
    
    return merged