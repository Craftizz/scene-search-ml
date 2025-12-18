
import asyncio
from dataclasses import dataclass
import logging
from typing import Any, List, Optional

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

logger = logging.getLogger("scene_search")


@dataclass
class EmbedderConfig:
    model_name: str = "google/siglip-base-patch16-224"
    device: str | None = None    
    batch_size: int = 8


@dataclass
class EmbedImageResult:
    vector: list[float]


@dataclass
class EmbedTextResult:
    vector: list[float]

from app.models import EmbeddingSmoothingConfig


class Embedder:

    def __init__(self, config: EmbedderConfig) -> None:
        self.config: EmbedderConfig = config

        if config.device:
            self.device: torch.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._lock: asyncio.Lock = asyncio.Lock()
        
        try:
            # Prefer a locally downloaded model snapshot if provided (built into image)
            local_dir = os.environ.get("SIGLIP_LOCAL_DIR") or None
            
            # Verify local directory exists and has required files
            use_local = False
            if local_dir:
                if os.path.isdir(local_dir):
                    if os.path.exists(os.path.join(local_dir, "config.json")):
                        use_local = True
                        logger.info(f"Using local SigLIP from: {local_dir}")
                    else:
                        logger.warning(f"Local dir {local_dir} exists but missing config.json")
                else:
                    logger.warning(f"SIGLIP_LOCAL_DIR set but path doesn't exist: {local_dir}")
            
            if use_local:
                self.processor: Any = AutoProcessor.from_pretrained(
                    local_dir, use_fast=True, local_files_only=True, trust_remote_code=True
                )
                self.model: Any = AutoModel.from_pretrained(
                    local_dir, local_files_only=True, trust_remote_code=True
                )
            else:
                # Fall back to HF hub (online) if no local snapshot is present
                logger.info(f"Loading SigLIP from HuggingFace: {config.model_name}")
                self.processor: Any = AutoProcessor.from_pretrained(config.model_name, use_fast=True, trust_remote_code=True)
                self.model: Any = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SigLIP model '{config.model_name}': {e}\n"
                "Ensure the model name is correct and dependencies are installed."
            )
        
        self.model.to(self.device)  # type: ignore
        self.model.eval()


    async def embed_images(self, images: List[Image.Image]) -> List[EmbedImageResult]:
        """Generate embeddings for a list of PIL Images.

        Processes the list in chunks of `batch_size` and returns a list of
        `EmbedImageResult` in the same order as the input images.
        """

        if not isinstance(images, list):
            raise TypeError("`images` must be a list of PIL.Image instances")

        if len(images) == 0:
            return []


        # validate inputs
        for i, im in enumerate(images):
            if not isinstance(im, Image.Image):
                raise TypeError(f"images[{i}] is not a PIL.Image instance")

        results: List[EmbedImageResult] = []

        for i in range(0, len(images), self.config.batch_size):
            batch = images[i : i + self.config.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")  # type: ignore
            
            # Move ALL inputs to device at once
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            async with self._lock:
                with torch.no_grad():
                    # Use **inputs to unpack all processor outputs
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # convert each vector in the batch to a python list
            batch_numpy = image_features.detach().cpu().numpy()
            for idx, vec in enumerate(batch_numpy):
                results.append(EmbedImageResult(vector=vec.tolist()))
            
            # Validation logging for first batch only
            if i == 0 and len(batch_numpy) > 0:
                import numpy as np
                norm = np.linalg.norm(batch_numpy[0])
                logger.info(f"[Embedder] Image embedding dim: {len(batch_numpy[0])}, norm: {norm:.6f}")

        return results
    

    async def embed_text(self, text: str) -> EmbedTextResult:
        """Generate an embedding for a single text string.

        Uses the Hugging Face SigLIP processor and model to generate embeddings.
        """

        # CRITICAL: SigLIP was trained with padding="max_length"
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length")  # type: ignore
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        async with self._lock:
            with torch.no_grad():
                # SigLIP get_text_features returns properly projected embeddings
                text_embeddings = self.model.get_text_features(**inputs)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # Use indexing instead of squeeze to ensure shape [embedding_dim]
        embedding = text_embeddings[0].cpu().numpy()
        
        # Validation logging
        import numpy as np
        norm = np.linalg.norm(embedding)
        logger.info(f"[Embedder] Text embedding dim: {len(embedding)}, norm: {norm:.6f}")
        
        return EmbedTextResult(vector=embedding.tolist())


class EmbeddingSmoother:
    """Apply exponential moving average smoothing to embeddings."""
    
    def __init__(self, config: EmbeddingSmoothingConfig):
        """Initialize smoother.
        
        Args:
            config: Smoothing configuration
        """
        self.config = config
        self._ema: Optional[Any] = None  # numpy array
    
    def smooth_embeddings(self, embeddings: List[Any]) -> List[Any]:
        """Apply EMA smoothing to embeddings.
        
        Args:
            embeddings: List of numpy arrays
            
        Returns:
            List of smoothed numpy arrays
        """
        if self.config.window_size <= 1:
            return embeddings
            
        import numpy as np
        alpha = 2.0 / (self.config.window_size + 1)
        smoothed = []
        
        for emb in embeddings:
            emb_array = np.array(emb, dtype=float)
            if self._ema is None:
                self._ema = emb_array.copy()
            else:
                self._ema = alpha * emb_array + (1.0 - alpha) * self._ema
            smoothed.append(self._ema.copy())
            
        return smoothed
    
    def reset(self) -> None:
        """Reset the smoothing state."""
        self._ema = None
