
import asyncio
from dataclasses import dataclass
import logging

import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

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


class Embedder:

    def __init__(self, config: EmbedderConfig):
        self.config = config

        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._lock = asyncio.Lock()
        
        try:
            self.processor = SiglipProcessor.from_pretrained(config.model_name)
            self.model = SiglipModel.from_pretrained(config.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SigLIP model '{config.model_name}': {e}\n"
                "Ensure the model name is correct and dependencies are installed."
            )
        
        self.model.to(self.device)  # type: ignore
        self.model.eval()


    async def embed_images(self, images: list[Image.Image]) -> list[EmbedImageResult]:
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

        results: list[EmbedImageResult] = []

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
