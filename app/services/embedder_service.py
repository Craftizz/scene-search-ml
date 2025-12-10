
import asyncio
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


@dataclass
class EmbedderConfig:

    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    
    batch_size: int = 8


@dataclass
class EmbedImageResult:
    vector: list[float]


@dataclass
class EmbedTextResult:
    vector: np.ndarray


class Embedder:

    def __init__(self, config: EmbedderConfig):
        self.config = config

        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._lock = asyncio.Lock()
        self.processor = CLIPProcessor.from_pretrained(config.model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(config.model_name)
        
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            async with self._lock:
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize per-row
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # convert each vector in the batch to a python list
            for vec in image_features.detach().cpu().numpy():
                results.append(EmbedImageResult(vector=vec.tolist()))

        return results
    

    async def embed_text(self, text: str) -> EmbedTextResult:
        """Generate an embedding for a single text string.

        Uses the Hugging Face CLIP processor and model to generate embeddings.
        """

        inputs = self.processor(text=[text], return_tensors="pt", padding=True)  # type: ignore
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        async with self._lock:
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embedding = text_features.squeeze().cpu().numpy()
        return EmbedTextResult(vector=embedding)
