
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import asyncio
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


@dataclass
class CaptionerConfig:

    processor_name: str = "Salesforce/blip-image-captioning-base"
    model_name: str = "Salesforce/blip-image-captioning-base"

    device: Optional[str] = None
    use_fp16: bool = False
    batch_size: int = 4
    max_length: int = 50


@dataclass
class CaptionResult:
    text: str


class Captioner:


    def __init__(self, config: CaptionerConfig):
        self.config = config

        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For thread/async-safety when using shared model
        self._lock = asyncio.Lock()
        self.processor = BlipProcessor.from_pretrained(config.processor_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(config.model_name)

        self.model.to(self.device)  # type: ignore
        if config.use_fp16 and str(self.device).startswith("cuda"):
            try:
                self.model.half()
            except Exception:
                pass

        self.model.eval()


    async def caption(self, images: List[Image.Image]) -> List[CaptionResult]:
        """Generate captions for a list of PIL Images.

        This method requires `images` to be a list (even for a single image).
        It processes images in batches according to `config.batch_size`, runs
        `model.generate()` under the shared asyncio lock to avoid concurrent
        GPU usage, and returns a list of `CaptionResult` preserving input order.
        """

        if not isinstance(images, list):
            raise TypeError("`images` must be a list of PIL.Image instances")

        if len(images) == 0:
            return []

        # validate items
        for idx, im in enumerate(images):
            if not isinstance(im, Image.Image):
                raise TypeError(f"images[{idx}] is not a PIL.Image instance")

        results: List[CaptionResult] = []

        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i : i + self.config.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")  # type: ignore[arg-type]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # generate under lock to avoid concurrent GPU usage issues
            async with self._lock:
                with torch.no_grad():
                    out_ids = self.model.generate(**inputs, max_length=self.config.max_length)  # type: ignore

            captions = self.processor.batch_decode(out_ids, skip_special_tokens=True)
            results.extend(CaptionResult(text=t) for t in captions)

        return results