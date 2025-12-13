
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import asyncio
import torch
import logging
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger("scene_search")


@dataclass
class CaptionerConfig:
    # Use Florence-2-base by default for richer, less-hallucinating captions
    processor_name: str = "microsoft/Florence-2-base"
    model_name: str = "microsoft/Florence-2-base"

    device: Optional[str] = None
    use_fp16: bool = False
    batch_size: int = 4
    max_new_tokens: int = 1024



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

        # Load model with proper dtype
        torch_dtype = torch.float16 if config.use_fp16 and self.device.type == "cuda" else torch.float32
        
        try:
            # Load model configuration first to set attention implementation
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
            # Disable SDPA to avoid compatibility issues
            model_config._attn_implementation = "eager"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name, 
                config=model_config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            self.processor = AutoProcessor.from_pretrained(
                config.processor_name, 
                trust_remote_code=True
            )
            
            # Move to device if not using device_map
            if self.device.type != "cuda":
                self.model.to(self.device)  # type: ignore
                
            logger.info("Florence2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Florence2 model: {e}")
            raise


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
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)

        return results


    async def _process_batch(self, images: List[Image.Image]) -> List[CaptionResult]:
        """Process a batch of images for captioning."""
        
        try:
            # generate under lock to avoid concurrent GPU usage issues
            async with self._lock:

                captions = []
                for idx, image in enumerate(images):
                    try:
                        caption = await self._generate_single_caption(image)
                        captions.append(caption)
                    except Exception as e:
                        logger.warning(f"Failed to generate caption for image {idx}: {e}")
                        captions.append("An error occurred")

            return [CaptionResult(text=caption) for caption in captions]
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [CaptionResult(text="An error occurred") for _ in images]


    async def _generate_single_caption(self, image: Image.Image) -> str:
        """Generate caption for a single image."""
        
        # Florence2 prompt for captioning
        prompt = "<CAPTION>"
        
        try:

            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate caption - Florence2 specific approach
            with torch.no_grad():
                # Use the model's generate method without problematic parameters
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    use_cache=False
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=prompt, 
                image_size=(image.width, image.height)
            )
            
            if isinstance(parsed_answer, dict) and prompt in parsed_answer:
                caption = parsed_answer[prompt]
                if caption and isinstance(caption, str):
                    return caption.strip()
            
            return self._extract_caption_fallback(generated_text, prompt)
            
        except Exception as e:
            logger.warning(f"Error in single caption generation: {e}")
            return "An error occurred"


    def _extract_caption_fallback(self, generated_text: str, prompt: str) -> str:
        """Fallback method to extract caption from generated text."""
        
        try:
            # Florence2 typically returns text in format: <prompt>caption</prompt>
            start_marker = prompt
            end_marker = "</s>"
            
            start_idx = generated_text.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = generated_text.find(end_marker, start_idx)
                if end_idx != -1:
                    caption = generated_text[start_idx:end_idx].strip()
                    if caption:
                        return caption
            
            # If parsing fails, return a cleaned version
            return generated_text.strip() or "Unable to generate caption"
            
        except Exception:
            return "Unable to generate caption"