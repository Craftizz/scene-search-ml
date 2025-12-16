from PIL import Image
from typing import Optional

from app.core.config import settings


def validate_image_dimensions(img: Image.Image) -> None:
    """Validate `img` dimensions against configured limits in `settings`.

    Raises:
        ValueError: if the image exceeds configured width/height/pixel limits.
    """
    try:
        w, h = img.size
    except Exception as e:
        raise ValueError("unable to determine image size") from e

    max_w: Optional[int] = getattr(settings, "max_image_width", None)
    max_h: Optional[int] = getattr(settings, "max_image_height", None)
    max_pixels: Optional[int] = getattr(settings, "max_image_pixels", None)

    if (max_w is not None and w > max_w) or (
        max_h is not None and h > max_h
    ) or (max_pixels is not None and (w * h) > max_pixels):
        raise ValueError("image dimensions exceed allowed limits")
