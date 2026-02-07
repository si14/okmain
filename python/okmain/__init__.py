from __future__ import annotations

from PIL import Image

from okmain._core import dominant_color_from_rgb_bytes

__all__ = ["dominant_color"]


def dominant_color(image: Image.Image) -> tuple[int, int, int]:
    """Extract the dominant color from a PIL Image.

    The image must be in RGB mode. RGBA, grayscale, and other modes
    are rejected with a ``ValueError``.
    """
    if image.mode != "RGB":
        raise ValueError(f"expected RGB image, got {image.mode}")
    rgb_bytes = image.tobytes()
    return dominant_color_from_rgb_bytes(rgb_bytes)
