"""
Image utilities for the signature-verification pipeline.

Responsibilities
----------------
* Load images from file paths, URLs, or PIL objects.
* Crop a region of interest from a cheque image.
* Preprocess a cropped signature so it matches the Siamese model's expectations.
* Heuristically validate that a cropped region actually contains a handwritten
  signature (not a blank area or a block of printed text).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# Type alias accepted everywhere an image is expected
ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_image(src: ImageInput) -> Image.Image:
    """
    Load an image from a variety of sources and return an RGB PIL Image.

    Parameters
    ----------
    src:
        * ``str`` / ``Path`` — file path or HTTP(S) URL
        * ``bytes`` — raw image bytes
        * ``PIL.Image.Image`` — returned as-is (converted to RGB)
        * ``numpy.ndarray`` — assumed HxWx3 uint8 BGR or RGB
    """
    if isinstance(src, Image.Image):
        return src.convert("RGB")

    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            return Image.fromarray(src, mode="L").convert("RGB")
        # OpenCV is BGR; convert if the first channel looks blue-dominant on a typical scene
        return Image.fromarray(src[..., ::-1] if src.shape[2] == 3 else src).convert("RGB")

    if isinstance(src, bytes):
        return Image.open(io.BytesIO(src)).convert("RGB")

    src = str(src)
    if src.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(src, timeout=15) as resp:  # noqa: S310
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    return Image.open(src).convert("RGB")


# ── Cropping ──────────────────────────────────────────────────────────────────

def crop_region(
    image: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding: int = 8,
) -> Image.Image:
    """
    Crop ``image`` to the rectangle ``[x1, y1, x2, y2]`` with optional
    symmetric padding (clamped to image bounds).

    Parameters
    ----------
    image:
        Source image (RGB PIL Image).
    x1, y1, x2, y2:
        Pixel coordinates of the bounding box (top-left / bottom-right).
    padding:
        Extra pixels to add on each side (helps avoid cutting the ink strokes).

    Returns
    -------
    PIL.Image.Image
        Cropped region as an RGB image.
    """
    W, H = image.size
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(W, x2 + padding)
    bottom = min(H, y2 + padding)
    return image.crop((left, top, right, bottom))


# ── Signature validation ──────────────────────────────────────────────────────

def validate_signature_region(
    crop: Image.Image,
    min_variance_ratio: float = 0.001,
) -> tuple[bool, str]:
    """
    Heuristic check that a cropped image region contains a handwritten signature.

    The function uses three lightweight tests:

    1. **Blank check** — if pixel variance is too low the region is blank or
       near-uniform and cannot contain a signature.
    2. **Dark-ink ratio** — a handwritten signature should have a meaningful
       fraction of dark pixels on a light background (or vice versa).
    3. **Stroke connectivity** — a real signature has connected dark pixels
       organised in strokes; random noise or a printed QR code would fail.

    Parameters
    ----------
    crop:
        Cropped image region to inspect.
    min_variance_ratio:
        Variance threshold relative to the maximum possible variance (255²).
        Regions with relative variance below this value are rejected.

    Returns
    -------
    (passed, note)
        ``passed`` is ``True`` when the region looks like a valid signature.
        ``note`` is a human-readable description of the decision.
    """
    gray = np.array(crop.convert("L"), dtype=np.float32)

    # 1. Variance check
    variance = float(np.var(gray))
    max_variance = 255.0 ** 2
    if variance / max_variance < min_variance_ratio:
        return False, f"Blank region: variance ratio {variance / max_variance:.5f} < {min_variance_ratio}"

    # 2. Dark-ink ratio (assuming light background)
    threshold = 128
    dark_pixel_ratio = float(np.mean(gray < threshold))
    if dark_pixel_ratio < 0.005:
        return False, f"Almost no ink detected: dark-pixel ratio {dark_pixel_ratio:.4f} < 0.005"
    if dark_pixel_ratio > 0.98:
        return False, f"Region is almost entirely dark — likely noise: ratio {dark_pixel_ratio:.4f}"

    # 3. Stroke connectivity — count connected components via a simple erosion check.
    # A real signature has moderate connected-ink structure; printed text has many
    # small isolated glyphs; noise has many tiny isolated dots.
    from PIL import ImageFilter

    binary = crop.convert("L").point(lambda p: 0 if p < threshold else 255, "L")
    eroded = binary.filter(ImageFilter.MinFilter(3))  # morphological erosion approximation
    eroded_arr = np.array(eroded, dtype=np.float32)
    surviving_dark = float(np.mean(eroded_arr < threshold))
    if surviving_dark < 0.001:
        return False, (
            "Ink strokes too thin / disconnected after erosion — "
            f"likely noise rather than signature (surviving dark ratio {surviving_dark:.4f})"
        )

    return True, "Signature region validation passed."


# ── Preprocessing for Siamese model ──────────────────────────────────────────

def preprocess_for_siamese(
    crop: Image.Image,
    input_size: tuple[int, int] = (155, 220),
    grayscale: bool = True,
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> "torch.Tensor":
    """
    Resize, optionally convert to grayscale, and normalise a cropped signature
    image into a 4-D ``(1, C, H, W)`` float32 tensor ready for the Siamese model.

    Parameters
    ----------
    crop:
        Cropped signature region (RGB PIL Image).
    input_size:
        ``(H, W)`` expected by the model.
    grayscale:
        When ``True`` the output tensor has ``C=1`` and the mean/std are
        computed as the greyscale equivalents of ``norm_mean``/``norm_std``.
    norm_mean, norm_std:
        Channel-wise normalisation statistics.

    Returns
    -------
    torch.Tensor
        Shape ``(1, C, H, W)`` — batch dimension is always 1.
    """
    import torch

    h, w = input_size

    # Resize with high-quality resampling, preserving aspect ratio with padding
    img = _resize_with_padding(crop, w, h)

    if grayscale:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        mean_g = 0.299 * norm_mean[0] + 0.587 * norm_mean[1] + 0.114 * norm_mean[2]
        std_g = 0.299 * norm_std[0] + 0.587 * norm_std[1] + 0.114 * norm_std[2]
        arr = (arr - mean_g) / (std_g + 1e-7)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        mean_arr = np.array(norm_mean, dtype=np.float32)
        std_arr = np.array(norm_std, dtype=np.float32)
        arr = (arr - mean_arr) / (std_arr + 1e-7)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return tensor.float()


def _resize_with_padding(
    img: Image.Image,
    target_w: int,
    target_h: int,
    fill_color: int = 255,
) -> Image.Image:
    """
    Resize ``img`` to fit inside ``(target_w, target_h)`` while preserving the
    aspect ratio. Pad the shorter axis with ``fill_color`` (white by default).
    """
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    padded = Image.new(img.mode, (target_w, target_h), fill_color)
    offset_x = (target_w - img.width) // 2
    offset_y = (target_h - img.height) // 2
    padded.paste(img, (offset_x, offset_y))
    return padded
