"""
utils/processor.py
Preprocessing pipeline for the AirDefenseML inference engine.
Converts raw image bytes → normalised NumPy batch tensor (1, H, W, 3).
"""

import io
import numpy as np
from PIL import Image, ImageOps

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE   = (224, 224)   # Width × Height expected by the model
NORM_SCALE    = 1.0 / 255.0  # Normalise pixel values to [0, 1]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Load raw image bytes and return a (1, 224, 224, 3) float32 array.

    Steps
    -----
    1. Decode bytes → PIL Image
    2. Convert to RGB  (handles RGBA, grayscale, palette images)
    3. Resize to 224×224 with high-quality Lanczos resampling
    4. Scale pixel values from [0, 255] → [0.0, 1.0]
    5. Add batch dimension: (H, W, 3) → (1, H, W, 3)

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image file.

    Returns
    -------
    np.ndarray
        Batch-ready float32 tensor of shape (1, 224, 224, 3).

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as a valid image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Unable to decode image: {exc}") from exc

    # Ensure 3-channel RGB regardless of source format
    img = img.convert("RGB")

    # Auto-rotate based on EXIF orientation (fixes mobile camera photos)
    img = ImageOps.exif_transpose(img)

    # Resize preserving quality
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # → float32 array (H, W, 3), keep in [0, 255] range
    # EfficientNetB3 has an internal normalization layer — do NOT rescale here
    arr = np.array(img, dtype=np.float32)

    # Add batch dimension
    return np.expand_dims(arr, axis=0)
