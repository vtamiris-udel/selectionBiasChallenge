"""decAnalytics selection-bias challenge — block letter generator

Utilities to create a centered block-letter mask sized to a target image.

The primary function `create_block_letter_s` returns a 2D NumPy array with
values normalized to the [0, 1] range where 0.0 represents the drawn
letter (black) and 1.0 the background (white). This is useful for creating
binary masks or synthetic overlays that match an image's dimensions.

Example
-------
>>> mask = create_block_letter_s(200, 300, letter="A")
>>> mask.shape
(200, 300)

The module relies on Pillow for drawing text; if a TrueType font matching
the preferred names cannot be found, the code falls back to Pillow's
default bitmap font.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font at the requested size from common locations.

    The function probes a list of typical font filenames and platform
    locations (Linux, macOS, common user installs) and returns the first
    usable TTF font. If no suitable file is found, the Pillow default font
    is returned.

    Parameters
    ----------
    size : int
        Requested font size in pixels. The caller is responsible for ensuring
        ``size`` is positive; this function will pass the value directly to
        Pillow's ``truetype`` loader.

    Returns
    -------
    ImageFont.ImageFont
        A Pillow font instance sized approximately to ``size``. On fallback
        the returned object may be a small bitmap font provided by Pillow.
    """
    font_candidates = [
        "DejaVuSans-Bold.ttf",  # Pillow ships this on most installs
        "Arial Bold.ttf",
        "Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for font_path in font_candidates:
        try:
            candidate = Path(font_path)
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9
) -> np.ndarray:
    """
    Create a block letter mask sized to the target image dimensions.
    
    Parameters
    ----------
    height : int
        Height of the output array.
    width : int
        Width of the output array.
    letter : str
        Letter to draw (default "S").
    font_size_ratio : float
        Fraction of the smallest image dimension used to size the font.
        Must be in (0.0, 1.0].
    
    Returns
    -------
    np.ndarray
        2D array (height × width) with values in [0, 1]:
        - 0.0 where the letter is drawn (black)
        - 1.0 for the background (white)
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    if not letter:
        raise ValueError("letter must be a non-empty string")
    if font_size_ratio <= 0.0 or font_size_ratio > 1.0:
        raise ValueError("font_size_ratio must be between 0.0 and 1.0")
    
    # Create a white canvas
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    
    # Size the font relative to the smaller dimension
    font_size = max(1, int(min(height, width) * font_size_ratio))
    font = _load_font(font_size)
    
    # Resize downward until the text fits comfortably within the target area
    max_width = max(1, int(width * font_size_ratio))
    max_height = max(1, int(height * font_size_ratio))
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    while (text_w > max_width or text_h > max_height) and font_size > 1:
        font_size = max(1, int(font_size * 0.9))
        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Center the letter
    x = (width - text_w) // 2 - bbox[0]
    y = (height - text_h) // 2 - bbox[1]
    draw.text((x, y), letter, fill=0, font=font)
    
    # Convert to normalized numpy array in [0, 1]
    return np.array(canvas, dtype=np.float32) / 255.0
