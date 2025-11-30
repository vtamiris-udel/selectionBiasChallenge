"""
Create a four-panel statistics meme illustrating selection bias.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def _prepare_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Normalize to [0,1], convert to grayscale, and resize to target_size."""
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise ValueError("Images must be 2D or convertible to grayscale")
    pil_img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    return np.array(pil_img, dtype=np.float32) / 255.0


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white"
) -> None:
    """
    Assemble a 1x4 panel meme and save to disk.
    
    Parameters
    ----------
    original_img : np.ndarray
        Prepared grayscale image (reality panel).
    stipple_img : np.ndarray
        Stippled image (model panel).
    block_letter_img : np.ndarray
        Block letter mask (selection bias panel).
    masked_stipple_img : np.ndarray
        Stippled image with mask applied (estimate panel).
    output_path : str
        Where to save the PNG.
    dpi : int
        Output DPI; higher = sharper.
    background_color : str
        Figure background color.
    """
    # Use the original image size as the canonical panel size
    if original_img.ndim == 3:
        base_arr = np.mean(original_img, axis=2)
    else:
        base_arr = np.asarray(original_img, dtype=np.float32)
    if base_arr.ndim != 2:
        raise ValueError("original_img must be 2D or convertible to grayscale")
    base_arr = np.clip(base_arr, 0.0, 1.0)
    base_h, base_w = base_arr.shape
    
    panels = [
        ("Reality", base_arr),
        ("Your Model", _prepare_image(stipple_img, (base_h, base_w))),
        ("Selection Bias", _prepare_image(block_letter_img, (base_h, base_w))),
        ("Estimate", _prepare_image(masked_stipple_img, (base_h, base_w))),
    ]
    
    fig_w = 16
    fig_h = 4.5
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), constrained_layout=False)
    fig.patch.set_facecolor(background_color)
    
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")
        ax.set_facecolor(background_color)
    
    plt.tight_layout(pad=2.0)
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=background_color)
    plt.close(fig)
