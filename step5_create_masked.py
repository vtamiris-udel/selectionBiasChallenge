"""
Step 5: Apply the block letter mask to the stippled image to simulate selection bias.
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply a block-letter mask to a stippled image.
    
    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image (0.0 = black dots, 1.0 = white background).
    mask_img : np.ndarray
        Mask image in [0, 1]; darker values indicate regions to remove.
    threshold : float
        Pixels in the mask below this value are treated as masked-out areas.
    
    Returns
    -------
    np.ndarray
        Masked stippled image with the same shape as the inputs.
    """
    stipple = np.asarray(stipple_img, dtype=np.float32)
    mask = np.asarray(mask_img, dtype=np.float32)
    
    if stipple.shape != mask.shape:
        raise ValueError("stipple_img and mask_img must have the same shape")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be within [0, 1]")
    
    masked = stipple.copy()
    
    # Where the mask is dark, remove stipples by setting to white
    masked[mask < threshold] = 1.0
    
    return np.clip(masked, 0.0, 1.0)
