"""Zone highlighting on image previews.

The zone map is computed on a small copy (256 px).  It is upsampled
smoothly to the displayed image, everything outside the zone is darkened,
and the zone is outlined.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from stealzalut.core.palette import upsample_mask

#: Outline colour (RGB), the app's accent blue.
OUTLINE_RGB = (79, 128, 255)
#: Brightness kept outside the highlighted zone.
DIM_FACTOR = 0.28


def highlight_zone(image_rgb: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Darken *image_rgb* outside *mask* (any size) and outline the zone.

    Args:
        image_rgb: Displayed image, RGB uint8 ``(H, W, 3)``.
        mask: Zone mask, boolean, any resolution with the same aspect ratio.

    Returns:
        A new RGB uint8 image.
    """
    h, w = image_rgb.shape[:2]
    full = upsample_mask(mask, w, h)
    out = image_rgb.copy()
    out[~full] = (out[~full].astype(np.float32) * DIM_FACTOR).astype(np.uint8)
    contours, _ = cv2.findContours(
        full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    thickness = max(1, round(min(h, w) / 220))
    cv2.drawContours(out, contours, -1, OUTLINE_RGB, thickness, lineType=cv2.LINE_AA)
    return out
