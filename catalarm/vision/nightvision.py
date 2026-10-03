import cv2
import numpy as np


def is_night_vision(image_bgr: np.ndarray, tolerance: int = 6, color_ratio: float = 0.1) -> bool:
    """True when the frame is (near) grayscale, i.e. the camera runs in IR night mode.

    Ported from ``DetectNigthVision``: counts pixels whose channels differ. Works on a
    small thumbnail and tolerates compression noise between channels.
    """
    if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
        return True
    small = cv2.resize(image_bgr, (80, 45), interpolation=cv2.INTER_AREA).astype(np.int16)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    diff = (
        np.count_nonzero(np.abs(r - g) > tolerance)
        + np.count_nonzero(np.abs(r - b) > tolerance)
        + np.count_nonzero(np.abs(g - b) > tolerance)
    )
    return bool(diff / small.size <= color_ratio)
