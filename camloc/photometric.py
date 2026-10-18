"""Image normalization modes for the feature front-end (pure numpy + opencv, no torch).

The keyframe maps are built from night images; in daylight the global shading reverses (lit stands and
lamps go dark, previously unlit surfaces go bright) and the strongest night landmarks (lamps, floodlights)
are off, so absolute-intensity matching collapses. These modes let the same preprocessing be applied when
BUILDING a map and when QUERYING against it, so ALIKED/LightGlue (and RoMa) see comparable images.

normalize(gray, mode) -> uint8 single-channel image of the same shape, values in [0, 255].

Modes:
  'none'         : identity (passthrough).
  'clahe'        : the pipeline default, CLAHE clipLimit 2.0, 8x8 tiles. Preserves current night behavior.
  'clahe_strong' : harsher local contrast (clipLimit 4.0, 4x4 tiles) for high-dynamic-range daylight.
  'gradient'     : a shading-REVERSAL-robust representation. The local gradient magnitude (Sobel) depends on
                   the structure edges, not on whether a surface is lit or shadowed, so it is (up to a small
                   tolerance) invariant to a global intensity inversion g -> 255 - g. Normalized to [0, 255].

Keep every mode returning uint8 single-channel so the existing extract path (mask, resize, ALIKED) is
unchanged.
"""
import cv2
import numpy as np

MODES = ('none', 'clahe', 'clahe_strong', 'gradient')

# module-level CLAHE objects (cv2.createCLAHE is stateless across apply() calls; reuse avoids re-alloc)
_CLAHE = {
    'clahe': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)),
    'clahe_strong': cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)),
}


def _as_gray_u8(gray):
    g = np.asarray(gray)
    if g.ndim != 2:
        raise ValueError(f'normalize expects a single-channel (H, W) image, got shape {g.shape}')
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def gradient(gray):
    """Sobel gradient magnitude, per-image min-max normalized to uint8 [0, 255].

    Because Sobel is a (signed) linear derivative and we take its magnitude, replacing g by 255 - g only
    flips the sign of the derivative, leaving the magnitude unchanged. The result is therefore nearly
    invariant to a global intensity inversion, which is exactly the night<->day shading reversal we need
    the keypoint detector to be robust to.
    """
    g = _as_gray_u8(gray).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max())
    if m < 1e-6:
        return np.zeros_like(g, dtype=np.uint8)
    out = mag * (255.0 / m)
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize(gray, mode='clahe'):
    """Return a uint8 single-channel normalization of `gray` for the given mode. See module docstring."""
    if mode == 'none':
        return _as_gray_u8(gray)
    if mode in _CLAHE:
        return _CLAHE[mode].apply(_as_gray_u8(gray))
    if mode == 'gradient':
        return gradient(gray)
    raise ValueError(f'unknown feat-norm mode {mode!r}; valid: {MODES}')
