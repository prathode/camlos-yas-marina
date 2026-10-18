"""CPU tests for camloc.photometric.normalize (pure numpy + opencv, no torch, no dataset).

camloc.features imports torch + lightglue at module top, which are not installed here, so the pure
image-normalization logic lives in the torch-free camloc.photometric module and is tested directly.

Checks for every mode: uint8, single-channel, same shape, values in [0, 255]. 'none' is identity. The
'gradient' mode is near-invariant to a GLOBAL intensity inversion (g vs 255 - g), which is the key
property for matching night maps against daylight queries (where the shading reverses)."""
import numpy as np

from camloc.photometric import MODES, normalize


def _synthetic_image():
    """A night-like grayscale frame: a smooth gradient plus a few bright 'lamp' blobs."""
    h, w = 240, 320
    xs = np.linspace(0, 200, w, dtype=np.float32)[None, :]
    ys = np.linspace(0, 40, h, dtype=np.float32)[:, None]
    g = np.clip(xs + ys, 0, 255).astype(np.uint8)
    for (cy, cx) in [(40, 80), (120, 200), (200, 280), (60, 300)]:
        g[cy - 6:cy + 6, cx - 6:cx + 6] = 255
    return g


def test_every_mode_returns_uint8_single_channel_same_shape_in_range():
    g = _synthetic_image()
    for mode in MODES:
        out = normalize(g, mode)
        assert isinstance(out, np.ndarray), mode
        assert out.dtype == np.uint8, mode
        assert out.ndim == 2, mode
        assert out.shape == g.shape, mode
        assert out.min() >= 0 and out.max() <= 255, mode


def test_none_is_identity():
    g = _synthetic_image()
    assert np.array_equal(normalize(g, 'none'), g)


def test_clahe_default_matches_legacy_behavior():
    """The default 'clahe' mode must reproduce the old fixed CLAHE (clipLimit 2.0, 8x8) exactly, so night
    runs are byte-for-byte unchanged."""
    import cv2
    g = _synthetic_image()
    legacy = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    assert np.array_equal(normalize(g, 'clahe'), legacy)


def test_gradient_is_near_invariant_to_global_intensity_inversion():
    """Feeding g and 255 - g must give nearly identical gradient-normalized images (shading reversal)."""
    g = _synthetic_image()
    a = normalize(g, 'gradient').astype(np.int16)
    b = normalize((255 - g).astype(np.uint8), 'gradient').astype(np.int16)
    diff = np.abs(a - b)
    # the Sobel magnitude only flips sign under inversion, so the magnitudes match up to rounding
    assert float(diff.mean()) < 1.0
    assert float(np.mean(diff > 5)) < 0.01


def test_gradient_differs_from_raw_intensity():
    """Sanity: 'gradient' is a structural representation, not a passthrough of intensity."""
    g = _synthetic_image()
    out = normalize(g, 'gradient')
    assert not np.array_equal(out, g)


def test_unknown_mode_raises():
    g = _synthetic_image()
    try:
        normalize(g, 'nope')
    except ValueError:
        return
    raise AssertionError('expected ValueError for an unknown mode')
