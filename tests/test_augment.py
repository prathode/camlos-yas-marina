"""CPU tests for camloc.augment: every TRANSFORM keeps a uint8 image same-shaped and in [0,255];
invert is an exact photographic negative; 'none' is the identity. No dataset or torch required."""
import numpy as np

from camloc.augment import TRANSFORMS, daylight, invert


def _synthetic_image():
    """A night-like grayscale frame: a smooth gradient plus a few bright 'lamp' blobs."""
    h, w = 728, 1506
    xs = np.linspace(0, 200, w, dtype=np.float32)[None, :]
    ys = np.linspace(0, 40, h, dtype=np.float32)[:, None]
    g = (xs + ys)
    g = np.clip(g, 0, 255).astype(np.uint8)
    # a handful of saturated lamp blobs
    for (cy, cx) in [(100, 200), (300, 900), (500, 1300), (650, 400)]:
        g[cy - 8:cy + 8, cx - 8:cx + 8] = 255
    return g


def test_all_transforms_preserve_shape_dtype_and_range():
    g = _synthetic_image()
    for name, fn in TRANSFORMS.items():
        out = fn(g)
        assert isinstance(out, np.ndarray), name
        assert out.dtype == np.uint8, name
        assert out.shape == g.shape, name
        assert out.min() >= 0 and out.max() <= 255, name


def test_invert_is_exact_negative():
    g = _synthetic_image()
    out = invert(g)
    assert np.array_equal(out, (255 - g).astype(np.uint8))


def test_none_is_identity():
    g = _synthetic_image()
    out = TRANSFORMS['none'](g)
    assert np.array_equal(out, g)


def test_daylight_proxy_registered_and_valid():
    """The synthetic 'daylight' proxy is in TRANSFORMS and returns a valid uint8 frame."""
    assert 'daylight' in TRANSFORMS
    g = _synthetic_image()
    out = daylight(g)
    assert out.dtype == np.uint8
    assert out.shape == g.shape
    assert out.min() >= 0 and out.max() <= 255


def test_daylight_removes_lamp_blobs_and_brightens():
    """The daylight proxy should turn the saturated night 'lamps' off (no longer 255) and lift the overall
    brightness, unlike the untouched night frame."""
    g = _synthetic_image()
    out = daylight(g)
    lamp = out[100 - 8:100 + 8, 200 - 8:200 + 8]
    assert lamp.max() < 255                 # the lamp blob is no longer saturated
    assert out.mean() > g.mean()            # the scene is brighter overall
