"""CPU tests for camloc.pf.ParticleFilter: pure-numpy predict/update_fix semantics.

A tiny stub TrackMap satisfies the ParticleFilter interface without a real map: dist_outside and
border_dir both return zeros, so update_map (not exercised here) would be a no-op constraint and the
filter math stays well-defined. These lock the gating semantics FEAT-002 will extend."""
import numpy as np

from camloc.pf import ParticleFilter, PFParams


class StubTrackMap:
    """Minimal track prior: everything is inside the drivable area and borders have zero heading."""

    def dist_outside(self, x, y):
        return np.zeros_like(np.asarray(x, dtype=float))

    def border_dir(self, x, y):
        return np.zeros_like(np.asarray(x, dtype=float))


def _make_pf(p=None, seed=0):
    tm = StubTrackMap()
    pose0 = (0.0, 0.0, 0.0)
    return ParticleFilter(tm, pose0, p=p or PFParams(n=4000), seed=seed)


def test_predict_advances_mean_along_straight_increment():
    pf = _make_pf()
    mx0, my0, myaw0 = pf.estimate()
    # straight-line body-frame increment of +1.0 m along x, no lateral, no rotation
    pf.predict(dx=1.0, dy=0.0, dyaw=0.0, dt=0.1)
    mx1, my1, myaw1 = pf.estimate()
    # the mean advances by ~1 m in x (scale ~ 1), stays near zero in y and yaw
    assert abs((mx1 - mx0) - 1.0) < 0.05
    assert abs(my1 - my0) < 0.05
    assert abs(myaw1 - myaw0) < np.radians(2.0)


def test_update_fix_gates_out_far_fix_and_accepts_near_fix():
    pf = _make_pf()
    mx, my, myaw = pf.estimate()
    # a fix well beyond the default fix_gate_pos (10 m) must be rejected
    assert pf.update_fix(mx + 50.0, my, myaw) is False
    # a fix inside the gate must be accepted
    assert pf.update_fix(mx + 1.0, my + 0.5, myaw) is True


def test_in_gate_fix_pulls_estimate_toward_fix():
    pf = _make_pf()
    mx0, my0, _ = pf.estimate()
    fx, fy = mx0 + 2.0, my0 + 0.0
    assert pf.update_fix(fx, fy, 0.0) is True
    mx1, my1, _ = pf.estimate()
    # the weighted estimate moves toward the fix (positive x direction) without overshooting it
    assert mx1 > mx0
    assert mx1 <= fx + 1e-6


def test_per_source_gate_override_accepts_fix_beyond_default_gate():
    # a fix beyond the default position gate but inside a larger override gate is accepted
    pf = _make_pf()
    mx, my, myaw = pf.estimate()
    far = mx + 20.0  # beyond the default 10 m gate
    assert pf.update_fix(far, my, myaw) is False
    pf2 = _make_pf()
    mx, my, myaw = pf2.estimate()
    assert pf2.update_fix(far, my, myaw, gate_pos=30.0) is True


def test_per_source_sigma_overrides_are_honored():
    # a looser-sigma fix source should still accept and reweight without error
    pf = _make_pf()
    mx, my, myaw = pf.estimate()
    assert pf.update_fix(mx + 1.0, my + 0.5, myaw, sig_lat=2.0, sig_lon=4.0) is True


def test_gate_scale_default_matches_current_behavior():
    # gate_scale=1.0 (the default) must reproduce the un-widened gating exactly
    pf = _make_pf()
    mx, my, myaw = pf.estimate()
    assert pf.update_fix(mx + 50.0, my, myaw, gate_scale=1.0) is False     # same as no-arg call
    pf2 = _make_pf()
    assert pf2.update_fix(mx + 1.0, my + 0.5, myaw, gate_scale=1.0) is True


def test_widened_gate_accepts_far_fix_that_default_rejects():
    # a correct-but-far fix rejected by the default gate is accepted once the gate is widened (escalation)
    pf = _make_pf()
    mx, my, myaw = pf.estimate()
    far = mx + 15.0                                   # beyond the default 10 m position gate
    assert pf.update_fix(far, my, myaw) is False      # default gate rejects it
    pf2 = _make_pf()
    mx, my, myaw = pf2.estimate()
    assert pf2.update_fix(far, my, myaw, gate_scale=2.0) is True   # widened gate accepts it


def test_widened_yaw_gate_accepts_far_yaw_fix():
    # a fix with a large heading error is rejected by default but accepted with a widened gate
    pf = _make_pf()
    mx, my, myaw = pf.estimate()
    bad_yaw = myaw + np.radians(8.0)                  # beyond the default 5 deg yaw gate
    assert pf.update_fix(mx + 0.5, my, bad_yaw) is False
    pf2 = _make_pf()
    mx, my, myaw = pf2.estimate()
    assert pf2.update_fix(mx + 0.5, my, bad_yaw, gate_scale=2.0) is True
