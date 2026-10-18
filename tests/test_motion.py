"""CPU test for camloc.motion.increments: integrating a constant forward speed with zero lateral speed
and zero yaw-rate yields a straight line (dx = vx*dt, dy ~ 0, dyaw ~ 0), and the first output row covers
the window from t0 to the first image stamp (causality of the integration window)."""
import numpy as np
import pandas as pd

from camloc.motion import increments


def _write_state_csv(tmp_path):
    """Constant vx = 2 m/s, zero vy/wz over a few state stamps (10 Hz)."""
    base = 1_000_000_000  # 1 s in ns, arbitrary epoch
    step = 100_000_000     # 0.1 s in ns
    rows = []
    for i in range(20):
        rows.append({
            'timestamp_ns': base + i * step,
            'vx_mps': 2.0,
            'vy_mps': 0.0,
            'wz_radps': 0.0,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / 'state.csv'
    df.to_csv(path, index=False)
    return path, base, step


def test_increments_integrates_constant_vx_to_straight_line(tmp_path):
    state_csv, base, step = _write_state_csv(tmp_path)
    t0 = base                      # first pose at the first state stamp
    # image stamps at 0.5 s and 1.0 s after t0
    img_t = np.array([t0, t0 + 5 * step, t0 + 10 * step], dtype=np.int64)

    df = increments(state_csv, img_t, t0)

    # only stamps strictly greater than t0 produce rows
    assert list(df.timestamp_ns.values) == [t0 + 5 * step, t0 + 10 * step]

    # constant vx = 2 m/s, straight line: dx = vx * dt, dy ~ 0, dyaw ~ 0
    for _, row in df.iterrows():
        assert abs(row['dx'] - 2.0 * row['dt']) < 1e-6
        assert abs(row['dy']) < 1e-9
        assert abs(row['dyaw']) < 1e-9
        assert abs(row['dist'] - 2.0 * row['dt']) < 1e-6

    # each window is 0.5 s long
    assert abs(df.iloc[0]['dt'] - 0.5) < 1e-6
    assert abs(df.iloc[1]['dt'] - 0.5) < 1e-6


def test_first_row_covers_t0_to_first_image_stamp(tmp_path):
    state_csv, base, step = _write_state_csv(tmp_path)
    t0 = base
    first = t0 + 3 * step          # first image stamp 0.3 s after t0
    img_t = np.array([first, first + 2 * step], dtype=np.int64)

    df = increments(state_csv, img_t, t0)

    # the first output row integrates the window t0 -> first image stamp
    assert df.iloc[0]['timestamp_ns'] == first
    assert abs(df.iloc[0]['dt'] - (first - t0) * 1e-9) < 1e-9
    assert abs(df.iloc[0]['dx'] - 2.0 * (first - t0) * 1e-9) < 1e-6
