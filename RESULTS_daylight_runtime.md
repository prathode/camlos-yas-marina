# Daylight robustness and runtime (notes for the P5200 run)

This note is the deliverable summary for two pieces of work on top of the night submission
(validation laps 3 / 7 at 0.21 / 0.27 m, see [RESULTS17.10morning.md](RESULTS17.10morning.md)):

1. **Daylight robustness.** The relocalization maps are built from night images; the afternoon
   daylight lap `rosbag_2026-10-13_16-08-46_cameras_lap.zip` reverses the shading (lit stands and
   lamps go dark, previously unlit surfaces go bright) and turns the strongest night landmarks off,
   so absolute-intensity matching collapses. We add opt-in preprocessing and an adaptive fallback so
   the same night maps keep working.
2. **Runtime.** The night config runs at ~200 ms/frame on the RTX 2070. The target hardware is an
   NVIDIA Quadro P5200 (16 GB, Pascal, compute capability 6.1) and the goal is at least 10 FPS, i.e.
   at most 100 ms/frame. We add structural speed knobs and a recommended fast configuration.

Everything new defaults to the current night behavior, so the night submission is unchanged unless a
switch is set. Real daylight RMSE and GPU timing were NOT measured here (see
[Limitations](#7-honest-limitations)); the numbers to read on the P5200 are given below.

---

## 1. Summary of what changed

- **FEAT-002 (daylight):** `camloc/photometric.py` normalization modes plus a `--feat-norm` flag on
  `run.py`, `scripts/build_kf_map.py` and `scripts/export_kf_images.py`; an adaptive RoMa escalation
  (`--reloc-fail-escalate` with `--escalate-period` / `--escalate-cert` / `--escalate-gate`); a widened
  particle-filter fix gate while escalated; and a synthetic `daylight` proxy in `camloc/augment.py`.
- **FEAT-003 (runtime):** structural speed knobs on `run.py` (`--merge`, `--n-kf`, `--kpts`,
  `--resize`, `--every`) and two Pascal-unsafe half-precision switches (`--lg-mp`, `--aliked-amp`,
  both default off), plus a recommended fast configuration `configs/fl_rl_fast.json`.

---

## 2. Daylight handling

### Feature normalization (`--feat-norm`)

`camloc/photometric.py` adds `normalize(gray, mode)` with modes `none`, `clahe`, `clahe_strong`,
`gradient`. The flag is exposed on:

- `run.py --feat-norm` (default `clahe`, i.e. the current night behavior),
- `scripts/build_kf_map.py --feat-norm` (default `clahe`),
- `scripts/export_kf_images.py --feat-norm` (default `none`, which keeps existing RoMa keyframe images
  unchanged).

**Recommended for daylight: `gradient`.** It replaces the absolute-intensity image with the Sobel
gradient magnitude. Because the magnitude of a (signed) derivative is unchanged when the image is
globally inverted (`g` to `255 - g`), the representation is near-invariant to the night-to-day shading
reversal, so ALIKED / LightGlue (and RoMa) see structure edges instead of brightness.

**The map and the queries must use the SAME `--feat-norm`.** A keyframe map built with `clahe` cannot
be matched against `gradient` queries. Either reuse the night (`clahe`) maps with `run.py --feat-norm
clahe` or `none`, or build daylight maps with `--feat-norm gradient` and run with `--feat-norm
gradient`. For the RoMa fallback the exported keyframe images (`export_kf_images.py --feat-norm`) must
match too.

### Adaptive RoMa escalation (`--reloc-fail-escalate`)

`run.py --reloc-fail-escalate N` (default `0` = off) keeps a causal counter of consecutive image sets
with no accepted LightGlue fix. After `N` such sets it temporarily:

- shortens the RoMa period to `--escalate-period` (default `0` = half of `--roma-period`),
- loosens the RoMa certainty threshold to `--escalate-cert` (default `0` = half of `--roma-cert`),
- widens the particle-filter fix gate by `--escalate-gate` (default `2.0`, applied to position and yaw).

It reverts the moment a sparse LightGlue fix is accepted again. The counter uses only past sets (no
ground truth), so causality is preserved. With `N = 0` nothing changes.

### Synthetic daylight proxy

`camloc/augment.py` adds a `daylight` transform (used by `scripts/stress_test.py --conds`) that combines
the dominant night-to-day effects (lamps off, overall brightening, reduced contrast, a mild local shading
reversal). It is a stress proxy only, not a real night-to-day translation; `invert` remains the worst-case
reversal proxy.

---

## 3. Runtime

### Speed knobs on `run.py` (all default to current behavior)

| Flag | Default | Effect |
|---|---|---|
| `--n-kf` | 2 | number of nearest keyframes matched per camera (fewer = faster) |
| `--merge` | 0 | 1 = match the `--n-kf` keyframes concatenated in ONE LightGlue call |
| `--kpts` | 2048 | ALIKED max keypoints (fewer = faster, less robust) |
| `--resize` | 1024 | ALIKED input resize (smaller = faster) |
| `--every` | '' | per-camera localize stride, e.g. `fl:1,rl:2` localizes `rl` every other set; a pose is still emitted every set, so coverage stays 100 % |

### Pascal-unsafe half-precision switches

| Flag | Default | Note |
|---|---|---|
| `--lg-mp` | 0 | LightGlue mixed precision (fp16) |
| `--aliked-amp` | 0 | ALIKED fp16 autocast |

**Do NOT enable fp16 by default on the P5200.** Pascal (compute capability 6.1) has no tensor cores and
weak fp16, so half precision is often SLOWER than fp32. `run.py` prints a warning if these are requested
on a pre-Volta (cc < 7.0) device. If you try them, benchmark BOTH ways (see the P5200 checklist). fp16 is
NOT part of the recommended config.

### Recommended fast configuration (fp32)

Paired with `configs/fl_rl_fast.json` (identical PF params to `configs/fl_rl.json`, so filter accuracy is
unchanged; the speed comes from the structural flags, not from retuning the filter):

```sh
--cams fl,rl --n-kf 1 --merge 1 --kpts 1536 --resize 1024 --every fl:1,rl:2 --roma-fallback 0
```

---

## 4. Exact commands (run on the P5200)

### Datasets (fetch on your own machine; both return HTTP 403 from the sandbox)

- Night training run:
  [c0_2026-10-13_22-53-42_camloc.zip](https://eu1-s3.virtuozzo.com/dev-auto-mobility-data/hackaton_2026/c0_2026-10-13_22-53-42_camloc.zip)
- Daylight lap: `rosbag_2026-10-13_16-08-46_cameras_lap.zip` from the SharePoint link the organizers
  shared (forms authentication, cannot be downloaded headlessly).

### Prepare the daylight lap

```sh
unzip rosbag_2026-10-13_16-08-46_cameras_lap.zip        # -> rosbag_2026-10-13_16-08-46_cameras_lap.mcap
DAY=rosbag_2026-10-13_16-08-46_cameras_lap.mcap
python tools/export_cache.py $DAY cache_day --img-every 10
python tools/extract_gt.py $DAY gt_day.csv              # only if the lap carries GT (needed to build/score)
python scripts/split_laps.py gt_day.csv cache_day       # only if GT is present
```

### Option A: reuse the night maps (no new map needed)

```sh
python run.py $DAY out/day/day.csv --maps maps --cams fl,rl --pf-config configs/fl_rl_fast.json \
    --feat-norm gradient --roma-fallback 1 --reloc-fail-escalate 3 \
    --n-kf 1 --merge 1 --kpts 1536 --resize 1024 --every fl:1,rl:2
```

Note: reusing the night maps with `--feat-norm gradient` requires the night maps to have been built with
`gradient` too (same `--feat-norm` on both sides). If your night maps are `clahe`, either run with
`--feat-norm clahe` (no reversal robustness) or build gradient maps as in Option B.

### Option B (optional): build daylight-specific keyframes (needs daylight GT)

Build the map and export its RoMa keyframe images with the SAME `--feat-norm gradient`:

```sh
python scripts/build_kf_map.py $DAY cache_day gt_day.csv maps_day/kf_fl.npz --cam fl --feat-norm gradient
python scripts/build_kf_map.py $DAY cache_day gt_day.csv maps_day/kf_rl.npz --cam rl --feat-norm gradient
python scripts/export_kf_images.py $DAY cache_day gt_day.csv maps_day/kf_fl.npz --cam fl --feat-norm gradient
python run.py $DAY out/day/day.csv --maps maps_day --cams fl,rl --pf-config configs/fl_rl_fast.json \
    --feat-norm gradient --roma-fallback 1 --reloc-fail-escalate 3 \
    --n-kf 1 --merge 1 --kpts 1536 --resize 1024 --every fl:1,rl:2
```

### Score the daylight run (needs daylight GT)

```sh
python scripts/eval_laps.py gt_day.csv out/day/day.csv cache_day
```

### Offline proxy sanity check (held-out laps 2 and 6 ONLY, never val 3 / 7)

Run this on the night bag before touching the real lap, to exercise the daylight code paths and compare
settings without the real daylight data:

```sh
python3 scripts/stress_test.py BAG --laps 2,6 --maps maps_no26 --conds daylight,invert --roma 0,1 \
    --run-args '--feat-norm gradient --reloc-fail-escalate 3'
```

(`BAG` is the night mcap; `maps_no26/` are the keyframe maps built without laps 2 and 6.)

---

## 5. Expected effect and exactly which numbers to read

**Daylight accuracy.** With `--feat-norm gradient` plus the RoMa escalation, matching should survive the
shading reversal and the daylight RMSE should drop, with the **lateral and yaw errors improving first**,
then the along-track (longitudinal) error. Read these from the `tools/evaluate.py` output printed by
`scripts/eval_laps.py`:

- **position RMSE**
- **lateral RMSE**
- **yaw RMSE**

(and longitudinal RMSE as the slower-to-improve term).

**Runtime.** Read `mean_ms` from `out/day/day_timing.json` (or `out/fast/fast_timing.json` for the fast
benchmark). The target is **`mean_ms` <= 100** (at least 10 FPS).

Fast benchmark on the held-out laps (accuracy must not drop noticeably vs the night config):

```sh
python scripts/eval_variant.py BAG cache --cams fl,rl --n-kf 1 --merge 1 --kpts 1536 \
    --every fl:1,rl:2 --tag fast
```

---

## 6. Verify on the P5200

- [ ] **(a) Night regression.** Rerun the night submission command (the plain `configs/fl_rl.json` run,
      no new flags) and confirm val laps 3 / 7 are still ~0.21 / 0.27 m. New behavior defaults off, so this
      should be unchanged by construction.
- [ ] **(b) Daylight accuracy.** Run Option A (or B) on `rosbag_2026-10-13_16-08-46_cameras_lap` and read
      position / lateral / yaw RMSE from `scripts/eval_laps.py`.
- [ ] **(c) Timing.** Confirm `mean_ms` <= 100 in the `*_timing.json` of the fast run.
- [ ] **(d) fp16 (only if tried).** Benchmark `--lg-mp 1 --aliked-amp 1` BOTH ways; on Pascal (cc 6.1) it
      may be SLOWER than fp32, so keep whichever is faster and do not assume fp16 wins.

---

## 7. Honest limitations

- **No real daylight measurement here.** The sandbox is CPU-only with no torch/GPU, no dataset, and both
  dataset URLs return HTTP 403. Real daylight RMSE and GPU timing were NOT measured. The numbers above are
  targets and the fields to read, not results.
- **Synthetic proxies are not real daylight.** The `daylight` and `invert` transforms exercise the code
  paths and let settings be compared on the held-out laps, but real daylight also changes shadows and which
  structures are visible.
- **Night results preserved by construction.** Every new switch defaults to the current night behavior
  (`--feat-norm clahe`, `--reloc-fail-escalate 0`, all speed/fp16 knobs at their current values), so the
  night submission is unchanged unless a flag is set.
- **Invariants maintained.** Causality (a pose at `t` uses only images and state with stamp <= `t`), GT
  isolation (ground truth only for map building and scoring; only the first pose used at inference), and the
  lap discipline (maps from laps 0,1,2,4,5,6,8; tune only on held-out laps 2 and 6; never tune on or leak
  val laps 3 and 7) all hold.
