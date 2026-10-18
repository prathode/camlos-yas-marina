"""Per-camera frame-stride scheduling for the online loop (pure python, no torch, no cv2).

The online loop in run.py emits exactly one pose per image set. To cut GPU work, a camera can be
localized only every k-th set (--every fl:1,rl:2 localizes rl on every other set) while the filter
still predicts and updates on every set, so output coverage stays 100%. The parsing and the
per-camera skip decision are kept here, free of torch/cv2, so they can be unit-tested on CPU without
the GPU stack or the dataset.
"""


def parse_every(spec, cams):
    """Parse a stride spec like 'fl:1,rl:2' into {cam: stride} for every camera in `cams`.

    Cameras not named in the spec keep stride 1 (localize on every set = current behavior). An empty
    or None spec means all strides are 1. Strides must be positive integers.

    >>> parse_every('fl:1,rl:2', ['fl', 'rl'])
    {'fl': 1, 'rl': 2}
    >>> parse_every('', ['fl', 'rl'])
    {'fl': 1, 'rl': 1}
    >>> parse_every('rl:3', ['fl', 'rl'])
    {'fl': 1, 'rl': 3}
    """
    every = {c: 1 for c in cams}
    for kv in (spec or '').split(','):
        kv = kv.strip()
        if not kv:
            continue
        cam, _, s = kv.partition(':')
        cam = cam.strip()
        stride = int(s)
        if stride < 1:
            raise ValueError(f'stride for {cam!r} must be a positive integer, got {stride}')
        if cam not in every:
            raise ValueError(f'--every names camera {cam!r} not in {cams}')
        every[cam] = stride
    return every


def localize_this_set(every, cam, set_index):
    """Whether camera `cam` is localized on image set number `set_index` (0-based) under strides `every`.

    A camera with stride k is localized when set_index % k == 0, so stride 1 localizes every set
    (current behavior) and stride 2 localizes sets 0, 2, 4, ...

    >>> e = parse_every('fl:1,rl:2', ['fl', 'rl'])
    >>> [localize_this_set(e, 'fl', i) for i in range(4)]
    [True, True, True, True]
    >>> [localize_this_set(e, 'rl', i) for i in range(4)]
    [True, False, True, False]
    """
    return set_index % every.get(cam, 1) == 0
