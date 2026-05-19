"""Pure-function tests for vggt4d.masks.dynamic_mask."""

from __future__ import annotations

import numpy as np

from vggt4d.masks.dynamic_mask import adaptive_multiotsu_variance


def test_adaptive_multiotsu_variance_bimodal():
    rng = np.random.default_rng(42)
    low = rng.normal(loc=20.0, scale=2.0, size=(64, 64))
    high = rng.normal(loc=200.0, scale=2.0, size=(64, 64))
    img = np.concatenate([low, high], axis=0).astype(np.float32)

    threshold = adaptive_multiotsu_variance(img)

    assert threshold is not None
    # The threshold must lie strictly between the two modes.
    assert 25.0 < float(threshold) < 195.0
