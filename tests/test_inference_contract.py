"""End-to-end contract test for VGGT4DInference.

Asserts the API shape matches the official ``vggt-pytorch-inference`` reference
so callers can swap between the two.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.conftest import needs_checkpoint, needs_cuda
from vggt4d.inference import InferenceResult


@pytest.mark.smoke
@needs_cuda
@needs_checkpoint
def test_vggt4d_inference_returns_inference_results() -> None:
    # Import lazily so the unit suite (no GPU) doesn't pay the model-load cost.
    from vggt4d.inference import VGGT4DInference

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (518, 518, 3), dtype=np.uint8) for _ in range(4)]

    model = VGGT4DInference()
    results = model(frames)

    assert isinstance(results, list)
    assert len(results) == len(frames)
    assert all(isinstance(r, InferenceResult) for r in results)

    expected_fields = {
        "image",
        "width",
        "height",
        "extrinsic",
        "intrinsic",
        "depth_map",
        "depth_conf",
        "point_map_by_unprojection",
        "dynamic_mask",
    }
    actual_fields = {f.name for f in dataclasses.fields(InferenceResult)}
    assert expected_fields <= actual_fields

    head = results[0]
    h, w = head.height, head.width
    assert head.image.shape == (h, w, 3) and head.image.dtype == np.uint8
    assert head.extrinsic.shape == (3, 4)
    assert head.intrinsic.shape == (3, 3)
    assert head.depth_map.shape == (h, w)
    assert head.depth_conf.shape == (h, w)
    assert head.point_map_by_unprojection.shape == (h, w, 3)
    assert head.dynamic_mask.shape == (h, w) and head.dynamic_mask.dtype == np.bool_
