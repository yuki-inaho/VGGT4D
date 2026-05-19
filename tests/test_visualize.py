"""Tests for vggt4d.visualize — verifies the rerun-side contract without GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vggt4d.inference import InferenceResult
from vggt4d.visualize import save_results_to_rrd


def _fake_result(seed: int, h: int = 32, w: int = 32) -> InferenceResult:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    extrinsic = np.eye(4)[:3, :4].astype(np.float32)
    intrinsic = np.array([[w * 0.7, 0, w / 2], [0, w * 0.7, h / 2], [0, 0, 1]], dtype=np.float32)
    depth = rng.uniform(0.1, 5.0, (h, w)).astype(np.float32)
    conf = rng.uniform(0.0, 1.0, (h, w)).astype(np.float32)
    points = np.stack(
        [
            np.tile(np.linspace(-1, 1, w), (h, 1)),
            np.tile(np.linspace(-1, 1, h)[:, None], (1, w)),
            depth,
        ],
        axis=-1,
    ).astype(np.float32)
    return InferenceResult(
        image=image,
        width=w,
        height=h,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth,
        depth_conf=conf,
        point_map_by_unprojection=points,
        dynamic_mask=rng.integers(0, 2, (h, w), dtype=np.uint8).astype(bool),
    )


def test_save_results_to_rrd_writes_file(tmp_path: Path):
    results = [_fake_result(i) for i in range(3)]
    rrd_path = tmp_path / "viz.rrd"
    out = save_results_to_rrd(results, rrd_path, filter_percent=30.0)
    assert out == rrd_path
    assert rrd_path.exists()
    # Recording payloads are non-trivial in size even for tiny synthetic data.
    assert rrd_path.stat().st_size > 1024
