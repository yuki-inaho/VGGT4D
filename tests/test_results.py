"""Tests for loading saved VGGT4D result directories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vggt4d.results import load_inference_results
from vggt4d.utils.store import save_tum_poses
from vggt4d.visualize import save_results_to_rrd


def _write_saved_results(root: Path, n_frames: int = 2, h: int = 6, w: int = 8) -> None:
    root.mkdir(parents=True, exist_ok=True)
    intrinsics = np.tile(np.eye(3, dtype=np.float32)[None], (n_frames, 1, 1))
    intrinsics[:, 0, 0] = 10.0
    intrinsics[:, 1, 1] = 10.0
    intrinsics[:, 0, 2] = w / 2
    intrinsics[:, 1, 2] = h / 2
    np.savetxt(root / "pred_intrinsics.txt", intrinsics.reshape(n_frames, 9), fmt="%f")

    cam2world = np.tile(np.eye(4, dtype=np.float32)[None], (n_frames, 1, 1))
    cam2world[:, 0, 3] = np.arange(n_frames, dtype=np.float32)
    save_tum_poses(root, cam2world)

    rng = np.random.default_rng(0)
    for idx in range(n_frames):
        image = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(image).save(root / f"frame_{idx:04d}.png")
        np.save(root / f"frame_{idx:04d}.npy", np.full((h, w), idx + 1, dtype=np.float32))
        np.save(root / f"conf_{idx:04d}.npy", np.full((h, w), 0.5, dtype=np.float32))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[idx : idx + 2, idx : idx + 2] = 255
        Image.fromarray(mask).save(root / f"dynamic_mask_{idx:04d}.png")


def test_load_inference_results_roundtrips_saved_artifacts(tmp_path: Path):
    _write_saved_results(tmp_path)

    results = load_inference_results(tmp_path, show_progress=False)

    assert len(results) == 2
    head = results[0]
    assert head.image.shape == (6, 8, 3)
    assert head.width == 8
    assert head.height == 6
    assert head.intrinsic.shape == (3, 3)
    assert head.extrinsic.shape == (3, 4)
    assert head.depth_map.shape == (6, 8)
    assert head.depth_conf.shape == (6, 8)
    assert head.point_map_by_unprojection.shape == (6, 8, 3)
    assert head.dynamic_mask.dtype == np.bool_


def test_load_inference_results_rejects_mismatched_file_counts(tmp_path: Path):
    _write_saved_results(tmp_path, n_frames=2)
    (tmp_path / "conf_0001.npy").unlink()

    with pytest.raises(ValueError, match="confidence"):
        load_inference_results(tmp_path, show_progress=False)


def test_loaded_inference_results_can_be_saved_to_rrd(tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_saved_results(results_dir)
    results = load_inference_results(results_dir, show_progress=False)

    rrd_path = tmp_path / "scene.rrd"
    save_results_to_rrd(results, rrd_path, show_progress=False)

    assert rrd_path.exists()
    assert rrd_path.stat().st_size > 1024
