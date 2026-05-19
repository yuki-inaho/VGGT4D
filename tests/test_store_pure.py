"""Pure-function tests for vggt4d.utils.store — no GPU, no model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vggt4d.utils.store import (
    c2w_to_tumpose,
    load_tum_poses,
    save_depth,
    save_depth_conf,
    save_intrinsic_txt,
    save_tum_poses,
    to_tum_poses,
)


def test_c2w_to_tumpose_identity():
    pose = c2w_to_tumpose(np.eye(4))
    np.testing.assert_allclose(pose[:3], [0.0, 0.0, 0.0])
    # identity rotation in (qw, qx, qy, qz)
    np.testing.assert_allclose(np.abs(pose[3]), 1.0)
    np.testing.assert_allclose(pose[4:], [0.0, 0.0, 0.0])


def test_save_load_tum_poses_roundtrip(tmp_path: Path):
    c2ws = np.tile(np.eye(4)[None], (4, 1, 1)).astype(np.float64)
    for i in range(4):
        c2ws[i, :3, 3] = [i, 2 * i, -i]

    save_tum_poses(tmp_path, c2ws)
    out = tmp_path / "pred_traj.txt"
    assert out.exists()

    loaded = load_tum_poses(tmp_path)
    assert loaded.shape == (4, 4, 4)
    np.testing.assert_allclose(loaded[:, :3, 3], c2ws[:, :3, 3], atol=1e-5)


def test_save_depth_writes_per_frame(tmp_path: Path):
    depths = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    save_depth(tmp_path, depths)
    for i in range(3):
        arr = np.load(tmp_path / f"frame_{i:04d}.npy")
        np.testing.assert_array_equal(arr, depths[i])


def test_save_depth_conf_writes_per_frame(tmp_path: Path):
    conf = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    save_depth_conf(tmp_path, conf)
    assert (tmp_path / "conf_0000.npy").exists()
    assert (tmp_path / "conf_0001.npy").exists()


def test_save_intrinsic_txt_shape(tmp_path: Path):
    intrinsic = np.zeros((2, 3, 3), dtype=np.float32)
    intrinsic[:, 0, 0] = [100.0, 200.0]
    intrinsic[:, 1, 1] = [100.0, 200.0]
    save_intrinsic_txt(tmp_path, intrinsic)
    loaded = np.loadtxt(tmp_path / "pred_intrinsics.txt")
    assert loaded.shape == (2, 9)
    np.testing.assert_allclose(loaded[:, 0], [100.0, 200.0])


def test_to_tum_poses_matches_save(tmp_path: Path):
    c2ws = np.tile(np.eye(4)[None], (2, 1, 1))
    traj, ts = to_tum_poses(c2ws)
    assert traj.shape == (2, 7)
    np.testing.assert_allclose(ts, [0.0, 1.0])
