"""Tests for exported VGGT4D player assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vggt4d.inference import InferenceResult
from vggt4d.player_assets import (
    camera_to_world,
    export_results_to_player_assets,
    load_frame_cloud,
    select_result_frames,
)


def _fake_result(h: int = 8, w: int = 10) -> InferenceResult:
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[..., 0] = np.arange(w, dtype=np.uint8)
    image[..., 1] = np.arange(h, dtype=np.uint8)[:, None]
    depth = np.linspace(0.1, 2.0, h * w, dtype=np.float32).reshape(h, w)
    conf = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    yy, xx = np.mgrid[:h, :w]
    points = np.stack([xx, yy, depth], axis=-1).astype(np.float32)
    dynamic_mask = np.zeros((h, w), dtype=bool)
    dynamic_mask[:, : w // 2] = True
    return InferenceResult(
        image=image,
        width=w,
        height=h,
        extrinsic=np.eye(4, dtype=np.float32)[:3, :4],
        intrinsic=np.array(
            [[10.0, 0.0, w / 2], [0.0, 10.0, h / 2], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        depth_map=depth,
        depth_conf=conf,
        point_map_by_unprojection=points,
        dynamic_mask=dynamic_mask,
    )


def test_select_result_frames_keeps_source_indices():
    results = [_fake_result() for _ in range(5)]

    selected = select_result_frames(results, stride=2)

    assert [frame.source_index for frame in selected] == [0, 2, 4]
    assert [id(frame.result) for frame in selected] == [
        id(results[0]),
        id(results[2]),
        id(results[4]),
    ]


def test_camera_to_world_inverts_extrinsic():
    result = _fake_result()
    result.extrinsic[:, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    c2w = camera_to_world(result)

    np.testing.assert_allclose(c2w[:3, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_export_results_to_player_assets_writes_manifest_and_files(tmp_path: Path):
    results = [_fake_result() for _ in range(3)]

    manifest = export_results_to_player_assets(
        results,
        output_dir=tmp_path,
        stride=2,
        max_points_per_frame=7,
        write_ply=True,
        show_progress=False,
    )

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["frame_count"] == 2
    assert manifest["frames"][0]["source_index"] == 0
    assert manifest["frames"][1]["source_index"] == 2

    frame = manifest["frames"][0]
    for key in ("rgb", "depth", "depth_raw", "mask", "mask_overlay", "point_cloud"):
        assert (tmp_path / frame[key]).exists()
    assert (tmp_path / frame["point_cloud_ply"]).exists()

    cloud = load_frame_cloud(tmp_path, frame)
    assert cloud.points.shape == (7, 3)
    assert cloud.colors.shape == (7, 3)
