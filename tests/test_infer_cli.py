"""Unit tests for the single-sequence inference CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.infer import _select_image_paths, _write_frame_manifest, _write_metadata
from vggt4d.pipeline import list_scene_image_paths


def test_list_scene_image_paths_uses_natural_sort_and_common_suffixes(tmp_path: Path):
    for name in ["frame_10.jpg", "frame_2.JPG", "frame_1.jpeg", "ignore.txt"]:
        (tmp_path / name).write_text("x")

    assert [p.name for p in list_scene_image_paths(tmp_path)] == [
        "frame_1.jpeg",
        "frame_2.JPG",
        "frame_10.jpg",
    ]


def test_select_image_paths_applies_slice_stride_and_cap(tmp_path: Path):
    paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(10)]

    selected = _select_image_paths(paths, start=1, end=9, stride=2, max_frames=3)

    assert [p.name for p in selected] == ["frame_0001.jpg", "frame_0003.jpg", "frame_0005.jpg"]


def test_select_image_paths_rejects_empty_selection(tmp_path: Path):
    with pytest.raises(ValueError, match="No frames selected"):
        _select_image_paths([tmp_path / "frame_0001.jpg"], start=5)


def test_manifest_and_metadata_record_selected_frames(tmp_path: Path):
    input_dir = tmp_path / "frames"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    selected = [input_dir / "frame_0001.jpg", input_dir / "frame_0002.jpg"]

    _write_frame_manifest(output_dir, input_dir, selected)
    _write_metadata(
        output_dir,
        input_dir=input_dir,
        selected_paths=selected,
        preprocessed_shape=(2, 3, 294, 518),
        preprocess_mode="crop",
        checkpoint=Path("ckpts/model.pt"),
    )

    assert (output_dir / "frames.txt").read_text().splitlines() == [
        "0000\tframe_0001.jpg",
        "0001\tframe_0002.jpg",
    ]
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["num_selected_frames"] == 2
    assert metadata["preprocessed_shape_nchw"] == [2, 3, 294, 518]
    assert metadata["outputs"]["camera_trajectory_tum"] == "pred_traj.txt"
