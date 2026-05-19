"""End-to-end smoke test: run demo_vggt4d on a tiny synthetic scene."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, needs_checkpoint, needs_cuda


@pytest.mark.smoke
@needs_cuda
@needs_checkpoint
def test_demo_e2e_generates_expected_outputs(
    synthetic_scene_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "demo_vggt4d.py",
            "--input_dir",
            str(synthetic_scene_dir),
            "--output_dir",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"demo failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    scene_out = out_dir / "scene1"
    assert scene_out.is_dir()

    # README-documented artifacts must all be present.
    expected_globs = [
        "frame_*.npy",
        "conf_*.npy",
        "frame_*.png",
        "dynamic_mask_*.png",
        "pred_intrinsics.txt",
        "pred_traj.txt",
    ]
    for pattern in expected_globs:
        matches = list(scene_out.glob(pattern))
        assert matches, f"no files match {pattern} in {scene_out}"
