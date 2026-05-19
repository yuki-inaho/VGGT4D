"""Run VGGT4D inference on one directory of sequential image frames.

This CLI is intentionally simpler than ``demo_vggt4d.py``: it treats the input
directory itself as one scene, runs the three-stage VGGT4D pipeline once, and
writes camera poses plus dense outputs into ``--output``.

Examples::

    uv run python -m scripts.infer --input frames_dir --output outputs/my_scene
    uv run python -m scripts.infer --input frames_dir --output outputs/my_scene \
        --stride 5 --max-frames 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from vggt4d.pipeline import (
    DEFAULT_CHECKPOINT,
    VGGT4DPipeline,
    list_scene_image_paths,
    load_image_paths,
)
from vggt4d.preprocess import PreprocessMode


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _select_image_paths(
    paths: Sequence[Path],
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[Path]:
    selected = list(paths[start:end:stride])
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("No frames selected; adjust --start/--end/--stride/--max-frames")
    return selected


def _write_frame_manifest(
    output_dir: Path, input_dir: Path, selected_paths: Sequence[Path]
) -> None:
    with (output_dir / "frames.txt").open("w") as f:
        for idx, path in enumerate(selected_paths):
            f.write(f"{idx:04d}\t{path.relative_to(input_dir)}\n")


def _write_metadata(
    output_dir: Path,
    input_dir: Path,
    selected_paths: Sequence[Path],
    preprocessed_shape: Sequence[int],
    preprocess_mode: PreprocessMode,
    checkpoint: Path,
) -> None:
    metadata = {
        "input_dir": str(input_dir.resolve()),
        "checkpoint": str(checkpoint),
        "preprocess_mode": preprocess_mode,
        "preprocessed_shape_nchw": list(preprocessed_shape),
        "num_selected_frames": len(selected_paths),
        "first_frame": selected_paths[0].name,
        "last_frame": selected_paths[-1].name,
        "outputs": {
            "camera_trajectory_tum": "pred_traj.txt",
            "intrinsics_flattened": "pred_intrinsics.txt",
            "rgb_frames": "frame_*.png",
            "depth_maps": "frame_*.npy",
            "depth_confidence": "conf_*.npy",
            "dynamic_masks": "dynamic_mask_*.png",
            "frame_manifest": "frames.txt",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_inference(
    input_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    preprocess_mode: PreprocessMode,
    start: int,
    end: int | None,
    stride: int,
    max_frames: int | None,
) -> Path:
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("VGGT4D inference currently requires CUDA in this code path")

    paths = list_scene_image_paths(input_dir)
    if not paths:
        raise SystemExit(f"No jpg/jpeg/png images found under {input_dir}")
    selected_paths = _select_image_paths(
        paths, start=start, end=end, stride=stride, max_frames=max_frames
    )

    print(f"Found {len(paths)} frames under {input_dir}")
    print(
        f"Selected {len(selected_paths)} frames ({selected_paths[0].name} -> {selected_paths[-1].name})"
    )
    if len(selected_paths) > 64:
        print(
            "Warning: many frames can require substantial GPU memory; use --stride or --max-frames if needed"
        )

    pipeline = VGGT4DPipeline(checkpoint_path=checkpoint)
    images = load_image_paths(selected_paths, device=pipeline.device, mode=preprocess_mode)
    print(f"Preprocessed tensor shape: {tuple(images.shape)}")

    result = pipeline.run_scene(images)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir)
    _write_frame_manifest(output_dir, input_dir, selected_paths)
    _write_metadata(
        output_dir,
        input_dir=input_dir,
        selected_paths=selected_paths,
        preprocessed_shape=images.shape,
        preprocess_mode=preprocess_mode,
        checkpoint=checkpoint,
    )
    print(f"Wrote inference outputs to {output_dir}")
    print(f"Camera poses: {output_dir / 'pred_traj.txt'}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Directory containing sequential frames"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for inference artifacts"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--mode", choices=("crop", "pad"), default="crop", help="VGGT preprocessing mode"
    )
    parser.add_argument(
        "--start", type=_non_negative_int, default=0, help="0-based inclusive frame index"
    )
    parser.add_argument(
        "--end", type=_non_negative_int, default=None, help="0-based exclusive frame index"
    )
    parser.add_argument("--stride", type=_positive_int, default=1, help="Keep every Nth frame")
    parser.add_argument(
        "--max-frames", type=_positive_int, default=None, help="Cap selected frame count"
    )
    args = parser.parse_args(argv)

    run_inference(
        input_dir=args.input,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        preprocess_mode=args.mode,
        start=args.start,
        end=args.end,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
