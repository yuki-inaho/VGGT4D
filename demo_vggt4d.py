"""CLI entrypoint: run VGGT4D over every scene under ``--input_dir``."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt4d.pipeline import VGGT4DPipeline, load_scene_images


def _iter_scene_dirs(input_dir: Path) -> list[Path]:
    scene_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not scene_dirs:
        raise ValueError(f"No scene directories found in {input_dir}")
    return scene_dirs


def _process_scene(pipeline: VGGT4DPipeline, scene_dir: Path, output_dir: Path) -> None:
    images = load_scene_images(scene_dir, pipeline.device)
    if images is None:
        print(f"Warning: No images found in {scene_dir}, skipping this scene")
        return

    print(f"Processing scene: {scene_dir.name} ({images.shape[0]} images)")
    print("  Stage 1: predict depth map and dynamic map")
    print("  Stage 2: refine extrinsics by dynamic map")
    print("  Stage 3: refine dynamic map")

    result = pipeline.run_scene(images)
    print(f"  Saving predictions to {output_dir}\n")
    result.save(output_dir)


def main(input_dir: str, output_dir: str) -> None:
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    scene_dirs = _iter_scene_dirs(in_root)
    print(f"Found {len(scene_dirs)} scenes, starting processing...\n")

    pipeline = VGGT4DPipeline()
    for scene_dir in scene_dirs:
        _process_scene(pipeline, scene_dir, out_root / scene_dir.name)
    print(f"All scenes processed! Results saved to {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT4D demo script")
    parser.add_argument("--input_dir", type=str, required=True, help="Input data directory path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output result directory path")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
