"""Load saved VGGT4D inference artifacts back into the public result API."""

from __future__ import annotations

from pathlib import Path
from typing import Sized

import numpy as np
from PIL import Image

from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt4d.inference import InferenceResult
from vggt4d.utils.store import load_tum_poses


def _required_paths(results_dir: Path) -> dict[str, Path]:
    paths = {
        "intrinsics": results_dir / "pred_intrinsics.txt",
        "trajectory": results_dir / "pred_traj.txt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing saved VGGT4D result files: {missing}")
    return paths


def _load_intrinsics(path: Path) -> np.ndarray:
    intrinsics = np.loadtxt(path, dtype=np.float32)
    intrinsics = np.atleast_2d(intrinsics)
    if intrinsics.shape[1] != 9:
        raise ValueError(f"Expected flattened 3x3 intrinsics in {path}, got {intrinsics.shape}")
    return intrinsics.reshape(-1, 3, 3)


def _load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(Image.open(path).convert("L"))
    return mask > 0


def _glob_required(results_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match {pattern} in {results_dir}")
    return paths


def _validate_count(name: str, items: Sized, expected: int) -> None:
    if len(items) != expected:
        raise ValueError(f"Expected {expected} {name} entries, found {len(items)}")


def _validate_frame_arrays(
    image: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    dynamic_mask: np.ndarray,
    source: Path,
) -> None:
    expected_hw = image.shape[:2]
    for name, array in (
        ("depth", depth),
        ("confidence", conf),
        ("dynamic mask", dynamic_mask),
    ):
        if array.shape != expected_hw:
            raise ValueError(
                f"{name} shape {array.shape} does not match image shape {expected_hw} for {source}"
            )


def load_inference_results(results_dir: str | Path) -> list[InferenceResult]:
    """Load artifacts written by ``SceneResult.save`` / ``scripts.infer``.

    The returned objects can be passed directly to
    :func:`vggt4d.visualize.visualize_results` or
    :func:`vggt4d.visualize.save_results_to_rrd`.
    """
    root = Path(results_dir)
    paths = _required_paths(root)

    image_paths = _glob_required(root, "frame_*.png")
    depth_paths = _glob_required(root, "frame_*.npy")
    conf_paths = _glob_required(root, "conf_*.npy")
    mask_paths = sorted(root.glob("dynamic_mask_*.png"))

    n_frames = len(image_paths)
    _validate_count("depth", depth_paths, n_frames)
    _validate_count("confidence", conf_paths, n_frames)
    if mask_paths:
        _validate_count("dynamic mask", mask_paths, n_frames)

    intrinsics = _load_intrinsics(paths["intrinsics"])
    cam2world = load_tum_poses(root)
    _validate_count("intrinsic", intrinsics, n_frames)
    _validate_count("camera pose", cam2world, n_frames)

    world_to_camera = np.linalg.inv(cam2world)[:, :3, :4].astype(np.float32)
    depths = [np.load(path).astype(np.float32) for path in depth_paths]
    point_maps = unproject_depth_map_to_point_map(
        np.stack(depths, axis=0)[..., None], world_to_camera, intrinsics
    ).astype(np.float32)

    results: list[InferenceResult] = []
    for idx, image_path in enumerate(image_paths):
        image = _load_png(image_path)
        height, width = image.shape[:2]
        depth = depths[idx]
        conf = np.load(conf_paths[idx]).astype(np.float32)
        mask_path = mask_paths[idx] if mask_paths else root / "__missing_dynamic_mask__.png"
        dynamic_mask = _load_mask(mask_path, shape=(height, width))
        _validate_frame_arrays(image, depth, conf, dynamic_mask, image_path)

        results.append(
            InferenceResult(
                image=image,
                width=int(width),
                height=int(height),
                extrinsic=world_to_camera[idx],
                intrinsic=intrinsics[idx],
                depth_map=depth,
                depth_conf=conf,
                point_map_by_unprojection=point_maps[idx],
                dynamic_mask=dynamic_mask,
            )
        )
    return results
