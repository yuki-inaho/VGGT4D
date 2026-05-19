"""Export and play lightweight VGGT4D viewer assets.

The asset directory is intentionally made of ordinary files: sampled point
clouds, RGB PNGs, colorized depth PNGs, mask overlays, and a JSON manifest.
This keeps playback separate from inference and avoids baking a video too early.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from tqdm.auto import tqdm

from vggt4d.inference import InferenceResult
from vggt4d.results import load_inference_results

SCHEMA = "vggt4d-player-assets-v1"
WINDOW_TITLE = "VGGT4D RGB / Depth / Dynamic mask"


@dataclass(frozen=True)
class FrameCloud:
    points: np.ndarray
    colors: np.ndarray


@dataclass(frozen=True)
class SelectedFrame:
    source_index: int
    result: InferenceResult


def camera_to_world(result: InferenceResult) -> np.ndarray:
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :4] = result.extrinsic
    return np.linalg.inv(world_to_camera).astype(np.float32)


def select_result_frames(
    results: Sequence[InferenceResult],
    stride: int,
    max_frames: int | None = None,
) -> list[SelectedFrame]:
    selected = [
        SelectedFrame(source_index=source_index, result=result)
        for source_index, result in enumerate(results)
        if source_index % stride == 0
    ]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        scaled = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.zeros(depth.shape, dtype=np.float32)
        normalized[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0, 1)
        scaled = (normalized * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    overlay[mask.astype(bool)] = (255, 64, 64)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def _sample_indices(mask: np.ndarray, max_points: int) -> np.ndarray:
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size <= max_points:
        return idx
    take = np.linspace(0, idx.size - 1, max_points, dtype=np.int64)
    return idx[take]


def sample_frame_cloud(
    result: InferenceResult,
    filter_percent: float,
    max_points: int,
    hide_dynamic: bool,
) -> FrameCloud:
    threshold = np.percentile(result.depth_conf, filter_percent)
    keep = np.isfinite(result.point_map_by_unprojection).all(axis=-1)
    keep &= result.depth_conf >= threshold
    if hide_dynamic:
        keep &= ~result.dynamic_mask.astype(bool)

    idx = _sample_indices(keep, max_points=max_points)
    points = result.point_map_by_unprojection.reshape(-1, 3)[idx].astype(np.float32)
    colors = result.image.reshape(-1, 3)[idx].astype(np.uint8)
    return FrameCloud(points=points, colors=colors)


def write_binary_ply(path: Path, cloud: FrameCloud) -> None:
    points = np.asarray(cloud.points, dtype="<f4")
    colors = np.asarray(cloud.colors, dtype=np.uint8)
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    if len(points):
        vertices["x"] = points[:, 0]
        vertices["y"] = points[:, 1]
        vertices["z"] = points[:, 2]
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        vertices.tofile(f)


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _write_gray_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def export_results_to_player_assets(
    results: Sequence[InferenceResult],
    output_dir: Path,
    stride: int = 1,
    max_frames: int | None = None,
    filter_percent: float = 70.0,
    max_points_per_frame: int = 10_000,
    hide_dynamic: bool = False,
    write_ply: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_result_frames(results, stride=stride, max_frames=max_frames)
    if not selected:
        raise ValueError("No frames selected for player asset export")

    frames: list[dict[str, Any]] = []
    iterator = tqdm(
        enumerate(selected),
        total=len(selected),
        desc="Exporting player assets",
        unit="frame",
        disable=not show_progress,
    )
    for export_index, selected_frame in iterator:
        result = selected_frame.result
        stem = f"frame_{export_index:06d}"
        rgb_path = output_dir / "rgb" / f"{stem}.png"
        depth_path = output_dir / "depth" / f"{stem}.png"
        depth_raw_path = output_dir / "depth_raw" / f"{stem}.npy"
        mask_path = output_dir / "mask" / f"{stem}.png"
        mask_overlay_path = output_dir / "mask_overlay" / f"{stem}.png"
        cloud_npz_path = output_dir / "point_clouds" / f"{stem}.npz"
        cloud_ply_path = output_dir / "point_clouds" / f"{stem}.ply"

        cloud = sample_frame_cloud(
            result,
            filter_percent=filter_percent,
            max_points=max_points_per_frame,
            hide_dynamic=hide_dynamic,
        )
        c2w = camera_to_world(result)

        _write_rgb_png(rgb_path, result.image)
        _write_rgb_png(depth_path, colorize_depth(result.depth_map))
        _write_gray_png(mask_path, result.dynamic_mask.astype(np.uint8) * 255)
        _write_rgb_png(mask_overlay_path, mask_overlay(result.image, result.dynamic_mask))
        depth_raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_raw_path, result.depth_map.astype(np.float32))
        cloud_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cloud_npz_path, points=cloud.points, colors=cloud.colors)
        if write_ply:
            write_binary_ply(cloud_ply_path, cloud)

        frame = {
            "frame_index": export_index,
            "source_index": selected_frame.source_index,
            "width": result.width,
            "height": result.height,
            "rgb": _rel(rgb_path, output_dir),
            "depth": _rel(depth_path, output_dir),
            "depth_raw": _rel(depth_raw_path, output_dir),
            "mask": _rel(mask_path, output_dir),
            "mask_overlay": _rel(mask_overlay_path, output_dir),
            "point_cloud": _rel(cloud_npz_path, output_dir),
            "point_cloud_ply": _rel(cloud_ply_path, output_dir) if write_ply else None,
            "point_count": int(len(cloud.points)),
            "camera_to_world": c2w.tolist(),
            "world_to_camera": result.extrinsic.tolist(),
            "intrinsic": result.intrinsic.tolist(),
        }
        frames.append(frame)

    manifest = {
        "schema": SCHEMA,
        "source_frame_count": len(results),
        "frame_count": len(frames),
        "stride": stride,
        "max_frames": max_frames,
        "filter_percent": filter_percent,
        "max_points_per_frame": max_points_per_frame,
        "hide_dynamic": hide_dynamic,
        "frames": frames,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def export_saved_results_to_player_assets(
    results_dir: Path,
    output_dir: Path,
    stride: int = 1,
    max_frames: int | None = None,
    filter_percent: float = 70.0,
    max_points_per_frame: int = 10_000,
    hide_dynamic: bool = False,
    write_ply: bool = True,
) -> dict[str, Any]:
    results = load_inference_results(results_dir)
    return export_results_to_player_assets(
        results,
        output_dir=output_dir,
        stride=stride,
        max_frames=max_frames,
        filter_percent=filter_percent,
        max_points_per_frame=max_points_per_frame,
        hide_dynamic=hide_dynamic,
        write_ply=write_ply,
    )


def load_manifest(assets_dir: Path) -> dict[str, Any]:
    with (assets_dir / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported player asset schema: {manifest.get('schema')}")
    return manifest


def load_frame_cloud(assets_dir: Path, frame: dict[str, Any]) -> FrameCloud:
    cloud_path = assets_dir / frame["point_cloud"]
    data = np.load(cloud_path)
    return FrameCloud(
        points=data["points"].astype(np.float32),
        colors=data["colors"].astype(np.uint8),
    )


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image at {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    out = panel.copy()
    cv2.putText(
        out,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_asset_panel(assets_dir: Path, frame: dict[str, Any], display_index: int, n_frames: int):
    rgb = _load_rgb(assets_dir / frame["rgb"])
    depth = _load_rgb(assets_dir / frame["depth"])
    mask = _load_rgb(assets_dir / frame["mask_overlay"])
    panels = [
        _label_panel(rgb, f"RGB {display_index + 1}/{n_frames}"),
        _label_panel(depth, "Depth"),
        _label_panel(mask, "Dynamic mask"),
    ]
    return np.concatenate(panels, axis=1)


def _make_polyline(points: np.ndarray):
    import pyvista as pv

    poly = pv.PolyData(points.astype(np.float32))
    if len(points) >= 2:
        lines = []
        for idx in range(len(points) - 1):
            lines.extend([2, idx, idx + 1])
        poly.lines = np.array(lines, dtype=np.int64)
    return poly


def _make_camera_lines(frames: Sequence[dict[str, Any]], camera_scale: float):
    import pyvista as pv

    points: list[np.ndarray] = []
    lines: list[int] = []

    def add_segment(a: np.ndarray, b: np.ndarray) -> None:
        start = len(points)
        points.extend([a, b])
        lines.extend([2, start, start + 1])

    for frame in frames:
        c2w = np.asarray(frame["camera_to_world"], dtype=np.float32)
        intrinsic = np.asarray(frame["intrinsic"], dtype=np.float32)
        corners_px = np.array(
            [
                [0.0, 0.0, 1.0],
                [float(frame["width"]), 0.0, 1.0],
                [float(frame["width"]), float(frame["height"]), 1.0],
                [0.0, float(frame["height"]), 1.0],
            ],
            dtype=np.float32,
        )
        rays = (np.linalg.inv(intrinsic) @ corners_px.T).T
        rays *= camera_scale / np.maximum(rays[:, 2:3], 1e-6)
        corners_world = (c2w[:3, :3] @ rays.T).T + c2w[:3, 3]
        center = c2w[:3, 3]
        for corner in corners_world:
            add_segment(center, corner)
        for idx in range(4):
            add_segment(corners_world[idx], corners_world[(idx + 1) % 4])

    poly = pv.PolyData(np.asarray(points, dtype=np.float32))
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def _add_cloud(plotter: Any, cloud: FrameCloud, name: str, point_size: int):
    import pyvista as pv

    if len(cloud.points) == 0:
        return None
    poly = pv.PolyData(cloud.points)
    poly["rgb"] = cloud.colors
    return plotter.add_mesh(
        poly,
        scalars="rgb",
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
        name=name,
    )


def play_player_assets(
    assets_dir: Path,
    play_fps: float = 4.0,
    camera_scale: float = 0.08,
    point_size: int = 5,
    background: str = "black",
) -> None:
    import pyvista as pv

    manifest = load_manifest(assets_dir)
    frames = manifest["frames"]
    if not frames:
        raise ValueError(f"No player frames found in {assets_dir}")

    centers = np.stack(
        [np.asarray(frame["camera_to_world"], dtype=np.float32)[:3, 3] for frame in frames],
        axis=0,
    )
    plotter: Any = pv.Plotter(window_size=[1200, 800], title="VGGT4D Asset Player")
    plotter.set_background(background)
    plotter.add_axes()
    plotter.add_mesh(_make_camera_lines(frames, camera_scale), color="white", line_width=1)
    plotter.add_mesh(_make_polyline(centers), color="yellow", line_width=3)

    frame_idx = 0
    playing = False
    last_tick = time.monotonic()
    current_actor = None
    marker_actor = None

    plotter.show(auto_close=False, interactive_update=True)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    def update_frame(new_idx: int) -> None:
        nonlocal current_actor, marker_actor, frame_idx
        frame_idx = new_idx % len(frames)
        if current_actor is not None:
            plotter.remove_actor(current_actor)
        if marker_actor is not None:
            plotter.remove_actor(marker_actor)
        frame = frames[frame_idx]
        current_actor = _add_cloud(
            plotter,
            load_frame_cloud(assets_dir, frame),
            name="current_frame_points",
            point_size=point_size,
        )
        marker_actor = plotter.add_mesh(
            pv.Sphere(radius=camera_scale * 0.12, center=centers[frame_idx]),
            color="red",
            name="current_camera",
        )
        panel = make_asset_panel(assets_dir, frame, frame_idx, len(frames))
        cv2.imshow(WINDOW_TITLE, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        plotter.update()

    try:
        update_frame(frame_idx)
        while True:
            plotter.update()
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), 83):
                update_frame(frame_idx + 1)
                playing = False
            elif key in (ord("p"), 81):
                update_frame(frame_idx - 1)
                playing = False
            elif key == ord(" "):
                playing = not playing

            if playing and time.monotonic() - last_tick >= 1.0 / play_fps:
                update_frame(frame_idx + 1)
                last_tick = time.monotonic()
    finally:
        cv2.destroyWindow(WINDOW_TITLE)
        plotter.close()
