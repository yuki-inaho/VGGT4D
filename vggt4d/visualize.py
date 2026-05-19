"""Rerun visualization for VGGT4DInference results.

The blueprint mirrors the official VGGT inference reference
(``vggt-pytorch-inference/vggt_inference/visualize.py``) with one extra panel
for the VGGT4D dynamic mask.

For headless / CI use, prefer :func:`save_results_to_rrd` which writes a
``.rrd`` file without launching the viewer; the rerun web viewer can then be
spawned separately and driven by tools like Playwright.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from vggt4d.inference import InferenceResult

DEFAULT_APP_ID = "vggt4d"
PointCloud = tuple[np.ndarray, np.ndarray]


def _default_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="body/pose", contents="body/**"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="body/cam/image"),
                rrb.Spatial2DView(origin="body/cam/depth_map"),
                rrb.Spatial2DView(origin="body/cam/dynamic_mask"),
            ),
        )
    )


def draw_pose(transform: np.ndarray, name: str, static: bool = False) -> None:
    rr.log(
        name,
        rr.Arrows3D(
            origins=[0, 0, 0],
            vectors=[[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=[0.001, 0.001, 0.001],
        ),
        static=static,
    )
    rr.log(
        name,
        rr.Transform3D(
            translation=transform[:3, 3],
            mat3x3=transform[:3, :3],
        ),
        static=static,
    )


def _filter_by_confidence(
    points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    dyn_mask: np.ndarray,
    filter_percent: float,
) -> tuple[PointCloud, PointCloud, np.ndarray]:
    """Return dynamic-aware point clouds plus the 2D confidence mask."""
    threshold = np.percentile(conf, filter_percent)
    keep_2d = conf >= threshold
    keep_flat = keep_2d.reshape(-1)
    pts = points.reshape(-1, 3)[keep_flat]
    rgb = colors.reshape(-1, 3)[keep_flat]
    dyn_flat = dyn_mask.reshape(-1).astype(bool)
    static_keep = keep_flat & ~dyn_flat
    static_pts = points.reshape(-1, 3)[static_keep]
    static_rgb = colors.reshape(-1, 3)[static_keep]
    return (pts, rgb), (static_pts, static_rgb), keep_2d


def _validate_filter_percent(filter_percent: float) -> None:
    if not 0.0 <= filter_percent <= 100.0:
        raise ValueError("filter_percent must be between 0 and 100")


def _log_frame(idx: int, result: InferenceResult, filter_percent: float) -> None:
    rr.set_time("frame", sequence=idx)

    world_to_camera = np.eye(4)
    world_to_camera[:3, :4] = result.extrinsic
    camera_to_world = np.linalg.inv(world_to_camera)

    rr.log(
        "body/cam",
        rr.Pinhole(
            image_from_camera=result.intrinsic,
            width=result.width,
            height=result.height,
            image_plane_distance=0.02,
        ),
    )
    rr.log(
        "body/cam",
        rr.Transform3D(
            translation=camera_to_world[:3, 3],
            mat3x3=camera_to_world[:3, :3],
        ),
    )

    draw_pose(camera_to_world, f"body/pose{idx}", static=True)
    draw_pose(camera_to_world, "body/pose")

    (pts, rgb), (static_pts, static_rgb), keep_2d = _filter_by_confidence(
        points=result.point_map_by_unprojection,
        colors=result.image,
        conf=result.depth_conf,
        dyn_mask=result.dynamic_mask,
        filter_percent=filter_percent,
    )

    rr.log(f"body/points{idx}", rr.Points3D(pts, colors=rgb, radii=0.0003), static=True)
    rr.log(
        f"body/static_points{idx}",
        rr.Points3D(static_pts, colors=static_rgb, radii=0.0003),
        static=True,
    )

    depth_for_viz = result.depth_map.copy()
    depth_for_viz[~keep_2d] = 0
    rr.log("body/cam/image", rr.Image(result.image))
    rr.log("body/cam/depth_map", rr.DepthImage(depth_for_viz))
    rr.log(
        "body/cam/dynamic_mask",
        rr.SegmentationImage(result.dynamic_mask.astype(np.uint8)),
    )


def visualize_results(
    results: Sequence[InferenceResult],
    filter_percent: float = 50.0,
    app_id: str = DEFAULT_APP_ID,
    spawn: bool = True,
) -> None:
    """Stream VGGT4D results to a (possibly spawned) rerun viewer."""
    _validate_filter_percent(filter_percent)
    rr.init(app_id, spawn=spawn)
    rr.send_blueprint(_default_blueprint())
    for i, result in enumerate(results):
        _log_frame(i, result, filter_percent)


def save_results_to_rrd(
    results: Sequence[InferenceResult],
    rrd_path: Path,
    filter_percent: float = 50.0,
    app_id: str = DEFAULT_APP_ID,
) -> Path:
    """Write the same visualization to an ``.rrd`` file without opening a viewer."""
    _validate_filter_percent(filter_percent)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init(app_id, spawn=False)
    rr.save(str(rrd_path), default_blueprint=_default_blueprint())
    for i, result in enumerate(results):
        _log_frame(i, result, filter_percent)
    rr.disconnect()
    return rrd_path
