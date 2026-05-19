"""I/O helpers for VGGT4D predictions.

All public helpers accept numpy arrays or torch tensors interchangeably; the
``_as_numpy`` adapter is the single coercion point so the rest of the module
stays free of tensor/array branching.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import open3d as o3d
import torch
from einops import rearrange
from evo.core.trajectory import PoseTrajectory3D
from jaxtyping import Float
from scipy.spatial.transform import Rotation
from torchvision.utils import save_image

ArrayLike = np.ndarray | torch.Tensor

C2W = Float[np.ndarray, "4 4"]
C2WBatch = Float[np.ndarray, "n_img 4 4"]
TumPose = Float[np.ndarray, "7"]  # [x, y, z, qw, qx, qy, qz]
TumPoseBatch = Float[np.ndarray, "n_img 7"]


def _as_numpy(array: ArrayLike) -> np.ndarray:
    """Coerce a torch tensor (any device) to a numpy array; pass arrays through."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def _save_per_frame_npy(data_dir: Path, array: ArrayLike, prefix: str) -> None:
    array = _as_numpy(array)
    for i in range(array.shape[0]):
        np.save(data_dir / f"{prefix}_{i:04d}.npy", array[i])


def _c2ws_to_tum_traj(
    c2ws: ArrayLike,
) -> tuple[TumPoseBatch, Float[np.ndarray, "n_img"]]:
    """Return TUM-format poses ``(N, 7)`` and synthetic timestamps ``(N,)``."""
    c2ws = _as_numpy(c2ws)
    tum_poses = np.stack([c2w_to_tumpose(c) for c in c2ws], axis=0)
    timestamps = np.arange(c2ws.shape[0], dtype=float)
    return tum_poses, timestamps


def save_dynamic_masks(data_dir: Path, masks: Iterable[ArrayLike]) -> None:
    for i, dynamic_mask in enumerate(masks):
        img_path = data_dir / f"dynamic_mask_{i:04d}.png"
        mask_uint8 = (_as_numpy(dynamic_mask) * 255).astype(np.uint8)
        cv2.imwrite(str(img_path), mask_uint8)


def save_intrinsic_txt(data_dir: Path, intrinsic: ArrayLike) -> None:
    intrinsic = rearrange(_as_numpy(intrinsic), "n_img h w -> n_img (h w)")
    np.savetxt(data_dir / "pred_intrinsics.txt", intrinsic, fmt="%f")


def save_rgb(data_dir: Path, images: torch.Tensor) -> None:
    n_img = images.shape[0]
    for i in range(n_img):
        save_image(images[i], data_dir / f"frame_{i:04d}.png")


def save_depth(data_dir: Path, depths: ArrayLike) -> None:
    _save_per_frame_npy(data_dir, depths, prefix="frame")


def save_depth_conf(data_dir: Path, conf: ArrayLike) -> None:
    _save_per_frame_npy(data_dir, conf, prefix="conf")


def c2w_to_tumpose(c2w: ArrayLike) -> TumPose:
    """Convert a 4x4 camera-to-world matrix to ``[x, y, z, qw, qx, qy, qz]``."""
    c2w = _as_numpy(c2w)
    xyz = c2w[:3, -1]
    qx, qy, qz, qw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    return np.concatenate([xyz, [qw, qx, qy, qz]])


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, (tuple, list)):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def to_tum_poses(c2ws: ArrayLike) -> list:
    """Return ``[tum_poses (N,7), timestamps (N,)]`` for downstream consumers."""
    tum_poses, timestamps = _c2ws_to_tum_traj(c2ws)
    return [tum_poses, timestamps]


def save_tum_poses(data_dir: Path, c2ws: ArrayLike) -> None:
    traj = make_traj(list(_c2ws_to_tum_traj(c2ws)))
    with (data_dir / "pred_traj.txt").open("w") as f:
        for i in range(traj.num_poses):
            xyz = " ".join(map(str, traj.positions_xyz[i]))
            wxyz = " ".join(map(str, traj.orientations_quat_wxyz[i]))
            f.write(f"{traj.timestamps[i]} {xyz} {wxyz}\n")


def load_tum_poses(data_dir: Path) -> C2WBatch:
    data = np.loadtxt(data_dir / "pred_traj.txt")
    pred_pose = np.zeros((data.shape[0], 4, 4))
    pred_pose[:, :3, 3] = data[:, 1:4]
    pred_pose[:, :3, :3] = Rotation.from_quat(
        data[:, 4:], scalar_first=True
    ).as_matrix()
    pred_pose[:, 3, 3] = 1.0
    return pred_pose.astype(np.float32)


def enlarge_seg_masks(data_dir: Path, kernel_size: int = 5) -> None:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for mask_path in sorted(data_dir.glob("dynamic_mask_*.png")):
        frame_id = int(mask_path.stem.split("_")[-1])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        enlarged = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(str(data_dir / f"enlarged_dynamic_mask_{frame_id:04d}.png"), enlarged)


def save_pts_ply(data_dir: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    ply_path = data_dir / "points.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ply_path.absolute()), pcd)


def save_vggt4d_result(
    data_dir: Path,
    cam2world: np.ndarray,
    intrinsic: np.ndarray,
    images: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    dyn_masks: np.ndarray | None = None,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "cam2world.npy", cam2world)
    np.save(data_dir / "intrinsic.npy", intrinsic)
    np.save(data_dir / "images.npy", images)
    np.save(data_dir / "depth.npy", depth)
    np.save(data_dir / "conf.npy", conf)
    if dyn_masks is not None:
        np.save(data_dir / "dyn_masks.npy", dyn_masks)


def load_vggt4d_result(data_dir: Path):
    cam2world = np.load(data_dir / "cam2world.npy")
    intrinsic = np.load(data_dir / "intrinsic.npy")
    images = np.load(data_dir / "images.npy")
    depth = np.load(data_dir / "depth.npy")
    conf = np.load(data_dir / "conf.npy")
    dyn_masks_path = data_dir / "dyn_masks.npy"
    dyn_masks = np.load(dyn_masks_path) if dyn_masks_path.exists() else None
    return cam2world, intrinsic, images, depth, conf, dyn_masks
