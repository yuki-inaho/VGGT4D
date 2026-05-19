"""Image preprocessing helpers (numpy-array centric).

Mirrors the official VGGT inference API
(`vggt-pytorch-inference/vggt_inference/preprocess.py`) so that callers can
move between the two repos with the same call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np
import torch
from jaxtyping import Float, UInt8

PreprocessMode = Literal["crop", "pad"]

# Resolution contract:
# - The pretrained VGGT/VGGT4D path follows the official 518px preprocessing convention.
# - Patch embedding is shape-flexible, but both H and W must be divisible by 14.
# - crop mode fixes W=518 and keeps H <= 518 after center-cropping when needed.
# - pad mode preserves the full image by padding to 518x518.
TARGET_SIZE = 518
PATCH_SIZE = 14


def _validate_downsample_factor(downsample_factor: int) -> None:
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be greater than or equal to 1")


def load_images(
    image_paths: Iterable[str | Path], downsample_factor: int = 1
) -> list[UInt8[np.ndarray, "h w 3"]]:
    """Load BGR images from disk via OpenCV (matches official ref)."""
    _validate_downsample_factor(downsample_factor)
    images: list[np.ndarray] = []
    paths = list(image_paths)
    for path in paths[::downsample_factor]:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image at {path}")
        images.append(img)
    return images


def read_images_from_video(
    video_path: str | Path, downsample_factor: int = 1
) -> list[UInt8[np.ndarray, "h w 3"]]:
    """Read BGR frames from a video file (matches official ref)."""
    _validate_downsample_factor(downsample_factor)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    frames: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % downsample_factor == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def _resize_single(
    img_bgr: UInt8[np.ndarray, "h w 3"], mode: PreprocessMode
) -> Float[torch.Tensor, "3 h w"]:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if mode == "pad":
        if w >= h:
            new_w = TARGET_SIZE
            new_h = round(h * (new_w / w) / PATCH_SIZE) * PATCH_SIZE
        else:
            new_h = TARGET_SIZE
            new_w = round(w * (new_h / h) / PATCH_SIZE) * PATCH_SIZE
    else:  # crop
        new_w = TARGET_SIZE
        new_h = round(h * (new_w / w) / PATCH_SIZE) * PATCH_SIZE

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    if mode == "crop" and new_h > TARGET_SIZE:
        start = (new_h - TARGET_SIZE) // 2
        tensor = tensor[:, start : start + TARGET_SIZE, :]
    elif mode == "pad":
        h_pad = TARGET_SIZE - tensor.shape[1]
        w_pad = TARGET_SIZE - tensor.shape[2]
        if h_pad > 0 or w_pad > 0:
            tensor = torch.nn.functional.pad(
                tensor,
                (w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2),
                mode="constant",
                value=1.0,
            )
    return tensor


def _pad_to_max(
    tensors: list[Float[torch.Tensor, "3 h w"]],
) -> list[Float[torch.Tensor, "3 h w"]]:
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded: list[torch.Tensor] = []
    for t in tensors:
        h_pad = max_h - t.shape[1]
        w_pad = max_w - t.shape[2]
        if h_pad or w_pad:
            t = torch.nn.functional.pad(
                t,
                (w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2),
                mode="constant",
                value=1.0,
            )
        padded.append(t)
    return padded


def preprocess_images(
    images: list[UInt8[np.ndarray, "h w 3"]], mode: PreprocessMode = "crop"
) -> Float[torch.Tensor, "n_img 3 h w"]:
    """Preprocess BGR images into a batched ``(N, 3, H, W)`` tensor in [0, 1].

    This is the numpy-in / tensor-out counterpart of
    ``vggt.utils.load_fn.load_and_preprocess_images`` (which loads from paths).
    """
    if not images:
        raise ValueError("At least 1 image is required")
    if mode not in ("crop", "pad"):
        raise ValueError("Mode must be either 'crop' or 'pad'")

    processed = [_resize_single(img, mode) for img in images]
    if len({(t.shape[1], t.shape[2]) for t in processed}) > 1:
        processed = _pad_to_max(processed)
    return torch.stack(processed)
