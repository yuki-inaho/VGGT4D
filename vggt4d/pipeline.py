"""Reusable inference pipeline for VGGT4D.

The pipeline is split into the three stages described in the paper:

1. Predict per-frame depth and a (coarse) dynamic map.
2. Refine the camera extrinsics by masking out dynamic regions.
3. Refine the dynamic mask itself using the geometrically-consistent extrinsics.

The CLI in ``demo_vggt4d.py`` is a thin wrapper around :class:`VGGT4DPipeline`;
notebooks and tests can also import it directly without any side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from vggt.utils.load_fn import load_and_preprocess_images
from vggt4d.masks.dynamic_mask import (
    adaptive_multiotsu_variance,
    cluster_attention_maps,
    extract_dyn_map,
)
from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.preprocess import PreprocessMode
from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt4d.utils.store import (
    save_depth,
    save_depth_conf,
    save_dynamic_masks,
    save_intrinsic_txt,
    save_rgb,
    save_tum_poses,
)

DEFAULT_CHECKPOINT = Path("./ckpts/model_tracker_fixed_e20.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PATCH_SIZE = 14
_NATURAL_PART_RE = re.compile(r"(\d+)")


@dataclass
class SceneResult:
    """Numpy outputs of one VGGT4D scene; everything is host-side / GPU-free.

    Shape conventions (``N`` = number of frames, ``H``/``W`` = image height/width):
        * ``images``        — ``(N, 3, H, W)`` torch tensor on inference device
        * ``intrinsic``     — ``(N, 3, 3)``
        * ``extrinsic``     — ``(N, 3, 4)``
        * ``cam2world``     — ``(N, 4, 4)``
        * ``depth``         — ``(N, H, W)``
        * ``depth_conf``    — ``(N, H, W)``
        * ``refined_dyn_mask`` — ``(N, H, W)`` boolean tensor
    """

    images: Float[torch.Tensor, "n_img 3 h w"]
    intrinsic: Float[np.ndarray, "n_img 3 3"]
    extrinsic: Float[np.ndarray, "n_img 3 4"]
    cam2world: Float[np.ndarray, "n_img 4 4"]
    depth: Float[np.ndarray, "n_img h w"]
    depth_conf: Float[np.ndarray, "n_img h w"]
    refined_dyn_mask: Bool[torch.Tensor, "n_img h w"]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_intrinsic_txt(output_dir, self.intrinsic)
        save_rgb(output_dir, self.images)
        save_depth(output_dir, self.depth)
        save_depth_conf(output_dir, self.depth_conf)
        save_tum_poses(output_dir, self.cam2world)
        save_dynamic_masks(output_dir, self.refined_dyn_mask)


def autodetect_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _natural_path_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = _NATURAL_PART_RE.split(path.name)
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower()) for part in parts
    )


def list_scene_image_paths(scene_dir: Path) -> list[Path]:
    """Return naturally sorted jpg/jpeg/png frame paths from one image directory."""
    return sorted(
        (
            p
            for p in scene_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=_natural_path_key,
    )


def load_image_paths(
    image_paths: Sequence[str | Path],
    device: torch.device,
    mode: PreprocessMode = "crop",
) -> Float[torch.Tensor, "n_img 3 h w"]:
    """Load explicit image paths with the same preprocessing used by the pipeline."""
    paths = [str(p) for p in image_paths]
    if not paths:
        raise ValueError("At least 1 image is required")
    images = load_and_preprocess_images(paths, mode=mode)
    return images.to(device)


def load_scene_images(
    scene_dir: Path, device: torch.device, mode: PreprocessMode = "crop"
) -> Float[torch.Tensor, "n_img 3 h w"] | None:
    """Load all jpg/png images in ``scene_dir``; return ``None`` if empty."""
    image_paths = list_scene_image_paths(scene_dir)
    if not image_paths:
        return None
    return load_image_paths(image_paths, device=device, mode=mode)


class VGGT4DPipeline:
    """Three-stage VGGT4D inference pipeline."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or autodetect_device()
        self.model = self._build_model(checkpoint_path, self.device)

    @staticmethod
    def _build_model(checkpoint_path: Path, device: torch.device) -> VGGTFor4D:
        model = VGGTFor4D()
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)

    def _stage1_depth_and_dyn(
        self, images: Float[torch.Tensor, "n_img 3 h w"]
    ) -> tuple[dict, Bool[torch.Tensor, "n_img h w"]]:
        """Run base inference and derive a binary dynamic mask."""
        predictions, qk_dict, enc_feat, agg_tokens_list = inference(self.model, images)
        del agg_tokens_list

        n_img, _, h_img, w_img = images.shape
        h_tok, w_tok = h_img // PATCH_SIZE, w_img // PATCH_SIZE

        qk_dict = organize_qk_dict(qk_dict, n_img)
        dyn_maps = extract_dyn_map(qk_dict, images)
        feat_map = rearrange(
            enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok
        )
        norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)

        upsampled = F.interpolate(
            rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"),
            size=(h_img, w_img),
            mode="bilinear",
            align_corners=False,
        )
        upsampled = rearrange(upsampled, "n_img 1 h w -> n_img h w")
        threshold = adaptive_multiotsu_variance(upsampled.cpu().numpy())
        dyn_masks = upsampled > threshold

        del enc_feat, feat_map
        torch.cuda.empty_cache()
        return predictions, dyn_masks

    def _stage2_refine_extrinsics(
        self,
        images: Float[torch.Tensor, "n_img 3 h w"],
        dyn_masks: Bool[torch.Tensor, "n_img h w"],
    ) -> dict:
        """Re-run inference with dyn_masks to obtain robust camera extrinsics."""
        predictions, _, _, _ = inference(
            self.model, images, dyn_masks.to(self.device)
        )
        torch.cuda.empty_cache()
        return predictions

    def _stage3_refine_dyn_mask(
        self,
        images: Float[torch.Tensor, "n_img 3 h w"],
        depths: Float[np.ndarray, "n_img h w"],
        dyn_masks: Bool[torch.Tensor, "n_img h w"],
        cam2world: Float[np.ndarray, "n_img 4 4"],
        intrinsic: Float[np.ndarray, "n_img 3 3"],
    ) -> Bool[torch.Tensor, "n_img h w"]:
        refiner = RefineDynMask(
            images,
            torch.tensor(depths).to(self.device),
            dyn_masks.to(self.device),
            torch.tensor(cam2world).float().to(self.device),
            torch.tensor(intrinsic).to(self.device),
            self.device,
        )
        refined = refiner.refine_masks()
        del refiner
        torch.cuda.empty_cache()
        return refined

    def run_scene(
        self, images: Float[torch.Tensor, "n_img 3 h w"]
    ) -> SceneResult:
        predictions1, dyn_masks = self._stage1_depth_and_dyn(images)
        predictions2 = self._stage2_refine_extrinsics(images, dyn_masks)

        intrinsic = predictions1["intrinsic"]
        cam2world = predictions2["cam2world"]
        depth = predictions1["depth"]
        depth_conf = predictions1["depth_conf"]

        refined_mask = self._stage3_refine_dyn_mask(
            images, depth, dyn_masks, cam2world, intrinsic
        )

        return SceneResult(
            images=images,
            intrinsic=intrinsic,
            extrinsic=predictions2["extrinsic"],
            cam2world=cam2world,
            depth=depth,
            depth_conf=depth_conf,
            refined_dyn_mask=refined_mask,
        )
