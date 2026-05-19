"""Reusable inference pipeline for VGGT4D.

The pipeline is split into the three stages described in the paper:

1. Predict per-frame depth and a (coarse) dynamic map.
2. Refine the camera extrinsics by masking out dynamic regions.
3. Refine the dynamic mask itself using the geometrically-consistent extrinsics.

The CLI in ``demo_vggt4d.py`` is a thin wrapper around :class:`VGGT4DPipeline`;
notebooks and tests can also import it directly without any side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from vggt.utils.load_fn import load_and_preprocess_images
from vggt4d.masks.dynamic_mask import (
    adaptive_multiotsu_variance,
    cluster_attention_maps,
    extract_dyn_map,
)
from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
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
PATCH_SIZE = 14


@dataclass
class SceneResult:
    """Numpy outputs of one VGGT4D scene; everything is host-side / GPU-free."""

    images: torch.Tensor
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    cam2world: np.ndarray
    depth: np.ndarray
    depth_conf: np.ndarray
    refined_dyn_mask: torch.Tensor

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


def load_scene_images(scene_dir: Path, device: torch.device) -> torch.Tensor | None:
    """Load all jpg/png images in ``scene_dir``; return ``None`` if empty."""
    image_paths = sorted(
        [*scene_dir.glob("*.jpg"), *scene_dir.glob("*.png")]
    )
    if not image_paths:
        return None
    images = load_and_preprocess_images([str(p) for p in image_paths])
    return images.to(device)


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
        self, images: torch.Tensor
    ) -> tuple[dict, torch.Tensor]:
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
        self, images: torch.Tensor, dyn_masks: torch.Tensor
    ) -> dict:
        """Re-run inference with dyn_masks to obtain robust camera extrinsics."""
        predictions, _, _, _ = inference(
            self.model, images, dyn_masks.to(self.device)
        )
        torch.cuda.empty_cache()
        return predictions

    def _stage3_refine_dyn_mask(
        self,
        images: torch.Tensor,
        depths: np.ndarray,
        dyn_masks: torch.Tensor,
        cam2world: np.ndarray,
        intrinsic: np.ndarray,
    ) -> torch.Tensor:
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

    def run_scene(self, images: torch.Tensor) -> SceneResult:
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
