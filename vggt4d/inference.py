"""Official-VGGT-style inference API for VGGT4D.

Mirrors the surface area of
``vggt-pytorch-inference/vggt_inference/vggt_inference.py`` (VGGTInference +
InferenceResult), so call sites stay portable across the two projects. The
added value over plain VGGT is the per-frame ``dynamic_mask`` produced by
VGGT4D's three-stage pipeline.

Typical use::

    from vggt4d.inference import VGGT4DInference
    from vggt4d.preprocess import load_images

    model = VGGT4DInference()
    images = load_images(["scene/img001.jpg", "scene/img002.jpg"])
    results = model(images)              # list[InferenceResult]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, UInt8

from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt4d.pipeline import DEFAULT_CHECKPOINT, VGGT4DPipeline, autodetect_device
from vggt4d.preprocess import PreprocessMode, preprocess_images


@dataclass
class InferenceResult:
    """Per-frame VGGT4D output.

    Fields image / extrinsic / intrinsic / depth_map / depth_conf /
    point_map_by_unprojection mirror
    ``vggt_inference.vggt_inference.InferenceResult`` exactly.
    ``dynamic_mask`` is the VGGT4D-specific addition.
    """

    image: UInt8[np.ndarray, "h w 3"]
    width: int
    height: int
    extrinsic: Float[np.ndarray, "3 4"]
    intrinsic: Float[np.ndarray, "3 3"]
    depth_map: Float[np.ndarray, "h w"]
    depth_conf: Float[np.ndarray, "h w"]
    point_map_by_unprojection: Float[np.ndarray, "h w 3"]
    dynamic_mask: Bool[np.ndarray, "h w"]


class VGGT4DInference(nn.Module):
    """``nn.Module`` wrapper presenting the official VGGT API for VGGT4D."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or autodetect_device()
        self.pipeline = VGGT4DPipeline(checkpoint_path=checkpoint_path, device=self.device)

    @property
    def model(self) -> nn.Module:
        """Underlying VGGT4D backbone (for parity with VGGTInference.model)."""
        return self.pipeline.model

    @torch.inference_mode()
    def forward(
        self,
        input_images: list[UInt8[np.ndarray, "h w 3"]],
        mode: PreprocessMode = "crop",
    ) -> list[InferenceResult]:
        images = preprocess_images(input_images, mode=mode).to(self.device)
        scene = self.pipeline.run_scene(images)
        return _scene_result_to_inference_results(scene, images)


def _scene_result_to_inference_results(scene, images) -> list[InferenceResult]:
    n_img = images.shape[0]
    height, width = images.shape[-2:]

    point_map = unproject_depth_map_to_point_map(
        scene.depth[..., None], scene.extrinsic, scene.intrinsic
    )
    masks = scene.refined_dyn_mask.detach().cpu().numpy().astype(bool)
    rgb_uint8 = (images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    results: list[InferenceResult] = []
    for i in range(n_img):
        results.append(
            InferenceResult(
                image=rgb_uint8[i],
                width=int(width),
                height=int(height),
                extrinsic=scene.extrinsic[i],
                intrinsic=scene.intrinsic[i],
                depth_map=scene.depth[i],
                depth_conf=scene.depth_conf[i],
                point_map_by_unprojection=point_map[i],
                dynamic_mask=masks[i],
            )
        )
    return results
