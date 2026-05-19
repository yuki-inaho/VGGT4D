"""Tests for vggt4d.preprocess (numpy-in image helpers)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vggt4d.preprocess import (
    PATCH_SIZE,
    TARGET_SIZE,
    load_images,
    preprocess_images,
    read_images_from_video,
)


def _bgr(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_preprocess_crop_mode_returns_target_height():
    out = preprocess_images([_bgr(700, 1000)], mode="crop")
    assert out.dtype == torch.float32
    assert out.shape[0] == 1 and out.shape[1] == 3
    assert out.shape[3] == TARGET_SIZE
    assert out.shape[2] <= TARGET_SIZE  # cropped to target


def test_preprocess_pad_mode_returns_square():
    out = preprocess_images([_bgr(700, 1000)], mode="pad")
    assert out.shape[2] == TARGET_SIZE and out.shape[3] == TARGET_SIZE


def test_preprocess_batches_mixed_shapes():
    out = preprocess_images([_bgr(700, 1000), _bgr(800, 600)], mode="crop")
    assert out.shape[0] == 2
    # both frames must share the same H/W after padding-to-max
    assert out.shape[1:] == out[0].shape


def test_preprocess_rejects_empty_list():
    with pytest.raises(ValueError):
        preprocess_images([], mode="crop")


def test_preprocess_rejects_invalid_mode():
    with pytest.raises(ValueError):
        preprocess_images([_bgr(700, 1000)], mode="bogus")  # type: ignore[arg-type]


def test_preprocess_resize_is_patch_aligned():
    out = preprocess_images([_bgr(700, 1000)], mode="crop")
    # cropping happens AFTER resize; only the resize step must be patch-aligned.
    assert out.shape[3] % PATCH_SIZE == 0


def test_io_helpers_reject_non_positive_downsample_factor():
    with pytest.raises(ValueError, match="downsample_factor"):
        load_images([], downsample_factor=0)
    with pytest.raises(ValueError, match="downsample_factor"):
        read_images_from_video("missing.mp4", downsample_factor=0)
