from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from tests.conftest import CHECKPOINT, needs_checkpoint, needs_cuda
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.utils.model_utils import inference


@pytest.fixture
def device():
    """VGGT4DはCUDA専用実装なので、GPUがない環境ではスキップ。"""
    return torch.device("cuda")


@pytest.fixture
def model(device):
    """VGGT4D本体をロードして推論モードにセット。"""
    model = VGGTFor4D().to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@pytest.fixture
def sample_images(tmp_path: Path):
    """
    簡易なサンプル画像を生成し、モデル入力と同じTensorに前処理する。
    サイズは56x56 (14で割り切れる) として計算量を抑える。
    """
    img_paths = []
    for idx, color in enumerate([(60, 60, 220), (120, 30, 180)]):
        img = Image.new("RGB", (56, 56), color=color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 45, 45], outline=(255, 255, 0), width=2)
        path = tmp_path / f"frame_{idx}.png"
        img.save(path)
        img_paths.append(str(path))

    images, _ = load_and_preprocess_images_square(img_paths, target_size=56)
    return images


@pytest.mark.smoke
@needs_cuda
@needs_checkpoint
def test_inference_smoke(model, device, sample_images):
    """
    事前学習済みVGGT4Dで最小構成のシーンを推論し、
    主な出力が妥当な形状で得られることを確認するスモークテスト。
    """
    images = sample_images.to(device)
    preds, qk_dict, enc_feat, agg_tokens = inference(model, images)

    n_img, _, height, width = sample_images.shape

    assert preds["depth"].shape == (n_img, height, width)
    assert preds["depth_conf"].shape == (n_img, height, width)
    assert preds["intrinsic"].shape == (n_img, 3, 3)
    assert preds["extrinsic"].shape == (n_img, 3, 4)
    assert preds["cam2world"].shape == (n_img, 4, 4)

    # 代表的な値がNaNになっていないことを確認
    assert not np.isnan(preds["depth"]).any()
    assert not np.isnan(preds["intrinsic"]).any()
    assert not np.isnan(preds["extrinsic"]).any()

    # 中間特徴がシーン枚数と揃っていることを確認
    assert enc_feat.shape[0] == n_img
    assert "global_q" in qk_dict and "frame_q" in qk_dict
