import numpy as np
import pytest
import sys
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.utils.model_utils import inference
from vggt.utils.load_fn import load_and_preprocess_images_square


CKPT_REPO = "facebook/VGGT_tracker_fixed"
CKPT_FILE = "model_tracker_fixed_e20.pt"


@pytest.fixture(scope="session")
def device():
    """VGGT4DはCUDA専用実装なので、GPUがない環境ではスキップ。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for VGGT4D inference")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def checkpoint_path():
    """事前学習済みウェイトをckpts配下に用意する。"""
    ckpt_dir = Path("ckpts")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / CKPT_FILE
    if not ckpt_path.exists():
        ckpt_path = Path(
            hf_hub_download(
                repo_id=CKPT_REPO,
                filename=CKPT_FILE,
                local_dir=ckpt_dir,
                local_dir_use_symlinks=False,
            )
        )
    return ckpt_path


@pytest.fixture(scope="session")
def model(device, checkpoint_path):
    """VGGT4D本体をロードして推論モードにセット。"""
    model = VGGTFor4D().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@pytest.fixture(scope="session")
def sample_images():
    """
    簡易なサンプル画像を生成し、モデル入力と同じTensorに前処理する。
    サイズは56x56 (14で割り切れる) として計算量を抑える。
    """
    tmp_dir = Path("tests") / "data" / "sample_scene"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for idx, color in enumerate([(60, 60, 220), (120, 30, 180)]):
        img = Image.new("RGB", (56, 56), color=color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 45, 45], outline=(255, 255, 0), width=2)
        path = tmp_dir / f"frame_{idx}.png"
        img.save(path)
        img_paths.append(str(path))

    images, _ = load_and_preprocess_images_square(img_paths, target_size=56)
    return images


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
