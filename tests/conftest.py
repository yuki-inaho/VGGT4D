from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "ckpts" / "model_tracker_fixed_e20.pt"


def _has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.init()
        torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return True


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason=f"checkpoint missing at {CHECKPOINT}"
)


@pytest.fixture(scope="session")
def synthetic_scene_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A reusable 3-frame synthetic scene with a moving square."""
    root = tmp_path_factory.mktemp("scenes") / "scene1"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        base = rng.integers(0, 255, (518, 518, 3), dtype=np.uint8)
        base[200 + i * 10 : 260 + i * 10, 200:260] = 255
        Image.fromarray(base).save(root / f"image{i:03d}.png")
    return root.parent
