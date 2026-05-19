"""Unit tests for scripts.visualize helper behavior."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.visualize import (
    _image_paths_from_dir,
    _load_inputs,
    _positive_int,
    _viewer_url,
    _wait_for_tcp_port,
)


def test_image_paths_from_dir_supports_common_case_variants(tmp_path: Path):
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "b.JPG")
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "a.jpeg")
    (tmp_path / "notes.txt").write_text("ignore me")

    assert [p.name for p in _image_paths_from_dir(tmp_path)] == ["a.jpeg", "b.JPG"]


def test_load_inputs_rejects_directory_without_images(tmp_path: Path):
    with pytest.raises(SystemExit, match="No images found"):
        _load_inputs(tmp_path, downsample=1)


def test_positive_int_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")


def test_viewer_url_quotes_rerun_proxy_url():
    url = _viewer_url(web_port=19090, grpc_port=19876)
    assert url == "http://127.0.0.1:19090?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A19876%2Fproxy"


def test_wait_for_tcp_port_detects_open_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        _wait_for_tcp_port("127.0.0.1", port, timeout_seconds=1.0)
