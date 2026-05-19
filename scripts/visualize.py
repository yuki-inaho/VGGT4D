"""End-to-end visualization CLI.

Pipeline:
    1. Load a scene (image dir or video).
    2. Run :class:`vggt4d.VGGT4DInference`.
    3. Either spawn a rerun viewer interactively (``--mode viewer``) or write
       a ``.rrd`` file plus a static screenshot driven by the Playwright CLI
       (``--mode rrd`` / ``--mode screenshot``).

Examples::

    # Interactive viewer (requires a display)
    uv run python -m scripts.visualize --input ./datasets/test_scene/scene1

    # Headless: write an .rrd file only
    uv run python -m scripts.visualize --input ./datasets/test_scene/scene1 \
        --mode rrd --rrd outputs/scene1.rrd

    # Headless: write .rrd and grab a screenshot of the rerun web viewer
    uv run python -m scripts.visualize --input ./datasets/test_scene/scene1 \
        --mode screenshot --rrd outputs/scene1.rrd --screenshot outputs/scene1.png
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Literal, TextIO

from vggt4d.inference import VGGT4DInference
from vggt4d.preprocess import load_images, read_images_from_video
from vggt4d.visualize import save_results_to_rrd, visualize_results

Mode = Literal["viewer", "rrd", "screenshot"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCREENSHOT_VIEWPORT = "1600,900"
SCREENSHOT_TIMEOUT_MS = "60000"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _image_paths_from_dir(input_path: Path) -> list[Path]:
    return sorted(
        p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _load_inputs(input_path: Path, downsample: int):
    if input_path.is_dir():
        paths = _image_paths_from_dir(input_path)
        if not paths:
            raise SystemExit(f"No images found under {input_path}")
        return load_images([str(p) for p in paths], downsample_factor=downsample)
    if input_path.is_file():
        return read_images_from_video(str(input_path), downsample_factor=downsample)
    raise SystemExit(f"Input path does not exist: {input_path}")


def _run_inference(images, checkpoint: Path):
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    model = VGGT4DInference(checkpoint_path=checkpoint)
    return model(images)


def _viewer_url(web_port: int, grpc_port: int) -> str:
    viewer_data_url = f"rerun+http://localhost:{grpc_port}/proxy"
    encoded_url = urllib.parse.quote(viewer_data_url, safe="")
    return f"http://127.0.0.1:{web_port}?url={encoded_url}"


def _read_log_tail(log_file: TextIO, limit: int = 4000) -> str:
    log_file.flush()
    log_file.seek(0)
    return log_file.read()[-limit:].strip()


def _wait_for_tcp_port(
    host: str,
    port: int,
    timeout_seconds: float,
    proc: subprocess.Popen | None = None,
    log_file: TextIO | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            detail = f"\n{_read_log_tail(log_file)}" if log_file is not None else ""
            raise RuntimeError(f"rerun exited before serving port {port}{detail}")
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _playwright_cli_screenshot(url: str, png_path: Path, wait_seconds: float) -> None:
    playwright = shutil.which("playwright")
    if playwright is None:
        raise RuntimeError("playwright CLI not found; run `uv sync --group dev` first")
    try:
        subprocess.run(
            [
                playwright,
                "screenshot",
                "--browser",
                "chromium",
                "--viewport-size",
                SCREENSHOT_VIEWPORT,
                "--wait-for-timeout",
                str(int(wait_seconds * 1000)),
                "--timeout",
                SCREENSHOT_TIMEOUT_MS,
                url,
                str(png_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Playwright CLI screenshot failed") from exc


def _screenshot_rrd(
    rrd_path: Path,
    png_path: Path,
    wait_seconds: float,
    web_port: int = 9090,
    grpc_port: int = 9876,
    serve_timeout_seconds: float = 30.0,
) -> None:
    """Spawn ``rerun --serve-web <rrd>`` and screenshot it via Playwright CLI.

    The web viewer is hosted at ``http://127.0.0.1:<web_port>`` and is told to
    connect to the in-process gRPC proxy at ``rerun+http://localhost:<grpc_port>/proxy``.
    """
    if not rrd_path.exists():
        raise FileNotFoundError(rrd_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    rerun = shutil.which("rerun")
    if rerun is None:
        raise RuntimeError("rerun CLI not found; run `uv sync --group dev` first")

    with tempfile.TemporaryFile("w+", encoding="utf-8") as rerun_log:
        proc = subprocess.Popen(
            [
                rerun,
                "--serve-web",
                "--web-viewer-port",
                str(web_port),
                "--port",
                str(grpc_port),
                str(rrd_path),
            ],
            stdout=rerun_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_tcp_port(
                "127.0.0.1",
                web_port,
                serve_timeout_seconds,
                proc=proc,
                log_file=rerun_log,
            )
            _playwright_cli_screenshot(_viewer_url(web_port, grpc_port), png_path, wait_seconds)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="Image directory or video file")
    parser.add_argument(
        "--mode",
        choices=("viewer", "rrd", "screenshot"),
        default="rrd",
        help="viewer: spawn rerun viewer; rrd: save .rrd only; screenshot: also save PNG via Playwright",
    )
    parser.add_argument("--rrd", type=Path, default=Path("outputs/vggt4d.rrd"))
    parser.add_argument("--screenshot", type=Path, default=Path("outputs/vggt4d.png"))
    parser.add_argument(
        "--downsample", type=_positive_int, default=1, help="Frame downsample factor"
    )
    parser.add_argument("--filter-percent", type=float, default=50.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./ckpts/model_tracker_fixed_e20.pt"),
    )
    parser.add_argument(
        "--wait",
        type=_non_negative_float,
        default=3.0,
        help="Seconds to wait for the viewer to render before screenshotting",
    )
    parser.add_argument("--web-viewer-port", type=_positive_int, default=9090)
    parser.add_argument("--grpc-port", type=_positive_int, default=9876)
    parser.add_argument(
        "--serve-timeout",
        type=_non_negative_float,
        default=30.0,
        help="Seconds to wait for rerun --serve-web to start accepting connections",
    )
    args = parser.parse_args(argv)

    images = _load_inputs(args.input, args.downsample)
    print(f"Loaded {len(images)} frames from {args.input}")

    results = _run_inference(images, args.checkpoint)
    print(f"Inference produced {len(results)} per-frame results")

    if args.mode == "viewer":
        visualize_results(results, filter_percent=args.filter_percent, spawn=True)
        time.sleep(args.wait)  # keep process alive briefly for the viewer to flush
        return 0

    save_results_to_rrd(results, args.rrd, filter_percent=args.filter_percent)
    print(f"Wrote {args.rrd}")

    if args.mode == "screenshot":
        _screenshot_rrd(
            args.rrd,
            args.screenshot,
            wait_seconds=args.wait,
            web_port=args.web_viewer_port,
            grpc_port=args.grpc_port,
            serve_timeout_seconds=args.serve_timeout,
        )
        print(f"Wrote {args.screenshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
