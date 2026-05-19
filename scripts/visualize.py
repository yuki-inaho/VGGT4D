"""End-to-end visualization CLI.

Pipeline:
    1. Load a scene (image dir or video).
    2. Run :class:`vggt4d.VGGT4DInference`.
    3. Either spawn a rerun viewer interactively (``--mode viewer``) or write
       a ``.rrd`` file plus a static screenshot driven by the Playwright
       browser (``--mode rrd`` / ``--mode screenshot``).

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
import time
from pathlib import Path
from typing import Literal

from vggt4d.inference import VGGT4DInference
from vggt4d.preprocess import load_images, read_images_from_video
from vggt4d.visualize import save_results_to_rrd, visualize_results


Mode = Literal["viewer", "rrd", "screenshot"]


def _load_inputs(input_path: Path, downsample: int):
    if input_path.is_dir():
        paths = sorted(
            [*input_path.glob("*.jpg"), *input_path.glob("*.png")]
        )
        if not paths:
            raise SystemExit(f"No images found under {input_path}")
        return load_images([str(p) for p in paths], downsample_factor=downsample)
    if input_path.is_file():
        return read_images_from_video(str(input_path), downsample_factor=downsample)
    raise SystemExit(f"Input path does not exist: {input_path}")


def _run_inference(images, checkpoint: Path):
    model = VGGT4DInference(checkpoint_path=checkpoint)
    return model(images)


def _screenshot_rrd(
    rrd_path: Path,
    png_path: Path,
    wait_seconds: float,
    web_port: int = 9090,
    grpc_port: int = 9876,
) -> None:
    """Spawn ``rerun --serve-web <rrd>`` and screenshot it via Playwright.

    The web viewer is hosted at ``http://127.0.0.1:<web_port>`` and is told to
    connect to the in-process gRPC proxy at ``rerun+http://localhost:<grpc_port>/proxy``.
    """
    import subprocess
    import urllib.parse

    from playwright.sync_api import sync_playwright

    png_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "rerun",
            "--serve-web",
            "--web-viewer-port",
            str(web_port),
            "--port",
            str(grpc_port),
            str(rrd_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        viewer_data_url = f"rerun+http://localhost:{grpc_port}/proxy"
        url = f"http://127.0.0.1:{web_port}?url={urllib.parse.quote(viewer_data_url, safe='')}"
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(int(wait_seconds * 1000))
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Image directory or video file")
    parser.add_argument(
        "--mode",
        choices=("viewer", "rrd", "screenshot"),
        default="rrd",
        help="viewer: spawn rerun viewer; rrd: save .rrd only; screenshot: also save PNG via Playwright",
    )
    parser.add_argument("--rrd", type=Path, default=Path("outputs/vggt4d.rrd"))
    parser.add_argument("--screenshot", type=Path, default=Path("outputs/vggt4d.png"))
    parser.add_argument("--downsample", type=int, default=1, help="Frame downsample factor")
    parser.add_argument("--filter-percent", type=float, default=50.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./ckpts/model_tracker_fixed_e20.pt"),
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=3.0,
        help="Seconds to wait for the viewer to render before screenshotting",
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
        _screenshot_rrd(args.rrd, args.screenshot, wait_seconds=args.wait)
        print(f"Wrote {args.screenshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
