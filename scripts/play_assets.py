"""Play exported VGGT4D viewer assets with PyVista + OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt4d.player_assets import play_player_assets


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Player asset directory")
    parser.add_argument("--play-fps", type=_positive_float, default=4.0)
    parser.add_argument("--camera-scale", type=_positive_float, default=0.08)
    parser.add_argument("--point-size", type=_positive_int, default=5)
    parser.add_argument("--background", default="black")
    args = parser.parse_args(argv)

    play_player_assets(
        assets_dir=args.assets,
        play_fps=args.play_fps,
        camera_scale=args.camera_scale,
        point_size=args.point_size,
        background=args.background,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
