"""Export saved VGGT4D results into reusable player assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt4d.player_assets import export_saved_results_to_player_assets


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Saved VGGT4D result directory")
    parser.add_argument("--output", type=Path, required=True, help="Output player asset directory")
    parser.add_argument("--stride", type=_positive_int, default=1, help="Export every nth frame")
    parser.add_argument("--max-frames", type=_positive_int, default=None)
    parser.add_argument("--filter-percent", type=float, default=70.0)
    parser.add_argument("--max-points-per-frame", type=_positive_int, default=10_000)
    parser.add_argument("--hide-dynamic", action="store_true", help="Exclude dynamic-mask points")
    parser.add_argument("--no-ply", action="store_true", help="Skip standard .ply point clouds")
    args = parser.parse_args(argv)

    manifest = export_saved_results_to_player_assets(
        results_dir=args.results,
        output_dir=args.output,
        stride=args.stride,
        max_frames=args.max_frames,
        filter_percent=args.filter_percent,
        max_points_per_frame=args.max_points_per_frame,
        hide_dynamic=args.hide_dynamic,
        write_ply=not args.no_ply,
    )
    print(
        f"Exported {manifest['frame_count']} player frames "
        f"from {manifest['source_frame_count']} saved frames to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
