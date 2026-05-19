set shell := ["bash", "-cu"]

PKG := "vggt4d/pipeline.py vggt4d/utils/store.py vggt4d/inference.py vggt4d/preprocess.py vggt4d/results.py vggt4d/visualize.py vggt4d/player_assets.py scripts demo_vggt4d.py"
PKG_ALL := "vggt vggt4d demo_vggt4d.py vis_vggt4d.py eval_mask.py"
TESTS := "tests"

default:
    @just --list

install:
    uv sync --group dev

fmt:
    uv run ruff format {{PKG}} {{TESTS}}

lint:
    uv run ruff check {{PKG}} {{TESTS}}

lint-all:
    uv run ruff check {{PKG_ALL}} {{TESTS}}

lint-fix:
    uv run ruff check --fix {{PKG}} {{TESTS}}

typecheck:
    uv run ty check {{PKG}}

test:
    uv run pytest -q

test-smoke:
    uv run pytest -q -m smoke

cov:
    uv run pytest --cov=vggt4d --cov-report=term-missing

radon-cc:
    uv run radon cc -s -a -nb vggt4d demo_vggt4d.py

radon-mi:
    uv run radon mi -s vggt4d demo_vggt4d.py

radon-hal:
    uv run radon hal vggt4d demo_vggt4d.py

radon: radon-cc radon-mi

check: lint typecheck test radon-cc

demo input output:
    uv run python demo_vggt4d.py --input_dir {{input}} --output_dir {{output}}

infer input output:
    uv run python -m scripts.infer --input {{input}} --output {{output}}

export-player-assets results output stride="1" max_points_per_frame="10000":
    uv run python -m scripts.export_player_assets --results {{results}} --output {{output}} --stride {{stride}} --max-points-per-frame {{max_points_per_frame}}

play-assets assets play_fps="4":
    uv run python -m scripts.play_assets --assets {{assets}} --play-fps {{play_fps}}

viz-rrd input rrd="outputs/vggt4d.rrd":
    uv run python -m scripts.visualize --input {{input}} --mode rrd --rrd {{rrd}}

viz-results-rrd results rrd="outputs/vggt4d.rrd":
    uv run python -m scripts.visualize --results {{results}} --mode rrd --rrd {{rrd}}

viz-screenshot input rrd="outputs/vggt4d.rrd" png="outputs/vggt4d.png" wait="3" web_port="9090" grpc_port="9876":
    uv run python -m scripts.visualize --input {{input}} --mode screenshot --rrd {{rrd}} --screenshot {{png}} --wait {{wait}} --web-viewer-port {{web_port}} --grpc-port {{grpc_port}}

viz-results-screenshot results rrd="outputs/vggt4d.rrd" png="outputs/vggt4d.png" wait="3" web_port="9090" grpc_port="9876":
    uv run python -m scripts.visualize --results {{results}} --mode screenshot --rrd {{rrd}} --screenshot {{png}} --wait {{wait}} --web-viewer-port {{web_port}} --grpc-port {{grpc_port}}

viz-viewer input:
    uv run python -m scripts.visualize --input {{input}} --mode viewer

viz-results-viewer results:
    uv run python -m scripts.visualize --results {{results}} --mode viewer

clean:
    rm -rf .pytest_cache .ruff_cache .coverage outputs/test_scene
