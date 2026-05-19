set shell := ["bash", "-cu"]

PKG := "vggt vggt4d demo_vggt4d.py vis_vggt4d.py eval_mask.py"
TESTS := "tests"

default:
    @just --list

install:
    uv sync --group dev

fmt:
    uv run ruff format {{PKG}} {{TESTS}}

lint:
    uv run ruff check {{PKG}} {{TESTS}}

lint-fix:
    uv run ruff check --fix {{PKG}} {{TESTS}}

typecheck:
    uv run ty check vggt4d demo_vggt4d.py

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

clean:
    rm -rf .pytest_cache .ruff_cache .coverage outputs/test_scene
