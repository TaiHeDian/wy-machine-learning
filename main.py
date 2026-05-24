from __future__ import annotations

from pathlib import Path

from src import run_pipeline


if __name__ == "__main__":
    raise SystemExit(run_pipeline(Path(__file__).resolve().parent))
