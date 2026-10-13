"""Build the pre-training dataset from the public review logs.

Usage (from api/):
    uv run --group ml python -m forglemmigej_api.training.build_public

Reads data/raw/fsrs-dataset/revlog_history-*.tsv, computes features with the
shared feature function and writes data/processed/public_examples.parquet.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from forglemmigej_api.training.dataset import to_rows
from forglemmigej_api.training.public import public_examples

REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO_ROOT / "data/raw/fsrs-dataset"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data/processed/public_examples.parquet"
    )
    args = parser.parse_args()

    frames = []
    for path in sorted(args.raw_dir.glob("revlog_history-*.tsv")):
        started = time.perf_counter()
        frame = pd.DataFrame(to_rows(public_examples(path)))
        frame.insert(0, "user", path.stem.removeprefix("revlog_history-"))
        frames.append(frame)
        print(
            f"{path.name}: {len(frame):,} examples in {time.perf_counter() - started:.0f}s",
            flush=True,
        )

    data = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.out, index=False)
    print(f"wrote {len(data):,} examples to {args.out}")


if __name__ == "__main__":
    main()
