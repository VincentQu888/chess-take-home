"""Collect verified Titled Tuesday games with safe prior-game ratings.

Python 3.10+; pip install requests pandas python-chess
    python collect_data.py
    python collect_data.py --self-test
    python collect_data.py --output /path/to/games.csv

Implementation lives in chess_pipeline/, not in this CLI entry point.
The full --self-test suite also requires the notebook's ML dependencies.
"""

import argparse
import json
from pathlib import Path
import unittest

from chess_pipeline.config import DEFAULT_CACHE, DEFAULT_END, DEFAULT_START
from chess_pipeline.pipeline import collect_dataset


def self_test() -> None:
    """Preserve the original command while running the extracted offline suite."""
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="First history date (inclusive)")
    parser.add_argument("--end", default=DEFAULT_END, help="Last event date (inclusive)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, help="Optional normalized CSV destination")
    parser.add_argument("--self-test", action="store_true", help="Offline regression checks only")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    frame = collect_dataset(args.start, args.end, args.cache_dir)
    print(json.dumps(frame.attrs["audit"], indent=2))
    print(frame.groupby("event_id")["outcome"].value_counts().unstack(fill_value=0))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Saved {len(frame):,} unique games to {args.output}")


if __name__ == "__main__":
    main()
