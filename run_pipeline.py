"""Run the complete pipeline without a notebook.

    python run_pipeline.py                 # collect, prepare, train, evaluate
    python run_pipeline.py --data-only     # stop after eligible time splits
    python run_pipeline.py --data-only --output /path/to/prepared.csv

Defaults match train_eval.ipynb. API cache is outside the repository. Only an
explicitly requested CSV is written; no model artifacts are saved.
"""

import argparse
import json
from pathlib import Path

from chess_pipeline.api import ChessComClient
from chess_pipeline.config import DEFAULT_CACHE, PipelineConfig
from chess_pipeline.pipeline import run_data_pipeline, run_experiment


def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-start", default=defaults.history_start)
    parser.add_argument("--target-start", default=defaults.target_start)
    parser.add_argument("--end", default=defaults.end_date)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional prepared-table CSV destination")
    args = parser.parse_args()
    config = PipelineConfig(history_start=args.history_start, target_start=args.target_start,
                            end_date=args.end)
    client = ChessComClient(cache_dir=args.cache_dir)
    if args.data_only:
        data = run_data_pipeline(config, client)
    else:
        experiment = run_experiment(config, client)
        data = experiment.data
        print("Probability metrics:\n", experiment.evaluation.metrics.round(5).to_string())
        print("Accuracy comparison:\n", experiment.evaluation.accuracy_comparison.round(5).to_string())
        print("Gain importance:\n", experiment.importance.to_string(index=False))
    print(json.dumps(data.raw.attrs["audit"], indent=2))
    print(data.coverage.groupby("event_id").agg(
        collected=("game_id", "size"), excluded_missing_prior=("missing_prior_rating", "sum")))
    print("Eligible games by split:\n", data.table.groupby("split", sort=False).size())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.table.to_csv(args.output, index=False)
        print(f"Saved {len(data.table):,} eligible games to {args.output}")


if __name__ == "__main__":
    main()
