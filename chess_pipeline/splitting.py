"""
Chronological splits keep whole UTC tournament dates together.
"""

import pandas as pd

SPLITS = ("train", "validation", "test")


def split_chronologically(table: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(table["event_start"], unit="s", utc=True).dt.date
    unique_dates = sorted(dates.unique())
    if len(unique_dates) < 5:
        raise ValueError("Need at least five target tournament dates for train/validation/test")
    held_out = max(1, round(len(unique_dates) * 0.1))
    train_end = len(unique_dates) - 2*held_out
    assignment = {day: ("train" if i < train_end else "validation"
                        if i < train_end+held_out else "test")
                  for i, day in enumerate(unique_dates)}
    split = table.assign(split=dates.map(assignment))
    assert split.groupby("event_id")["split"].nunique().max() == 1
    assert split.groupby("game_id")["split"].nunique().max() == 1
    train, validation, test = [split.loc[split["split"].eq(name)] for name in SPLITS]
    assert train["event_start"].max() < validation["event_start"].min()
    assert validation["event_start"].max() < test["event_start"].min()
    return split


def split_frames(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: table.loc[table["split"].eq(name)].copy() for name in SPLITS}
