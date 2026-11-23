# Predicting chess outcomes before play

A small, reproducible pipeline for predicting **White win, draw, or White loss** in Chess.com's Titled Tuesday tournaments, using only information available before each game.

## Run the pipeline

Requires **Python 3.10+**. From the repository root, install the CLI dependencies once:

```bash
python -m pip install requests pandas numpy python-chess lightgbm==4.7.0 scikit-learn
```

Then run the entire pipeline in one command:

```bash
python run_pipeline.py
```

This collects data, constructs features, splits, trains, and prints metrics and feature importance.

Other entry points:

```bash
python run_pipeline.py --data-only                       # Stop after data preparation
python collect_data.py --output games.csv                # Export collected games and prior ratings
python run_pipeline.py --data-only --output prepared.csv # Export the eligible feature table
python -B -m unittest discover -s tests -v                # Run offline tests
```

CSV exports are optional; the default run does not save a dataset or model artifact.

For the executed tables, confusion matrix, and importance plot, see [train_eval.ipynb](train_eval.ipynb). To regenerate it:

```bash
python -m pip install matplotlib nbformat nbclient nbconvert ipykernel
python -m jupyter nbconvert --to notebook --execute --inplace train_eval.ipynb
```

## Main decisions and tradeoffs

- **Sticking with 1 LightGBM model instead of searching for the best performing model.** One model family keeps this take home focused on the actual pipeline.
- **Dropping features, e.g. players' preferred openings.** Kept features to rating,historical draw rate, and recent form. Other features, like players' preferred openings were sparse and had low feature importance. Removing them reduced complexity; the same features with richer histories might still help.
- **Splitting data chronologically, keeping tournament dates together.** The model uses four dates to train, and it uses March 3 and March 10 for validation and testing respectively. This produces **7,478 / 1,932 / 1,982 games**, roughly **66/17/17**. Originally I decided on an 80/10/10 train/test/val split, but 66/17/17 is close enough while maintaining whole-tournament boundaries. Since the task is to predict future games, splitting train/test/val chronologically also helps data leakage.
- **Comparison against two rating baselines.** I chose to test the full-feature model against both a rating-only model and a deterministic rating-only rule for fair evaluation. VS. rating-only rule shows if the model is useful beyond just looking at both players' ratings, and VS. rating-only model shows if the other non-rating features are actually useful.
- **Choosing useful, sufficiently supported pre-game features from the API.** I tried to use as much useful information as the API supported while avoiding sparse features and obvious duplication. The seven features cover different signals, though they are not assumed to be statistically independent:
  - **Rating difference:** White's rating minus Black's, capturing relative strength.
  - **Mean rating:** the average of both ratings, distinguishing equal-strength games at different playing levels. Both rating features use each player's latest collected game completed strictly before the target starts—not the target game's post-game rating.
  - **Round number:** the tournament round, taken from the verified API round URL.
  - **White and Black historical form (two features):** performance relative to opponent strength. Each is `sum(actual score - Elo-style expected score) / (valid historical games + 10)`, where wins score 1, draws 0.5, and losses 0. Expected scores use the historical game's own prior ratings.
  - **White and Black historical draw rates (two features):** each player's proportion of draws, smoothed as `(draws + 1.5) / (historical games + 10)`.
  - Form and draw rates use completed collected games within **60 days before tournament start**, pooling both colors.
  

## Results and model quality

On the March 10 holdout:

| Model | Accuracy | Log loss ↓ |
|---|---:|---:|
| Training class prior | 46.77% | 0.90914 |
| Higher-rating rule | 69.58% | — |
| Rating-only LightGBM | 69.63% | **0.76619** |
| Full LightGBM | 69.68% | 0.76636 |

**The additional features do not demonstrate meaningful improvement over ratings alone.** The full model has slightly worse primary test log loss than rating-only LightGBM; its small accuracy advantage amounts to one additional correct prediction overall.

**Draw recall is zero** under the full model's argmax decision rule: it never chooses draw as the most probable class. This does not mean it assigns zero draw probability, but it is an important weakness hidden by overall accuracy.

The pipeline also reports macro one-vs-one ROC-AUC, multiclass Brier score, and built-in LightGBM gain importance.

- **Test ROC-AUC:** 0.68337 for the full model versus 0.68496 for rating-only LightGBM (higher is better).
- **Test Brier score:** 0.44511 versus 0.44502, respectively (lower is better). Neither probability metric improves with the additional features.
- **Gain importance:** rating difference dominates, followed by mean rating; round and historical features contribute much less.


## Code layout

- [`api.py`](chess_pipeline/api.py), [`collection.py`](chess_pipeline/collection.py), and [`processing.py`](chess_pipeline/processing.py): requests/cache, tournament traversal, normalization, and deduplication.
- [`features.py`](chess_pipeline/features.py) and [`splitting.py`](chess_pipeline/splitting.py): prior ratings, historical features, provenance checks, and chronological splits.
- [`modeling.py`](chess_pipeline/modeling.py) and [`evaluation.py`](chess_pipeline/evaluation.py): training, baselines, importance, and metrics.
- [`pipeline.py`](chess_pipeline/pipeline.py): data and experiment orchestration. Defaults live in [`config.py`](chess_pipeline/config.py); shared boundary helpers live in [`helpers.py`](chess_pipeline/helpers.py).
- [`tests/`](tests/): offline collection, feature/leakage, training, and evaluation checks. The notebook imports the same implementation as the CLI.


## Time spent and AI usage

Approximately **2 hours 50 minutes**:

| Activity | Time |
|---|---:|
| Thinking through the approach | 20 minutes |
| Initial data collection, training, and evaluation experiment | 1 hour |
| Fixing mistakes | 30 minutes |
| Refactoring into a clean pipeline and reviewing AI-generated code | 1 hour |

AI assistance was used for implementation, testing, debugging, and refactoring, with time spent reviewing the generated code.

AI transcripts are attached in the submission email.