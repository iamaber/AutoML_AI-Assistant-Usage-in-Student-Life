# AutoML AI Assistant Usage in Student Life

This project uses AutoML (PyCaret) to analyze and predict student
behavior regarding AI assistant usage in their academic life. The target
is `UsedAgain`: whether a student came back to an AI assistant after a
session.

**Rendered report:** [huggingface.co/spaces/iamaber/AI-Assistant-Usage-Student-AutoML](https://huggingface.co/spaces/iamaber/AI-Assistant-Usage-Student-AutoML) — a static HTML export of `autoML_cleaned.ipynb`, with no server or live prediction.

**Results:** see [`METRICS.md`](METRICS.md). The shipped model reaches
0.6714 AUC on a held-out 20% split, against 0.5 for a constant predictor
and a 0.7065 majority-class accuracy baseline. Treat the signal as weak.

## Project Structure

- `autoML.ipynb`: Jupyter notebook for data analysis, model training, and exporting the best pipeline.
- `autoML_cleaned.ipynb`: the notebook exported for the rendered HF Space report; has the same cells as `autoML.ipynb`.
- `autoML_cleaned.html`: static HTML export of `autoML_cleaned.ipynb`, tracked because it is the published report.
- `main.py`: Script to load the trained model and make predictions on new data.
- `my_best_pipeline.pkl`: Saved PyCaret pipeline/model (joblib format).
- `METRICS.md`: measured CV and holdout metrics plus the baseline comparison.
- `tests/`: pytest tests covering the `main.py` inference contract.
- `README.md`: Project documentation.

## Requirements

PyCaret 3.3.2 only supports Python 3.9–3.11 and raises on import for
3.12+, so `.python-version` pins 3.11.

```bash
uv sync
```

## Dataset

`ai_assistant_usage_student_life.csv` has 10,000 rows and the columns
`SessionID`, `StudentLevel`, `Discipline`, `SessionDate`,
`SessionLengthMin`, `TotalPrompts`, `TaskType`, `AI_AssistanceLevel`,
`FinalOutcome`, `UsedAgain` (target) and `SatisfactionRating`.

**Provenance:** the file was supplied with this project. The repository
contains no citation, generation script or source URL for it, and it is
not stated whether the rows are real or synthetic. It looks synthetic
(the categories are evenly balanced and `SessionID`s are sequential), but
that is an observation, not a documented fact. Before reusing this data
anywhere else, establish its origin and licensing — the repository
license in `LICENSE` covers the code only, not the dataset.

## Workflow

1. **Data Preprocessing**
   - Load data and remove identifier columns (`SessionID`, `SessionDate`).
   - Explore and describe the dataset.

2. **Model Training (autoML.ipynb)**
   - Use PyCaret's `setup` to preprocess data and define categorical features,
     with a fixed `session_id=42` and an 80/20 train-test split.
   - Compare multiple models using `compare_models()`.
   - Tune the best model and report CV metrics against a majority baseline.
   - Save the final pipeline as `my_best_pipeline.pkl`.

3. **Prediction (main.py)**
   - Load the saved pipeline with `joblib`, after verifying its SHA-256 digest.
   - Validate the input schema and reject values not seen during training.
   - Predict the target (`UsedAgain`) plus its probability.

## How to Use

### 1. Train and Export Model

Open `autoML.ipynb` and run all cells to:
- Train models using PyCaret.
- Save the final model as `my_best_pipeline.pkl`.

Re-exporting changes the file's digest; update `MODEL_SHA256` in
`main.py` to the new `shasum -a 256 my_best_pipeline.pkl` value.

### 2. Make Predictions

`main.py` ships a working example row. Run:

```bash
uv run python main.py
```

Output:

```
Prediction: [ True]
Probability of reuse: [0.67476601]
```

To score your own sessions, import the helper and pass a DataFrame with
exactly these eight columns; `predict()` raises `ValueError` naming what
is wrong if any are missing, extra, non-numeric or unseen:

| Column | Type | Allowed values |
|---|---|---|
| `StudentLevel` | categorical | `Undergraduate`, `Graduate`, `High School` |
| `Discipline` | categorical | `Computer Science`, `Psychology`, `History`, `Business`, `Math`, `Engineering`, `Biology` |
| `TaskType` | categorical | `Writing`, `Studying`, `Homework Help`, `Coding`, `Brainstorming`, `Research` |
| `FinalOutcome` | categorical | `Assignment Completed`, `Idea Drafted`, `Confused`, `Gave Up` |
| `SessionLengthMin` | numeric | minutes, e.g. `25.0` |
| `TotalPrompts` | integer | e.g. `6` |
| `AI_AssistanceLevel` | integer | `1`–`5` |
| `SatisfactionRating` | numeric | `1.0`–`5.0` |

```python
import pandas as pd
from main import predict

print(predict(pd.DataFrame([{...}])))
```

### 3. Tests

```bash
uv run --group dev pytest
```

## Security note

`my_best_pipeline.pkl` is a pickle (joblib) stream, and loading a pickle
executes code embedded in that file. Only load artifacts you produced
yourself or otherwise trust. `main.py` refuses to load the file unless
its SHA-256 matches the pinned digest, which turns silent tampering into
an error — but that guard is a tripwire, not a sandbox.

## License

Code is MIT-licensed; see `LICENSE`. The dataset is not covered by that
license (see Provenance above).
