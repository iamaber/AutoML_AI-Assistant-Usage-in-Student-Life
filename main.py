import hashlib
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "my_best_pipeline.pkl"
MODEL_SHA256 = "cb3d5ee7615f2e59e914745c54cb69e46e59a5bc8321f470d30d77a522a22ae0"

NUMERIC_FEATURES = [
    "SessionLengthMin",
    "TotalPrompts",
    "AI_AssistanceLevel",
    "SatisfactionRating",
]

CATEGORICAL_FEATURES = {
    "StudentLevel": {"Undergraduate", "Graduate", "High School"},
    "Discipline": {
        "Computer Science",
        "Psychology",
        "History",
        "Business",
        "Math",
        "Engineering",
        "Biology",
    },
    "TaskType": {
        "Writing",
        "Studying",
        "Homework Help",
        "Coding",
        "Brainstorming",
        "Research",
    },
    "FinalOutcome": {
        "Assignment Completed",
        "Idea Drafted",
        "Confused",
        "Gave Up",
    },
}

FEATURES = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES)


def load_pipeline(path=MODEL_PATH):
    """Load the trained pipeline after checking the artifact is the expected one.

    The artifact is a pickle (joblib) stream, so loading it executes code stored
    in the file. Only load artifacts you produced yourself or otherwise trust;
    the digest check below turns silent tampering into a hard error.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(
            f"{path} does not match the expected checksum.\n"
            f"  expected: {MODEL_SHA256}\n"
            f"  actual:   {digest}\n"
            "If you retrained the pipeline on purpose, update MODEL_SHA256 in "
            "main.py to the new digest."
        )
    return joblib.load(path)


def validate(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {missing}")

    extra = [c for c in df.columns if c not in FEATURES]
    if extra:
        raise ValueError(f"Unexpected column(s) in input: {extra}")

    for column, allowed in CATEGORICAL_FEATURES.items():
        unknown = sorted(set(df[column]) - allowed)
        if unknown:
            raise ValueError(
                f"Column '{column}' contains value(s) not seen during training: "
                f"{unknown}. Allowed: {sorted(allowed)}"
            )

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be numeric.")
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")


def predict(df):
    validate(df)
    model = load_pipeline()
    return pd.DataFrame(
        {
            "UsedAgain": model.predict(df[FEATURES]),
            "UsedAgainProbability": model.predict_proba(df[FEATURES])[:, 1],
        }
    )


def main():
    sample_data = pd.DataFrame(
        [
            {
                "StudentLevel": "Undergraduate",
                "Discipline": "Engineering",
                "TaskType": "Studying",
                "FinalOutcome": "Assignment Completed",
                "SessionLengthMin": 25.0,
                "TotalPrompts": 6,
                "AI_AssistanceLevel": 3,
                "SatisfactionRating": 4.0,
            }
        ]
    )

    prediction = predict(sample_data)
    print("Prediction:", prediction["UsedAgain"].values)
    print("Probability of reuse:", prediction["UsedAgainProbability"].values)


if __name__ == "__main__":
    main()
