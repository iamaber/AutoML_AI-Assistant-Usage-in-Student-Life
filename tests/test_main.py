import pandas as pd
import pytest

import main

VALID_ROW = {
    "StudentLevel": "Undergraduate",
    "Discipline": "Engineering",
    "TaskType": "Studying",
    "FinalOutcome": "Assignment Completed",
    "SessionLengthMin": 25.0,
    "TotalPrompts": 6,
    "AI_AssistanceLevel": 3,
    "SatisfactionRating": 4.0,
}


def test_predict_accepts_full_feature_row():
    result = main.predict(pd.DataFrame([VALID_ROW]))

    assert list(result.columns) == ["UsedAgain", "UsedAgainProbability"]
    assert len(result) == 1
    assert result["UsedAgainProbability"].between(0, 1).all()


def test_predict_rejects_missing_numeric_feature():
    row = dict(VALID_ROW)
    row.pop("TotalPrompts")

    with pytest.raises(ValueError, match="Missing required feature column"):
        main.predict(pd.DataFrame([row]))


def test_predict_rejects_unseen_category():
    row = dict(VALID_ROW, FinalOutcome="Pass")

    with pytest.raises(ValueError, match="not seen during training"):
        main.predict(pd.DataFrame([row]))


def test_predict_rejects_non_numeric_feature():
    row = dict(VALID_ROW, TotalPrompts="six")

    with pytest.raises(ValueError, match="must be numeric"):
        main.predict(pd.DataFrame([row]))
