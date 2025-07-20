# AutoML AI Assistant Usage in Student Life

This project uses AutoML (PyCaret) to analyze and predict student behavior regarding AI assistant usage in their academic life. The workflow includes data preprocessing, model selection, tuning, evaluation, and deployment for prediction.

## Project Structure

- `autoML.ipynb`: Jupyter notebook for data analysis, model training, and exporting the best pipeline.
- `main.py`: Script to load the trained model and make predictions on new data.
- `my_best_pipeline.pkl`: Saved PyCaret pipeline/model.
- `README.md`: Project documentation.

## Dataset

The dataset (`ai_assistant_usage_student_life.csv`) contains information about students' interactions with AI assistants, including features like:
- StudentLevel
- Discipline
- TaskType
- FinalOutcome
- UsedAgain (target variable)

## Workflow

1. **Data Preprocessing**  
   - Load data and remove unnecessary columns (`SessionID`, `SessionDate`).
   - Explore and describe the dataset.

2. **Model Training (autoML.ipynb)**  
   - Use PyCaret's `setup` to preprocess data and define categorical features.
   - Compare multiple models using `compare_models()`.
   - Tune the best model.
   - Save the best pipeline as `my_best_pipeline.pkl`.

3. **Prediction (main.py)**  
   - Load the saved pipeline using `pickle`.
   - Prepare new sample data as a pandas DataFrame.
   - Predict the target (`UsedAgain`) for new samples.

## How to Use

### 1. Train and Export Model

Open `autoML.ipynb` and run all cells to:
- Train models using PyCaret.
- Save the best model as `my_best_pipeline.pkl`.

### 2. Make Predictions

Edit `main.py` to provide your own sample data, then run:

```bash
python main.py
```

You will see the prediction output in the terminal.

## Example: main.py

````python
import pickle
import pandas as pd

# Load the trained PyCaret pipeline
model = pickle.load(open('my_best_pipeline.pkl', 'rb'))

# Example input data for prediction
sample_data = pd.DataFrame([{
    'StudentLevel': 'Undergraduate',
    'Discipline': 'Engineering',
    'TaskType': 'Assignment',
    'FinalOutcome': 'Pass',
    # Add other required features if necessary
}])

# Make prediction
prediction = model.predict(sample_data)
print("Prediction:", prediction.values)