import pickle
import numpy as np
import pandas as pd

# Load the trained PyCaret pipeline
model = pickle.load(open('my_best_pipeline.pkl', 'rb'))


sample_data = pd.DataFrame([{
    'StudentLevel': 'Undergraduate',
    'Discipline': 'Engineering',
    'TaskType': 'Assignment',
    'FinalOutcome': 'Pass',
    
}])

# Make prediction
prediction = model.predict(sample_data)
print("Prediction:", prediction.values)