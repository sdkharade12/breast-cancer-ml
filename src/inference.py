import joblib
import pandas as pd

model = joblib.load('models/reduced_LR_model.pkl')
scaler = joblib.load('models/reduced_scaler_LR.pkl')
expected_columns = joblib.load('models/reduced_columns_LR.pkl')

def predict(input_df):
    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    return prediction, probability