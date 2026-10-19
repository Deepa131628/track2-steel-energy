"""Track 2 inference. Usage:
    python predict.py test.csv predictions.csv
or:  from predict import predict_next_usage; preds = predict_next_usage(df)
Model: RandomForestRegressor (150 trees, min_samples_leaf=5), features = current-row
telemetry + hour + one-hot (WeekStatus, Day_of_week, Load_Type). Needs scikit-learn, pandas, joblib.
"""
import os, sys
import joblib
import pandas as pd

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track2_energy_model.joblib")
_saved = joblib.load(_PATH)
_model, _cols = _saved["model"], _saved["columns"]

def predict_next_usage(input_df):
    dates = pd.to_datetime(input_df["date"], format="%d/%m/%Y %H:%M")
    X = input_df.drop(columns=["date", "next_interval_Usage_kWh"], errors="ignore").copy()
    X["hour"] = dates.dt.hour
    X = pd.get_dummies(X, columns=["WeekStatus", "Day_of_week", "Load_Type"], dtype=int)
    return _model.predict(X.reindex(columns=_cols, fill_value=0))

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    out = pd.DataFrame({"date": df["date"], "predicted_next_interval_Usage_kWh": predict_next_usage(df)})
    out.to_csv(sys.argv[2] if len(sys.argv) > 2 else "predictions.csv", index=False)
