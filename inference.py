"""Inference for the 4-layer Line 2 delay LSTM.

Replicates the exact feature pipeline and train-mask scaling from
12_train_cause_lstm.py, then runs models/lstm4_base.keras.

Usage:
    from inference import predict_for
    predict_for("2024-05-01")  # predicts 2024-05-02
"""
import os
import numpy as np
import pandas as pd
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = 30
BASE = ["total_delay", "incident_count", "max_delay", "avg_delay",
        "delay_lag1", "delay_lag7", "incident_lag1",
        "delay_roll7_mean", "delay_roll7_max", "incident_roll7_mean",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]


def _build_df():
    df = pd.read_csv(os.path.join(HERE, "data", "line2_daily_causes.csv"),
                     parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["dow"] = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["delay_lag1"] = df["total_delay"].shift(1)
    df["delay_lag7"] = df["total_delay"].shift(7)
    df["incident_lag1"] = df["incident_count"].shift(1)
    df["delay_roll7_mean"] = df["total_delay"].shift(1).rolling(7).mean()
    df["delay_roll7_max"] = df["total_delay"].shift(1).rolling(7).max()
    df["incident_roll7_mean"] = df["incident_count"].shift(1).rolling(7).mean()
    return df.iloc[SEQ:].reset_index(drop=True)  # warmup for lags/rolling/SEQ


_DF = _build_df()
_TRAIN_M = _DF["date"] < "2023-01-01"
_X = _DF[BASE].to_numpy(dtype=np.float32)
_MU = _X[_TRAIN_M].mean(0)
_SD = _X[_TRAIN_M].std(0) + 1e-8
_XS = (_X - _MU) / _SD
_Y = _DF["total_delay"].to_numpy(dtype=np.float32)
_TGT_MU = float(_Y[_TRAIN_M].mean())
_TGT_SD = float(_Y[_TRAIN_M].std())
_MODEL = keras.models.load_model(os.path.join(HERE, "models", "lstm4_base.keras"))
_DATES = _DF["date"].dt.strftime("%Y-%m-%d").tolist()

DATA_START = _DATES[0]
DATA_END = _DATES[-1]


def predict_for(date_str):
    """Predict total Line 2 delay minutes for the day AFTER date_str.

    date_str must be a YYYY-MM-DD inside the dataset with at least 30 days
    of history behind it. Returns dict with prediction, actual (if the next
    day exists in the data), and the 30-day history used.
    """
    if date_str not in _DATES:
        raise ValueError(f"date {date_str} not in dataset "
                         f"({DATA_START} to {DATA_END})")
    j = _DATES.index(date_str)
    if j < SEQ - 1:
        raise ValueError(f"need 30 days of history; earliest usable date "
                         f"is {_DATES[SEQ - 1]}")
    x = _XS[j - SEQ + 1: j + 1][np.newaxis, :, :]
    pred = float(_MODEL.predict(x, verbose=0)[0, 0] * _TGT_SD + _TGT_MU)
    pred = max(0.0, pred)
    next_day = (_DF["date"].iloc[j] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out = {
        "for_date": next_day,
        "predicted_minutes": round(pred, 1),
        "actual_minutes": None,
        "history": [
            {"date": _DATES[k], "total_delay": round(float(_Y[k]), 1)}
            for k in range(j - SEQ + 1, j + 1)
        ],
    }
    if j + 1 < len(_DF):
        out["actual_minutes"] = round(float(_Y[j + 1]), 1)
    return out


def backtest(date_start="2024-01-01", date_end=None):
    """Run the model over a date range; returns list of dicts."""
    rows = []
    for d in _DATES:
        if d < date_start:
            continue
        if date_end and d >= date_end:
            break
        try:
            r = predict_for(d)
        except ValueError:
            continue
        if r["actual_minutes"] is None:
            break
        rows.append({"date": r["for_date"],
                     "predicted": r["predicted_minutes"],
                     "actual": r["actual_minutes"]})
    return rows
