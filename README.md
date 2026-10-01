# TTC Line 2 Delay Predictor

Live Streamlit demo of a 4-layer LSTM trained to predict next-day total
delay minutes on the TTC Line 2 (Bloor-Danforth).

- **Model:** 4 stacked LSTM layers, 30-day window, trained 2014-2022,
  validated 2023, tested Jan 2024 - Aug 2026 (974 days).
- **Test MAE:** 40.3 minutes (train-mean baseline 42.8).
- **Data:** Toronto Open Data, TTC Subway Delay Data (267,842 incidents,
  2014-01-01 to 2026-08-31), Line 2 (`BD`) daily aggregates.

Pick a date, see the model's prediction with a severity category
(Minimal / Moderate / Heavy / Severe), then reveal what actually happened.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

Push to GitHub, then deploy at https://share.streamlit.io (New app ->
select repo -> `streamlit_app.py`).

Note: model weights are course project work. Keep the repo private until
all group members approve public release.
