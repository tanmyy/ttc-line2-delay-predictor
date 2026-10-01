"""TTC Line 2 Delay Predictor - Streamlit app.

Live inference with the trained 4-layer LSTM (models/lstm4_base.keras).
Pick a date, see the model's prediction + severity category, then reveal
what actually happened.
"""
import json
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from inference import predict_for, DATA_START, DATA_END

st.set_page_config(page_title="TTC Line 2 Delay Predictor", layout="centered")

CATS = [(30, "Minimal", "#00a651"), (75, "Moderate", "#ffd200"),
        (150, "Heavy", "#ff7a00"), (float("inf"), "Severe", "#e31837")]


def cat(v):
    for lim, name, color in CATS:
        if v <= lim:
            return name, color
    return CATS[-1][1], CATS[-1][2]


def badge(name, color):
    st.markdown(
        f"<span style='background:{color};color:#0b0b0d;font-weight:800;"
        f"font-size:13px;letter-spacing:1px;padding:5px 16px;border-radius:999px;"
        f"text-transform:uppercase;'>{name}</span>",
        unsafe_allow_html=True,
    )


st.markdown("## TTC Line 2 Delay Predictor")
st.caption("4-layer LSTM. Live inference: the model reads the 30 days before "
           "your date and predicts the next day's total delay minutes.")

d0 = date(2014, 1, 31)
d1 = date(*map(int, DATA_END.split("-")))
picked = st.date_input("History ends on", value=date(2026, 8, 30),
                       min_value=d0, max_value=d1)

if st.button("Predict next day", type="primary"):
    st.session_state["rec"] = predict_for(picked.isoformat())
    st.session_state["revealed"] = False

rec = st.session_state.get("rec")
if rec:
    name, color = cat(rec["predicted_minutes"])
    st.markdown("### Predicted")
    st.markdown(f"# {rec['predicted_minutes']:.0f} min")
    badge(name, color)
    st.caption(f"for {rec['for_date']}")

    hist = rec["history"]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot([h["date"][5:] for h in hist],
            [h["total_delay"] for h in hist],
            color="#8d8d96", label="last 30 days (actual)")
    ax.scatter([hist[-1]["date"][5:]], [rec["predicted_minutes"]],
               color="#ffd200", s=80, zorder=5,
               label=f"prediction for {rec['for_date']}")
    ax.tick_params(colors="#8d8d96", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2a2a2f")
    ax.set_ylabel("delay minutes", color="#8d8d96")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    if rec["actual_minutes"] is not None and not st.session_state.get("revealed"):
        if st.button("Reveal actual"):
            st.session_state["revealed"] = True
            st.rerun()

    if st.session_state.get("revealed") and rec["actual_minutes"] is not None:
        aname, acolor = cat(rec["actual_minutes"])
        err = abs(rec["predicted_minutes"] - rec["actual_minutes"])
        st.markdown("### Actual")
        st.markdown(f"# {rec['actual_minutes']:.0f} min")
        badge(aname, acolor)
        st.write(f"Error: **{err:.0f} min**")
    elif rec["actual_minutes"] is None:
        st.info("Actual unknown: this date is beyond the dataset.")

with st.expander("Backtest: 973 test days (Jan 2024 - Aug 2026)"):
    with open("demo_predictions.json") as f:
        preds = json.load(f)
    keys = sorted(r["date"] for r in preds)
    pm = {r["date"]: r for r in preds}
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")
    ax2.plot(keys, [pm[k]["actual"] for k in keys], color="#00a651",
             lw=1, label="actual")
    ax2.plot(keys, [pm[k]["predicted"] for k in keys], color="#ffd200",
             lw=1, label="predicted")
    ax2.tick_params(colors="#8d8d96", labelsize=8)
    for s in ax2.spines.values():
        s.set_color("#2a2a2f")
    ax2.legend(fontsize=8)
    ax2.set_title("Test MAE 40.3 min", color="#f2f2f2", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

st.caption("_Test MAE 40.3 min over 974 days; R2 is negative, so daily totals "
           "remain mostly noise. Research demo, not operations advice._")
