import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "/Users/Jiadong.Chen/Documents/New project 2/rf_model_state_fair.pkl"
METADATA_PATH = "/Users/Jiadong.Chen/Documents/New project 2/rf_model_state_fair_metadata.pkl"
MASTER_SHEET_PATH = "/Users/Jiadong.Chen/Desktop/MSEF/Master Sheet for MSEF.csv"
MAX_WEEKLY_SHIFT = 0.20


@st.cache_resource
def load_model_assets():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    metadata = {}
    try:
        with open(METADATA_PATH, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)
    except FileNotFoundError:
        metadata = {}

    master_sheet = pd.read_csv(MASTER_SHEET_PATH)
    master_sheet.columns = master_sheet.columns.str.strip()
    master_sheet["Week_Num"] = pd.to_numeric(
        master_sheet["Period"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    seasonality_map = (
        master_sheet.dropna(subset=["Week_Num", "Seasonality"])
        .drop_duplicates(subset=["Week_Num"])
        .set_index("Week_Num")["Seasonality"]
        .to_dict()
    )

    return model, metadata, seasonality_map


def parse_recent_prices(raw_text, window_size):
    prices = [float(piece.strip()) for piece in raw_text.split(",") if piece.strip()]
    if len(prices) < window_size:
        raise ValueError(f"Enter at least {window_size} recent prices.")
    return prices


def build_feature_row(
    week_num,
    seasonality,
    weekly_bushels,
    cumulative_harvest,
    is_harvesting,
    demand_ethanol,
    demand_livestock,
):
    return pd.DataFrame(
        {
            "Week_Num": [week_num],
            "Seasonality": [seasonality],
            "Weekly_Bushels_Produced": [weekly_bushels],
            "Cumulative_Harvest": [cumulative_harvest],
            "Is_Harvesting": [is_harvesting],
            "Demand_Ethanol": [demand_ethanol],
            "Demand_Livestock": [demand_livestock],
        }
    )


st.set_page_config(page_title="Harvest or Hold? Forecaster", layout="wide")

try:
    rf_model, metadata, seasonality_map = load_model_assets()
except Exception as exc:
    st.error(f"Could not load model assets: {exc}")
    st.stop()

window_size = int(metadata.get("window_size", 4))
feature_columns = metadata.get(
    "feature_columns",
    [
        "Week_Num",
        "Seasonality",
        "Weekly_Bushels_Produced",
        "Cumulative_Harvest",
        "Is_Harvesting",
        "Demand_Ethanol",
        "Demand_Livestock",
    ],
)

st.title("🌽 Harvest or Hold? Nebraska Corn Market Forecaster")

# ==========================================
# SIMPLE VS ADVANCED VIEW TOGGLE
# ==========================================
view_mode = st.radio(
    "Select Dashboard View:",
    ["👨‍🌾 Simple View", "📊 Advanced View"],
    horizontal=True
)
st.markdown("---")


# Sidebar remains visible for both so data can be inputted
st.sidebar.header("Market Control Panel")

current_week = st.sidebar.slider("Current Week Number", min_value=1, max_value=51, value=6)
target_week = st.sidebar.slider(
    "Target Forecast Week",
    min_value=current_week + 1,
    max_value=52,
    value=min(current_week + 6, 52),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Momentum Baseline")
st.sidebar.number_input(
    "Momentum Window Size (Weeks)",
    min_value=1,
    max_value=52,
    value=window_size,
    step=1,
    disabled=True,
    help="This is fixed at 4 weeks because the trained model was built on a 4-week momentum baseline.",
)
recent_prices_input = st.sidebar.text_input(
    f"Recent Prices (enter at least {window_size})",
    "3.78, 3.83, 3.70, 3.84",
)
st.sidebar.caption(
    "Use East Nebraska prices to stay consistent with the model's historical data."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Seasonality")
use_auto_seasonality = st.sidebar.checkbox("Auto-fill seasonality by forecast week", value=True)
manual_seasonality = st.sidebar.number_input(
    "Manual Seasonality Override",
    value=float(seasonality_map.get(current_week, 4.50)),
    format="%.2f",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Demand Factors")
demand_ethanol = st.sidebar.number_input(
    "Ethanol Demand (Thousand Barrels/Day)",
    value=990,
    step=1,
)
demand_livestock = st.sidebar.number_input(
    "Livestock Demand (Head)",
    value=2500000,
    step=10000,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Supply Factors")
is_harvesting = st.sidebar.selectbox("Is it harvest season?", options=[0, 1], index=0)
weekly_bushels = st.sidebar.number_input(
    "Weekly Bushels Produced",
    value=0.0,
    step=1000000.0,
    format="%.0f",
)
cumulative_harvest = st.sidebar.number_input(
    "Cumulative Harvest (Bushels)",
    value=0.0,
    step=1000000.0,
    format="%.0f",
)

st.sidebar.markdown("---")
clip_predictions = st.sidebar.checkbox(
    "Cap weekly deviation at ±$0.20",
    value=True,
)

# Only show the complex model text in the Advanced View
if view_mode == "📊 Advanced View":
    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.subheader("Model Logic")
        st.markdown(
            """
    - Baseline: 4-week moving average of recent prices
    - Deviation driver: supply, demand, and seasonality inputs
    - Forecast style: chained week-by-week projection
            """
        )
    with right_col:
        st.subheader("Model Inputs Used")
        st.code(", ".join(feature_columns), language="text")


if st.button("🚀 Run Chained Forecast", type="primary"):
    try:
        recent_prices = parse_recent_prices(recent_prices_input, window_size)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
        
    initial_average_price = np.mean(recent_prices[-window_size:])
    forecast_rows = []

    for week in range(current_week + 1, target_week + 1):
        moving_avg = float(np.mean(recent_prices[-window_size:]))
        seasonality_value = (
            float(seasonality_map.get(week, manual_seasonality))
            if use_auto_seasonality
            else float(manual_seasonality)
        )

        future_conditions = build_feature_row(
            week_num=week,
            seasonality=seasonality_value,
            weekly_bushels=weekly_bushels,
            cumulative_harvest=cumulative_harvest,
            is_harvesting=is_harvesting,
            demand_ethanol=demand_ethanol,
            demand_livestock=demand_livestock,
        )[feature_columns]

        raw_deviation = float(rf_model.predict(future_conditions)[0])
        deviation = (
            float(np.clip(raw_deviation, -MAX_WEEKLY_SHIFT, MAX_WEEKLY_SHIFT))
            if clip_predictions
            else raw_deviation
        )
        predicted_price = moving_avg + deviation
        recent_prices.append(predicted_price)

        forecast_rows.append(
            {
                "Week": week,
                "Seasonality": seasonality_value,
                "Momentum": round(moving_avg, 4),
                "Predicted Deviation": round(deviation, 4),
                "Predicted Price": round(predicted_price, 4),
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)
    final_price = forecast_df.iloc[-1]['Predicted Price']
    price_change = final_price - initial_average_price

    # Determine Recommendation Logic
    if price_change >= 0.05:
        recommendation = "HOLD 🛑"
        reason = "Prices are projected to trend upward. Waiting could yield higher profits."
    elif price_change <= -0.05:
        recommendation = "HARVEST / SELL NOW 🚜"
        reason = "Prices are projected to drop. Locking in current rates is advised."
    else:
        recommendation = "MONITOR 🔍"
        reason = "Prices are projected to remain relatively stable. Keep an eye on market shifts."

    # ==========================================
    # DISPLAY FOR SIMPLE VIEW
    # ==========================================
    if view_mode == "👨‍🌾 Simple View":
        st.header("🎯 Your Forecast Recommendation")
        st.subheader(f"{recommendation}")
        st.write(f"**Why?** {reason}")
        
        st.metric(
            label=f"Projected Price for Week {target_week}", 
            value=f"${final_price:.2f}", 
            delta=f"{price_change:.2f} vs current average"
        )
        
        st.info("Switch to the 'Advanced View' at the top of the page to see the week-by-week data breakdown and price trajectory charts.")

    # ==========================================
    # DISPLAY FOR ADVANCED VIEW
    # ==========================================
    elif view_mode == "📊 Advanced View":
        st.subheader(f"Forecast: Week {current_week + 1} to Week {target_week}")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.success(f"Final projected price for Week {target_week}: ${final_price:.2f}")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            forecast_df["Week"],
            forecast_df["Predicted Price"],
            marker="o",
            color="#C56A1A",
            linewidth=2.5,
        )
        ax.set_title("Forecasted Price Trajectory", fontweight="bold")
        ax.set_xlabel("Week Number")
        ax.set_ylabel("Price ($/Bushel)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

        st.caption(
            "Interpretation: the moving average sets the short-term baseline, and the model "
            "estimates whether economic conditions push price above or below that baseline."
        )

# ==========================================
# GLOBAL DISCLAIMER WARNING
# ==========================================
st.markdown("---")
st.warning(
    "**Disclaimer:** These projections are estimates based on historical trends and current inputs. "
    "They are not guaranteed. The model cannot anticipate 'black swan' events, such as extreme "
    "weather disasters, unpredictable geopolitical shifts, or sudden market crashes."
)
