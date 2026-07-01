import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from datetime import datetime

MODEL_PATH = "rf_model_state_fair.pkl"
METADATA_PATH = "rf_model_state_fair_metadata.pkl"
MASTER_SHEET_PATH = "Master Sheet for MSEF State.csv"
MAX_WEEKLY_SHIFT = 0.20
USDA_API_KEY = "F8E7FAB6-C2FA-3375-8A8B-7996AC634920"

# ==========================================
# AUTO DATA PIPELINE (USDA API)
# ==========================================
@st.cache_data(ttl=3600)  
def fetch_live_supply_data():
    """
    Fetches harvest progress, LAST WEEK's progress, and total annual production.
    Returns: (harvest_pct, last_week_pct, total_production, status_message)
    """
    current_year = datetime.now().year
    current_week_num = datetime.now().isocalendar()[1]
    url = "https://quickstats.nass.usda.gov/api/api_GET/"
    
    # Baseline fallback values
    harvest_pct = 0.0
    last_week_pct = 0.0
    total_production = 1800000000.0  
    status_msg = ""
    
    # --- 1. FETCH TOTAL ANNUAL PRODUCTION ---
    prod_payload = {
        "key": USDA_API_KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "PRODUCTION",
        "short_desc": "CORN, GRAIN - PRODUCTION, MEASURED IN BU",
        "state_name": "NEBRASKA",
        "freq_desc": "ANNUAL",
        "format": "JSON"
    }
    
    try:
        prod_response = requests.get(url, params=prod_payload, timeout=10)
        if prod_response.status_code == 200:
            prod_records = prod_response.json().get('data', [])
            if prod_records:
                newest_prod = max(prod_records, key=lambda x: x['year'])
                total_production = float(newest_prod['Value'].replace(',', ''))
    except Exception:
        pass  

    # --- 2. FETCH CURRENT & LAST WEEK PROGRESS ---
    harvest_payload = {
        "key": USDA_API_KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "PROGRESS",
        "short_desc": "CORN, GRAIN - PROGRESS, MEASURED IN PCT HARVESTED",
        "state_name": "NEBRASKA",
        "year__GE": str(current_year - 1), 
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=harvest_payload, timeout=10)
        if response.status_code == 200:
            records = response.json().get('data', [])
            
            exact_match_value = None
            last_week_match_value = None
            highest_value_this_year = 0
            
            for record in records:
                record_week = int(record['reference_period_desc'].split('#')[-1])
                record_val = float(record['Value'])
                
                if record_val > highest_value_this_year:
                    highest_value_this_year = record_val
                    
                if record_week == current_week_num:
                    exact_match_value = record_val
                if record_week == current_week_num - 1:
                    last_week_match_value = record_val
            
            # Apply dynamic fallbacks
            if exact_match_value is not None:
                harvest_pct = exact_match_value / 100.0
                last_week_pct = (last_week_match_value / 100.0) if last_week_match_value is not None else 0.0
                status_msg = f"🚜 Active Harvest: Using live USDA progress report ({exact_match_value}%)."
            elif len(records) == 0 or highest_value_this_year == 0:
                harvest_pct = 0.0
                last_week_pct = 0.0
                status_msg = "🌱 Pre-Harvest Season: No USDA reports published yet this year. Defaulting to 0%."
            else:
                harvest_pct = 1.0
                last_week_pct = 1.0
                status_msg = f"❄️ Post-Harvest Season: USDA reporting ended after hitting {highest_value_this_year}%. Defaulting to 100%."
        else:
            status_msg = "⚠️ Could not connect to USDA API. Using baseline defaults."
            
    except Exception:
        status_msg = "⚠️ Network error trying to fetch USDA data. Using baseline defaults."

    return harvest_pct, last_week_pct, total_production, status_msg

# ==========================================
# CACHED MACHINE LEARNING LOAD
# ==========================================
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
    week_num, seasonality, weekly_bushels, cumulative_harvest, is_harvesting, demand_ethanol, demand_livestock
):
    return pd.DataFrame({
        "Week_Num": [week_num], "Seasonality": [seasonality], "Weekly_Bushels_Produced": [weekly_bushels],
        "Cumulative_Harvest": [cumulative_harvest], "Is_Harvesting": [is_harvesting],
        "Demand_Ethanol": [demand_ethanol], "Demand_Livestock": [demand_livestock],
    })

st.set_page_config(page_title="Harvest or Hold? Forecaster", layout="wide")

try:
    rf_model, metadata, seasonality_map = load_model_assets()
except Exception as exc:
    st.error(f"Could not load model assets: {exc}")
    st.stop()

window_size = int(metadata.get("window_size", 4))
feature_columns = metadata.get(
    "feature_columns",
    ["Week_Num", "Seasonality", "Weekly_Bushels_Produced", "Cumulative_Harvest", "Is_Harvesting", "Demand_Ethanol", "Demand_Livestock"],
)

st.title("🌽 Harvest or Hold? Nebraska Corn Market Forecaster")

# ==========================================
# SIMPLE VS ADVANCED VIEW TOGGLE
# ==========================================
view_mode = st.radio("Select Dashboard View:", ["👨‍🌾 Simple View", "📊 Advanced View"], horizontal=True)
st.markdown("---")

# ==========================================
# MARKET CONTROL PANEL (SIDEBAR)
# ==========================================
st.sidebar.header("Market Control Panel")

current_week = st.sidebar.slider("Current Week Number", min_value=1, max_value=51, value=6)
target_week = st.sidebar.slider("Target Forecast Week", min_value=current_week + 1, max_value=52, value=min(current_week + 6, 52))

st.sidebar.markdown("---")
st.sidebar.subheader("Momentum Baseline")
st.sidebar.number_input("Momentum Window Size (Weeks)", min_value=1, max_value=52, value=window_size, step=1, disabled=True, help="Fixed at 4 weeks due to model training.")
recent_prices_input = st.sidebar.text_input(f"Recent Prices (enter at least {window_size})", "3.78, 3.83, 3.70, 3.84")

st.sidebar.markdown("---")
st.sidebar.subheader("Seasonality")
use_auto_seasonality = st.sidebar.checkbox("Auto-fill seasonality by forecast week", value=True)
manual_seasonality = st.sidebar.number_input("Manual Seasonality Override", value=float(seasonality_map.get(current_week, 4.50)), format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("Demand Factors")
demand_ethanol = st.sidebar.number_input("Ethanol Demand (Thousand Barrels/Day)", value=990, step=1)
demand_livestock = st.sidebar.number_input("Livestock Demand (Head)", value=2500000, step=10000)

# ==========================================
# AUTO DATA: SUPPLY FACTORS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Supply Factors")

auto_supply = st.sidebar.checkbox("📡 Auto-fetch Live USDA Data", value=True)

if auto_supply:
    with st.sidebar.status("Fetching Live USDA Data..."):
        harvest_pct, last_week_pct, live_production, status_text = fetch_live_supply_data()
    
    st.sidebar.info(status_text)
    
    # Calculate automated variables behind the scenes
    is_harvesting = 1 if 0.0 < harvest_pct < 1.0 else 0
    cumulative_harvest = harvest_pct * live_production
    
    # NEW STEP 5: Calculate weekly bushels safely
    weekly_pct_change = max(0.0, harvest_pct - last_week_pct) 
    weekly_bushels = weekly_pct_change * live_production
    
    # Clean UI: Display the computed values as metrics/read-only information
    st.sidebar.metric(label="Official Annual Production (Bu)", value=f"{live_production:,.0f}")
    st.sidebar.number_input("Cumulative Harvest (Bushels)", value=cumulative_harvest, disabled=True, format="%.0f")
    st.sidebar.number_input("Weekly Bushels Produced", value=weekly_bushels, disabled=True, format="%.0f")
else:
    is_harvesting = st.sidebar.selectbox("Is it harvest season? (0=No, 1=Yes)", options=[0, 1], index=0)
    cumulative_harvest = st.sidebar.number_input("Cumulative Harvest (Bushels)", value=0.0, step=1000000.0, format="%.0f")
    weekly_bushels = st.sidebar.number_input("Weekly Bushels Produced", value=0.0, step=1000000.0, format="%.0f")

st.sidebar.markdown("---")
clip_predictions = st.sidebar.checkbox("Cap weekly deviation at ±$0.20", value=True)

# ==========================================
# MAIN FORECASTING LOGIC
# ==========================================
if view_mode == "📊 Advanced View":
    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.subheader("Model Logic")
        st.markdown("- Baseline: 4-week moving average of recent prices\n- Deviation driver: supply, demand, and seasonality inputs\n- Forecast style: chained week-by-week projection")
    with right_col:
        st.subheader("Model Inputs Used")
        st.code(", ".join(feature_columns), language="text")

if st.button("🚀 Run Chained Forecast", type="primary"):
    try:
        try:
            recent_prices = parse_recent_prices(recent_prices_input, window_size)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
            
        initial_average_price = np.mean(recent_prices[-window_size:])
        forecast_rows = []

        for week in range(current_week + 1, target_week + 1):
            moving_avg = float(np.mean(recent_prices[-window_size:]))
            seasonality_value = float(seasonality_map.get(week, manual_seasonality)) if use_auto_seasonality else float(manual_seasonality)

            future_conditions = build_feature_row(
                week_num=week, seasonality=seasonality_value, weekly_bushels=weekly_bushels,
                cumulative_harvest=cumulative_harvest, is_harvesting=is_harvesting,
                demand_ethanol=demand_ethanol, demand_livestock=demand_livestock,
            )[feature_columns]

            raw_deviation = float(rf_model.predict(future_conditions)[0])
            deviation = float(np.clip(raw_deviation, -MAX_WEEKLY_SHIFT, MAX_WEEKLY_SHIFT)) if clip_predictions else raw_deviation
            predicted_price = moving_avg + deviation
            recent_prices.append(predicted_price)

            forecast_rows.append({
                "Week": week, "Seasonality": seasonality_value, "Momentum": round(moving_avg, 4),
                "Predicted Deviation": round(deviation, 4), "Predicted Price": round(predicted_price, 4),
            })

        forecast_df = pd.DataFrame(forecast_rows)
        final_price = forecast_df.iloc[-1]['Predicted Price']
        price_change = final_price - initial_average_price

        if price_change >= 0.05:
            recommendation, reason = "HOLD 🛑", "Prices are projected to trend upward. Waiting could yield higher profits."
        elif price_change <= -0.05:
            recommendation, reason = "HARVEST / SELL NOW 🚜", "Prices are projected to drop. Locking in current rates is advised."
        else:
            recommendation, reason = "MONITOR 🔍", "Prices are projected to remain relatively stable. Keep an eye on market shifts."

        if view_mode == "👨‍🌾 Simple View":
            st.header("🎯 Your Forecast Recommendation")
            st.subheader(f"{recommendation}")
            st.write(f"**Why?** {reason}")
            st.metric(label=f"Projected Price for Week {target_week}", value=f"${final_price:.2f}", delta=f"{price_change:.2f} vs current average")
            st.info("Switch to the 'Advanced View' at the top of the page to see the week-by-week data breakdown and price trajectory charts.")

        elif view_mode == "📊 Advanced View":
            st.subheader(f"Forecast: Week {current_week + 1} to Week {target_week}")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            st.markdown("<h3 style='text-align: center; color: gray;'>⬇️</h3>", unsafe_allow_html=True)
            st.success(f"Final projected price for Week {target_week}: ${final_price:.2f}")
            st.markdown("<h3 style='text-align: center; color: gray;'>⬇️</h3>", unsafe_allow_html=True)

            fig = px.line(forecast_df, x="Week", y="Predicted Price", title="Forecasted Price Trajectory", markers=True, labels={"Predicted Price": "Price ($/Bushel)", "Week": "Week Number"})
            fig.update_traces(line_color="#C56A1A", line_width=3, marker=dict(size=8, color="#C56A1A"))
            fig.update_layout(xaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'), yaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)', tickprefix="$"), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("⚠️ System Interruption Detected")
        st.info("⏳ We are currently waiting for the newest/updated data to sync, or a required field is missing. Please check back later or verify your inputs.")

st.markdown("---")
st.warning("**Disclaimer:** These projections are estimates based on historical trends and current inputs. They are not guaranteed to be 100% accurate. The model cannot effectively predict outliers caused by 'black swan' events, such as extreme weather disasters, unpredictable geopolitical shifts, or sudden market crashes.")
