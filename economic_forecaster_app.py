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

# --- API KEYS ---
USDA_API_KEY = "F8E7FAB6-C2FA-3375-8A8B-7996AC634920"
EIA_API_KEY = "hQjaCbkfOn9dttle5ho4oRu1aaffTZgJgmB7lqZx"
AMS_API_KEY = "oK/SXE39wQiRwoT0kHooLx7XYOLwAjHr" 

# ==========================================
# AUTO DATA PIPELINE: MOMENTUM (USDA AMS API)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_recent_prices():
    """
    Fetches the last 4 unique daily cash prices from Nebraska Elevators (Report 3225).
    Directly targets the USDA's official regional summary row to perfectly match the website.
    """
    fallback_prices = "3.78, 3.83, 3.70, 3.84"
    
    if not AMS_API_KEY:
        return fallback_prices, "⚠️ AMS API Key missing. Using manual baseline."
        
    url = "https://marsapi.ams.usda.gov/services/v1.2/reports/3225/Report%20Detail"
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        # Give the USDA server 30 full seconds to gather the spreadsheet
        response = session.get(url, auth=(AMS_API_KEY, ''), timeout=30)
        
        if response.status_code != 200:
            return fallback_prices, f"⚠️ USDA Server Error: Code {response.status_code}"
            
        data = response.json()
        records = data.get('results', []) if isinstance(data, dict) else data
            
        flat_records = []
        for item in records:
            if isinstance(item, dict) and 'results' in item:
                flat_records.extend(item['results'])
            else:
                flat_records.append(item)
                
        if not flat_records:
            return fallback_prices, "⚠️ Report Detail returned empty data."
            
        corn_records = []
        for r in flat_records:
            commodity = str(r.get('commodity', '')).upper()
            # The USDA puts the regional summary name directly in market_location_name
            market_loc = str(r.get('market_location_name', '')).upper()
            
            # Look EXACTLY for the official East region summary row
            if "CORN" in commodity and market_loc == "EAST":
                date_val = r.get('published_date', r.get('report_date', ''))
                p_avg = r.get('avg_price')
                
                # Grab their pre-calculated official average
                if date_val and p_avg is not None and str(p_avg).strip() != "":
                    try:
                        corn_records.append({
                            'date': date_val,
                            'price': float(p_avg)
                        })
                    except ValueError:
                        pass
                    
        if not corn_records:
            return fallback_prices, "⚠️ Found 0 official East Region summary rows."
            
        df = pd.DataFrame(corn_records)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # We don't need to mathematically average them anymore! 
        # Just grab the newest 4 unique dates from their official list.
        daily_avg = df.drop_duplicates(subset=['date']).sort_values('date', ascending=False)
        unique_prices = daily_avg['price'].round(2).tolist()[:4]
        
        if len(unique_prices) < 4:
            return fallback_prices, f"⚠️ Only found {len(unique_prices)} valid days of data. Need 4."
            
        # Reverse for chronological order (Week 1, Week 2, Week 3, Week 4)
        unique_prices.reverse()
        price_str = ", ".join(map(str, unique_prices))
        return price_str, "📈 Live USDA Official Prices Fetched (East Region Summary)!"
        
    except requests.exceptions.Timeout:
        return fallback_prices, "⚠️ USDA API timed out after 30 seconds. Using baseline."
    except Exception as e:
        return fallback_prices, f"⚠️ Python Error: {str(e)}"

# ==========================================
# AUTO DATA PIPELINE: SUPPLY (USDA API)
# ==========================================
@st.cache_data(ttl=3600)  
def fetch_live_supply_data():
    current_year = datetime.now().year
    current_week_num = datetime.now().isocalendar()[1]
    url = "https://quickstats.nass.usda.gov/api/api_GET/"
    
    harvest_pct = 0.0
    last_week_pct = 0.0
    total_production = 1800000000.0  
    status_msg_prod = "⚠️ Using hardcoded baseline production."
    status_msg_harv = ""
    
    forecast_payload = {
        "key": USDA_API_KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "PRODUCTION, FORECAST",
        "short_desc": "CORN, GRAIN - PRODUCTION, FORECAST, MEASURED IN BU",
        "agg_level_desc": "STATE", 
        "state_name": "NEBRASKA",
        "year": str(current_year),
        "format": "JSON"
    }
    
    production_found = False
    try:
        fc_response = requests.get(url, params=forecast_payload, timeout=10)
        if fc_response.status_code == 200:
            fc_records = fc_response.json().get('data', [])
            if fc_records:
                newest_fc = fc_records[0] 
                total_production = float(newest_fc['Value'].replace(',', ''))
                production_found = True
                status_msg_prod = f"📈 Using live {current_year} WASDE Production Forecast."
    except Exception:
        pass 

    if not production_found:
        prod_payload = {
            "key": USDA_API_KEY,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "group_desc": "FIELD CROPS",
            "commodity_desc": "CORN",
            "statisticcat_desc": "PRODUCTION",
            "short_desc": "CORN, GRAIN - PRODUCTION, MEASURED IN BU",
            "prodn_practice_desc": "ALL PRODUCTION PRACTICES",
            "agg_level_desc": "STATE", 
            "state_name": "NEBRASKA",
            "year": str(current_year - 1), 
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
                    status_msg_prod = f"📉 {current_year} forecasts unavailable. Using {current_year-1} final harvest as proxy."
        except Exception:
            pass 

    harvest_payload = {
        "key": USDA_API_KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "PROGRESS",
        "short_desc": "CORN, GRAIN - PROGRESS, MEASURED IN PCT HARVESTED",
        "agg_level_desc": "STATE", 
        "state_name": "NEBRASKA",
        "year__GE": str(current_year - 1), 
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=harvest_payload, timeout=10)
        if response.status_code == 200:
            records = response.json().get('data', [])
            exact_match_value, last_week_match_value = None, None
            highest_value_this_year, records_this_year = 0, 0
            
            for record in records:
                if int(record['year']) != current_year: continue
                records_this_year += 1
                record_week = int(record['reference_period_desc'].split('#')[-1])
                record_val = float(record['Value'])
                
                if record_val > highest_value_this_year: highest_value_this_year = record_val
                if record_week == current_week_num: exact_match_value = record_val
                if record_week == current_week_num - 1: last_week_match_value = record_val
            
            if exact_match_value is not None:
                harvest_pct = exact_match_value / 100.0
                last_week_pct = (last_week_match_value / 100.0) if last_week_match_value is not None else 0.0
                status_msg_harv = f"🚜 Active Harvest: USDA progress report ({exact_match_value}%)."
            elif records_this_year == 0 or highest_value_this_year == 0:
                harvest_pct, last_week_pct = 0.0, 0.0
                status_msg_harv = f"🌱 Pre-Harvest Season: Defaulting to 0%."
            else:
                harvest_pct, last_week_pct = 1.0, 1.0
                status_msg_harv = f"❄️ Post-Harvest: USDA hit {highest_value_this_year}%. Defaulting to 100%."
        else:
            status_msg_harv = "⚠️ USDA API Error."
    except Exception:
        status_msg_harv = "⚠️ Network Error."

    final_status_msg = f"{status_msg_harv}\n\n{status_msg_prod}"
    return harvest_pct, last_week_pct, total_production, final_status_msg

# ==========================================
# AUTO DATA PIPELINE: DEMAND (USDA & EIA APIs)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_livestock_demand():
    current_year = datetime.now().year
    url = "https://quickstats.nass.usda.gov/api/api_GET/"
    
    livestock_head = 2500000.0 
    status_msg = "⚠️ Using hardcoded baseline for livestock demand."
    
    payload = {
        "key": USDA_API_KEY,
        "short_desc": "CATTLE, ON FEED - INVENTORY",
        "state_name": "NEBRASKA",
        "year__GE": str(current_year - 1),
        "format": "JSON"
    }
    
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    try:
        response = requests.get(url, params=payload, timeout=10)
        if response.status_code == 200:
            records = response.json().get('data', [])
            if records:
                valid_records = []
                for r in records:
                    try:
                        year = int(r['year'])
                        period_upper = r['reference_period_desc'].upper()
                        month_num = 1
                        for m_name, m_val in month_map.items():
                            if m_name in period_upper:
                                month_num = m_val
                                break
                        valid_records.append((year, month_num, r))
                    except Exception:
                        continue
                
                if valid_records:
                    valid_records.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    newest_record = valid_records[0][2]
                    
                    livestock_head = float(newest_record['Value'].replace(',', ''))
                    record_month = newest_record['reference_period_desc']
                    record_year = newest_record['year']
                    
                    status_msg = f"🐄 Live USDA Cattle on Feed: {livestock_head:,.0f} head ({record_month.title()} {record_year})."
    except Exception:
        pass
        
    return livestock_head, status_msg

@st.cache_data(ttl=3600)
def fetch_ethanol_demand():
    url = "https://api.eia.gov/v2/petroleum/pnp/wprode/data/"
    ethanol_bpd = 990.0 
    status_msg = "⚠️ Using hardcoded baseline for ethanol demand."
    
    payload = {
        "api_key": EIA_API_KEY,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "W_EPOOXE_YOP_R20_MBBLD",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 10 
    }
    
    try:
        response = requests.get(url, params=payload, timeout=10)
        if response.status_code == 200:
            data = response.json().get('response', {}).get('data', [])
            if data:
                newest_record = data[0]
                ethanol_bpd = float(newest_record['value'])
                record_date = newest_record['period']
                status_msg = f"🏭 Live EIA Ethanol Production: {ethanol_bpd:,.0f}k Barrels/Day (Week of {record_date})."
    except Exception:
        pass
        
    return ethanol_bpd, status_msg

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
        master_sheet["Period"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
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

def build_feature_row(week_num, seasonality, weekly_bushels, cumulative_harvest, is_harvesting, demand_ethanol, demand_livestock):
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
feature_columns = metadata.get("feature_columns", ["Week_Num", "Seasonality", "Weekly_Bushels_Produced", "Cumulative_Harvest", "Is_Harvesting", "Demand_Ethanol", "Demand_Livestock"])

st.title("🌽 Harvest or Hold? Nebraska Corn Market Forecaster")

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
st.sidebar.number_input("Momentum Window Size (Weeks)", min_value=1, max_value=52, value=window_size, step=1, disabled=True)

auto_momentum = st.sidebar.checkbox("📡 Auto-fetch Live Regional Prices", value=True)

if auto_momentum:
    with st.sidebar.status("Fetching Live Price Data..."):
        live_prices_str, price_status = fetch_recent_prices()
    
    st.sidebar.info(price_status)
    recent_prices_input = st.sidebar.text_input(f"Recent Prices (enter at least {window_size})", live_prices_str, disabled=True)
else:
    recent_prices_input = st.sidebar.text_input(f"Recent Prices (enter at least {window_size})", "3.78, 3.83, 3.70, 3.84")

st.sidebar.markdown("---")
st.sidebar.subheader("Seasonality")
use_auto_seasonality = st.sidebar.checkbox("Auto-fill seasonality by forecast week", value=True)
manual_seasonality = st.sidebar.number_input("Manual Seasonality Override", value=float(seasonality_map.get(current_week, 4.50)), format="%.2f")

# ==========================================
# AUTO DATA: DEMAND FACTORS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Demand Factors")

auto_demand = st.sidebar.checkbox("📡 Auto-fetch Live Demand Data", value=True)

if auto_demand:
    with st.sidebar.status("Fetching Live Demand Data..."):
        live_cattle, cattle_status = fetch_livestock_demand()
        live_ethanol, ethanol_status = fetch_ethanol_demand()
    
    st.sidebar.info(cattle_status)
    demand_livestock = st.sidebar.number_input("Livestock Demand (Head)", value=live_cattle, disabled=True, format="%.0f")
    
    st.sidebar.info(ethanol_status)
    demand_ethanol = st.sidebar.number_input("Ethanol Demand (Thousand Barrels/Day)", value=live_ethanol, disabled=True, format="%.0f")
else:
    demand_livestock = st.sidebar.number_input("Livestock Demand (Head)", value=2500000.0, step=10000.0, format="%.0f")
    demand_ethanol = st.sidebar.number_input("Ethanol Demand (Thousand Barrels/Day)", value=990.0, step=1.0, format="%.0f")

# ==========================================
# AUTO DATA: SUPPLY FACTORS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Supply Factors")

auto_supply = st.sidebar.checkbox("📡 Auto-fetch Live Supply Data", value=True)

if auto_supply:
    with st.sidebar.status("Fetching Live USDA Data..."):
        harvest_pct, last_week_pct, live_production, status_text = fetch_live_supply_data()
    
    st.sidebar.info(status_text)
    
    is_harvesting = 1 if 0.0 < harvest_pct < 1.0 else 0
    cumulative_harvest = harvest_pct * live_production
    weekly_pct_change = max(0.0, harvest_pct - last_week_pct) 
    weekly_bushels = weekly_pct_change * live_production
    
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
        # Flattened validation check to prevent SyntaxErrors
        recent_prices = parse_recent_prices(recent_prices_input, window_size)
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

    except ValueError as val_err:
        st.error(str(val_err))
    except Exception as e:
        st.error("⚠️ System Interruption Detected")
        st.info("⏳ We are currently waiting for the newest/updated data to sync, or a required field is missing. Please check back later or verify your inputs.")

st.markdown("---")
st.warning("**Disclaimer:** These projections are estimates based on historical trends and current inputs. They are not guaranteed to be 100% accurate. The model cannot effectively predict outliers caused by 'black swan' events, such as extreme weather disasters, unpredictable geopolitical shifts, or sudden market crashes.")
