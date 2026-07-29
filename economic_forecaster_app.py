import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

# ==========================================
# WEEK TO CALENDAR MONTH MAPPING HELPER
# ==========================================
def get_week_calendar_label(week_num):
    # If the user toggled the next crop year, week_num might be > 52. Wrap it for logic.
    display_week = ((week_num - 1) % 52) + 1
    
    month_map = {
        (1, 4): "January (Post-Harvest Storage)",
        (5, 8): "February (Pre-Planting Planning)",
        (9, 13): "March (USDA Planting Intentions)",
        (14, 17): "April (Field Prep & Early Planting)",
        (18, 22): "May (Planting & Emergence)",
        (23, 26): "June (Crop Growth & Weather Watch)",
        (27, 30): "July (Pollination & Critical Window)",
        (31, 35): "August (Grain Fill & Crop Progress)",
        (36, 39): "September (Early Harvest & Husker Harvest Days)",
        (40, 44): "October (Peak Harvest)",
        (45, 48): "November (Late Harvest & Post-Harvest)",
        (49, 52): "December (Year-End Grain Marketing)"
    }
    
    year_label = " (Next Year)" if week_num > 52 else ""
    for (start, end), label in month_map.items():
        if start <= display_week <= end:
            return f"Week {display_week} — {label}{year_label}"
    return f"Week {display_week}{year_label}"

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
    fallback_prices = "3.78, 3.83, 3.70, 3.84"
    
    if not AMS_API_KEY:
        return fallback_prices, "⚠️ AMS API Key missing. Using manual baseline."
        
    url = "https://marsapi.ams.usda.gov/services/v1.2/reports/3225/Report%20Detail"
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
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
            
            if "CORN" in commodity:
                date_val = r.get('published_date', r.get('report_date', ''))
                p_min = r.get('price Min')
                p_max = r.get('price Max')
                p_avg = r.get('avg_price')
                
                final_price = None
                
                if p_avg is not None and str(p_avg).strip() != "":
                    final_price = float(p_avg)
                elif p_min is not None and p_max is not None:
                    try:
                        final_price = (float(p_min) + float(p_max)) / 2.0
                    except ValueError:
                        pass
                        
                if date_val and final_price is not None:
                    location_string = str(r.get('trade_loc', '')) + " " + str(r.get('market_location_name', ''))
                    corn_records.append({
                        'date': date_val,
                        'price': final_price,
                        'location': location_string.upper()
                    })
                    
        if not corn_records:
            return fallback_prices, "⚠️ Found 0 Corn rows with valid numerical prices."
            
        east_records = [row for row in corn_records if "EAST" in row['location']]
        
        if len(east_records) > 0:
            target_records = east_records
            status_message = "📈 Live USDA AMS Cash Prices Fetched (East Region True Mean)!"
        else:
            target_records = corn_records
            status_message = "📈 Live USDA AMS Cash Prices Fetched (Statewide True Mean)!"
            
        df = pd.DataFrame(target_records)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        daily_avg = df.groupby('date')['price'].mean().reset_index()
        daily_avg = daily_avg.sort_values('date', ascending=False)
        
        unique_prices = daily_avg['price'].round(2).tolist()[:4]
        
        if len(unique_prices) < 4:
            return fallback_prices, f"⚠️ Only found {len(unique_prices)} valid days of data. Need 4."
            
        unique_prices.reverse()
        price_str = ", ".join(map(str, unique_prices))
        return price_str, status_message
        
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

# --- CALCULATE SEASONALITY AVERAGE FOR PERCENTAGE INDEXING ---
valid_seasonalities = [v for k, v in seasonality_map.items() if not pd.isna(v)]
seasonality_grand_avg = sum(valid_seasonalities) / len(valid_seasonalities) if valid_seasonalities else 4.23

window_size = int(metadata.get("window_size", 4))
feature_columns = metadata.get("feature_columns", ["Week_Num", "Seasonality", "Weekly_Bushels_Produced", "Cumulative_Harvest", "Is_Harvesting", "Demand_Ethanol", "Demand_Livestock"])

# 🔒 PRIVACY GUARANTEE BANNER & TRANSPARENCY
st.markdown("""
<div style="background-color: #f0f7f4; border-left: 5px solid #2e7d32; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
    <strong style="color: #2e7d32; font-size: 16px;">🔒 100% Private & Open-Source Decision Support</strong><br>
    <span style="color: #333; font-size: 14px;">No user account required. We do not collect, store, or sell your farm's financial, yield, or acreage data. Built as a free Capstone project for Nebraska growers.</span>
</div>
""", unsafe_allow_html=True)

st.title("🌽 Harvest or Hold? Nebraska Corn Market Forecaster")

view_mode = st.radio("Select Dashboard View:", ["👨‍🌾 Simple View", "📊 Advanced View"], horizontal=True)
st.markdown("---")

# ==========================================
# MARKET CONTROL PANEL (SIDEBAR)
# ==========================================
st.sidebar.header("Market Control Panel")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Your Economics (Break-Even)")
user_breakeven = st.sidebar.number_input(
    "Cost of Production / Break-Even ($/bu)",
    value=4.50, step=0.05,
    help="Your personal break-even price. The recommendation engine ties predictions directly to this number."
)
monthly_storage_cost = st.sidebar.number_input(
    "Est. Monthly Storage / Holding Cost ($/bu/month)",
    value=0.05, step=0.01,
    help="Includes elevator fees, utility shrink, or operating loan interest while holding grain."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Timeline Planner")

# NEXT YEAR TOGGLE: This allows users to push the target week into next year (Weeks 53-104)
plan_next_year = st.sidebar.checkbox("🔀 Plan into Next Crop Year")

current_week = st.sidebar.slider("Current Week Number", min_value=1, max_value=51, value=6)

if plan_next_year:
    target_week = st.sidebar.slider("Target Forecast Week", min_value=current_week + 1, max_value=104, value=current_week + 52)
else:
    target_week = st.sidebar.slider("Target Forecast Week", min_value=current_week + 1, max_value=52, value=min(current_week + 6, 52))

st.sidebar.info(f"📆 Current: {get_week_calendar_label(current_week)}\n📆 Target: {get_week_calendar_label(target_week)}")

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
    with st.sidebar.expander("🔍 Guide: Manual Price Fetching"):
        st.markdown("""
        1. Open the [USDA My Market News (Report 3225)](https://mymarketnews.ams.usda.gov/viewReport/3225).
        2. **Note:** Please use the price listed for **EAST**, as the entire forecast model is built on this specific region's structural baseline.
        3. Extract the last 4 unique daily prices and type them into the field separated by commas.
        """)

st.sidebar.markdown("---")
st.sidebar.subheader("Seasonality Percentage Index")
use_auto_seasonality = st.sidebar.checkbox("Auto-fill seasonality by forecast week", value=True)

# Compute the default index for the current week. Wrap logic if in Next Year.
wrapped_curr_week = ((current_week - 1) % 52) + 1
default_raw_seasonality = float(seasonality_map.get(wrapped_curr_week, seasonality_grand_avg))
default_index = default_raw_seasonality / seasonality_grand_avg

manual_season_index = st.sidebar.number_input("Manual Seasonality Index (1.0 = Average)", value=default_index, step=0.01, format="%.2f")

st.sidebar.markdown(f"""
💡 **Tip:** The model now uses a Percentage Index rather than raw dollars. An index of `1.03` means the historical price for this week is 3% higher than the baseline average.
*(Baseline average used: **${seasonality_grand_avg:.2f}**)*
""")

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
    with st.sidebar.expander("🔍 Guide: Manual Livestock Demand"):
        st.markdown("""
        1. Open the [USDA QuickStats Tool](https://quickstats.nass.usda.gov/).
        2. Apply the following strict filters:
           - **Program:** Survey
           - **Sector:** Animals & Products
           - **Group:** Livestock
           - **Commodity:** Cattle
           - **Category:** Inventory
           - **Data Item:** CATTLE, ON FEED - INVENTORY
           - **Geographic Level:** State → Nebraska
        3. Press **Get Data** and copy/paste the most recent reporting value.
        """)
        
    demand_ethanol = st.sidebar.number_input("Ethanol Demand (Thousand Barrels/Day)", value=990.0, step=1.0, format="%.0f")
    with st.sidebar.expander("🔍 Guide: Manual Ethanol Demand"):
        st.markdown("""
        1. Head over to the official [EIA Weekly Ethanol Production Stream](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=W_EPOOXE_YOP_R20_MBBLD&f=W).
        2. Locate the most current week's entry on the ledger and input it above.
        """)

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
    
    with st.sidebar.expander("🔍 Guide: Manual Supply Computations"):
        st.markdown("""
        **Step 1:** Head to the [USDA QuickStats Interface](https://quickstats.nass.usda.gov/).
        
        **Step 2:** Isolate via the criteria below:
        - **Program:** Survey 
        - **Sector:** Crops 
        - **Commodity:** Corn 
        - **Category:** Progress 
        - **Data Item:** CORN, GRAIN - PROGRESS, MEASURED IN PCT HARVESTED 
        - **State:** Nebraska
        
        **Step 3:** Record the exact percentage completion noted for your target observation week.
        
        **Step 4:** Multiply that percentage (rendered as a formal decimal) against the newest operational annual baseline production metrics (*e.g., 2,027,300,000 bushels for the 2025 seasonal harvest cyclical tracking loop*) to extract your **Cumulative Harvest**.
        
        **Step 5:** Compute the net raw dynamic differentiation by subtracting last week's recorded Cumulative Harvest value from this week's current result to isolate your **Weekly Bushels Produced**.
        """)

st.sidebar.markdown("---")
clip_predictions = st.sidebar.checkbox("Cap weekly deviation at ±$0.20", value=True)

# ==========================================
# MAIN FORECASTING LOGIC
# ==========================================
if view_mode == "📊 Advanced View":
    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.subheader("Model Logic")
        st.markdown("- Baseline: 4-week moving average of recent prices\n- Deviation driver: supply, demand, and seasonality inputs\n- Forecast style: chained week-by-week projection (Live Model Inference)")
    with right_col:
        st.subheader("Model Inputs Used")
        st.code(", ".join(feature_columns), language="text")

if st.button("🚀 Run Chained Forecast", type="primary"):
    try:
        recent_prices = parse_recent_prices(recent_prices_input, window_size)
        initial_average_price = np.mean(recent_prices[-window_size:])
        forecast_rows = []

        for week in range(current_week + 1, target_week + 1):
            moving_avg = float(np.mean(recent_prices[-window_size:]))
            
            # Wrap week back to 1-52 if it goes into next year, so seasonality map doesn't break
            model_week = ((week - 1) % 52) + 1
            
            if use_auto_seasonality:
                raw_season_val = float(seasonality_map.get(model_week, seasonality_grand_avg))
                seasonality_value = raw_season_val / seasonality_grand_avg
            else:
                seasonality_value = float(manual_season_index)

            # --- TRUE MODEL PREDICTION FIX ---
            future_conditions = build_feature_row(
                week_num=model_week, seasonality=seasonality_value, weekly_bushels=weekly_bushels,
                cumulative_harvest=cumulative_harvest, is_harvesting=is_harvesting,
                demand_ethanol=demand_ethanol, demand_livestock=demand_livestock,
            )[feature_columns]

            raw_model_output = float(rf_model.predict(future_conditions)[0])
            
            # Automatically detect if the model is predicting an absolute price (e.g., $4.10) or a deviation (e.g., -$0.10)
            if raw_model_output > 1.0:
                raw_deviation = raw_model_output - moving_avg
            else:
                raw_deviation = raw_model_output
            
            try:
                # Calculate probability intervals via ensemble spread
                all_tree_preds = [tree.predict(future_conditions.values)[0] for tree in rf_model.estimators_]
                tree_std = float(np.std(all_tree_preds))
            except Exception:
                tree_std = 0.05

            # Apply the +/- $0.20 limit ONLY to the true deviation
            deviation = float(np.clip(raw_deviation, -MAX_WEEKLY_SHIFT, MAX_WEEKLY_SHIFT)) if clip_predictions else raw_deviation
            predicted_price = moving_avg + deviation
            
            # Update the chained rolling window with our new predicted price
            recent_prices.append(predicted_price)

            forecast_rows.append({
                "Week": week, "Seasonality Index": round(seasonality_value, 4), "Momentum": round(moving_avg, 4),
                "Predicted Deviation": round(deviation, 4), "Predicted Price": round(predicted_price, 4), "StdDev": tree_std,
            })

        forecast_df = pd.DataFrame(forecast_rows)
        final_price = forecast_df.iloc[-1]['Predicted Price']
        price_change = final_price - initial_average_price

        # Dynamic Recommendation Logic tied to Personal Economics
        months_held = max(0, target_week - current_week) / 4.33
        total_storage_cost = months_held * monthly_storage_cost
        net_margin = final_price - user_breakeven
        
        if net_margin > (total_storage_cost * 1.5):
            recommendation = "🟢 SELL / FORWARD CONTRACT"
            reason = f"The predicted market price of **${final_price:.2f}** is **${net_margin:.2f} above your break-even** (${user_breakeven:.2f}) and comfortably covers your estimated storage holding costs (${total_storage_cost:.2f} total). Securing price coverage at this level is strongly advised."
        elif net_margin >= 0:
            recommendation = "🟡 MONITOR CLOSELY / HOLD SHORT-TERM"
            reason = f"The predicted price of **${final_price:.2f}** covers your break-even (${user_breakeven:.2f}) with a thin margin of **${net_margin:.2f}**. Keep an eye on cumulative storage costs (${total_storage_cost:.2f}) to ensure delay doesn't erode profits."
        else:
            recommendation = "🔴 HOLD / EVALUATE RISK MANAGEMENT"
            reason = f"The predicted price of **${final_price:.2f}** is **${abs(net_margin):.2f} BELOW your break-even** (${user_breakeven:.2f}). Holding grain or evaluating crop insurance/basis protections is recommended before locking in sales."

        if view_mode == "👨‍🌾 Simple View":
            st.header("🎯 Your Forecast Recommendation")
            st.subheader(f"{recommendation}")
            st.write(f"**The Bottom Line:** {reason}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Price", f"${final_price:.2f} / bu", delta=f"{price_change:+.2f} vs current")
            with col2:
                st.metric("Your Break-Even Price", f"${user_breakeven:.2f} / bu")
            with col3:
                st.metric("Predicted Net Profit Margin", f"${net_margin:+.2f} / bu", delta_color="normal" if net_margin >= 0 else "inverse")
                
            st.info("Switch to the 'Advanced View' at the top of the page to see probability bands and trajectory charts.")

        elif view_mode == "📊 Advanced View":
            st.subheader(f"📊 Advanced Risk Analytics & Probability Intervals")
            
            # Show top metrics
            final_std = forecast_df.iloc[-1]['StdDev']
            conf_lower = max(0.0, final_price - final_std)
            conf_upper = final_price + final_std
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(f"Point Forecast (Week {target_week})", f"${final_price:.2f}")
            with c2:
                st.metric("70% Probability Interval", f"${conf_lower:.2f} — ${conf_upper:.2f}")
            with c3:
                st.metric("Model Ensemble Spread (Std Dev)", f"±${final_std:.2f}")

            st.markdown("### 📈 Price Prediction Distribution (70% Confidence Band)")
            
            # Calculate bands
            forecast_df['Upper_Bound'] = forecast_df['Predicted Price'] + forecast_df['StdDev']
            forecast_df['Lower_Bound'] = (forecast_df['Predicted Price'] - forecast_df['StdDev']).clip(lower=0)
            
            fig = go.Figure()
            
            # Shaded probability area
            fig.add_trace(go.Scatter(
                x=forecast_df['Week'].tolist() + forecast_df['Week'].tolist()[::-1],
                y=forecast_df['Upper_Bound'].tolist() + forecast_df['Lower_Bound'].tolist()[::-1],
                fill='toself', fillcolor='rgba(46, 125, 50, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='70% Probability Range'
            ))
            
            # Main price line
            fig.add_trace(go.Scatter(
                x=forecast_df['Week'], y=forecast_df['Predicted Price'],
                mode='lines+markers', line=dict(color='#2e7d32', width=3),
                name='Forecast Price'
            ))
            
            fig.update_layout(
                title=f"Forecasted Price Trajectory ({get_week_calendar_label(current_week)} to {get_week_calendar_label(target_week)})",
                xaxis_title="Week Number",
                yaxis_title="Price ($/Bushel)",
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)', tickprefix="$")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Show Raw Data Table"):
                display_df = forecast_df.drop(columns=['StdDev', 'Upper_Bound', 'Lower_Bound'])
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ==========================================
        # HISTORICAL MARKET CONTEXT SECTION
        # ==========================================
        st.markdown("---")
        st.subheader("📜 Historical Market Context: Eastern Nebraska")

        # Display the historical price trend image
        try:
            st.image(
                "eastern_nebraska_historical_corn_prices.png", 
                caption="Historical Corn Prices in Eastern Nebraska (2016 - 2026)",
                use_container_width=True
            )
        except Exception:
            st.warning("⚠️ Historical price chart image (`eastern_nebraska_historical_corn_prices.png`) not found in directory.")

        # Explanatory text under the chart
        st.info(
            "**Market Insight:** Historical price trends demonstrate clear seasonal movements driven "
            "by harvest pressure, storage cycles, and spring/summer planting uncertainties. "
            "The model integrates these long-term historical dynamics alongside current live supply/demand signals "
            "to baseline its forward projections."
        )

    except ValueError as val_err:
        st.error(str(val_err))
    except Exception as e:
        st.error("⚠️ System Interruption Detected")
        st.info(f"Details: {str(e)}")

st.markdown("---")
st.caption("Looking for the legacy framework? Access the [Original Nebraska Corn Price Predictor Deployment V1](https://nebraska-corn-market-price-predictor.streamlit.app/) archive map.")
st.warning("**Disclaimer:** These projections are estimates based on historical trends and current inputs. They are not guaranteed to be 100% accurate. The model cannot effectively predict outliers caused by 'black swan' events, such as extreme weather disasters, unpredictable geopolitical shifts, or sudden market crashes.")
