import streamlit as st
import hopsworks
import joblib
import pandas as pd
import requests
import os
from datetime import datetime, timedelta, time, UTC # Use UTC
import warnings

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🍃",
    layout="wide"
)
st.title("AQI Prediction Dashboard 🍃")
st.info("This dashboard uses a trained XGBoost model to forecast the US AQI for the next 3 days.")

# --- Suppress specific warnings for a cleaner UI ---
warnings.filterwarnings("ignore", category=UserWarning, message="The installed hopsworks client version")

# --- 2. Load Secrets (from Streamlit Secrets) ---
# When you deploy, set these in Streamlit Cloud's 'Settings' -> 'Secrets'
try:
    HOPSWORKS_KEY = os.environ.get("HOPSWORKS_API_KEY", st.secrets["HOPSWORKS_API_KEY"])
    API_KEY = os.environ.get("OPENWEATHER_API_KEY", st.secrets["OPENWEATHER_API_KEY"])
except KeyError:
    st.error("🔴 API keys not found. Please set HOPSWORKS_API_KEY and OPENWEATHER_API_KEY in your Streamlit secrets.")
    st.stop()
    
LAT = 24.8607
LON = 67.0011

# --- 3. Helper Functions (Copied directly from your script) ---
breakpoints = {
    "pm2_5": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)],
    "o3": [(0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150), (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)]
}
def calc_aqi(conc, pollutant):
    for (C_low, C_high, I_low, I_high) in breakpoints[pollutant]:
        if C_low <= conc <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (conc - C_low) + I_low
    return None
def compute_us_aqi(row):
    aqi_values = []
    for pol in ["pm2_5", "pm10", "o3", "no2", "so2", "co"]:
        val = row.get(pol)
        if val is not None: aqi = calc_aqi(val, pol);
        if aqi is not None: aqi_values.append(aqi)
    return max(aqi_values) if aqi_values else None

def fetch_weather(start, end):
    url = f"https://history.openweathermap.org/data/2.5/history/city?lat={LAT}&lon={LON}&type=hour&start={start}&end={end}&appid={API_KEY}"
    r = requests.get(url); r.raise_for_status(); data = r.json()
    rows = []
    for entry in data.get("list", []):
        rows.append({"dt": entry["dt"], "temp": entry["main"]["temp"], "humidity": entry["main"]["humidity"], "wind_speed": entry["wind"]["speed"]})
    return pd.DataFrame(rows)

def fetch_pollution(start, end):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}"
    r = requests.get(url); r.raise_for_status(); data = r.json()
    rows = []
    for entry in data.get("list", []):
        rows.append({
            "dt": entry["dt"], "aqi": entry["main"]["aqi"], "co": entry["components"]["co"], "no2": entry["components"]["no2"],
            "o3": entry["components"]["o3"], "so2": entry["components"]["so2"], "pm2_5": entry["components"]["pm2_5"],
            "pm10": entry["components"]["pm10"], "nh3": entry["components"]["nh3"],
        })
    return pd.DataFrame(rows)

# --- 4. Caching Data Loading Functions ---
# @st.cache_resource tells Streamlit to run this func once and save the result (the model)
@st.cache_resource(ttl=600) # Cache for 10 minutes
def get_model_and_history():
    """Connects to Hopsworks, gets the latest model and 10 days of data."""
    with st.spinner("Connecting to Hopsworks and loading model..."):
        # project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
        project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
        fs = project.get_feature_store()
        
        mr = project.get_model_registry()
        model = mr.get_model(name="aqi_predictor") # Gets latest version
        model_dir = model.download()
        
        print("Downloaded model directory:", model_dir)
        print("Contents:", os.listdir(model_dir))
        
        model = joblib.load(model_dir + "/aqi_gb_model.pkl")
        st.write(f"✅ Model version {model.version} loaded.")
        
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        ten_days_ago = datetime.now(UTC).date() - timedelta(days=10)
        history_df = aqi_fg.filter(aqi_fg.date > ten_days_ago).read()
        history_df['date'] = pd.to_datetime(history_df['date']).dt.tz_convert('UTC')
        history_df = history_df.sort_values('date').dropna() # Drop NaNs
        st.write(f"✅ Historical data loaded (up to {history_df['date'].max().date()}).")
        
    return model, history_df

# @st.cache_data tells Streamlit to save the *data* this func returns
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_future_weather_and_pollutants():
    """
    Fetches 5-day weather & pollution forecast and aggregates to daily.
    This provides the *INPUTS* for our model.
    """
    with st.spinner("Fetching 3-day weather & pollutant forecast..."):
        weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}"
        r_weather = requests.get(weather_url); r_weather.raise_for_status()
        weather_data = r_weather.json()

        poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={API_KEY}"
        r_poll = requests.get(poll_url); r_poll.raise_for_status()
        poll_data = r_poll.json()

        weather_rows = []
        for entry in weather_data.get("list", []):
            weather_rows.append({"dt": entry["dt"], "temp": entry["main"]["temp"], "humidity": entry["main"]["humidity"], "wind_speed": entry["wind"]["speed"]})
        weather_df = pd.DataFrame(weather_rows)
        weather_df['date'] = pd.to_datetime(weather_df['dt'], unit='s').dt.date

        poll_rows = []
        for entry in poll_data.get("list", []):
            poll_rows.append({
                "dt": entry["dt"], "aqi": entry["main"]["aqi"], "co": entry["components"]["co"], "no2": entry["components"]["no2"],
                "o3": entry["components"]["o3"], "so2": entry["components"]["so2"], "pm2_5": entry["components"]["pm2_5"],
                "pm10": entry["components"]["pm10"], "nh3": entry["components"]["nh3"],
            })
        poll_df = pd.DataFrame(poll_rows)
        poll_df['date'] = pd.to_datetime(poll_df['dt'], unit='s').dt.date

        daily_weather_df = weather_df.groupby("date").mean(numeric_only=True).reset_index()
        daily_poll_df = poll_df.groupby("date").mean(numeric_only=True).reset_index()
        
        forecast_df = pd.merge(daily_weather_df, daily_poll_df, on="date", how="inner")
        
        forecast_df["us_aqi"] = None # Placeholder
        forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.tz_localize('UTC')
        
        # Fill any missing forecast values
        forecast_df = forecast_df.ffill().bfill()
        st.write("✅ Future input data fetched.")
        
    return forecast_df.set_index('date')

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_todays_estimated_data(history_df):
    """
    Runs a 'mini-ETL' to get today's data *so far*.
    """
    with st.spinner("Fetching and estimating data for 'today'..."):
        today_utc = datetime.now(UTC).date()
        start_dt = datetime.combine(today_utc, time.min)
        end_dt = datetime.combine(today_utc, time.max)
        START_UNIX = int(start_dt.timestamp())
        END_UNIX = int(end_dt.timestamp())

        weather_df = fetch_weather(START_UNIX, END_UNIX)
        pollution_df = fetch_pollution(START_UNIX, END_UNIX)

        if weather_df.empty or pollution_df.empty:
            st.warning(f"No data available *yet* for today ({today_utc}). Using yesterday's data as 'today'.")
            return history_df.iloc[-1:]

        hourly_df = pd.merge(weather_df, pollution_df, on="dt", how="inner")
        hourly_df["date"] = pd.to_datetime(hourly_df["dt"], unit='s').dt.date
        todays_row = hourly_df.groupby("date").agg({
            "temp": "mean", "humidity": "mean", "wind_speed": "mean", "aqi": "mean",
            "co": "mean", "no2": "mean", "o3": "mean", "so2": "mean",
            "pm2_5": "mean", "pm10": "mean", "nh3": "mean"
        }).reset_index()
        
        todays_row["date"] = pd.to_datetime(todays_row["date"]).dt.tz_localize('UTC')
        todays_row["us_aqi"] = todays_row.apply(compute_us_aqi, axis=1)
        # Fill any missing values
        todays_row = todays_row.ffill().bfill() 
        st.write("✅ 'Today's' data estimated.")
        
    return todays_row

# --- 5. Main Prediction Logic ---
# This is the exact code from your test script, now wrapped in UI elements
try:
    model, history_df = get_model_and_history()
    future_inputs_df = get_future_weather_and_pollutants()
    todays_row = get_todays_estimated_data(history_df)
    
    model_features = model.feature_names_in_
    
    if todays_row.iloc[0]['date'].date() != history_df.iloc[-1]['date'].date():
        base_data = pd.concat([history_df, todays_row]).set_index('date')
    else:
        base_data = history_df.set_index('date')

    predictions = []
    
    st.header("🚀 3-Day AQI Forecast (Predicted by your Model)")
    cols = st.columns(3) # Create 3 columns for the 3 days

    with st.spinner("Running 3-Day Autoregressive Forecast..."):
        for i in range(3):
            target_date = (datetime.now(UTC).date() + timedelta(days=i+1))
            target_date_ts = pd.Timestamp(target_date).tz_localize('UTC')
            
            # --- Build the Feature Row ---
            feature_row = {}
            
            data_lag_1 = base_data.loc[base_data.index == (target_date_ts - timedelta(days=1))].iloc[0]
            data_lag_3 = base_data.loc[base_data.index == (target_date_ts - timedelta(days=3))].iloc[0]
            data_lag_7 = base_data.loc[base_data.index == (target_date_ts - timedelta(days=7))].iloc[0]

            forecasted_inputs = future_inputs_df.loc[future_inputs_df.index.date == target_date].iloc[0]

            # Add non-lagged features
            feature_row['temp'] = forecasted_inputs['temp']
            feature_row['humidity'] = forecasted_inputs['humidity']
            feature_row['wind_speed'] = forecasted_inputs['wind_speed']
            feature_row['aqi'] = forecasted_inputs['aqi'] # We provide this as it was trained on it
            feature_row['co'] = forecasted_inputs['co']
            feature_row['no2'] = forecasted_inputs['no2']
            feature_row['o3'] = forecasted_inputs['o3']
            feature_row['so2'] = forecasted_inputs['so2']
            feature_row['pm2_5'] = forecasted_inputs['pm2_5']
            feature_row['pm10'] = forecasted_inputs['pm10']
            feature_row['nh3'] = forecasted_inputs['nh3']

            # Populate all lag features
            feature_row['aqi_lag_1_day'] = data_lag_1['us_aqi']
            feature_row['aqi_lag_3_day'] = data_lag_3['us_aqi']
            feature_row['aqi_lag_7_day'] = data_lag_7['us_aqi']
            
            feature_row['temp_lag_1_day'] = data_lag_1['temp']
            feature_row['temp_lag_3_day'] = data_lag_3['temp']
            feature_row['temp_lag_7_day'] = data_lag_7['temp']
            
            feature_row['wind_lag_1_day'] = data_lag_1['wind_speed']
            feature_row['wind_lag_3_day'] = data_lag_3['wind_speed']
            feature_row['wind_lag_7_day'] = data_lag_7['wind_speed']
            
            feature_row['humidity_lag_1_day'] = data_lag_1['humidity']
            feature_row['humidity_lag_3_day'] = data_lag_3['humidity']
            feature_row['humidity_lag_7_day'] = data_lag_7['humidity']
            
            feature_row['so2_lag_1_day'] = data_lag_1['so2']
            feature_row['so2_lag_3_day'] = data_lag_3['so2']
            feature_row['so2_lag_7_day'] = data_lag_7['so2']
            
            feature_row['nh3_lag_1_day'] = data_lag_1['nh3']
            feature_row['nh3_lag_3_day'] = data_lag_3['nh3']
            feature_row['nh3_lag_7_day'] = data_lag_7['nh3']
            
            feature_row['no2_lag_1_day'] = data_lag_1['no2']
            feature_row['no2_lag_3_day'] = data_lag_3['no2']
            feature_row['no2_lag_7_day'] = data_lag_7['no2']
            
            feature_row['co_lag_1_day'] = data_lag_1['co']
            feature_row['co_lag_3_day'] = data_lag_3['co']
            feature_row['co_lag_7_day'] = data_lag_7['co']
            
            feature_row['pm10_lag_1_day'] = data_lag_1['pm10']
            feature_row['pm10_lag_3_day'] = data_lag_3['pm10']
            feature_row['pm10_lag_7_day'] = data_lag_7['pm10']
            
            feature_row['pm2_5_lag_1_day'] = data_lag_1['pm2_5']
            feature_row['pm2_5_lag_3_day'] = data_lag_3['pm2_5']
            feature_row['pm2_5_lag_7_day'] = data_lag_7['pm2_5']
            
            feature_row['month'] = target_date_ts.month

            # --- Predict! ---
            feature_df = pd.DataFrame([feature_row], columns=model_features)

            prediction = model.predict(feature_df)
            predicted_aqi = round(prediction[0])
            predictions.append({'date': target_date, 'predicted_aqi': predicted_aqi})
            
            # --- Display in its own column ---
            with cols[i]:
                st.metric(label=f"**{target_date.strftime('%A, %b %d')}**", value=int(predicted_aqi))

            # --- Autoregressive Step ---
            forecasted_inputs_row = future_inputs_df.loc[future_inputs_df.index.date == target_date].iloc[0]
            new_row_data = forecasted_inputs_row.to_dict()
            new_row_data['us_aqi'] = predicted_aqi # Add the value we just predicted
            new_row_df = pd.DataFrame([new_row_data], index=[target_date_ts])
            base_data = pd.concat([base_data, new_row_df])

    # --- 7. Final Output ---
    st.success("✅ Forecast Complete!")
    
    st.subheader("Forecast Chart")
    forecast_results_df = pd.DataFrame(predictions).set_index('date')
    st.line_chart(forecast_results_df, y='predicted_aqi')
    
    st.subheader("Raw Data")
    with st.expander("Show Latest Historical Data from Hopsworks"):
        st.dataframe(history_df.tail(10), use_container_width=True)
    with st.expander("Show Future Input Data from API"):
        st.dataframe(future_inputs_df, use_container_width=True)


except Exception as e:
    st.error(f"--- 🔴 An error occurred ---")
    st.exception(e) # This will print the full traceback to the UI
