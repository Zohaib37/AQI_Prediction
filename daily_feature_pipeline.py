import requests
import pandas as pd
from datetime import datetime, timedelta, time
import hopsworks
import os  # Added for environment variables

# --- Config ---
LAT = 24.8607
LON = 67.0011
# Get API keys from environment variables (for GitHub Actions)
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
HOPSWORKS_KEY = os.environ.get("HOPSWORKS_API_KEY")

if not API_KEY or not HOPSWORKS_KEY:
    raise Exception("API keys (OPENWEATHER_API_KEY, HOPSWORKS_API_KEY) not found in environment variables.")


def fetch_weather(start, end):
    url = (
        f"https://history.openweathermap.org/data/2.5/history/city"
        f"?lat={LAT}&lon={LON}&type=hour&start={start}&end={end}&appid={API_KEY}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    rows = []
    for entry in data.get("list", []):
        rows.append({
            "dt": entry["dt"],
            "temp": entry["main"]["temp"],
            "humidity": entry["main"]["humidity"],
            "wind_speed": entry["wind"]["speed"]
        })
    return pd.DataFrame(rows)

def fetch_pollution(start, end):
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    rows = []
    for entry in data.get("list", []):
        rows.append({
            "dt": entry["dt"],
            "aqi": entry["main"]["aqi"],
            "co": entry["components"]["co"],
            "no2": entry["components"]["no2"],
            "o3": entry["components"]["o3"],
            "so2": entry["components"]["so2"],
            "pm2_5": entry["components"]["pm2_5"],
            "pm10": entry["components"]["pm10"],
            "nh3": entry["components"]["nh3"],
        })
    return pd.DataFrame(rows)

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
        if val is not None:
            aqi = calc_aqi(val, pol)
            if aqi is not None:
                aqi_values.append(aqi)
    return max(aqi_values) if aqi_values else None

def run_daily_feature_pipeline():
    today_utc = datetime.utcnow().date()
    yesterday_utc = today_utc - timedelta(days=1)
    
    # Start is 00:00:00 on "yesterday"
    start_dt = datetime.combine(yesterday_utc, time.min)
    # End is 23:59:59 on "yesterday"
    end_dt = datetime.combine(yesterday_utc, time.max)

    START_UNIX = int(start_dt.timestamp())
    END_UNIX = int(end_dt.timestamp())

    print(f"--- Running Daily Feature Pipeline for {yesterday_utc} ---")
    print(f"Fetching data from {start_dt} (UTC) to {end_dt} (UTC)")

    # 2. Fetch hourly data for yesterday
    weather_df = fetch_weather(START_UNIX, END_UNIX)
    pollution_df = fetch_pollution(START_UNIX, END_UNIX)

    if weather_df.empty or pollution_df.empty:
        print(f"No data returned from APIs for {yesterday_utc}. Exiting.")
        return

    # 3. Combine and aggregate into ONE daily row
    hourly_df = pd.merge(weather_df, pollution_df, on="dt", how="inner")
    hourly_df["date"] = pd.to_datetime(hourly_df["dt"], unit="s").dt.date
    
    # Aggregate by day
    new_daily_row = hourly_df.groupby("date").agg({
        "temp": "mean", "humidity": "mean", "wind_speed": "mean", "aqi": "mean",
        "co": "mean", "no2": "mean", "o3": "mean", "so2": "mean",
        "pm2_5": "mean", "pm10": "mean", "nh3": "mean"
    }).reset_index()

    new_daily_row["date"] = pd.to_datetime(new_daily_row["date"]).dt.tz_localize('UTC')
    
    # Calculate US AQI
    new_daily_row["us_aqi"] = new_daily_row.apply(compute_us_aqi, axis=1)

    if new_daily_row.empty:
        print(f"Data aggregation failed for {yesterday_utc}. Exiting.")
        return

    print(f"Successfully aggregated data for {yesterday_utc}.")

    # 4. Connect to Hopsworks and get history for feature engineering
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)

    # Fetch last 10 days of data to calculate new lags/rolling windows
    ten_days_ago = yesterday_utc - timedelta(days=10)
    history_df = aqi_fg.filter(aqi_fg.date > ten_days_ago).read()
    history_df['date'] = pd.to_datetime(history_df['date']).dt.tz_convert('UTC')
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df = history_df.sort_values('date')

    print(f"Fetched {len(history_df)} rows of history from Hopsworks.")

    # 5. Perform Feature Engineering
    # Combine history with the new row to calculate features correctly
    combined_df = pd.concat([history_df, new_daily_row], ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)

    # combined_df['temp_1_day_lag'] = combined_df['temp'].shift(1)
    # combined_df['pm2_5_1_day_lag'] = combined_df['pm2_5'].shift(1)
    # combined_df['humidity_1_day_lag'] = combined_df['humidity'].shift(1)
    # combined_df['wind_speed_1_day_lag'] = combined_df['wind_speed'].shift(1)
    
    # combined_df['pm2_5_7_day_avg'] = combined_df['pm2_5'].shift(1).rolling(window=7, min_periods=1).mean()
    # combined_df['temp_7_day_avg'] = combined_df['temp'].shift(1).rolling(window=7, min_periods=1).mean()

    combined_df['aqi_lag_1_day'] = combined_df['us_aqi'].shift(1)
    combined_df['aqi_lag_3_day'] = combined_df['us_aqi'].shift(3)
    combined_df['aqi_lag_7_day'] = combined_df['us_aqi'].shift(7)

    combined_df['temp_lag_1_day'] = combined_df['temp'].shift(1)
    combined_df['temp_lag_3_day'] = combined_df['temp'].shift(3)
    combined_df['temp_lag_7_day'] = combined_df['temp'].shift(7)

    combined_df['wind_lag_1_day'] = combined_df['wind_speed'].shift(1)
    combined_df['wind_lag_3_day'] = combined_df['wind_speed'].shift(3)
    combined_df['wind_lag_7_day'] = combined_df['wind_speed'].shift(7)

    combined_df['humidity_lag_1_day'] = combined_df['humidity'].shift(1)
    combined_df['humidity_lag_3_day'] = combined_df['humidity'].shift(3)
    combined_df['humidity_lag_7_day'] = combined_df['humidity'].shift(7)

    combined_df['so2_lag_1_day'] = combined_df['so2'].shift(1)
    combined_df['so2_lag_3_day'] = combined_df['so2'].shift(3)
    combined_df['so2_lag_7_day'] = combined_df['so2'].shift(7)

    combined_df['nh3_lag_1_day'] = combined_df['nh3'].shift(1)
    combined_df['nh3_lag_3_day'] = combined_df['nh3'].shift(3)
    combined_df['nh3_lag_7_day'] = combined_df['nh3'].shift(7)

    combined_df['no2_lag_1_day'] = combined_df['no2'].shift(1)
    combined_df['no2_lag_3_day'] = combined_df['no2'].shift(3)
    combined_df['no2_lag_7_day'] = combined_df['no2'].shift(7)

    combined_df['co_lag_1_day'] = combined_df['co'].shift(1)
    combined_df['co_lag_3_day'] = combined_df['co'].shift(3)
    combined_df['co_lag_7_day'] = combined_df['co'].shift(7)

    combined_df['pm10_lag_1_day'] = combined_df['pm10'].shift(1)
    combined_df['pm10_lag_3_day'] = combined_df['pm10'].shift(3)
    combined_df['pm10_lag_7_day'] = combined_df['pm10'].shift(7)

    combined_df['pm2_5_lag_1_day'] = combined_df['pm2_5'].shift(1)
    combined_df['pm2_5_lag_3_day'] = combined_df['pm2_5'].shift(3)
    combined_df['pm2_5_lag_7_day'] = combined_df['pm2_5'].shift(7)

    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['month'] = combined_df['date'].dt.month

    # 6. Isolate and insert the new, fully-featured row
    final_new_row = combined_df.tail(1)

    # Check if the row is for yesterday (safety check)
    if final_new_row.iloc[0]['date'].date() != yesterday_utc:
        print(f"Error: Final row date ({final_new_row.iloc[0]['date'].date()}) does not match yesterday ({yesterday_utc}).")
        return

    print(f"Inserting new fully-featured row for {yesterday_utc}:")
    print(final_new_row)
    
    aqi_fg.insert(final_new_row, write_options={"wait_for_job": True})
    
    print("✅ Daily feature pipeline completed successfully.")

if __name__ == "__main__":

    run_daily_feature_pipeline()
