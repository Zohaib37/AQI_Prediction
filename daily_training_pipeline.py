import pandas as pd
import hopsworks
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Connect to Hopsworks
project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# Get the Feature Group
aqi_fg = fs.get_feature_group(name="aqi_features", version=1)

# Read data and prepare
feature_df = aqi_fg.read()
feature_df = feature_df.sort_values('date').reset_index(drop=True)
feature_df = feature_df.dropna() # Drop any rows with NaN lags (e.g., first few days)

X = feature_df.drop(columns=['date', 'us_aqi'])
y = feature_df['us_aqi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.15, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'New Model Trained. MSE: {mse}, R2: {r2}')

# Save the model artifact
joblib.dump(xgb_model, 'aqi_xgb_model.pkl')

# Save to Model Registry
model_registry = project.get_model_registry()

aqi_model = model_registry.python.create_model(
    name="aqi_predictor",
    metrics={"mse": mse, "r2": r2},
    description="XGBoost model for AQI prediction"
)

aqi_model.save('aqi_xgb_model.pkl')
print("New model version saved to registry.")