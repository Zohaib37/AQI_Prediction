# AQI_Prediction
A serverless machine learning system that predicts Air Quality Index (AQI) for the next 3 days using historical weather and pollutant data. The system integrates automated data pipelines, feature engineering, model training, and deployment through a web application.
# Project Overview
This project implements an end-to-end MLOps pipeline for AQI forecasting. Historical weather data (temperature, humidity, wind speed) and pollutant concentrations (PM2.5, PM10, CO, NO2, SO2, O3, NH3) are collected from OpenWeather API. The system performs feature engineering to create lag features and temporal features, trains machine learning models, and provides 3-day forecasts through an interactive Streamlit web application.
The pipeline is fully automated using GitHub Actions, with hourly data collection and daily model retraining. Features and models are stored in Hopsworks Feature Store and Model Registry respectively, enabling version control and reproducibility.
