# Pearls AQI Predictor

I built this for my internship at 10Pearls. It is a serverless system that predicts
air quality three days ahead for five Pakistani cities, Karachi, Lahore, Faisalabad,
Islamabad and Peshawar.

**Try it live:** https://pearls-aqi-predictor-mohid.streamlit.app

## How it works

An hourly pipeline pulls pollution data from OpenWeather and weather data from
Open-Meteo, computes the real US EPA Air Quality Index from the raw pollutant
readings and writes everything to a Hopsworks feature store. A random forest model
retrains daily on that data and forecasts the AQI three days out. The dashboard shows
the current reading, the three day forecast, a hazard alert, and a SHAP breakdown of
what is driving each prediction.

## Stack

Python, scikit-learn, Hopsworks for the feature store and model registry, GitHub
Actions for the hourly and daily pipelines, and Streamlit for the dashboard.
