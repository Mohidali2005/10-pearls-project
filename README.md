# Pearls AQI Predictor

I built this for my internship at 10Pearls. It predicts the Air Quality Index three days
ahead for Karachi, Lahore, Faisalabad, Islamabad and Peshawar. It runs on free services
and keeps itself up to date, so there is nothing to host.

Live: https://pearls-aqi-predictor-mohid.streamlit.app

## How it works

An hourly GitHub Actions job pulls pollution data from OpenWeather and weather from
Open-Meteo, computes the US EPA AQI from the raw pollutant readings and saves it to a
Hopsworks feature store. A daily job retrains the model and only registers it if it beats
the current one. The Streamlit dashboard reads from the same store and shows the current
AQI, the three day forecast, a health alert and a SHAP chart of what drove the prediction.

I compute the EPA AQI myself rather than use OpenWeather's 1 to 5 rating, which is too
coarse to forecast. That means converting the gas pollutants to the right units,
averaging each one over its EPA window and looking the values up in the 2024 breakpoint
table. PM2.5 turns out to be the dominant pollutant almost every hour.

## The model

I tested persistence, ridge, SARIMAX, an LSTM, XGBoost and a random forest on the last 60
days. The random forest won at every horizon, about RMSE 10 one day out against 18 for
just repeating today's value. It is a single model for all five cities with the city as a
one hot feature, and the train/test split is by date so the future never leaks in.

## Running it

Put your keys in a .env file:

    OPENWEATHER_API_KEY=...
    HOPSWORKS_API_KEY=...
    HOPSWORKS_PROJECT=...

Then:

    pip install -r requirements.txt
    streamlit run app.py

AQI_Predictor.ipynb is the full build from data pull to trained model. feature_pipeline.py
and training_pipeline.py are the scheduled jobs. REPORT.pdf is the writeup.

## Notes

OpenWeather's pollution numbers come from a model, not ground stations, so they read
smoother than reality and will not match aqicn.org exactly. GitHub's scheduled runs can
be late and switch off after 60 days without a commit.
