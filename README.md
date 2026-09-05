# Pearls AQI Predictor

I built this for my internship at 10Pearls. It forecasts the Air Quality Index three days
ahead for Karachi, Lahore, Faisalabad, Islamabad and Peshawar. Everything runs on free
services and the pipeline keeps itself up to date, so there is nothing to host and nothing
to run by hand.

Live dashboard: https://pearls-aqi-predictor-mohid.streamlit.app

## What it does

The brief was to build a serverless end to end ML system: a feature pipeline, a feature
store, a training pipeline, a model registry, automated CI/CD and a dashboard, with EDA,
SHAP explanations and health alerts on top. This repo is all of that.

An hourly GitHub Actions job pulls pollution data from OpenWeather and weather from
Open-Meteo, computes the US EPA AQI from the raw pollutant readings and writes it to a
Hopsworks feature store. A second job runs once a day, retrains the model on the full
history and only registers the new one if it beats the model already in the registry. The
Streamlit dashboard reads from the same feature store and shows the current AQI, the three
day forecast, a colour coded health alert, a 30 day trend, a side by side comparison of all
five cities and a SHAP breakdown of what drove the prediction.

Nothing runs on a server I pay for. Hopsworks stores the features and the model, GitHub
Actions runs the two schedules and Streamlit Community Cloud hosts the dashboard, all on
free tiers.

## How the AQI is calculated

OpenWeather returns its own air quality index as a number from 1 to 5. That is far too
coarse to forecast against, so I compute the US EPA AQI from the raw pollutant
concentrations instead, which is the 0 to 500 scale that aqicn.org and most real dashboards
show.

Getting there takes a few steps. The gas pollutants come back in micrograms per cubic metre
and the EPA table expects parts per billion or parts per million, so ozone, carbon
monoxide, sulphur dioxide and nitrogen dioxide are converted first. Each pollutant is then
averaged over the window the EPA specifies, 24 hours for PM2.5 and PM10, 8 hours for ozone
and carbon monoxide and 1 hour for the rest. The averaged value is looked up in the 2024
breakpoint table with a straight line interpolation, and the overall AQI is the highest of
the individual pollutant sub indices. I also record which pollutant produced that highest
value, and in Pakistan it is almost always PM2.5.

## The model

Models are trained on a daily table, one row per city per day. The features are the daily
mean and max of every pollutant, the daily weather, AQI lags at 1, 2, 3 and 7 days, PM2.5
lags, 3 and 7 day rolling mean and standard deviation, a day to day change rate, the PM2.5
to PM10 ratio, cyclical encodings of the day of week and month, and a one hot column per
city. Open-Meteo also gives a free 7 day weather forecast, so the forecast temperature,
wind, rain and humidity for each of the next three days go in as features too, since they
are genuinely known at prediction time and weather moves pollution around a lot.

"Three days ahead" is three separate targets, one for each day, rather than one model run
forward three times, so a small early error does not compound.

I trained persistence, ridge regression, a per city SARIMAX, a two layer LSTM, XGBoost and
a random forest, with the random forest and XGBoost tuned by Optuna. The test set is the
most recent 60 days and validation uses a time series split, never a random one, because a
random split on lagged data leaks the future and gives a fake high score. Average RMSE
across the five cities:

| model | 1 day | 2 days | 3 days |
|---|---|---|---|
| persistence | 18.45 | 26.63 | 29.56 |
| ridge | 17.69 | 27.70 | 31.01 |
| random forest | 10.50 | 20.45 | 23.52 |
| xgboost | 12.79 | 23.46 | 23.99 |
| sarimax | 17.51 | 24.27 | 26.08 |
| lstm | 13.10 | 22.82 | 26.65 |

The random forest wins at every horizon and is the one in production. It is a single model
for all five cities with the city as a feature, which gives it five times the training data
and one artifact to serve, and I still report the error per city so a bad city cannot hide
inside the average. SHAP says today's PM2.5 reading carries most of the signal, well ahead
of everything derived from it.

One caveat worth stating. During the historical backfill the forecast weather columns are
filled from the ERA5 archive, which is a perfect forecast, so the backtest numbers are a
little more optimistic than live performance will be. This is standard practice but it is
worth knowing.

## What is in the repo

AQI_Predictor.ipynb is the whole build in one notebook, from the first data pull through
the EDA, feature engineering, all six models, the SHAP analysis and registering the winner.
It is meant to be read top to bottom. feature_pipeline.py and training_pipeline.py are the
two scheduled jobs, and they are a copy of the relevant notebook cells rather than an
import, so the notebook stays self contained. app.py is the dashboard. The two files under
.github/workflows run the jobs on a schedule. REPORT.pdf is the written report and outputs
holds the figures and the model comparison table.

## Running it locally

Put your keys in a .env file next to app.py:

    OPENWEATHER_API_KEY=your_key
    HOPSWORKS_API_KEY=your_key
    HOPSWORKS_PROJECT=your_project_name

Then install and start the dashboard:

    pip install -r requirements.txt
    streamlit run app.py

The two pipeline scripts run the same way with python feature_pipeline.py and python
training_pipeline.py. They do not read anything from disk, so they work from a clean
checkout. Running the feature pipeline twice in a row is safe, it upserts by city and
timestamp so nothing doubles up.

To run the schedule on GitHub you need the same three values added as repository secrets
under Settings, Secrets and variables, Actions. Both workflows also have a manual trigger
if you want to fire them without waiting for the cron.

## Notes and limitations

OpenWeather's pollution figures come from the CAMS atmospheric model, not ground stations,
so they read smoother than reality and will not line up exactly with aqicn.org. This is a
known trade for having free hourly history back to 2020 for every city.

GitHub's scheduled runs are best effort and can be 5 to 20 minutes late, and GitHub
disables scheduled workflows after 60 days with no commits to the repo, so a long quiet
period needs a nudge.

Hopsworks free tier allows one project and a small number of feature groups, which is why
everything lives in just two, aqi_hourly and aqi_daily.
