# Pearls AQI Predictor

I built this during my internship at 10Pearls. It forecasts the Air Quality Index
three days ahead for five Pakistani cities: Karachi, Lahore, Faisalabad, Islamabad
and Peshawar. Everything runs on free tiers and updates itself, so there is no server
to keep alive.

**Live dashboard:** https://pearls-aqi-predictor-mohid.streamlit.app

## What it does

Every hour a pipeline fetches the latest pollution and weather readings, works out the
real US EPA air quality index for each city and saves it to a feature store. Once a day
a second pipeline retrains the forecasting model on the full history. The dashboard
reads from the same feature store and shows the current air quality, a three day
forecast, a health warning that matches the EPA category and a chart that explains what
pushed each prediction up or down.

## Computing the AQI

OpenWeather gives you an `aqi` field but it is just a number from 1 to 5, which is far
too coarse to train a regression on. So the project computes the actual US EPA AQI, the
0 to 500 scale that aqicn.org and most real dashboards show, from the raw pollutant
concentrations.

That means:

- converting O3, CO, SO2 and NO2 from µg/m³ into the ppm or ppb the EPA tables expect
- rolling each pollutant over the window the EPA asks for, which is 24 hours for PM2.5
  and PM10, 8 hours for O3 and CO, and 1 hour for SO2 and NO2
- looking each value up in the 2024 EPA breakpoint table and interpolating
- taking the highest sub-index as the overall AQI and recording which pollutant it
  came from

In Pakistan that pollutant is PM2.5 almost every hour of the year.

## The pipeline

```
OpenWeather Air Pollution  ─┐
                            ├─►  feature_pipeline.py  ─►  Hopsworks feature store
Open-Meteo weather         ─┘     (hourly, GitHub Actions)    aqi_hourly, aqi_daily
                                                                     │
                                  training_pipeline.py  ────────────►  Hopsworks model registry
                                  (daily, GitHub Actions)             best model + metrics
                                                                     │
                                  app.py (Streamlit)  ◄──────────────┘
```

The feature pipeline runs on a GitHub Actions cron every hour. It pulls a recent window
of data, recomputes the features and upserts two feature groups in Hopsworks. `aqi_hourly`
keeps the hourly readings and preserves the daily cycle. `aqi_daily` rolls those up per
city per day and adds lag features, rolling averages and the next three days of forecast
weather.

The training pipeline runs once a day. It reads `aqi_daily` back from the feature store,
retrains the model and only registers the new version if it beats the one already in the
registry, so a bad retrain can never replace a good model.

Both scripts are self contained. A GitHub Actions runner starts from nothing every time,
so neither one reads any local files or assumes a previous run happened.

## The model

I tried six approaches and scored them on the most recent 60 days, which none of the
models see while training. Average RMSE across the five cities:

| model | 1 day | 2 days | 3 days |
|---|---|---|---|
| persistence (just repeat today) | 18.4 | 26.6 | 29.6 |
| ridge regression | 17.7 | 27.7 | 31.0 |
| SARIMAX | 17.5 | 24.3 | 26.1 |
| LSTM | 13.1 | 22.8 | 26.6 |
| XGBoost | 12.8 | 23.5 | 24.0 |
| random forest | 10.5 | 20.4 | 23.5 |

The random forest wins at every horizon so that is what runs in production. Ridge is the
one worth a comment: it actually does worse than just repeating today's value at two and
three days out, because a straight line through 58 mostly correlated columns has nowhere
near the room a tree ensemble has to correct itself further out.

A few choices that shaped the results:

- **One model for all five cities**, with the city as a one hot feature. That gives five
  times the training data and one artifact to serve. Metrics are still broken out per
  city so a weak city cannot hide inside the average.
- **Time based splits only.** The test set is the last stretch of the data and validation
  uses time-series cross-validation. A random split on lagged time series data leaks the
  future and hands you a fake R² around 0.95. This is the most common way this kind of
  project goes wrong.
- **Forecast weather as a feature.** Open-Meteo publishes free weather forecasts a week
  out, and wind, rain and temperature really do help predict AQI. Wind disperses
  particulates and rain washes them out. Those columns exist at prediction time so it is
  fair to feed them to the model.

## The dashboard

`app.py` is a Streamlit app. For the chosen city it shows a gauge with the current AQI
shaded by the six EPA bands, the category and dominant pollutant, a live pollutant
readout and a coloured hazard alert with matching health advice. Under that are the
three day forecast cards, a 30 day history chart with the forecast added on the end, and
a map and bar chart comparing all five cities right now. Two tabs at the bottom hold a
SHAP breakdown of the current prediction and the full model comparison table.

Each section renders inside its own guard. If one chart or one library breaks after a
redeploy, that block shows a short notice and the rest of the page keeps working.

## Repo layout

```
AQI_Predictor.ipynb    the whole build in one notebook, meant to be read top to bottom
feature_pipeline.py    the hourly job, run by GitHub Actions
training_pipeline.py   the daily retrain, run by GitHub Actions
app.py                 the Streamlit dashboard
.github/workflows/     the two cron schedules
requirements.txt       pinned where a version drift has bitten before
REPORT.pdf             the written report
```

The notebook defines every function inline so it reads as one continuous piece of work
rather than a set of scripts. The two pipeline files are a copy of the cells they need,
which is roughly 150 repeated lines. That is a deliberate trade for a notebook you can
open and run start to end.

## Running it yourself

You need a free OpenWeather API key and a free Hopsworks account. Put them in a `.env`
file in the project root:

```
OPENWEATHER_API_KEY=your_key
HOPSWORKS_API_KEY=your_key
HOPSWORKS_PROJECT=your_project_name
```

Then:

```
pip install -r requirements.txt
jupyter notebook AQI_Predictor.ipynb    # to walk through the whole build
streamlit run app.py                    # to run the dashboard locally
```

On its first run the notebook backfills three years of history, which is slow, so it
saves parquet checkpoints along the way and a crash never costs you the whole pull.

## Things to know

- OpenWeather's pollution data comes from the CAMS model, not ground stations. It is
  smoother than reality and will not line up exactly with aqicn.org. That is a known
  limitation, not a bug, and the dashboard says so too.
- During the historical backfill the forecast weather columns are filled from the ERA5
  archive, which is effectively a perfect forecast. Backtest scores are a little
  optimistic next to real live performance because of that.
- GitHub Actions cron is best effort. A scheduled run is often 5 to 20 minutes late, and
  GitHub switches scheduled workflows off after 60 days with no commits to the repo.
- The Hopsworks free tier has storage limits, which is why the project keeps to two
  feature groups.
