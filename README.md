# Pearls AQI Predictor

This is my internship project at 10Pearls. It predicts the Air Quality Index three days
ahead for Karachi, Lahore, Faisalabad, Islamabad and Peshawar, and it runs entirely on
free services that refresh themselves, so nothing has to be hosted or kept alive by hand.

Live dashboard: https://pearls-aqi-predictor-mohid.streamlit.app

## What it does

Once an hour a script pulls the latest pollution numbers from OpenWeather and the latest
weather from Open-Meteo, turns them into a proper US EPA air quality index for each city
and stores the result in a Hopsworks feature store. Once a day another script retrains
the forecasting model on everything collected so far. The dashboard reads from the same
store and shows the current air quality, the three day forecast, a health warning and a
small chart explaining which pollutant or weather variable moved the prediction.

## About the AQI number

OpenWeather does return an air quality value but it is only a 1 to 5 rating, which is not
something you can sensibly train a regression on. So the project builds the real EPA AQI
instead, the familiar 0 to 500 scale, from the raw pollutant concentrations.

Getting there takes a few steps. Ozone, carbon monoxide, sulphur dioxide and nitrogen
dioxide have to be converted from micrograms per cubic metre into the parts per million
or parts per billion the EPA tables use. Each pollutant then gets averaged over its own
window, which is 24 hours for the two particulate sizes, 8 hours for ozone and carbon
monoxide and 1 hour for the other two. After that it is a lookup in the 2024 EPA
breakpoint table with linear interpolation, and the overall AQI is just the worst of the
individual pollutant scores. The project also records which pollutant that was, and in
Pakistan it is PM2.5 nearly every hour of the year.

## How the pieces fit together

There are two GitHub Actions workflows on a schedule. The hourly one runs the feature
script, which fetches a recent window of data, rebuilds the features and writes them to
two feature groups in Hopsworks. One group holds the hourly readings and the other rolls
them up to a single row per city per day with lag columns, rolling averages and the
coming three days of forecast weather attached.

The daily workflow runs the training script. It reads the daily table back out of the
feature store, retrains a random forest and compares it against whatever model is
already registered. It only keeps the new one if it actually scored better, so a bad day
in the data can never push a worse model into production.

Neither script holds any state between runs. A GitHub runner is a blank machine every
time, so both scripts fetch what they need and rebuild everything from scratch on every
run.

## The model

I tried a plain persistence baseline, ridge regression, SARIMAX, an LSTM, XGBoost and a
random forest, and scored all of them on the last sixty days of data that none of them
saw during training. The random forest was clearly the best. It lands around an RMSE of
10 on the one day forecast, where simply repeating today's value scores about 18, and it
stays ahead at two and three days as well. XGBoost came second. Ridge was a bit of a
surprise, it does worse than the do-nothing baseline past the first day, because a
linear model does not have the room a tree ensemble does to keep correcting itself as
the horizon grows.

One model covers all five cities rather than one model each. The city goes in as a plain
one hot column. That way it trains on five times the data and there is a single thing to
deploy, and I still report the error separately per city so a weak one cannot hide in
the average.

The train and test split is by date, never random. Validation uses time series cross
validation. If you shuffle rows on lagged time series data you leak future information
into training and get an R² around 0.95 that means nothing, and that is the usual way a
project like this quietly goes wrong.

The forecast weather columns are worth a note. Open-Meteo gives a free seven day weather
forecast, and wind, rain and temperature genuinely help. Wind spreads the particulates
out and rain drags them down. A real forecast for those is available at the moment of
prediction, so it is fair to feed them to the model as inputs.

## The dashboard

app.py is a Streamlit app. Pick a city and it shows a gauge for the current AQI coloured
by the six EPA bands, the category and the dominant pollutant, the live pollutant
readings and a coloured health alert. Under that are the three forecast days, a thirty
day history with the forecast drawn on the end, and a small map and bar chart putting
all five cities next to each other. Two tabs hold the SHAP explanation for the current
prediction and the full table of every model I tested.

If a chart or a library breaks after a redeploy, that one part of the page shows a short
message and the rest still loads.

## Running it

You will need a free OpenWeather key and a free Hopsworks account. Make a file called
.env in the project folder with three lines:

    OPENWEATHER_API_KEY=your_key
    HOPSWORKS_API_KEY=your_key
    HOPSWORKS_PROJECT=your_project_name

Then install the requirements and open whichever part you want:

    pip install -r requirements.txt
    jupyter notebook AQI_Predictor.ipynb
    streamlit run app.py

The notebook is the full story from the data pull to the trained model and is meant to
be read straight through. feature_pipeline.py and training_pipeline.py are the two
scheduled jobs. app.py is the dashboard. The written report is REPORT.pdf. The first
notebook run backfills three years of history and is slow, but it saves checkpoints to
disk as it goes so a crash does not cost you the whole download.

## Things that are not perfect

OpenWeather's pollution data comes out of the CAMS model rather than physical ground
stations, so it is smoother than the real air and will not match a site like aqicn.org
exactly. During the historical backfill the forecast weather is actually taken from the
archive, which is a perfect forecast, so the backtest scores flatter the model a little
next to live use. GitHub's scheduled runs are best effort and can be twenty minutes
late, and GitHub turns them off after sixty days without a commit. The Hopsworks free
tier also limits how much you can store, which is why there are only two feature groups.
