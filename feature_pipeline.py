import os
import math
import pathlib
import tempfile
from datetime import datetime,timedelta,timezone
import requests
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from tenacity import retry,stop_after_attempt,wait_exponential

load_dotenv()
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

CITIES = {
    "Karachi":(24.86,67.01),
    "Lahore":(31.55,74.35),
    "Faisalabad":(31.41,73.07),
    "Islamabad":(33.72,73.06),
    "Peshawar":(34.01,71.57),
}

POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARS = [
    "temperature_2m","relative_humidity_2m","dew_point_2m","precipitation",
    "surface_pressure","wind_speed_10m","wind_direction_10m","cloud_cover",
]

# conc low, conc high, aqi low, aqi high
BREAKPOINTS = {
    # pm2_5 is the 2024 revised table
    "pm2_5":[
        (0.0,9.0,0,50),(9.1,35.4,51,100),(35.5,55.4,101,150),
        (55.5,125.4,151,200),(125.5,225.4,201,300),(225.5,325.4,301,500),
    ],
    "pm10":[
        (0,54,0,50),(55,154,51,100),(155,254,101,150),
        (255,354,151,200),(355,424,201,300),(425,604,301,500),
    ],
    "o3":[
        (0.000,0.054,0,50),(0.055,0.070,51,100),(0.071,0.085,101,150),
        (0.086,0.105,151,200),(0.106,0.200,201,300),
    ],
    "co":[
        (0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
        (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500),
    ],
    "so2":[
        (0,35,0,50),(36,75,51,100),(76,185,101,150),
        (186,304,151,200),(305,604,201,300),(605,1004,301,500),
    ],
    "no2":[
        (0,53,0,50),(54,100,51,100),(101,360,101,150),
        (361,649,151,200),(650,1249,201,300),(1250,2049,301,500),
    ],
}

AVERAGING_HOURS = {"pm2_5":24,"pm10":24,"o3":8,"co":8,"so2":1,"no2":1}

# o3 and co need ppm and the other two need ppb
MOLECULAR_WEIGHTS = {"o3":48.0,"co":28.01,"so2":64.06,"no2":46.01}

# epa truncates the concentration to this many decimals before the table lookup
TRUNCATE_DECIMALS = {"pm2_5":1,"pm10":0,"o3":3,"co":1,"so2":0,"no2":0}

# days of daily rows this run will refresh in aqi_daily
UPDATE_DAYS = 10
# extra days of history fetched only to seed the 7 day lag and rolling features
LAG_BUFFER_DAYS = 7
# extra hours fetched only to seed the 24 hour pm2_5 and pm10 rolling window
WARMUP_HOURS = 24

@retry(stop=stop_after_attempt(3),wait=wait_exponential(min=2,max=10))
def fetch_pollution(lat,lon,start,end):
    r = requests.get(POLLUTION_HISTORY_URL,params={"lat":lat,"lon":lon,"start":int(start.timestamp()),"end":int(end.timestamp()),"appid":OPENWEATHER_KEY},timeout=30)
    r.raise_for_status()
    return r.json()

@retry(stop=stop_after_attempt(3),wait=wait_exponential(min=2,max=10))
def fetch_weather(lat,lon,start,end):
    r = requests.get(WEATHER_ARCHIVE_URL,params={"latitude":lat,"longitude":lon,"start_date":start.strftime("%Y-%m-%d"),"end_date":end.strftime("%Y-%m-%d"),"hourly":",".join(WEATHER_VARS),"timezone":"UTC"},timeout=30)
    r.raise_for_status()
    return r.json()

# hudi upserts by primary key so a retried insert overwrites with the same values instead of duplicating
@retry(stop=stop_after_attempt(3),wait=wait_exponential(min=5,max=30))
def insert_fg(fg,df):
    fg.insert(df,wait=True)

def convert_units(conc,pollutant):
    if pollutant not in MOLECULAR_WEIGHTS:
        return conc
    ppb = conc * 24.45 / MOLECULAR_WEIGHTS[pollutant]
    if pollutant in ("o3","co"):
        return ppb / 1000
    return ppb

def truncate(conc,pollutant):
    factor = 10 ** TRUNCATE_DECIMALS[pollutant]
    return math.floor(conc*factor) / factor

def calc_aqi(conc,pollutant):
    for bp_lo,bp_hi,aqi_lo,aqi_hi in BREAKPOINTS[pollutant]:
        if bp_lo <= conc <= bp_hi:
            return (aqi_hi-aqi_lo) / (bp_hi-bp_lo) * (conc-bp_lo) + aqi_lo
    return 500

def compute_aqi(concentrations):
    sub_indices = {}
    for pollutant,conc in concentrations.items():
        converted = convert_units(conc,pollutant)
        truncated = truncate(converted,pollutant)
        sub_indices[pollutant] = calc_aqi(truncated,pollutant)
    dominant_pollutant = max(sub_indices,key=sub_indices.get)
    aqi = round(sub_indices[dominant_pollutant])
    return aqi,dominant_pollutant

def cyclical_encode(values,period):
    radians = 2*np.pi*values/period
    return np.sin(radians),np.cos(radians)

now = datetime.now(timezone.utc)
fetch_start = now - timedelta(days=UPDATE_DAYS+LAG_BUFFER_DAYS,hours=WARMUP_HOURS)

pollution_rows = []
for city,(lat,lon) in CITIES.items():
    data = fetch_pollution(lat,lon,fetch_start,now)
    for item in data["list"]:
        row = item["components"]
        row["timestamp_utc"] = pd.to_datetime(item["dt"],unit="s",utc=True)
        row["city"] = city
        pollution_rows.append(row)
pollution_raw = pd.DataFrame(pollution_rows)

weather_frames = []
for city,(lat,lon) in CITIES.items():
    data = fetch_weather(lat,lon,fetch_start,now)
    df = pd.DataFrame(data["hourly"])
    df["city"] = city
    weather_frames.append(df)
weather_raw = pd.concat(weather_frames,ignore_index=True)
weather_raw["time"] = pd.to_datetime(weather_raw["time"],utc=True)

print(f"fetched pollution {pollution_raw.shape} | weather {weather_raw.shape}")

pollution_raw = pollution_raw.drop_duplicates(subset=["city","timestamp_utc"])
weather_raw = weather_raw.rename(columns={"time":"timestamp_utc"}).drop_duplicates(subset=["city","timestamp_utc"])
hourly = pd.merge(pollution_raw,weather_raw,on=["city","timestamp_utc"],how="outer")

frames = []
for city in CITIES:
    sub = hourly[hourly["city"]==city].drop(columns="city").set_index("timestamp_utc").sort_index()
    full_range = pd.date_range(sub.index.min(),sub.index.max(),freq="h")
    sub = sub.reindex(full_range).interpolate(limit=3)
    sub["city"] = city
    frames.append(sub)
hourly = pd.concat(frames).rename_axis("timestamp_utc").reset_index()

rolled = hourly.copy()
for city in CITIES:
    mask = rolled["city"]==city
    for pollutant,window in AVERAGING_HOURS.items():
        rolled.loc[mask,pollutant] = rolled.loc[mask,pollutant].rolling(window,min_periods=window).mean()

aqi_col = []
dominant_col = []
for row in rolled[list(AVERAGING_HOURS)].itertuples(index=False):
    values = row._asdict()
    if any(pd.isna(v) for v in values.values()):
        aqi_col.append(np.nan)
        dominant_col.append(None)
    else:
        aqi,dominant = compute_aqi(values)
        aqi_col.append(aqi)
        dominant_col.append(dominant)

hourly["epa_aqi"] = aqi_col
hourly["dominant_pollutant"] = dominant_col
hourly["timestamp_pkt"] = hourly["timestamp_utc"] + pd.Timedelta(hours=5)

hourly["hour"] = hourly["timestamp_pkt"].dt.hour
hourly["dow"] = hourly["timestamp_pkt"].dt.dayofweek
hourly["month"] = hourly["timestamp_pkt"].dt.month
hourly["hour_sin"],hourly["hour_cos"] = cyclical_encode(hourly["hour"],24)
hourly["dow_sin"],hourly["dow_cos"] = cyclical_encode(hourly["dow"],7)
hourly["month_sin"],hourly["month_cos"] = cyclical_encode(hourly["month"],12)
hourly["wind_dir_sin"],hourly["wind_dir_cos"] = cyclical_encode(hourly["wind_direction_10m"],360)

# drop the warmup hours that only exist to seed the 24 hour aqi rolling window
hourly_keep_start = fetch_start + pd.Timedelta(hours=WARMUP_HOURS)
hourly = hourly[hourly["timestamp_utc"]>=hourly_keep_start].reset_index(drop=True)
print(f"hourly rows to upsert: {len(hourly)} | epa_aqi missing {hourly['epa_aqi'].isna().sum()}")

hourly["date"] = hourly["timestamp_pkt"].dt.date
daily = hourly.groupby(["city","date"]).agg(
    pm2_5_mean=("pm2_5","mean"),pm2_5_max=("pm2_5","max"),
    pm10_mean=("pm10","mean"),pm10_max=("pm10","max"),
    o3_mean=("o3","mean"),o3_max=("o3","max"),
    no2_mean=("no2","mean"),no2_max=("no2","max"),
    so2_mean=("so2","mean"),so2_max=("so2","max"),
    co_mean=("co","mean"),co_max=("co","max"),
    no_mean=("no","mean"),no_max=("no","max"),
    nh3_mean=("nh3","mean"),nh3_max=("nh3","max"),
    temp_mean=("temperature_2m","mean"),
    humidity_mean=("relative_humidity_2m","mean"),
    wind_mean=("wind_speed_10m","mean"),
    pressure_mean=("surface_pressure","mean"),
    cloud_mean=("cloud_cover","mean"),
    precip_sum=("precipitation","sum"),
    aqi=("epa_aqi","mean"),
).reset_index()

daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values(["city","date"]).reset_index(drop=True)

for city in CITIES:
    mask = daily["city"]==city

    for lag in [1,2,3,7]:
        daily.loc[mask,f"aqi_lag_{lag}"] = daily.loc[mask,"aqi"].shift(lag)
    for lag in [1,2,3]:
        daily.loc[mask,f"pm25_lag_{lag}"] = daily.loc[mask,"pm2_5_mean"].shift(lag)

    daily.loc[mask,"aqi_roll_mean_3"] = daily.loc[mask,"aqi"].rolling(3).mean()
    daily.loc[mask,"aqi_roll_mean_7"] = daily.loc[mask,"aqi"].rolling(7).mean()
    daily.loc[mask,"aqi_roll_std_7"] = daily.loc[mask,"aqi"].rolling(7).std()
    daily.loc[mask,"aqi_change_rate"] = daily.loc[mask,"aqi"].pct_change()

    for h in [1,2,3]:
        daily.loc[mask,f"fc_temp_d{h}"] = daily.loc[mask,"temp_mean"].shift(-h)
        daily.loc[mask,f"fc_wind_d{h}"] = daily.loc[mask,"wind_mean"].shift(-h)
        daily.loc[mask,f"fc_precip_d{h}"] = daily.loc[mask,"precip_sum"].shift(-h)
        daily.loc[mask,f"fc_humidity_d{h}"] = daily.loc[mask,"humidity_mean"].shift(-h)
        daily.loc[mask,f"aqi_d{h}"] = daily.loc[mask,"aqi"].shift(-h)

daily["pm25_pm10_ratio"] = daily["pm2_5_mean"] / daily["pm10_mean"]
daily["dow"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["doy"] = daily["date"].dt.dayofyear
daily["doy_sin"],daily["doy_cos"] = cyclical_encode(daily["doy"],365)
daily["is_weekend"] = daily["dow"].isin([5,6]).astype("int32")

city_dummies = pd.get_dummies(daily["city"],prefix="city").astype("int32")
daily = pd.concat([daily,city_dummies],axis=1)

# drop the lag buffer days that only exist to seed the 7 day lag and rolling features
cutoff_date = daily["date"].max() - pd.Timedelta(days=UPDATE_DAYS-1)
daily = daily[daily["date"]>=cutoff_date].reset_index(drop=True)
print(f"daily rows to upsert: {len(daily)}")

project = hopsworks.login(host="eu-west.cloud.hopsworks.ai",api_key_value=HOPSWORKS_KEY,project=HOPSWORKS_PROJECT,cert_folder=tempfile.gettempdir())
fs = project.get_feature_store()

hourly_fg = fs.get_or_create_feature_group(
    name="aqi_hourly",
    version=1,
    description="hourly pollutant and weather features per city",
    primary_key=["city","timestamp_utc"],
    event_time="timestamp_utc",
    time_travel_format="HUDI",
    statistics_config=False,
)
insert_fg(hourly_fg,hourly)

daily_fg = fs.get_or_create_feature_group(
    name="aqi_daily",
    version=1,
    description="daily aggregates, lags, rolling stats and the three day ahead aqi targets",
    primary_key=["city","date"],
    event_time="date",
    time_travel_format="HUDI",
    statistics_config=False,
)
insert_fg(daily_fg,daily)

print("upserted aqi_hourly and aqi_daily")
