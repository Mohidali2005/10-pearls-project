import os
import math
import pathlib
import tempfile
import requests
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import shap
import hopsworks
from dotenv import load_dotenv
from tenacity import retry,stop_after_attempt,wait_exponential

load_dotenv()
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

DARK_BG = "#0E1117"  # matches backgroundColor in .streamlit/config.toml
plt.style.use("dark_background")

st.set_page_config(page_title="Pearls AQI Predictor",page_icon="🌫️",layout="wide",initial_sidebar_state="expanded")

CITIES = {
    "Karachi":(24.8607,67.0011),
    "Lahore":(31.5204,74.3587),
    "Faisalabad":(31.4504,73.1350),
    "Islamabad":(33.6844,73.0479),
    "Peshawar":(34.0151,71.5249),
}

AQI_CATEGORIES = [
    (50,"Good","green"),
    (100,"Moderate","gold"),
    (150,"Unhealthy for Sensitive Groups","orange"),
    (200,"Unhealthy","red"),
    (300,"Very Unhealthy","purple"),
    (500,"Hazardous","darkred"),
]

HEALTH_GUIDANCE = {
    "Good":"Air quality is satisfactory and poses little or no risk.",
    "Moderate":"Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups":"Sensitive groups may experience health effects, the general public is not likely to be affected.",
    "Unhealthy":"Everyone may begin to experience health effects, sensitive groups may experience more serious effects.",
    "Very Unhealthy":"Health alert, everyone may experience more serious health effects.",
    "Hazardous":"Health warning of emergency conditions, the entire population is more likely to be affected.",
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

st.markdown("""
<style>
.block-container{padding-top:2rem;padding-bottom:3rem;}
div[data-testid="stMetric"]{background-color:#161B22;border:1px solid #30363D;border-radius:0.5rem;padding:0.75rem 1rem;}
</style>
""",unsafe_allow_html=True)

def category_for(aqi):
    for upper,name,color in AQI_CATEGORIES:
        if aqi<=upper:
            return name,color
    return AQI_CATEGORIES[-1][1],AQI_CATEGORIES[-1][2]

def alert_color(aqi):
    if aqi<=100:
        return "green"
    if aqi<=200:
        return "orange"
    if aqi<=300:
        return "red"
    return "maroon"

# same EPA breakpoint arithmetic as the feature pipeline, used only to name the dominant
# pollutant when the dashboard has to fall back to the daily averages
BREAKPOINTS = {
    "pm2_5":[(0.0,9.0,0,50),(9.1,35.4,51,100),(35.5,55.4,101,150),(55.5,125.4,151,200),(125.5,225.4,201,300),(225.5,325.4,301,500)],
    "pm10":[(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,604,301,500)],
    "o3":[(0.000,0.054,0,50),(0.055,0.070,51,100),(0.071,0.085,101,150),(0.086,0.105,151,200),(0.106,0.200,201,300)],
    "co":[(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500)],
    "so2":[(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300),(605,1004,301,500)],
    "no2":[(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,2049,301,500)],
}
MOLECULAR_WEIGHTS = {"o3":48.0,"co":28.01,"so2":64.06,"no2":46.01}
TRUNCATE_DECIMALS = {"pm2_5":1,"pm10":0,"o3":3,"co":1,"so2":0,"no2":0}

def convert_units(conc,pollutant):
    if pollutant not in MOLECULAR_WEIGHTS:
        return conc
    ppb = conc*24.45/MOLECULAR_WEIGHTS[pollutant]
    if pollutant in ("o3","co"):
        return ppb/1000
    return ppb

def calc_aqi(conc,pollutant):
    factor = 10**TRUNCATE_DECIMALS[pollutant]
    conc = math.floor(conc*factor)/factor
    for bp_lo,bp_hi,aqi_lo,aqi_hi in BREAKPOINTS[pollutant]:
        if bp_lo<=conc<=bp_hi:
            return (aqi_hi-aqi_lo)/(bp_hi-bp_lo)*(conc-bp_lo)+aqi_lo
    return 500

def dominant_pollutant(concentrations):
    sub_indices = {p:calc_aqi(convert_units(c,p),p) for p,c in concentrations.items()}
    return max(sub_indices,key=sub_indices.get)

@contextmanager
def section(label):
    # a broken chart or a library that changed its api on redeploy should take down its
    # own block and nothing else, so every display section runs inside this
    try:
        yield
    except Exception as err:
        st.warning(f"{label} is unavailable right now.")
        print(f"section '{label}' failed: {err!r}")

def latest_conditions(city,hourly,daily):
    h = hourly[hourly["city"]==city].dropna(subset=["epa_aqi"]).sort_values("timestamp_utc")
    if len(h):
        r = h.iloc[-1]
        return {"aqi":r["epa_aqi"],"dominant":r["dominant_pollutant"],"pm2_5":r["pm2_5"],"pm10":r["pm10"],"o3":r["o3"],"no2":r["no2"],"as_of":f"{r['timestamp_pkt']:%d %b %Y, %H:%M} PKT","stale":False}
    d = daily[daily["city"]==city].dropna(subset=["aqi"]).sort_values("date")
    if len(d):
        r = d.iloc[-1]
        conc = {"pm2_5":r["pm2_5_mean"],"pm10":r["pm10_mean"],"o3":r["o3_mean"],"co":r["co_mean"],"so2":r["so2_mean"],"no2":r["no2_mean"]}
        return {"aqi":r["aqi"],"dominant":dominant_pollutant(conc),"pm2_5":r["pm2_5_mean"],"pm10":r["pm10_mean"],"o3":r["o3_mean"],"no2":r["no2_mean"],"as_of":f"{r['date']:%d %b %Y} daily average","stale":True}
    return None

@st.cache_resource
def get_project():
    return hopsworks.login(host="eu-west.cloud.hopsworks.ai",api_key_value=HOPSWORKS_KEY,project=HOPSWORKS_PROJECT,cert_folder=tempfile.gettempdir())

@st.cache_resource
def load_model():
    mr = get_project().get_model_registry()
    best = mr.get_best_model(name="aqi_random_forest",metric="rmse_aqi_d1",direction="min")
    bundle = joblib.load(pathlib.Path(best.download())/"random_forest.pkl")
    return bundle["model"],bundle["features"]

@st.cache_data(ttl=3600)
def load_daily():
    fg = get_project().get_feature_store().get_feature_group(name="aqi_daily",version=1)
    cutoff = (datetime.now(timezone.utc)-timedelta(days=40)).strftime("%Y-%m-%d")
    daily = fg.filter(fg.date>=cutoff).read()
    return daily.sort_values(["city","date"]).reset_index(drop=True)

@st.cache_data(ttl=3600)
def load_hourly():
    fg = get_project().get_feature_store().get_feature_group(name="aqi_hourly",version=1)
    # 10 days rather than 2 so a late or skipped feature pipeline run still leaves a recent row to show
    cutoff = (datetime.now(timezone.utc)-timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    hourly = fg.filter(fg.timestamp_utc>=cutoff).read()
    return hourly.sort_values(["city","timestamp_utc"]).reset_index(drop=True)

@retry(stop=stop_after_attempt(3),wait=wait_exponential(min=2,max=10))
def fetch_forecast_weather(lat,lon):
    r = requests.get(FORECAST_URL,params={"latitude":lat,"longitude":lon,"hourly":"temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m","forecast_days":7,"timezone":"Asia/Karachi"},timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def load_forecast_weather(lat,lon):
    data = fetch_forecast_weather(lat,lon)
    df = pd.DataFrame(data["hourly"])
    df["date"] = pd.to_datetime(df["time"]).dt.date
    return df.groupby("date").agg(temp=("temperature_2m","mean"),humidity=("relative_humidity_2m","mean"),wind=("wind_speed_10m","mean"),precip=("precipitation","sum")).reset_index()

def build_row(city,lat,lon,daily,features):
    core_features = [f for f in features if not f.startswith("fc_")]
    city_daily = daily[daily["city"]==city].rename(columns={f.lower():f for f in features}).sort_values("date")
    # one upstream day with no computable aqi blanks the 7 day rolling aqi columns for a
    # week or more, so carry the last good value forward rather than discarding every
    # recent row and falling back to a base date from last month
    roll_cols = [f for f in core_features if "roll" in f]
    city_daily[roll_cols] = city_daily[roll_cols].ffill().bfill()
    city_daily = city_daily.dropna(subset=core_features)
    row = city_daily.iloc[[-1]].copy()
    base_date = row["date"].iloc[0]
    fc = load_forecast_weather(lat,lon)
    fc = fc[fc["date"]>base_date.date()].sort_values("date").head(3).reset_index(drop=True)
    for h in range(1,4):
        row[f"fc_temp_d{h}"] = fc.loc[h-1,"temp"]
        row[f"fc_wind_d{h}"] = fc.loc[h-1,"wind"]
        row[f"fc_precip_d{h}"] = fc.loc[h-1,"precip"]
        row[f"fc_humidity_d{h}"] = fc.loc[h-1,"humidity"]
    return base_date,row[features]

try:
    model,features = load_model()
    daily = load_daily()
    hourly = load_hourly()
except Exception as err:
    st.error("The dashboard cannot reach its data or model right now. This is usually temporary, please refresh in a minute.")
    print(f"startup load failed: {err!r}")
    st.stop()

with st.sidebar:
    st.markdown("## Pearls AQI Predictor")
    st.caption("Three day AQI forecasts for five Pakistani cities, built on OpenWeather pollution and Open-Meteo weather data.")
    city = st.selectbox("City",list(CITIES))
    lat,lon = CITIES[city]
    st.divider()
    try:
        cond = latest_conditions(city,hourly,daily)
    except Exception as err:
        cond = None
        print(f"latest_conditions failed: {err!r}")
    if cond is None:
        st.error("No recent data in the feature store for this city. The feature pipeline may be behind, check back shortly.")
        st.stop()
    st.caption(f"data as of {cond['as_of']}")
    if cond["stale"]:
        st.warning("The hourly feed is behind so the current conditions below are the most recent daily average.")
    with st.expander("About this dashboard"):
        st.markdown("The AQI shown here is the US EPA AQI, calculated from raw pollutant concentrations and not OpenWeather's coarse 1 to 5 index. OpenWeather's pollution data comes from the CAMS atmospheric model and not ground stations, so it reads smoother than a ground monitor and will not match aqicn.org exactly. The dominant pollutant is whichever one produces the highest AQI after EPA's own conversion table, not whichever has the biggest raw number, which is why pm2.5 usually wins even when pm10's reading looks higher.")

current_aqi = cond["aqi"]
dominant = cond["dominant"]
category,color = category_for(current_aqi)
band_color = alert_color(current_aqi)

forecast = None
try:
    base_date,X = build_row(city,lat,lon,daily,features)
    pred = model.predict(X)[0]
    forecast = (base_date,X,pred)
except Exception as err:
    print(f"forecast build failed: {err!r}")

st.title(f"{city} air quality")

with section("Current conditions"):
    gauge_col,info_col = st.columns([1,2])
    with gauge_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            number={"font":{"size":44}},
            gauge={
                "axis":{"range":[0,400]},
                "bar":{"color":color},
                "bgcolor":DARK_BG,
                "borderwidth":0,
                "steps":[
                    {"range":[0,50],"color":"rgba(0,228,0,0.35)"},
                    {"range":[50,100],"color":"rgba(255,215,0,0.35)"},
                    {"range":[100,150],"color":"rgba(255,126,0,0.35)"},
                    {"range":[150,200],"color":"rgba(255,0,0,0.35)"},
                    {"range":[200,300],"color":"rgba(143,63,151,0.35)"},
                    {"range":[300,400],"color":"rgba(126,0,35,0.35)"},
                ],
            },
        ))
        gauge.update_layout(height=260,margin=dict(l=40,r=40,t=10,b=30),template="plotly_dark",paper_bgcolor=DARK_BG,font=dict(color="#E6EDF3"))
        st.plotly_chart(gauge,width="stretch")
    with info_col:
        st.markdown(f"#### {category}")
        st.caption(f"dominant pollutant : {dominant}")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("pm2.5",f"{cond['pm2_5']:.0f}")
        m2.metric("pm10",f"{cond['pm10']:.0f}")
        m3.metric("o3",f"{cond['o3']:.0f}")
        m4.metric("no2",f"{cond['no2']:.0f}")
        st.markdown(f"""
        <div style="background-color:{band_color};padding:0.9rem 1.2rem;border-radius:0.5rem;color:white;margin-top:0.5rem;">
        <strong>Hazard alert : {category}</strong><br>{HEALTH_GUIDANCE[category]}
        </div>
        """,unsafe_allow_html=True)

st.subheader("Three day forecast")
if forecast is None:
    st.info("The forecast is unavailable right now, check back shortly.")
else:
    base_date,X,pred = forecast
    with section("Three day forecast"):
        cols = st.columns(3)
        for i,col in enumerate(cols):
            day = base_date+pd.Timedelta(days=i+1)
            aqi_val = pred[i]
            cat,cat_color = category_for(aqi_val)
            with col:
                st.markdown(f"""
                <div style="border:1px solid #30363D;background-color:#161B22;border-radius:0.5rem;padding:1rem;text-align:center;">
                <div style="font-size:0.85rem;color:#8B949E;">{day:%A, %d %b}</div>
                <div style="font-size:2.2rem;font-weight:700;color:{cat_color};">{aqi_val:.0f}</div>
                <div style="font-size:0.9rem;">{cat}</div>
                </div>
                """,unsafe_allow_html=True)

with section("30 day trend and forecast"):
    st.subheader("30 day trend and forecast")
    city_daily = daily[daily["city"]==city].sort_values("date").tail(30)
    trend = go.Figure()
    bands = [(0,50,"green"),(50,100,"gold"),(100,150,"orange"),(150,200,"red"),(200,300,"purple"),(300,400,"darkred")]
    for lo,hi,c in bands:
        trend.add_hrect(y0=lo,y1=hi,fillcolor=c,opacity=0.13,line_width=0)
    trend.add_trace(go.Scatter(x=city_daily["date"],y=city_daily["aqi"],mode="lines+markers",name="actual",line=dict(color="steelblue",width=2)))
    if forecast is not None:
        base_date,X,pred = forecast
        future_dates = [base_date+pd.Timedelta(days=i+1) for i in range(3)]
        trend.add_trace(go.Scatter(x=[base_date]+future_dates,y=[city_daily["aqi"].iloc[-1]]+list(pred),mode="lines+markers",name="forecast",line=dict(color="darkorange",width=2,dash="dash")))
    trend.update_layout(height=420,margin=dict(l=20,r=20,t=10,b=10),yaxis_title="AQI",legend=dict(orientation="h",yanchor="bottom",y=1.02),template="plotly_dark",paper_bgcolor=DARK_BG,plot_bgcolor=DARK_BG)
    st.plotly_chart(trend,width="stretch")

with section("All cities comparison"):
    st.subheader("All cities right now")
    comp_rows = []
    for c,(la,lo) in CITIES.items():
        cc = latest_conditions(c,hourly,daily)
        if cc is None:
            continue
        cat,cat_color = category_for(cc["aqi"])
        comp_rows.append({"city":c,"lat":la,"lon":lo,"aqi":cc["aqi"],"category":cat})
    comp = pd.DataFrame(comp_rows)
    color_map = {name:color for _,name,color in AQI_CATEGORIES}
    map_col,bar_col = st.columns(2)
    with map_col:
        fig_map = px.scatter_map(comp,lat="lat",lon="lon",size="aqi",color="category",color_discrete_map=color_map,hover_name="city",hover_data={"aqi":":.0f","lat":False,"lon":False,"category":False},zoom=4.2,height=380)
        fig_map.update_layout(map_style="carto-darkmatter",margin=dict(l=0,r=0,t=0,b=0),legend=dict(orientation="h",yanchor="bottom",y=1.02),paper_bgcolor=DARK_BG,font=dict(color="#E6EDF3"))
        st.plotly_chart(fig_map,width="stretch")
    with bar_col:
        fig_bar = px.bar(comp.sort_values("aqi"),x="aqi",y="city",orientation="h",color="category",color_discrete_map=color_map,text="aqi")
        fig_bar.update_traces(texttemplate="%{text:.0f}",textposition="outside")
        fig_bar.update_layout(height=380,margin=dict(l=20,r=20,t=10,b=10),showlegend=False,xaxis_title="AQI",yaxis_title=None,template="plotly_dark",paper_bgcolor=DARK_BG,plot_bgcolor=DARK_BG)
        st.plotly_chart(fig_bar,width="stretch")

tab1,tab2 = st.tabs(["Explainability","Model performance"])

with tab1:
    st.markdown(f"What is driving tomorrow's forecast for {city}.")
    if forecast is None:
        st.info("Explainability needs the forecast, which is unavailable right now.")
    else:
        with section("Explainability"):
            base_date,X,pred = forecast
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            sv = shap_values[:,:,0]
            explanation = shap.Explanation(values=sv[0],base_values=explainer.expected_value[0],data=X.iloc[0].values,feature_names=features)
            plt.figure(figsize=(9,6))
            shap.plots.waterfall(explanation,show=False,max_display=12)
            fig = plt.gcf()
            fig.patch.set_facecolor(DARK_BG)
            for ax in fig.axes:
                ax.set_facecolor(DARK_BG)
            st.pyplot(fig,width="stretch",facecolor=DARK_BG)
            plt.close()

with tab2:
    with section("Model performance"):
        st.markdown("Average RMSE by model and forecast horizon, across all five cities on the last 60 day test period.")
        comparison = pd.read_csv("outputs/model_comparison.csv")
        summary = comparison.groupby(["model","horizon"])["rmse"].mean().unstack()[["aqi_d1","aqi_d2","aqi_d3"]].round(2)
        st.dataframe(summary,width="stretch")
        st.markdown("Per city RMSE for random forest, the model in production.")
        detail = comparison[comparison["model"]=="random_forest"].pivot(index="city",columns="horizon",values="rmse")[["aqi_d1","aqi_d2","aqi_d3"]].round(2)
        st.dataframe(detail,width="stretch")
