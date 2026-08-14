import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
import hopsworks
import joblib
import optuna
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

load_dotenv()
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

optuna.logging.set_verbosity(optuna.logging.WARNING)

model_dir = pathlib.Path("models")
model_dir.mkdir(exist_ok=True)

TARGETS = ["aqi_d1","aqi_d2","aqi_d3"]
CITIES = ["Karachi","Lahore","Faisalabad","Islamabad","Peshawar"]

project = hopsworks.login(host="eu-west.cloud.hopsworks.ai",api_key_value=HOPSWORKS_KEY,project=HOPSWORKS_PROJECT,cert_folder=tempfile.gettempdir())
fs = project.get_feature_store()
mr = project.get_model_registry()

daily = fs.get_feature_group(name="aqi_daily",version=1).read()

FEATURES = [c for c in daily.columns if c not in ["city","date"]+TARGETS]
daily = daily.dropna(subset=FEATURES+TARGETS).reset_index(drop=True)

test_start = daily["date"].max() - pd.Timedelta(days=59)
train_mask = daily["date"] < test_start
test_mask = ~train_mask

train = daily[train_mask].sort_values("date").reset_index(drop=True)
test = daily[test_mask].reset_index(drop=True)

X_train,y_train = train[FEATURES],train[TARGETS]
X_test,y_test = test[FEATURES],test[TARGETS]

print(f"train {len(X_train)} test {len(X_test)}")

tscv = TimeSeriesSplit(n_splits=3)

def rf_objective(trial):
    params = {
        "n_estimators":trial.suggest_int("n_estimators",100,400),
        "max_depth":trial.suggest_int("max_depth",5,25),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,10),
    }
    maes = []
    for tr_idx,va_idx in tscv.split(X_train):
        rf = RandomForestRegressor(**params,n_jobs=-1,random_state=100)
        rf.fit(X_train.iloc[tr_idx],y_train.iloc[tr_idx])
        maes.append(mean_absolute_error(y_train.iloc[va_idx],rf.predict(X_train.iloc[va_idx])))
    return np.mean(maes)

study_rf = optuna.create_study(direction="minimize")
study_rf.optimize(rf_objective,n_trials=20)
print(f"best MAE : {study_rf.best_value}")

random_forest = RandomForestRegressor(**study_rf.best_params,n_jobs=-1,random_state=100)
random_forest.fit(X_train,y_train)
preds = random_forest.predict(X_test)

results = []
for i,horizon in enumerate(TARGETS):
    truth = y_test[horizon].values
    pred = preds[:,i]
    for city in CITIES:
        mask = (test["city"]==city).values
        rmse = np.sqrt(mean_squared_error(truth[mask],pred[mask]))
        mae = mean_absolute_error(truth[mask],pred[mask])
        r2 = r2_score(truth[mask],pred[mask])
        results.append({"horizon":horizon,"city":city,"rmse":rmse,"mae":mae,"r2":r2})

comparison = pd.DataFrame(results)
horizon_metrics = comparison.groupby("horizon")[["rmse","mae","r2"]].mean()

metrics = {}
for horizon,row in horizon_metrics.iterrows():
    metrics[f"rmse_{horizon}"] = round(row["rmse"],3)
    metrics[f"mae_{horizon}"] = round(row["mae"],3)
    metrics[f"r2_{horizon}"] = round(row["r2"],3)

joblib.dump({"model":random_forest,"features":FEATURES},model_dir/"random_forest.pkl")

model_schema = ModelSchema(input_schema=Schema(X_test),output_schema=Schema(y_test))
incumbent = mr.get_best_model(name="aqi_random_forest",metric="rmse_aqi_d1",direction="min")

if incumbent is None or metrics["rmse_aqi_d1"] < incumbent.training_metrics["rmse_aqi_d1"]:
    registered = mr.sklearn.create_model(
        name="aqi_random_forest",
        metrics=metrics,
        description="random forest, three day ahead aqi forecast for all five cities",
        input_example=X_test.iloc[[0]],
        model_schema=model_schema,
    )
    registered.save(str(model_dir/"random_forest.pkl"))
    print(f"registered version {registered.version} | rmse_aqi_d1 {metrics['rmse_aqi_d1']}")
else:
    print(f"kept incumbent version {incumbent.version} | rmse_aqi_d1 {incumbent.training_metrics['rmse_aqi_d1']}")
