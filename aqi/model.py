"""Training, evaluation and saved models.

For each horizon (1–3 days) two gradient-boosted tree models are trained, one for
PM2.5 and one for PM10, on log-concentrations, for all cities together (the city is
a categorical feature). The AQI is then computed from the two predictions with the
city's national scale.

Evaluation is a strict time split: the model is trained on everything before
``test_start`` and scored on the following period, against
* persistence – tomorrow's AQI = today's AQI,
* 7-day mean – tomorrow's AQI = the mean of the last 7 days,
* the same model **without** the occasion features (ablation), to measure what the
  festival/season signals add.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .aqi_calc import aqi_array, category_index
from .cities import CITIES
from .features import HORIZONS, build_supervised, feature_columns

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def _regressor(cat_mask: list[bool], seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, categorical_features=cat_mask,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=40, random_state=seed,
    )


@dataclass
class HorizonModel:
    horizon: int
    features: list[str]
    pm25: HistGradientBoostingRegressor
    pm10: HistGradientBoostingRegressor

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        X = table[self.features]
        out = pd.DataFrame(index=table.index)
        out["pm2_5"] = np.expm1(self.pm25.predict(X)).clip(0)
        out["pm10"] = np.expm1(self.pm10.predict(X)).clip(0)
        return out


def fit_horizon(table: pd.DataFrame, h: int, use_occasions: bool = True) -> HorizonModel:
    feats = feature_columns(table, use_occasions)
    cat_mask = [c == "city_code" for c in feats]
    m25 = _regressor(cat_mask).fit(table[feats], np.log1p(table["y_pm25"]))
    m10 = _regressor(cat_mask).fit(table[feats], np.log1p(table["y_pm10"]))
    return HorizonModel(h, feats, m25, m10)


def _with_aqi(df: pd.DataFrame, pm25: str, pm10: str, name: str) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float, name=name)
    for city, idx in df.groupby("city").groups.items():
        std = CITIES[city].standard
        out.loc[idx] = aqi_array(df.loc[idx, pm25], df.loc[idx, pm10], std)
    return out


def _scores(true: pd.Series, pred: pd.Series, cities: pd.Series) -> dict:
    err = pred - true
    cat_hits = [category_index(t, CITIES[c].standard) == category_index(p, CITIES[c].standard)
                for t, p, c in zip(true, pred, cities)]
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "category_accuracy": float(np.mean(cat_hits)), "n": int(len(true))}


@dataclass
class Evaluation:
    test_start: str
    test_end: str
    results: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [f"Test period: {self.test_start} → {self.test_end} (trained only on earlier data)", ""]
        for h, res in self.results.items():
            lines += [f"### {h}-day-ahead AQI", "",
                      "| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |",
                      "|---|---|---|---|---|"]
            for name, s in res["overall"].items():
                ev = res["event_days"].get(name, {})
                lines.append(f"| {name} | {s['mae']:.1f} | {s['rmse']:.1f} | {s['category_accuracy']:.0%} | "
                             f"{ev.get('mae', float('nan')):.1f} |")
            lines += ["", "Per city (model, MAE): " + ", ".join(
                f"{c} {v['mae']:.1f}" for c, v in res["per_city"].items()), ""]
        return "\n".join(lines)


def evaluate(data: pd.DataFrame, test_start: str, test_end: str | None = None) -> Evaluation:
    ev = Evaluation(test_start, test_end or str(data["date"].max().date()))
    for h in HORIZONS:
        table = build_supervised(data, h)
        train = table[table["target_date"] < test_start]
        test = table[(table["issue_date"] >= test_start) & (table["target_date"] <= ev.test_end)].copy()
        full = fit_horizon(train, h, use_occasions=True)
        abl = fit_horizon(train, h, use_occasions=False)
        test[["m_pm25", "m_pm10"]] = full.predict(test).values
        test[["a_pm25", "a_pm10"]] = abl.predict(test).values
        test["aqi_true"] = _with_aqi(test, "y_pm25", "y_pm10", "aqi_true")
        test["aqi_model"] = _with_aqi(test, "m_pm25", "m_pm10", "aqi_model")
        test["aqi_no_occasions"] = _with_aqi(test, "a_pm25", "a_pm10", "aqi_no_occasions")
        test["aqi_persistence"] = _with_aqi(test, "pm25_lag0", "pm10_lag0", "p")
        mean7_pm10 = test.groupby("city")["pm10_lag0"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        test["mean7_pm10"] = mean7_pm10
        test["aqi_mean7"] = _with_aqi(test, "pm25_mean7", "mean7_pm10", "m7")
        methods = {"Model (history + weather + occasions)": "aqi_model",
                   "Model without occasion features": "aqi_no_occasions",
                   "Persistence (today's AQI)": "aqi_persistence",
                   "7-day mean": "aqi_mean7"}
        evt = test[test["ev_any"] > 0]
        ev.results[h] = {
            "overall": {n: _scores(test["aqi_true"], test[c], test["city"]) for n, c in methods.items()},
            "event_days": {n: _scores(evt["aqi_true"], evt[c], evt["city"]) for n, c in methods.items()}
            if len(evt) else {},
            "per_city": {city: _scores(g["aqi_true"], g["aqi_model"], g["city"])
                         for city, g in test.groupby("city")},
        }
        if h == 1:
            ev.predictions = test[["city", "issue_date", "target_date", "aqi_true", "aqi_model",
                                   "aqi_no_occasions", "aqi_persistence", "event_names"]]
    return ev


def train_all(data: pd.DataFrame) -> dict[int, HorizonModel]:
    return {h: fit_horizon(build_supervised(data, h), h) for h in HORIZONS}


def save_models(models: dict[int, HorizonModel], folder: Path = MODEL_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "aqi_models.joblib"
    joblib.dump(models, path)
    return path


def load_models(folder: Path = MODEL_DIR) -> dict[int, HorizonModel]:
    path = folder / "aqi_models.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m aqi train")
    return joblib.load(path)


def save_report(ev: Evaluation, folder: Path = REPORT_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(
        {"test_start": ev.test_start, "test_end": ev.test_end,
         "results": {str(k): v for k, v in ev.results.items()}}, indent=2))
    (folder / "metrics.md").write_text("# Evaluation\n\n" + ev.to_markdown())
    if hasattr(ev, "predictions"):
        ev.predictions.to_csv(folder / "test_predictions_1day.csv", index=False, float_format="%.1f")
