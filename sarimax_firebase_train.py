from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import firebase_admin
import joblib
import numpy as np
import pandas as pd
from firebase_admin import credentials, db
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app",
)
HISTORY_NODE = os.getenv("FIREBASE_HISTORY_NODE", "history")
MODEL_OUTPUT_PATH = Path(os.getenv("SARIMAX_MODEL_OUTPUT", "model_ai/sarimax_bundle.pkl"))
TRAIN_RATIO = float(os.getenv("SARIMAX_TRAIN_RATIO", "0.8"))
MIN_SAMPLES = int(os.getenv("SARIMAX_MIN_SAMPLES", "168"))
SEASONAL_PERIOD = int(os.getenv("SARIMAX_SEASONAL_PERIOD", "24"))
TARGET_NAME = os.getenv("SARIMAX_TARGET", "Daya")
USE_EXOG = os.getenv("SARIMAX_USE_EXOG", "true").lower() == "true"
AUTO_DROP_TEGANGAN_IF_SPARSE = True
TEGANGAN_NAN_THRESHOLD = 0.5


def initialize_firebase() -> None:
    firebase_key_json = os.getenv("FIREBASE_KEY")
    if not firebase_key_json:
        raise ValueError("FIREBASE_KEY tidak ditemukan di environment.")

    cred = credentials.Certificate(json.loads(firebase_key_json))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})


def fetch_history_dataframe() -> pd.DataFrame:
    raw_data = db.reference(HISTORY_NODE).get()
    if not raw_data:
        raise ValueError(f"Node '{HISTORY_NODE}' kosong atau tidak tersedia.")
    return pd.DataFrame.from_dict(raw_data, orient="index")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ["Arus", "Daya", "Energy", "Suhu", "Tegangan", "waktu"]
    available_cols = [column for column in expected_cols if column in df.columns]

    if "Daya" not in available_cols or "Energy" not in available_cols or "waktu" not in available_cols:
        raise ValueError("Kolom minimum yang dibutuhkan: Daya, Energy, waktu.")

    work = df[available_cols].copy()

    for column in ["Arus", "Daya", "Energy", "Suhu", "Tegangan"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if AUTO_DROP_TEGANGAN_IF_SPARSE and "Tegangan" in work.columns:
        tegangan_nan_ratio = float(work["Tegangan"].isna().mean())
        if tegangan_nan_ratio >= TEGANGAN_NAN_THRESHOLD:
            work = work.drop(columns=["Tegangan"])

    work["waktu"] = pd.to_numeric(work["waktu"], errors="coerce")
    work["waktu"] = pd.to_datetime(work["waktu"], unit="ms", errors="coerce")
    work = work.dropna(subset=["waktu"]).sort_values("waktu").set_index("waktu")
    work = work.replace([np.inf, -np.inf], np.nan).resample("1min").mean()

    for column in ["Daya", "Energy", "Arus"]:
        if column in work.columns:
            q1 = work[column].quantile(0.25)
            q3 = work[column].quantile(0.75)
            iqr = q3 - q1
            work[column] = work[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    for column in ["Daya", "Energy"]:
        if column in work.columns:
            work[column] = work[column].rolling(3, center=True).median()

    work = work.interpolate(method="time").ffill().bfill().dropna()
    if work.empty:
        raise ValueError("Dataset kosong setelah preprocessing.")

    return work


def build_target_series(processed_df: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name == "Daya":
        return processed_df["Daya"].resample("1H").mean().interpolate("time").ffill().bfill().dropna()

    if target_name == "Kwh_Jam":
        energy_hourly = processed_df["Energy"].resample("1H").mean().interpolate("time").ffill().bfill()
        return energy_hourly.diff().dropna().loc[lambda series: series >= 0]

    if target_name == "Rekap_Harian":
        energy_daily = processed_df["Energy"].resample("1D").mean().interpolate("time").ffill().bfill()
        return energy_daily.diff().dropna().loc[lambda series: series >= 0]

    raise ValueError(f"Target tidak dikenali: {target_name}")


def build_exogenous_features(processed_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    candidate_cols = [column for column in ["Tegangan", "Arus", "Suhu"] if column in processed_df.columns]
    if not USE_EXOG or not candidate_cols:
        return None

    exog = processed_df[candidate_cols].resample("1H").mean().interpolate("time").ffill().bfill()
    exog = exog.reindex(target_index).interpolate("time").ffill().bfill().dropna()
    return exog if not exog.empty else None


def split_train_test(series: pd.Series, exog: pd.DataFrame | None) -> Tuple[pd.Series, pd.Series, pd.DataFrame | None, pd.DataFrame | None]:
    split_idx = int(len(series) * TRAIN_RATIO)
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    if len(series) < MIN_SAMPLES or len(train_series) < 24 or len(test_series) < 2:
        raise ValueError(
            f"Data target belum cukup. Total={len(series)}, minimal={MIN_SAMPLES}, train={len(train_series)}, test={len(test_series)}"
        )

    if exog is None:
        return train_series, test_series, None, None

    return (
        train_series,
        test_series,
        exog.reindex(train_series.index).interpolate("time").ffill().bfill(),
        exog.reindex(test_series.index).interpolate("time").ffill().bfill(),
    )


def train_model(
    train_series: pd.Series,
    train_exog: pd.DataFrame | None,
    seasonal_period: int,
):
    auto_model = auto_arima(
        train_series,
        exogenous=train_exog,
        seasonal=True,
        m=seasonal_period,
        D=1,
        max_P=2,
        max_Q=2,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    fitted = SARIMAX(
        train_series,
        exog=train_exog,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    return fitted, auto_model


def evaluate_model(model_fit, test_series: pd.Series, test_exog: pd.DataFrame | None) -> Dict[str, float]:
    forecast = model_fit.get_forecast(steps=len(test_series), exog=test_exog).predicted_mean
    aligned = pd.DataFrame({"actual": test_series, "pred": forecast.values}, index=test_series.index).dropna()
    mse = float(mean_squared_error(aligned["actual"], aligned["pred"]))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(aligned["actual"], aligned["pred"]))
    return {"RMSE": rmse, "MAE": mae, "MSE": mse}


def main() -> None:
    print("🔌 Inisialisasi Firebase...")
    initialize_firebase()

    print("📥 Mengambil history dari Firebase...")
    raw_df = fetch_history_dataframe()
    processed_df = preprocess_dataframe(raw_df)
    target_series = build_target_series(processed_df, TARGET_NAME)
    exog_full = build_exogenous_features(processed_df, target_series.index)

    print(f"🎯 Target: {TARGET_NAME} | jumlah sampel: {len(target_series)}")
    train_series, test_series, train_exog, test_exog = split_train_test(target_series, exog_full)

    print("🤖 Menjalankan auto_arima + fit SARIMAX...")
    model_fit, auto_model = train_model(train_series, train_exog, SEASONAL_PERIOD)
    metrics = evaluate_model(model_fit, test_series, test_exog)

    print(f"order={auto_model.order} seasonal_order={auto_model.seasonal_order}")
    print(f"metrics={metrics}")

    print("💾 Fit final model dengan seluruh data...")
    final_model = SARIMAX(
        target_series,
        exog=exog_full,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    artifact = {
        "model_type": "statsmodels_sarimax",
        "model": final_model,
        "order": auto_model.order,
        "seasonal_order": auto_model.seasonal_order,
        "target_name": TARGET_NAME,
        "uses_exog": exog_full is not None,
        "exog_columns": list(exog_full.columns) if exog_full is not None else [],
        "train_ratio": TRAIN_RATIO,
        "metrics": metrics,
    }

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"✅ Model SARIMAX tersimpan di: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
