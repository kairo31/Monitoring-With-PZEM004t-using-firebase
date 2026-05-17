#!pip install pmdarima
#!pip uninstall gym -y

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json
import datetime
import pytz

# ==========================================
# KONFIGURASI PARAMETER 
# ==========================================
PATH_JSON = '/content/drive/MyDrive/SKRIPSHIT/firebase-project/firebase-project/serviceAccountKey.json'
DB_URL = 'https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app'
HISTORY_NODE = 'history'

SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 24)
FORECAST_STEPS = 168
AUTO_DROP_TEGANGAN_IF_SPARSE = True
TEGANGAN_NAN_THRESHOLD = 0.5
FORECAST_PLOT_GRANULARITY = 'D'
TARGET_SEASONALITY = {
    'Daya': 24,
    'Kwh_Jam': 24,
    'Rekap_Harian': 7,
}
MIN_SAMPLES_PER_TARGET = 168
STATSMODELS_MODEL_TYPE = "statsmodels_sarimax"

def install_required_packages() -> None:
    required = [
        'firebase-admin',
        'statsmodels',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pmdarima',
        'joblib',
    ]
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *required])

install_required_packages()

import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

def mount_google_drive() -> None:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    except ImportError:
        pass

def firebase_json_to_dataframe(raw_data: Dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(raw_data, orient='index')
    return df

def validate_expected_columns(df: pd.DataFrame) -> None:
    expected_cols = ['Arus', 'Daya', 'Energy', 'Suhu', 'Tegangan', 'waktu']
    for col in expected_cols:
        status = 'AVAILABLE' if col in df.columns else 'MISSING'

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ['Arus', 'Daya', 'Energy', 'Suhu', 'Tegangan', 'waktu']
    available_cols = [c for c in expected_cols if c in df.columns]

    if 'Energy' not in available_cols:
        raise ValueError("Column 'Energy' is required as the target variable.")
    if 'waktu' not in available_cols:
        raise ValueError("Column 'waktu' is required for datetime indexing.")

    work = df[available_cols].copy()

    for col in ['Arus', 'Daya', 'Energy', 'Suhu', 'Tegangan']:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors='coerce')

    if AUTO_DROP_TEGANGAN_IF_SPARSE and 'Tegangan' in work.columns:
        tegangan_nan_ratio = float(work['Tegangan'].isna().mean())
        if tegangan_nan_ratio >= TEGANGAN_NAN_THRESHOLD:
            work = work.drop(columns=['Tegangan'])

    work['waktu'] = pd.to_numeric(work['waktu'], errors='coerce')
    work['waktu'] = work['waktu'].replace([np.inf, -np.inf], np.nan)
    work.loc[(work['waktu'] > 3000000000000) | (work['waktu'] < 0), 'waktu'] = np.nan
    work['waktu'] = pd.to_datetime(work['waktu'], unit='ms', errors='coerce')

    work = work.dropna(subset=['waktu'])
    work = work.sort_values('waktu')
    work = work.set_index('waktu')

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.resample('1min').mean()

    for col in ['Daya', 'Energy', 'Arus']:
        if col in work.columns:
            q1 = work[col].quantile(0.25)
            q3 = work[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            work[col] = work[col].clip(lower, upper)

    for col in ['Daya', 'Energy']:
        if col in work.columns:
            work[col] = work[col].rolling(3, center=True).median()

    work = work.interpolate(method='time').ffill().bfill()
    work = work.dropna()

    if work.empty:
        raise ValueError('Processed dataset is empty after cleaning.')

    return work

def split_train_test(series: pd.Series, train_ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    if len(train) < 10 or len(test) < 2:
        raise ValueError('Dataset too small after split. Collect more data before modeling.')

    return train, test

def infer_time_frequency(index: pd.DatetimeIndex) -> str:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError('History index must be DatetimeIndex.')

    inferred = pd.infer_freq(index)
    if inferred:
        return inferred

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 'D'

    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return 'D'

    offset = pd.tseries.frequencies.to_offset(median_delta)
    return offset.freqstr

def infer_seasonal_period(index: pd.DatetimeIndex, fallback: int = 24) -> int:
    freq = infer_time_frequency(index)
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        delta = pd.Timedelta(offset)
    except Exception:
        return fallback

    if delta >= pd.Timedelta(days=1):
        return 7

    if delta <= pd.Timedelta(0):
        return fallback

    daily_points = int(round(pd.Timedelta(days=1) / delta))
    return max(2, daily_points)

def plot_basic_visualization(series: pd.Series, label: str = 'Energy') -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(series.index, series.values, color='tab:blue')
    plt.title(f'{label} Time Series (Raw/Cleaned)')
    plt.xlabel('Time')
    plt.ylabel(label)
    plt.tight_layout()

    output_dir = Path('/content/drive/MyDrive/SKRIPSHIT/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'basic_vis_{label.lower()}.png', dpi=150)
    plt.show()

def run_stationarity_test(series: pd.Series) -> Dict[str, float]:
    adf_stat, p_value, *_ = adfuller(series.dropna())
    result = {
        'adf_statistic': float(adf_stat),
        'p_value': float(p_value),
        'is_stationary': bool(p_value < 0.05),
    }
    return result

def plot_acf_pacf_analysis(series: pd.Series, lags: int = 40, label: str = 'Energy') -> None:
    series_clean = series.dropna()
    n_obs = len(series_clean)
    if n_obs < 4:
        print('Data terlalu sedikit untuk plot ACF/PACF (minimal 4 observasi).')
        return

    max_safe_lag = max(1, (n_obs // 2) - 1)
    safe_lags = min(lags, max_safe_lag)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(series_clean, lags=safe_lags, ax=axes[0])
    plot_pacf(series_clean, lags=safe_lags, ax=axes[1], method='ywm')
    axes[0].set_title(f'ACF - {label} (lags={safe_lags})')
    axes[1].set_title(f'PACF - {label} (lags={safe_lags})')
    plt.tight_layout()

    output_dir = Path('/content/drive/MyDrive/SKRIPSHIT/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'acf_pacf_{label.lower()}.png', dpi=150)
    plt.show()

def build_exogenous_features(processed_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    candidate_cols = [c for c in ['Tegangan', 'Arus', 'Suhu', 'AC', 'Magicom', 'Waterheater', 'Laptop', 'TV'] if c in processed_df.columns]
    if not candidate_cols:
        return None

    exog = processed_df[candidate_cols].copy()
    freq = infer_time_frequency(target_index)
    exog = exog.resample(freq).mean().interpolate('time').ffill().bfill()
    exog = exog.reindex(target_index).interpolate('time').ffill().bfill()
    exog = exog.dropna()
    if exog.empty:
        return None
    return exog

def build_target_series(processed_df: pd.DataFrame) -> Dict[str, pd.Series]:
    if 'Energy' not in processed_df.columns:
        raise ValueError("Column 'Energy' tidak ditemukan pada data yang sudah diproses.")
    if 'Daya' not in processed_df.columns:
        raise ValueError("Column 'Daya' tidak ditemukan pada data yang sudah diproses.")

    daya_series = processed_df['Daya'].resample('1H').mean().interpolate('time').ffill().bfill().dropna()

    energy_hourly_cumulative = processed_df['Energy'].resample('1H').mean().interpolate('time').ffill().bfill()
    kwh_jam_series = energy_hourly_cumulative.diff().dropna()
    kwh_jam_series = kwh_jam_series[kwh_jam_series >= 0]

    energy_daily_cumulative = processed_df['Energy'].resample('1D').mean().interpolate('time').ffill().bfill()
    rekap_harian_series = energy_daily_cumulative.diff().dropna()
    rekap_harian_series = rekap_harian_series[rekap_harian_series >= 0]

    targets = {
        'Daya': daya_series,
        'Kwh_Jam': kwh_jam_series,
        'Rekap_Harian': rekap_harian_series,
    }

    valid_targets = {}
    for name, series in targets.items():
        if series.empty or len(series) < MIN_SAMPLES_PER_TARGET:
            continue
        valid_targets[name] = series

    if not valid_targets:
        raise ValueError(f'Butuh minimal {MIN_SAMPLES_PER_TARGET} sampel per target.')

    return valid_targets

def train_sarimax(
    train_series: pd.Series,
    train_exog: pd.DataFrame | None = None,
    order: Tuple[int, int, int] = SARIMAX_ORDER,
    seasonal_order: Tuple[int, int, int, int] = SARIMAX_SEASONAL_ORDER,
):
    model = SARIMAX(
        train_series,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted

def evaluate_model(
    model_fit,
    test_series: pd.Series,
    test_exog: pd.DataFrame | None = None,
) -> Tuple[pd.Series, Dict[str, float]]:
    forecast_obj = model_fit.get_forecast(steps=len(test_series), exog=test_exog)
    preds = forecast_obj.predicted_mean

    preds = pd.Series(preds.values, index=test_series.index, name='predicted_energy')

    df_eval = pd.DataFrame({'actual': test_series, 'pred': preds}).dropna()
    if df_eval.empty:
        raise ValueError('Evaluation frame is empty after dropping NaN.')

    mse = float(mean_squared_error(df_eval['actual'], df_eval['pred']))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(df_eval['actual'], df_eval['pred']))
    mape = float(mean_absolute_percentage_error(df_eval['actual'], df_eval['pred']))
    r2 = float(r2_score(df_eval['actual'], df_eval['pred']))

    # Normalisasi rasio error (menyamakan skala dengan persentase)
    mean_actual = float(df_eval['actual'].mean())
    if mean_actual == 0:
        mean_actual = 1e-9
    nrmse = rmse / mean_actual
    nmae = mae / mean_actual

    metrics = {
        'RMSE': rmse,
        'NRMSE': nrmse,
        'MAE': mae,
        'NMAE': nmae,
        'MSE': mse,
        'MAPE': mape,
        'R2': r2
    }
    return preds, metrics

def walk_forward_validate(
    train_series: pd.Series,
    test_series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> Dict[str, float]:

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_train = model.fit(disp=False)

    res_updated = fitted_train.append(test_series, refit=False)
    forecast_obj = res_updated.get_prediction(start=test_series.index[0], end=test_series.index[-1], dynamic=False)
    pred_series = forecast_obj.predicted_mean

    mse = float(mean_squared_error(test_series, pred_series))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(test_series, pred_series))
    mape = float(mean_absolute_percentage_error(test_series, pred_series))
    r2 = float(r2_score(test_series, pred_series))

    mean_actual = float(test_series.mean())
    if mean_actual == 0:
        mean_actual = 1e-9
    nrmse = rmse / mean_actual
    nmae = mae / mean_actual

    return {
        'RMSE': rmse,
        'NRMSE': nrmse,
        'MAE': mae,
        'NMAE': nmae,
        'MSE': mse,
        'MAPE': mape,
        'R2': r2
    }

def plot_train_test_prediction(train: pd.Series, test: pd.Series, preds: pd.Series, label: str = 'Energy') -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train.values, label=f'Train ({label})', color='tab:blue')
    plt.plot(test.index, test.values, label=f'Test Actual ({label})', color='tab:green')
    plt.plot(test.index, preds.values, label=f'Test Prediction ({label})', color='tab:red', linestyle='--')
    plt.title(f'SARIMAX - Train vs Test vs Prediction ({label})')
    plt.xlabel('Time')
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()

    output_dir = Path('/content/drive/MyDrive/SKRIPSHIT/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'train_test_pred_{label.lower()}.png', dpi=150)
    plt.show()

def plot_future_forecast(
    history: pd.Series,
    forecast: pd.Series,
    granularity: str = FORECAST_PLOT_GRANULARITY,
    label: str = 'Energy',
) -> None:
    allowed = {'D', 'W', 'M'}
    if granularity not in allowed:
        raise ValueError(f"Invalid granularity '{granularity}'. Use one of {allowed}.")

    history_aligned = history.copy()
    forecast_aligned = forecast.copy()

    if not isinstance(history_aligned.index, pd.DatetimeIndex):
        history_aligned.index = pd.to_datetime(history_aligned.index, errors='coerce')
    if not isinstance(forecast_aligned.index, pd.DatetimeIndex):
        forecast_aligned.index = pd.to_datetime(forecast_aligned.index, errors='coerce')

    combined = pd.concat([
        history_aligned.rename('energy_actual'),
        forecast_aligned.rename('energy_forecast')
    ], axis=1)
    combined = combined[~combined.index.isna()].sort_index()

    aggregated = combined.resample(granularity).sum(min_count=1)

    aggregated['energy_actual'] = aggregated['energy_actual'].astype(float)
    aggregated['energy_forecast'] = aggregated['energy_forecast'].astype(float)

    x = np.arange(len(aggregated))

    plt.figure(figsize=(14, 6))
    plt.plot(x, aggregated['energy_actual'].values, label=f'Historical {label}', color='tab:blue')
    plt.plot(x, aggregated['energy_forecast'].values, label=f'{FORECAST_STEPS}-Step Forecast', color='tab:orange')

    granularity_label = {'D': 'day', 'W': 'week', 'M': 'month'}[granularity]
    plt.title(f'Future Forecast ({label}) - Aggregated per {granularity_label}')
    plt.xlabel(f'Period index since first data capture ({granularity_label})')
    plt.ylabel(f'{label} (aggregated)')

    tick_step = max(1, len(x) // 10)
    tick_positions = x[::tick_step]
    tick_labels = [aggregated.index[i].strftime('%Y-%m-%d') for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=30)

    plt.legend()
    plt.tight_layout()
    output_dir = Path('/content/drive/MyDrive/SKRIPSHIT/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    file_label = label.lower().replace(' ', '_')
    plt.savefig(output_dir / f'future_forecast_{file_label}.png', dpi=150)
    plt.show()

def generate_future_forecast(
    model_fit,
    history_series: pd.Series,
    steps: int = FORECAST_STEPS,
    future_exog: pd.DataFrame | None = None,
) -> pd.Series:
    future_obj = model_fit.get_forecast(steps=steps, exog=future_exog)
    future_pred = future_obj.predicted_mean

    if not isinstance(future_pred.index, pd.DatetimeIndex):
        history_index = history_series.index
        freq = infer_time_frequency(history_index)
        start_time = history_index.max() + pd.tseries.frequencies.to_offset(freq)
        future_index = pd.date_range(start=start_time, periods=steps, freq=freq)
        future_pred = pd.Series(future_pred.values, index=future_index, name='forecast_energy')

    return future_pred

def run_single_target_pipeline(
    target_name: str,
    target_series: pd.Series,
    processed_df: pd.DataFrame,
) -> Dict[str, object]:

    print(f"\n---> Memproses Target: {target_name} <---")
    stationarity_result = run_stationarity_test(target_series)

    plot_basic_visualization(target_series, label=target_name)
    plot_acf_pacf_analysis(target_series, label=target_name)

    seasonal_period = infer_seasonal_period(target_series.index, fallback=TARGET_SEASONALITY.get(target_name, 24))

    exog_full = build_exogenous_features(processed_df, target_series.index) if target_name == 'Daya' else None

    train_series, test_series = split_train_test(target_series, train_ratio=0.8)

    train_exog = None
    test_exog = None
    if exog_full is not None:
        train_exog = exog_full.reindex(train_series.index)
        test_exog = exog_full.reindex(test_series.index)
        train_exog = train_exog.interpolate('time').ffill().bfill()
        test_exog = test_exog.interpolate('time').ffill().bfill()

    print("Fitting SARIMAX Model...")
    try:
        model_fit = train_sarimax(
            train_series,
            train_exog=train_exog,
            order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL_ORDER,
        )
    except Exception as e:
        raise ValueError(f"SARIMAX gagal fitting model pada {target_name}. Error: {e}")

    test_preds, metrics = evaluate_model(model_fit, test_series, test_exog=test_exog)

    print("Melakukan Walk-Forward Validation...")
    wf_metrics = walk_forward_validate(
        train_series=train_series,
        test_series=test_series,
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
    )

    plot_train_test_prediction(train_series, test_series, test_preds, label=target_name)

    print("Fitting Final Model untuk Forecast...")
    final_model_fit = None
    try:
        final_model_fit = train_sarimax(
            target_series,
            train_exog=exog_full,
            order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL_ORDER,
        )
    except Exception as e:
        raise ValueError(f"SARIMAX gagal fitting final model pada {target_name}. Error: {e}")

    drive_model_dir = '/content/drive/MyDrive/SKRIPSHIT/models'
    os.makedirs(drive_model_dir, exist_ok=True)
    filename = f"{drive_model_dir}/sarimax_{target_name.lower()}.pkl"

    if final_model_fit is not None:
        artifact_bundle = {
            "model_type": STATSMODELS_MODEL_TYPE,
            "model": final_model_fit,
            "uses_exog": True if exog_full is not None else False,
            "exog_columns": list(exog_full.columns) if exog_full is not None else []
        }
        joblib.dump(artifact_bundle, filename, compress=3)
    else:
        raise ValueError(f"Model final {target_name} bernilai None setelah training.")

    future_exog = None
    if exog_full is not None:
        freq = infer_time_frequency(target_series.index)
        future_index = pd.date_range(
            start=target_series.index.max() + pd.tseries.frequencies.to_offset(freq),
            periods=FORECAST_STEPS,
            freq=freq,
        )
        future_exog = pd.DataFrame(
            [exog_full.iloc[-1].values] * len(future_index),
            index=future_index,
            columns=exog_full.columns,
        )

    future_forecast = generate_future_forecast(
        final_model_fit,
        target_series,
        steps=FORECAST_STEPS,
        future_exog=future_exog,
    )
    plot_future_forecast(
        target_series,
        future_forecast,
        granularity=FORECAST_PLOT_GRANULARITY,
        label=target_name,
    )

    return {'target': target_name, 'metrics': metrics, 'walk_forward': wf_metrics, 'stationarity': stationarity_result}

def main() -> None:
    mount_google_drive()

    # -----------------------------------------------------
    # MENGAMBIL DATA DARI FILE JSON LOKAL (Bypass Firebase Database Limit)
    # -----------------------------------------------------
    file_path = '/content/test-reading-the-pzem-default-rtdb-export (9).json'

    print(f"Membaca data lokal dari {file_path} ...")
    if not Path(file_path).exists():
         raise FileNotFoundError(f"File {file_path} tidak ditemukan! Pastikan kamu sudah upload ke Colab.")

    with open(file_path, 'r') as f:
        full_data = json.load(f)

    raw_data = full_data.get('history', full_data)
    print(f"Berhasil membaca data JSON!")

    df = firebase_json_to_dataframe(raw_data)
    validate_expected_columns(df)
    processed_df = preprocess_dataframe(df)
    target_map = build_target_series(processed_df)

    results = []
    for target_name, target_series in target_map.items():
        try:
            results.append(run_single_target_pipeline(target_name, target_series, processed_df))
        except Exception as exc:
            print(f"Skipping {target_name} due to error: {exc}")
            pass

    if not results:
        raise RuntimeError('Semua target gagal dijalankan.')

    comparison_df = pd.DataFrame([
        {
            'Target': r['target'],
            'RMSE': round(r['metrics']['RMSE'], 4),
            'NRMSE': round(r['metrics']['NRMSE'], 4),
            'MAE': round(r['metrics']['MAE'], 4),
            'NMAE': round(r['metrics']['NMAE'], 4),
            'MAPE': round(r['metrics']['MAPE'], 4),
            'R2': round(r['metrics']['R2'], 4),
            'WF_NRMSE': round(r['walk_forward']['NRMSE'], 4),
        }
        for r in results
    ]).sort_values('RMSE')

    print("\nRingkasan Evaluasi Model (Rasio Normalisasi Disamakan):")
    print(comparison_df.to_string(index=False))

    # ==========================================================
    # PENGIRIMAN DATA EVALUASI ERROR DAN STATUS MODEL KE FIREBASE
    # ==========================================================
    tz = pytz.timezone("Asia/Jakarta")
    waktu_skrg = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    firebase_eval_payload = {}
    for r in results:
        target = r['target']
        firebase_eval_payload[target] = {
            "Status_Stasioneritas": r['stationarity'],
            "Metrics_Pengujian": {
                "RMSE": round(r['metrics']['RMSE'], 4),
                "NRMSE": round(r['metrics']['NRMSE'], 4),
                "MAE": round(r['metrics']['MAE'], 4),
                "NMAE": round(r['metrics']['NMAE'], 4),
                "MSE": round(r['metrics']['MSE'], 4),
                "MAPE": round(r['metrics']['MAPE'], 4),
                "R-Squared": round(r['metrics']['R2'], 4)
            },
            "Metrics_Walk_Forward": {
                "RMSE": round(r['walk_forward']['RMSE'], 4),
                "NRMSE": round(r['walk_forward']['NRMSE'], 4),
                "MAE": round(r['walk_forward']['MAE'], 4),
                "NMAE": round(r['walk_forward']['NMAE'], 4),
                "MSE": round(r['walk_forward']['MSE'], 4),
                "MAPE": round(r['walk_forward']['MAPE'], 4),
                "R-Squared": round(r['walk_forward']['R2'], 4)
            },
            "Waktu_Training": waktu_skrg
        }

    try:
        # Inisiasi ulang ke DB Firebase jika butuh nulis evaluasi ke cloud
        if not firebase_admin._apps:
            cred = credentials.Certificate(PATH_JSON)
            firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})

        db.reference("Evaluasi_Model_SARIMAX").set(firebase_eval_payload)
        print("\n✅ Seluruh nilai error (Termasuk NRMSE & NMAE) dan status model berhasil dikirim ke Firebase!")
    except Exception as e:
        print(f"\n⚠️ Gagal mengirim evaluasi ke Firebase (bisa diabaikan jika hanya butuh hasil di Colab): {e}")

if __name__ == '__main__':
    main()
