from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

LEGACY_MODEL_TYPE = "pmdarima"
STATSMODELS_MODEL_TYPE = "statsmodels_sarimax"


def load_sarimax_artifact(model_path: str | Path) -> Dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")

    artifact = joblib.load(model_path)

    if isinstance(artifact, dict) and "model_type" in artifact:
        return artifact

    return {
        "model_type": LEGACY_MODEL_TYPE,
        "model": artifact,
        "uses_exog": True,
        "exog_columns": ["Suhu", "AC", "Magicom", "Waterheater", "Laptop", "TV"],
    }


def forecast_next_step(
    artifact: Dict[str, Any],
    latest_features: Dict[str, float | int],
    current_watt: float = None,
) -> float:
    model_type = artifact["model_type"]
    uses_exog = artifact.get("uses_exog", False)
    exog_columns = artifact.get("exog_columns", [])
    future_exog = None

    if uses_exog:
        future_exog = pd.DataFrame(
            [[latest_features.get(column, 0.0) for column in exog_columns]],
            columns=exog_columns,
        )

    if current_watt is not None:
        current_y = pd.Series([current_watt])
        try:
            if model_type == STATSMODELS_MODEL_TYPE:
                artifact["model"] = artifact["model"].apply(current_y, exog=future_exog)
            elif model_type == LEGACY_MODEL_TYPE:
                artifact["model"].update(current_y, X=future_exog)
        except Exception as e:
            print(f"⚠️ Gagal menyuntikkan data aktual ke SARIMAX: {e}")

    if model_type == STATSMODELS_MODEL_TYPE:
        forecast = artifact["model"].get_forecast(steps=1, exog=future_exog)
        return float(forecast.predicted_mean.iloc[0])

    if model_type == LEGACY_MODEL_TYPE:
        legacy_pred = artifact["model"].predict(n_periods=1, X=future_exog)
        if hasattr(legacy_pred, "iloc"):
            return float(legacy_pred.iloc[0])
        return float(legacy_pred[0])

    raise ValueError(f"Model type tidak dikenali: {model_type}")
