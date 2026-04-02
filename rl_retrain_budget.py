from __future__ import annotations

import calendar
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import firebase_admin
import numpy as np
import pandas as pd
import pytz
from firebase_admin import credentials, db
from gymnasium import Env, spaces
from stable_baselines3 import PPO

DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app",
)
HISTORY_NODE = os.getenv("FIREBASE_HISTORY_NODE", "history")
MODEL_OUTPUT_PATH = Path(os.getenv("RL_MODEL_OUTPUT", "model_ai/model_rl_budget.zip"))
DEFAULT_MONTHLY_TARGET_RP = float(os.getenv("DEFAULT_MONTHLY_TARGET_RP", "250000"))
TARIF_PER_KWH = float(os.getenv("TARIF_PER_KWH", "1352.0"))
TOTAL_TIMESTEPS = int(os.getenv("RL_TOTAL_TIMESTEPS", "40000"))


@dataclass
class TrainRow:
    daya: float
    suhu: float
    arus: float
    tegangan: float
    jam: float
    day_of_week: float
    is_weekend: float
    day_progress: float
    days_remaining: float


class BudgetEnergyEnv(Env):
    """Budget-aware RL env with predicted watt and budget gap in observation."""

    metadata = {"render_modes": []}

    def __init__(self, rows: list[TrainRow], monthly_target_rp: float):
        super().__init__()
        self.rows = rows
        self.monthly_target_rp = monthly_target_rp
        self.i = 0
        self.prev_action = 0

        self.action_space = spaces.Discrete(5)
        high = np.array(
            [
                5000,  # daya
                60,  # suhu
                1, 1, 1, 1,  # device flags
                23,  # jam
                6,  # day_of_week
                1,  # is_weekend
                1,  # is_peak_hour
                1,  # day_progress
                31,  # days_remaining
                5000,  # prediksi_watt
                2_000_000,  # monthly target
                2_000_000,  # gap to target
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                0,
                10,
                0, 0, 0, 0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10_000,
                -2_000_000,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _device_flags(self, daya: float) -> tuple[int, int, int, int]:
        ac = 1 if daya > 350 else 0
        wh = 1 if daya > 600 else 0
        magicom = 1 if 120 < daya < 450 else 0
        tv = 1 if 40 < daya < 220 else 0
        return ac, magicom, wh, tv

    def _build_state(self, row: TrainRow) -> np.ndarray:
        ac, magicom, wh, tv = self._device_flags(row.daya)
        prediksi_watt = float((0.6 * row.daya) + (0.25 * row.arus * row.tegangan) + (2.0 * row.suhu))
        prediksi_watt = max(prediksi_watt, 0.0)
        is_peak_hour = 1.0 if (17 <= row.jam <= 22) else 0.0

        est_daily_cost = (prediksi_watt / 1000.0) * 24 * TARIF_PER_KWH
        gap_to_target = (est_daily_cost * 30) - self.monthly_target_rp

        return np.array(
            [
                row.daya,
                row.suhu,
                ac,
                magicom,
                wh,
                tv,
                row.jam,
                row.day_of_week,
                row.is_weekend,
                is_peak_hour,
                row.day_progress,
                row.days_remaining,
                prediksi_watt,
                self.monthly_target_rp,
                gap_to_target,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        self.prev_action = 0
        return self._build_state(self.rows[self.i]), {}

    def step(self, action: int):
        row = self.rows[self.i]
        state = self._build_state(row)
        prediksi_watt = float(state[12])
        gap_to_target = float(state[14])

        reduction_factor = {0: 0.00, 1: 0.12, 2: 0.07, 3: 0.18, 4: 0.05}[int(action)]
        expected_after_action = prediksi_watt * (1.0 - reduction_factor)
        est_daily_cost_after = (expected_after_action / 1000.0) * 24 * TARIF_PER_KWH
        gap_after = (est_daily_cost_after * 30) - self.monthly_target_rp

        reward = -abs(gap_after) / max(self.monthly_target_rp, 1.0)
        if gap_to_target <= 0 and action != 0:
            reward -= 0.2
        if gap_to_target > 0 and action == 0:
            reward -= 0.2
        if int(action) != int(self.prev_action):
            reward -= 0.03
        if row.suhu >= 30 and int(action) == 1:
            reward -= 0.08
        if row.daya < 120 and int(action) in {1, 3}:
            reward -= 0.05

        self.i += 1
        self.prev_action = int(action)
        terminated = self.i >= len(self.rows) - 1
        next_state = self._build_state(self.rows[self.i if not terminated else -1])

        info = {
            "prediksi_watt": prediksi_watt,
            "gap_to_target": gap_to_target,
            "gap_after": gap_after,
        }
        return next_state, float(reward), terminated, False, info


def initialize_firebase() -> None:
    firebase_key_json = os.getenv("FIREBASE_KEY")
    if not firebase_key_json:
        raise ValueError("FIREBASE_KEY tidak ditemukan di environment.")

    cred = credentials.Certificate(json.loads(firebase_key_json))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})


def load_training_rows() -> list[TrainRow]:
    raw_data = db.reference(HISTORY_NODE).get()
    if not raw_data:
        raise ValueError(f"Node '{HISTORY_NODE}' kosong atau tidak tersedia.")

    df = pd.DataFrame.from_dict(raw_data, orient="index")
    required = ["Daya", "Suhu", "Arus", "Tegangan", "waktu"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib untuk retrain RL tidak lengkap: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["waktu"] = pd.to_datetime(df["waktu"], unit="ms", errors="coerce")
    df = df.dropna(subset=["waktu"]).sort_values("waktu")
    df = df.dropna(subset=["Daya", "Suhu", "Arus", "Tegangan"])

    if df.empty:
        raise ValueError("Data kosong setelah cleaning untuk retrain RL.")

    rows: list[TrainRow] = []
    tz = pytz.timezone("Asia/Jakarta")
    for _, row in df.tail(5000).iterrows():
        waktu = row["waktu"]
        if waktu.tzinfo is None:
            waktu = waktu.tz_localize("UTC").tz_convert(tz)
        else:
            waktu = waktu.tz_convert(tz)

        rows.append(
            TrainRow(
                daya=float(max(row["Daya"], 0.0)),
                suhu=float(np.clip(row["Suhu"], 10.0, 60.0)),
                arus=float(max(row["Arus"], 0.0)),
                tegangan=float(max(row["Tegangan"], 0.0)),
                jam=float(waktu.hour),
                day_of_week=float(waktu.weekday()),
                is_weekend=float(1 if waktu.weekday() >= 5 else 0),
                day_progress=float(waktu.day / max(calendar.monthrange(waktu.year, waktu.month)[1], 1)),
                days_remaining=float(max(calendar.monthrange(waktu.year, waktu.month)[1] - waktu.day, 0)),
            )
        )

    if len(rows) < 200:
        raise ValueError(f"Data untuk retrain RL terlalu sedikit ({len(rows)}), butuh >= 200 sampel.")

    return rows


def resolve_monthly_target() -> float:
    prefs = db.reference("user_preferences").get() or {}
    target = float(prefs.get("monthly_budget_rp", DEFAULT_MONTHLY_TARGET_RP) or DEFAULT_MONTHLY_TARGET_RP)

    now = datetime.now(pytz.timezone("Asia/Jakarta"))
    days = calendar.monthrange(now.year, now.month)[1]
    if days <= 0:
        return target
    return max(target, 10_000.0)


if __name__ == "__main__":
    print("Initialize Firebase for RL retraining...")
    initialize_firebase()

    print("Prepare RL dataset from Firebase history...")
    rows = load_training_rows()
    target = resolve_monthly_target()

    env = BudgetEnergyEnv(rows=rows, monthly_target_rp=target)
    model = PPO("MlpPolicy", env, verbose=1)

    print(f"Start PPO training | steps={TOTAL_TIMESTEPS} | samples={len(rows)}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUTPUT_PATH))

    print(f"Model RL saved at: {MODEL_OUTPUT_PATH}")
    print("Observation contains: prediksi_watt, target_bulanan_rp, gap_to_target.")
