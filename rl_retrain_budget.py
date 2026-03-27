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
from stable_baselines3 import PPO
from gymnasium import Env, spaces

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


class BudgetEnergyEnv(Env):
    """RL env dengan state yang memuat prediksi_watt, target_bulanan_rp, dan gap_to_target."""

    metadata = {"render_modes": []}

    def __init__(self, rows: list[TrainRow], monthly_target_rp: float):
        super().__init__()
        self.rows = rows
        self.monthly_target_rp = monthly_target_rp
        self.i = 0
        self.projected_month_cost = monthly_target_rp

        # action: 0 aman, 1 AC, 2 Magicom, 3 Waterheater, 4 TV
        self.action_space = spaces.Discrete(5)
        # state: [daya, suhu, ac, magicom, wh, tv, jam, prediksi_watt, target_bulanan_rp, gap_to_target]
        high = np.array([5000, 60, 1, 1, 1, 1, 23, 5000, 2_000_000, 2_000_000], dtype=np.float32)
        low = np.array([0, 10, 0, 0, 0, 0, 0, 0, 10_000, -2_000_000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _device_flags(self, daya: float) -> tuple[int, int, int, int]:
        # proxy sederhana untuk simulasi status device
        ac = 1 if daya > 350 else 0
        wh = 1 if daya > 600 else 0
        magicom = 1 if 120 < daya < 450 else 0
        tv = 1 if 40 < daya < 220 else 0
        return ac, magicom, wh, tv

    def _build_state(self, row: TrainRow) -> np.ndarray:
        ac, magicom, wh, tv = self._device_flags(row.daya)
        prediksi_watt = float((0.6 * row.daya) + (0.25 * row.arus * row.tegangan) + (2.0 * row.suhu))
        prediksi_watt = max(prediksi_watt, 0.0)

        est_daily_cost = (prediksi_watt / 1000.0) * 24 * TARIF_PER_KWH
        days_in_month = 30
        self.projected_month_cost = est_daily_cost * days_in_month
        gap_to_target = self.projected_month_cost - self.monthly_target_rp

        return np.array(
            [
                row.daya,
                row.suhu,
                ac,
                magicom,
                wh,
                tv,
                row.jam,
                prediksi_watt,
                self.monthly_target_rp,
                gap_to_target,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        return self._build_state(self.rows[self.i]), {}

    def step(self, action: int):
        row = self.rows[self.i]
        state = self._build_state(row)
        prediksi_watt = float(state[7])
        gap_to_target = float(state[9])

        reduction_factor = {0: 0.00, 1: 0.12, 2: 0.07, 3: 0.18, 4: 0.05}[int(action)]
        expected_after_action = prediksi_watt * (1.0 - reduction_factor)
        est_daily_cost_after = (expected_after_action / 1000.0) * 24 * TARIF_PER_KWH
        est_month_after = est_daily_cost_after * 30

        # reward: kurangi gap, tapi penalti aksi agresif saat tidak dibutuhkan
        gap_after = est_month_after - self.monthly_target_rp
        reward = -abs(gap_after) / max(self.monthly_target_rp, 1.0)

        if gap_to_target <= 0 and action != 0:
            reward -= 0.2
        if gap_to_target > 0 and action == 0:
            reward -= 0.2

        self.i += 1
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

    for col in ["Daya", "Suhu", "Arus", "Tegangan", "waktu"]:
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


def main() -> None:
<<<<<<< ours
    print("ðŸ”Œ Inisialisasi Firebase untuk retrain RL...")
    initialize_firebase()

    print("ðŸ“¥ Menyiapkan dataset RL dari history Firebase...")
=======
    print("í´Œ Inisialisasi Firebase untuk retrain RL...")
    initialize_firebase()

    print("í³¥ Menyiapkan dataset RL dari history Firebase...")
>>>>>>> theirs
    rows = load_training_rows()
    target = resolve_monthly_target()

    env = BudgetEnergyEnv(rows=rows, monthly_target_rp=target)
    model = PPO("MlpPolicy", env, verbose=1)

<<<<<<< ours
    print(f"ðŸ¤– Training PPO dimulai | steps={TOTAL_TIMESTEPS} | samples={len(rows)}")
=======
    print(f"í´– Training PPO dimulai | steps={TOTAL_TIMESTEPS} | samples={len(rows)}")
>>>>>>> theirs
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUTPUT_PATH))

    print(f"âœ… Model RL baru tersimpan di: {MODEL_OUTPUT_PATH}")
    print("State model sudah memuat: prediksi_watt, target_bulanan_rp, gap_to_target.")


if __name__ == "__main__":
    main()
