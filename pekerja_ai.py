"""
pekerja_ai.py — Inference Worker untuk RL Budget Energy v2
===========================================================
CHANGELOG dari versi lama:
  1. _build_state_from_firebase() — urutan state disamakan dengan
     BudgetEnergyEnv._build_state() di rl_budget_v2.py (OBS_DIM=16)
     Urutan lama (AC, Magicom, WH, TV) → baru (AC, Magicom, WH, TV, Laptop)
  2. ACTION_LABELS — identik dengan rl_budget_v2.py (6 aksi)
  3. load_rl_model() — bercabang:
       .pt  → custom ActorNetwork (model v2)
       .zip → PPO stable-baselines3 (model lama, backward-compat)
  4. Pengambilan data Firebase: tambah node "Laptop" di samping
     "AC", "Magicom", "WaterHeater", "TV"
"""

import os
import calendar
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytz

# ── Firebase ──────────────────────────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

DB_URL                 = os.getenv("FIREBASE_DB_URL",
    "https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app")
PATH_JSON              = os.getenv("FIREBASE_CREDENTIALS",
    "/content/drive/MyDrive/SKRIPSHIT/firebase-project/firebase-project/serviceAccountKey.json")
RL_MODEL_PATH          = os.getenv("RL_MODEL_PATH", "model_ai/model_rl_budget_v2.pt")
DEFAULT_MONTHLY_BUDGET = float(os.getenv("DEFAULT_MONTHLY_TARGET_RP", "250000"))
TARIF_PER_KWH          = float(os.getenv("TARIF_PER_KWH", "1352.0"))
TZ                     = pytz.timezone("Asia/Jakarta")

# ── Koreksi kalibrasi — WAJIB sama dengan rl_budget_v2.py ────────────────────
CALIBRATION_CURRENT_CORRECTION = 1.018
CALIBRATION_VOLTAGE_CORRECTION = 1.000

# ── Konstanta OBS & ACTION — WAJIB sama dengan rl_budget_v2.py ───────────────
OBS_DIM   = 16
N_ACTIONS = 6
HIDDEN_DIM = 128

# ── Action labels — WAJIB identik dengan rl_budget_v2.py ─────────────────────
ACTION_LABELS = {
    0: "Tidak ada tindakan",
    1: "Matikan / kurangi AC",
    2: "Tunda / hemat Magicom",
    3: "Matikan / kurangi WaterHeater",
    4: "Matikan TV",
    5: "Mode hemat Laptop",
}
ACTION_REDUCTION = {
    0: 0.00, 1: 0.30, 2: 0.10, 3: 0.20, 4: 0.05, 5: 0.05,
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIREBASE
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_firebase():
    if not FIREBASE_AVAILABLE:
        raise ImportError("firebase_admin tidak terinstall.")
    cred_path = Path(PATH_JSON)
    if not cred_path.exists():
        raise FileNotFoundError(f"Kredensial tidak ada: {PATH_JSON}")
    cred = credentials.Certificate(str(cred_path))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})


def get_latest_sensor_data() -> dict:
    """
    Ambil data terbaru dari Firebase.
    Node yang dibaca: history (PZEM), AC, Magicom, WaterHeater, TV, Laptop,
                      user_preferences.
    """
    history_ref  = db.reference("history")
    prefs_ref    = db.reference("user_preferences")

    history_raw  = history_ref.order_by_key().limit_to_last(1).get()
    prefs        = prefs_ref.get() or {}

    if not history_raw:
        raise ValueError("Tidak ada data di node 'history'.")

    latest = list(history_raw.values())[0]
    return {
        "daya":     float(latest.get("Daya", 0) or 0),
        "suhu":     float(np.clip(latest.get("Suhu", 25) or 25, 10, 60)),
        "arus":     float(latest.get("Arus", 0) or 0),
        "tegangan": float(latest.get("Tegangan", 0) or 0),
        "waktu_ms": latest.get("waktu", 0),
        "monthly_budget": float(
            prefs.get("monthly_budget_rp", DEFAULT_MONTHLY_BUDGET)
            or DEFAULT_MONTHLY_BUDGET
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STATE BUILDER — urutan WAJIB identik dengan BudgetEnergyEnv._build_state()
# ═══════════════════════════════════════════════════════════════════════════════

def _device_flags(daya: float) -> Tuple[int, int, int, int, int]:
    """
    Threshold dari DATA_KALIBRASI PZEM-004T (V × I):
      AC          avg 361.6 W → daya > 300
      Magicom     avg ~157  W → 100 < daya <= 220
      WaterHeater avg 246.5 W → 220 < daya <= 300
      TV          avg  53.0 W →  35 < daya <= 100
      Laptop      avg  76.7 W →  60 < daya <= 220  (bisa overlap Magicom)

    Return order: (flag_ac, flag_magicom, flag_wh, flag_tv, flag_laptop)
    → Sesuai idx 2–6 di state vector.
    """
    return (
        1 if daya > 300            else 0,   # flag_ac
        1 if 100 < daya <= 220     else 0,   # flag_magicom
        1 if 220 < daya <= 300     else 0,   # flag_waterheater
        1 if  35 < daya <= 100     else 0,   # flag_tv
        1 if  60 < daya <= 220     else 0,   # flag_laptop
    )


def _prediksi_watt(daya: float, arus: float, tegangan: float) -> float:
    """P = V × I + koreksi kalibrasi. Fallback ke daya jika sensor nol."""
    if tegangan > 0 and arus > 0:
        return max(tegangan * CALIBRATION_VOLTAGE_CORRECTION
                   * arus   * CALIBRATION_CURRENT_CORRECTION, 0.0)
    return max(daya, 0.0)


def build_state_from_firebase(sensor: dict) -> np.ndarray:
    """
    Bangun state vector 16-dim dari data sensor Firebase.
    Urutan idx 0–15 WAJIB identik dengan BudgetEnergyEnv._build_state().

    ┌─────┬──────────────────┐  ┌─────┬──────────────────┐
    │  0  │ daya             │  │  8  │ day_of_week      │
    │  1  │ suhu             │  │  9  │ is_weekend       │
    │  2  │ flag_ac          │  │ 10  │ is_peak_hour     │
    │  3  │ flag_magicom     │  │ 11  │ day_progress     │
    │  4  │ flag_waterheater │  │ 12  │ days_remaining   │
    │  5  │ flag_tv          │  │ 13  │ prediksi_watt    │
    │  6  │ flag_laptop      │  │ 14  │ monthly_budget   │
    │  7  │ jam              │  │ 15  │ gap_to_target    │
    └─────┴──────────────────┘  └─────┴──────────────────┘
    """
    daya     = sensor["daya"]
    suhu     = sensor["suhu"]
    arus     = sensor["arus"]
    tegangan = sensor["tegangan"]
    budget   = sensor["monthly_budget"]

    # Waktu
    waktu_ms = sensor.get("waktu_ms", 0)
    if waktu_ms:
        dt = datetime.fromtimestamp(waktu_ms / 1000, tz=TZ)
    else:
        dt = datetime.now(TZ)

    jam         = float(dt.hour)
    dow         = float(dt.weekday())
    is_weekend  = float(1 if dt.weekday() >= 5 else 0)
    is_peak     = float(1 if 17 <= dt.hour <= 22 else 0)
    days_in_mo  = calendar.monthrange(dt.year, dt.month)[1]
    day_prog    = float(dt.day / days_in_mo)
    days_rem    = float(max(days_in_mo - dt.day, 0))

    # Flags & prediksi
    fa, fmc, fwh, ftv, flp = _device_flags(daya)
    pred_watt = _prediksi_watt(daya, arus, tegangan)
    gap = ((pred_watt / 1000.0) * 24 * TARIF_PER_KWH * 30) - budget

    return np.array([
        daya, suhu,           # 0–1
        fa, fmc, fwh, ftv, flp,  # 2–6  (AC, Magicom, WH, TV, Laptop)
        jam, dow, is_weekend, is_peak,  # 7–10
        day_prog, days_rem,   # 11–12
        pred_watt,            # 13
        budget,               # 14
        gap,                  # 15
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# LOADER — bercabang .pt (custom v2) vs .zip (PPO SB3 lama)
# ═══════════════════════════════════════════════════════════════════════════════

def load_rl_model(model_path: Optional[str] = None):
    """
    Loader bercabang:
      .pt  → custom ActorNetwork dari rl_budget_v2.py
      .zip → PPO dari stable_baselines3 (model lama, backward-compat)

    Return: (model, model_type)
      model_type: "custom_v2" | "ppo_sb3"
    """
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    path = str(model_path or RL_MODEL_PATH)

    # ── Custom ActorNetwork (v2) ───────────────────────────────────────────
    if path.endswith(".pt"):
        class _Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(OBS_DIM, HIDDEN_DIM), nn.Tanh(),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
                    nn.Linear(HIDDEN_DIM, N_ACTIONS),
                )
            def forward(self, x):
                return Categorical(logits=self.net(x))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor  = _Actor().to(device)
        ckpt   = torch.load(path, map_location=device)

        # Checkpoint bisa berupa {"actor": state_dict} atau langsung state_dict
        sd = ckpt.get("actor", ckpt)
        actor.load_state_dict(sd)
        actor.eval()
        print(f"[Loader] ✅ custom ActorNetwork v2 dimuat dari {path}")
        print(f"         OBS_DIM={OBS_DIM}, N_ACTIONS={N_ACTIONS}")
        return actor, "custom_v2"

    # ── PPO stable-baselines3 (model lama) ────────────────────────────────
    else:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(path)
            print(f"[Loader] ✅ PPO SB3 dimuat dari {path}")
            print("         Pastikan state vector cocok dengan model lama (OBS_DIM=15, model lama; state akan di-slice otomatis).")
            return model, "ppo_sb3"
        except ImportError:
            raise ImportError(
                "stable_baselines3 tidak terinstall. "
                "Install: pip install stable-baselines3"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_action(state: np.ndarray,
                   model=None,
                   model_type: str = "custom_v2") -> dict:
    """
    Inferensi dari state vector 16-dim.
    Return dict: action, label, probabilities, gap_before, prediksi_watt.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "custom_v2":
        with torch.no_grad():
            t     = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist  = model(t)
            probs = dist.probs.squeeze(0).cpu().numpy()
        action = int(probs.argmax())
    else:
        # PPO SB3 — state harus OBS_DIM=15 (model lama)
        # Jika memakai model lama, potong state[:16]
        action, _ = model.predict(state[:16], deterministic=True)
        action    = int(action)
        probs     = np.zeros(N_ACTIONS); probs[action] = 1.0

    return {
        "action":         action,
        "label":          ACTION_LABELS[action],
        "reduction_pct":  ACTION_REDUCTION[action] * 100,
        "probabilities":  {ACTION_LABELS[i]: round(float(p), 4)
                           for i, p in enumerate(probs)},
        "gap_to_target":  float(state[15]),     # idx 15 = gap_to_target
        "prediksi_watt":  float(state[13]),     # idx 13 = prediksi_watt
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — jalankan satu siklus inferensi
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference_cycle(model_path: Optional[str] = None) -> dict:
    """
    Satu siklus lengkap:
      1. Ambil data Firebase
      2. Bangun state vector
      3. Inferensi → aksi
      4. Tulis rekomendasi ke Firebase
    """
    sensor = get_latest_sensor_data()
    state  = build_state_from_firebase(sensor)
    model, mtype = load_rl_model(model_path)
    result = predict_action(state, model, mtype)

    # Tulis rekomendasi ke Firebase
    db.reference("rekomendasi_rl").set({
        "action":        result["action"],
        "label":         result["label"],
        "reduction_pct": result["reduction_pct"],
        "gap_to_target": result["gap_to_target"],
        "prediksi_watt": result["prediksi_watt"],
        "timestamp_ms":  sensor["waktu_ms"],
    })

    print(f"\n[Inference] Aksi    : {result['action']} — {result['label']}")
    print(f"[Inference] Reduksi : {result['reduction_pct']:.0f}%")
    print(f"[Inference] Gap     : Rp {result['gap_to_target']:,.0f}")
    print(f"[Inference] Pred W  : {result['prediksi_watt']:.1f} W")
    return result


if __name__ == "__main__":
    initialize_firebase()
    run_inference_cycle()    if not firebase_key_json:
        raise ValueError("❌ GAGAL: Kunci FIREBASE_KEY tidak ditemukan di environment/secrets!")

    key_dict = json.loads(firebase_key_json)
    cred = credentials.Certificate(key_dict)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})
    print("✅ Berhasil login ke Firebase melalui jalur aman!")


def load_rl_model() -> PPO | RLBudgetV2Agent | None:
    candidate_paths = [RL_MODEL_PATH]
    if RL_MODEL_PATH != LEGACY_RL_MODEL_PATH:
        candidate_paths.append(LEGACY_RL_MODEL_PATH)

    for model_path in candidate_paths:
        if not os.path.exists(model_path):
            continue
        try:
            if model_path.endswith(".pt"):
                print(f"✅ Memuat RL v2 PyTorch dari {model_path}")
                return RLBudgetV2Agent.load(model_path)
            print(f"✅ Memuat RL legacy Stable-Baselines3 dari {model_path}")
            return PPO.load(model_path)
        except Exception as exc:
            print(f"⚠️ Model RL di {model_path} gagal dimuat ({exc}). Mencoba kandidat lain/fallback.")

    print(
        f"⚠️ Model RL tidak ditemukan di {RL_MODEL_PATH}. "
        "Menggunakan policy hemat berbasis aturan."
    )
    return None


def fetch_latest_monitoring_state() -> Dict[str, float]:
    hist_data = db.reference("history").order_by_key().limit_to_last(1).get()
    state = {
        "Daya": 0.0,
        "Suhu": 25.0,
        "Arus": 0.0,
        "Tegangan": 0.0,
    }
    if hist_data:
        for _, val in hist_data.items():
            state["Daya"] = float(val.get("Daya", 0) or 0)
            state["Suhu"] = float(val.get("Suhu", 25.0) or 25.0)
            state["Arus"] = float(val.get("Arus", 0) or 0)
            state["Tegangan"] = float(val.get("Tegangan", 0) or 0)
    return state


def fetch_device_state() -> Dict[str, int]:
    log_data = db.reference("log_konfirmasi").order_by_key().limit_to_last(1).get()
    device_state = {"AC": 0, "Magicom": 0, "Waterheater": 0, "TV": 0, "Laptop": 0}
    if log_data:
        for _, val in log_data.items():
            device_state = {
                "AC": 1 if val.get("AC") else 0,
                "Magicom": 1 if val.get("Magicom") else 0,
                "Waterheater": 1 if val.get("Waterheater") else 0,
                "TV": 1 if val.get("TV") else 0,
                "Laptop": 1 if val.get("Laptop") else 0,
            }
    return device_state


def fetch_budget_context(now: datetime) -> Dict[str, float | int | str]:
    prefs = db.reference("user_preferences").get() or {}
    dashboard = db.reference("dashboard_info").get() or {}
    history = db.reference("rekap_harian/history").order_by_key().limit_to_last(30).get() or {}

    monthly_target_rp = float(prefs.get("monthly_budget_rp", DEFAULT_MONTHLY_TARGET_RP) or DEFAULT_MONTHLY_TARGET_RP)
    today_cost_rp = float(dashboard.get("biaya_hari_ini", 0) or 0)
    today_kwh = float(dashboard.get("kwh_hari_ini", 0) or 0)
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    day_of_month = now.day
    daily_target_rp = monthly_target_rp / days_in_month if days_in_month else monthly_target_rp

    historical_costs = []
    today_key = now.strftime("%Y-%m-%d")
    for key, value in history.items():
        if key == today_key:
            continue
        if isinstance(value, dict):
            cost = float(value.get("biaya_rp", 0) or 0)
            if cost > 0:
                historical_costs.append(cost)

    average_daily_cost_rp = sum(historical_costs) / len(historical_costs) if historical_costs else today_cost_rp
    projected_monthly_cost_rp = average_daily_cost_rp * days_in_month
    current_month_run_rate_rp = (today_cost_rp / max(day_of_month, 1)) * days_in_month if today_cost_rp > 0 else projected_monthly_cost_rp
    reference_daily_cost_rp = average_daily_cost_rp if average_daily_cost_rp > 0 else daily_target_rp
    daily_saving_rp = max(reference_daily_cost_rp - today_cost_rp, 0.0)
    daily_saving_pct = (daily_saving_rp / reference_daily_cost_rp * 100) if reference_daily_cost_rp > 0 else 0.0

    return {
        "monthly_target_rp": monthly_target_rp,
        "daily_target_rp": daily_target_rp,
        "today_cost_rp": today_cost_rp,
        "today_kwh": today_kwh,
        "average_daily_cost_rp": average_daily_cost_rp,
        "projected_monthly_cost_rp": projected_monthly_cost_rp,
        "current_month_run_rate_rp": current_month_run_rate_rp,
        "days_in_month": days_in_month,
        "day_of_month": day_of_month,
        "daily_saving_rp": daily_saving_rp,
        "daily_saving_pct": daily_saving_pct,
    }


def fetch_recent_daya_stats() -> Dict[str, float]:
    history_data = db.reference("history").order_by_key().limit_to_last(120).get() or {}
    daya_values: list[float] = []
    for _, value in history_data.items():
        try:
            daya = float((value or {}).get("Daya", 0) or 0)
        except Exception:
            continue
        if daya >= 0:
            daya_values.append(daya)

    if not daya_values:
        return {"median": 0.0, "p95": 0.0, "mean": 0.0}

    arr = np.array(daya_values, dtype=np.float32)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
    }


def sanitize_predicted_watt(raw_prediction: float, current_watt: float, stats: Dict[str, float]) -> tuple[float, str]:
    pred = float(raw_prediction)
    reason = "ok"

    if pred < 0:
        pred = 0.0
        reason = "clamped_negative"

    p95 = max(float(stats.get("p95", 0.0)), current_watt, 100.0)
    dynamic_upper = max(p95 * 1.4, current_watt * 2.5, 300.0)
    if pred > dynamic_upper:
        pred = dynamic_upper
        reason = "clamped_upper"

    delta = abs(pred - current_watt)
    tolerance = max(150.0, current_watt * 0.8)
    if delta > tolerance:
        pred = (0.7 * current_watt) + (0.3 * pred)
        reason = "smoothed_outlier"

    return float(max(pred, 0.0)), reason

def map_rl_action_to_text(action: int) -> str:
    if action == 1:
        return "⚠️ Matikan atau naikkan setpoint AC 1-2°C untuk menekan beban puncak."
    if action == 2:
        return "⚠️ Kurangi durasi Waterheater karena ini salah satu beban terbesar."
    if action == 3:
        return "💡 Matikan TV saat tidak ditonton untuk menjaga konsumsi tetap hemat."
    if action == 4:
        return "⚠️ Gunakan Magicom seperlunya atau pindahkan ke mode warm saat nasi sudah matang."
    if action == 5:
        return "💡 Aktifkan mode hemat daya Laptop atau cabut charger saat baterai sudah cukup."
    return "✅ Pola beban saat ini masih aman. Pertahankan kebiasaan hemat hari ini."


def build_budget_recommendation(
    predicted_watt: float,
    device_state: Dict[str, int],
    budget_context: Dict[str, float | int | str],
    rl_action: int | None,
) -> Dict[str, Any]:
    daily_target_rp = float(budget_context["daily_target_rp"])
    monthly_target_rp = float(budget_context["monthly_target_rp"])
    today_cost_rp = float(budget_context["today_cost_rp"])
    projected_monthly_cost_rp = float(budget_context["projected_monthly_cost_rp"])
    current_month_run_rate_rp = float(budget_context["current_month_run_rate_rp"])
    average_daily_cost_rp = float(budget_context["average_daily_cost_rp"])
    daily_saving_pct = float(budget_context["daily_saving_pct"])
    daily_saving_rp = float(budget_context["daily_saving_rp"])

    predicted_daily_cost_rp = (max(predicted_watt, 0.0) / 1000.0) * 24 * TARIF_PER_KWH
    projected_reference_rp = max(projected_monthly_cost_rp, current_month_run_rate_rp)
    over_budget_rp = max(projected_reference_rp - monthly_target_rp, 0.0)
    target_status = "on_track" if over_budget_rp <= 0 else "over_budget"

    active_devices = [name for name, status in device_state.items() if status == 1]
    top_device_hint = active_devices[0] if active_devices else "beban non-prioritas"

    if rl_action is not None:
        base_advice = map_rl_action_to_text(rl_action)
    else:
        base_advice = "✅ Gunakan perangkat seperlunya dan prioritaskan beban yang benar-benar diperlukan."

    if predicted_daily_cost_rp > daily_target_rp or over_budget_rp > 0:
        saving_goal_today_rp = max(today_cost_rp - daily_target_rp, 0.0)
        urgency = "tinggi" if predicted_daily_cost_rp > (daily_target_rp * 1.2) else "sedang"
        budget_advice = (
            f"Prediksi SARIMAX menunjukkan potensi biaya harian sekitar Rp {predicted_daily_cost_rp:,.0f}. "
            f"Agar target bulanan Rp {monthly_target_rp:,.0f} tercapai, usahakan hemat sekitar Rp {saving_goal_today_rp:,.0f} hari ini "
            f"dengan mengurangi pemakaian {top_device_hint}."
        )
    else:
        urgency = "rendah"
        budget_advice = (
            f"Prediksi SARIMAX masih berada dalam batas aman target harian Rp {daily_target_rp:,.0f}. "
            "Pertahankan pola pemakaian saat ini agar target bulanan tetap tercapai."
        )

    saving_badge = (
        f"🟢 Hemat {daily_saving_pct:.1f}% dibanding rata-rata harian biasa"
        if daily_saving_pct > 0
        else "🟡 Belum ada penghematan signifikan dibanding rata-rata harian"
    )

    return {
        "target_status": target_status,
        "urgency": urgency,
        "ppo_advice": base_advice,
        "budget_advice": budget_advice,
        "combined_advice": f"{base_advice} {budget_advice}",
        "predicted_daily_cost_rp": round(predicted_daily_cost_rp, 2),
        "over_budget_rp": round(over_budget_rp, 2),
        "saving_badge": saving_badge,
        "daily_saving_pct": round(daily_saving_pct, 2),
        "daily_saving_rp": round(daily_saving_rp, 2),
        "average_daily_cost_rp": round(average_daily_cost_rp, 2),
    }


initialize_firebase()
print(" Memuat otak SARIMAX dan RL...")

model_path = SARIMAX_MODEL_PATH if os.path.exists(SARIMAX_MODEL_PATH) else LEGACY_SARIMAX_MODEL_PATH
sarimax_artifact = load_sarimax_artifact(model_path)
model_rl = load_rl_model()

tz = pytz.timezone("Asia/Jakarta")
now = datetime.now(tz)
jam_skrg = now.hour

monitoring_state = fetch_latest_monitoring_state()
device_state = fetch_device_state()
budget_context = fetch_budget_context(now)

latest_features = {
    "Suhu": monitoring_state["Suhu"],
    "AC": device_state["AC"],
    "Magicom": device_state["Magicom"],
    "Waterheater": device_state["Waterheater"],
    "Laptop": device_state["Laptop"],
    "TV": device_state["TV"],
    "Arus": monitoring_state["Arus"],
    "Tegangan": monitoring_state["Tegangan"],
}

# Ambil list prediksi untuk beberapa step ke depan
raw_prediksi_watt_list = forecast_next_step(
    artifact=sarimax_artifact,
    latest_features=latest_features,
    current_watt=monitoring_state["Daya"],
    steps=6  # Contoh: Ambil 6 langkah ke depan
)

recent_daya_stats = fetch_recent_daya_stats()

# Lakukan sanitasi untuk SETIAP elemen di dalam list
prediksi_watt_list = []
prediksi_adjustment_reasons = []

for raw_val in raw_prediksi_watt_list:
    clean_val, reason = sanitize_predicted_watt(
        raw_prediction=raw_val,
        current_watt=monitoring_state["Daya"],
        stats=recent_daya_stats,
    )
    prediksi_watt_list.append(round(clean_val, 2))
    prediksi_adjustment_reasons.append(reason)

# Gunakan index ke-0 (prediksi step terdekat) sebagai acuan untuk RL dan Budgeting
prediksi_watt = prediksi_watt_list[0]
raw_prediksi_watt = raw_prediksi_watt_list[0]
prediksi_adjustment = prediksi_adjustment_reasons[0]


rl_action = None
if model_rl is not None:
    gap_to_target = float(budget_context["current_month_run_rate_rp"]) - float(budget_context["monthly_target_rp"])
    is_peak_hour = 1.0 if 17 <= jam_skrg <= 22 else 0.0
    day_of_week = float(now.weekday())
    is_weekend = float(1 if now.weekday() >= 5 else 0)
    day_progress = float(now.day / max(int(budget_context["days_in_month"]), 1))
    days_remaining = float(max(int(budget_context["days_in_month"]) - now.day, 0))

    state_v2 = build_state_from_values(
        daya=monitoring_state["Daya"],
        suhu=monitoring_state["Suhu"],
        device_state=device_state,
        jam=jam_skrg,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        day_progress=day_progress,
        days_remaining=days_remaining,
        prediksi_watt=prediksi_watt,
        monthly_target_rp=float(budget_context["monthly_target_rp"]),
        gap_to_target=gap_to_target,
    )

    state_v1_10d = np.array(
        [
            monitoring_state["Daya"],
            monitoring_state["Suhu"],
            device_state["AC"],
            device_state["Magicom"],
            device_state["Waterheater"],
            device_state["TV"],
            jam_skrg,
            prediksi_watt,
            float(budget_context["monthly_target_rp"]),
            gap_to_target,
        ],
        dtype=np.float32,
    )

    state_v1 = np.array(
        [
            monitoring_state["Daya"],
            monitoring_state["Suhu"],
            device_state["AC"],
            device_state["Magicom"],
            device_state["Waterheater"],
            device_state["TV"],
            jam_skrg,
        ],
        dtype=np.float32,
    )

    try:
        expected_obs_dim = int(model_rl.observation_space.shape[0])
    except Exception:
        expected_obs_dim = len(state_v2)

    try:
        if expected_obs_dim == RL_V2_OBS_DIM:
            state = state_v2
        elif expected_obs_dim >= len(state_v1_10d):
            state = state_v1_10d
        else:
            state = state_v1
            
        rl_action, _ = model_rl.predict(state, deterministic=True)
        rl_action = int(rl_action)
    except Exception as exc:
        print(f"⚠️ Prediksi RL gagal ({exc}); fallback ke policy rule-based.")
        rl_action = None

recommendation = build_budget_recommendation(
    predicted_watt=prediksi_watt,
    device_state=device_state,
    budget_context=budget_context,
    rl_action=rl_action,
)

print(
    f" Daya Aktual: {monitoring_state['Daya']} W | Prediksi SARIMAX (raw): {raw_prediksi_watt:.2f} W | "
    f"Prediksi dipakai: {prediksi_watt:.2f} W | Target Bulanan: Rp {float(budget_context['monthly_target_rp']):,.0f}"
)
print(f" Proyeksi Masa Depan (6 Steps): {prediksi_watt_list}")
print(f" Rekomendasi Hemat: {recommendation['combined_advice']}")
print(f" Indikator hemat hari ini: {recommendation['saving_badge']}")

db.reference("Hasil_AI").set(
    {
        "prediksi_daya_selanjutnya": round(float(prediksi_watt), 2),
        "prediksi_masa_depan": prediksi_watt_list,
        "prediksi_daya_selanjutnya_raw": round(float(raw_prediksi_watt), 2),
        "prediksi_adjustment": prediksi_adjustment,
        "rekomendasi_rl": recommendation["combined_advice"],
        "rekomendasi_ppo": recommendation["ppo_advice"],
        "rekomendasi_budget": recommendation["budget_advice"],
        "target_bulanan_rp": round(float(budget_context["monthly_target_rp"]), 2),
        "target_harian_rp": round(float(budget_context["daily_target_rp"]), 2),
        "biaya_hari_ini_rp": round(float(budget_context["today_cost_rp"]), 2),
        "prediksi_biaya_harian_rp": recommendation["predicted_daily_cost_rp"],
        "proyeksi_bulanan_rp": round(float(budget_context["current_month_run_rate_rp"]), 2),
        "proyeksi_histori_bulanan_rp": round(float(budget_context["projected_monthly_cost_rp"]), 2),
        "selisih_target_bulanan_rp": recommendation["over_budget_rp"],
        "persen_hemat_hari_ini": recommendation["daily_saving_pct"],
        "nominal_hemat_hari_ini_rp": recommendation["daily_saving_rp"],
        "rata_rata_biaya_harian_rp": recommendation["average_daily_cost_rp"],
        "indikator_hemat": recommendation["saving_badge"],
        "status_target": recommendation["target_status"],
        "urgency": recommendation["urgency"],
        "rl_action": rl_action,
        "sarimax_model_type": sarimax_artifact.get("model_type"),
        "sarimax_model_path": model_path,
        "waktu_update": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
)

print("✅ Laporan AI hemat berbasis SARIMAX + target budget berhasil dikirim ke Firebase.")
