
import os
import json
import calendar
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pytz

# ── Firebase ──────────────────────────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

DB_URL               = os.getenv("FIREBASE_DB_URL", "https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app")
PATH_JSON            = os.getenv("FIREBASE_CREDENTIALS", "/content/drive/MyDrive/SKRIPSHIT/firebase-project/firebase-project/serviceAccountKey.json")
FIREBASE_KEY_JSON    = os.getenv("FIREBASE_KEY")

RL_MODEL_PATH        = os.getenv("RL_MODEL_PATH", "model_ai/model_rl_budget_v2.pt")
LEGACY_RL_MODEL_PATH = os.getenv("LEGACY_RL_MODEL_PATH", "model_ai/model_rl_budget.zip")
SARIMAX_MODEL_PATH   = os.getenv("SARIMAX_MODEL_PATH", "model_ai/sarimax_bundle.pkl")
LEGACY_SARIMAX_MODEL_PATH = os.getenv("LEGACY_SARIMAX_MODEL_PATH", "model_ai/legacy_sarimax.pkl")

DEFAULT_MONTHLY_BUDGET = float(os.getenv("DEFAULT_MONTHLY_TARGET_RP", "250000"))
TARIF_PER_KWH          = float(os.getenv("TARIF_PER_KWH", "1352.0"))
TZ                     = pytz.timezone("Asia/Jakarta")

# ── Kalibrasi — WAJIB sama dengan rl_budget_v2.py ────────────────────────────
CALIBRATION_CURRENT_CORRECTION = 1.018
CALIBRATION_VOLTAGE_CORRECTION = 1.000

# ── Konstanta OBS & ACTION — WAJIB sama dengan rl_budget_v2.py ───────────────
OBS_DIM    = 16
N_ACTIONS  = 6
HIDDEN_DIM = 128

# ── Normalisasi — WAJIB sama dengan BudgetEnergyEnv di rl_budget_v2.py ───────
_DAYA_MAX = 600.0
_SUHU_MAX = 60.0
_JAM_MAX  = 23.0
_DOW_MAX  = 6.0
_DAYS_MAX = 31.0
_WATT_MAX = 600.0

#  FIX Bug 5: definisikan DEVICE secara eksplisit
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ── SARIMAX mock fallback ─────────────────────────────────────────────────────
if "load_sarimax_artifact" not in globals():
    def load_sarimax_artifact(model_path):
        print("  load_sarimax_artifact tidak ditemukan, menggunakan mock.")
        return {"model_type": "mock_sarimax"}

if "forecast_next_step" not in globals():
    def forecast_next_step(artifact, latest_features, current_watt, steps):
        print("  forecast_next_step tidak ditemukan, menggunakan mock statis.")
        return [current_watt] * steps


# ═══════════════════════════════════════════════════════════════════════════════
# FIREBASE INIT & FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_firebase():
    if not FIREBASE_AVAILABLE:
        raise ImportError("firebase_admin tidak terinstall.")
    if firebase_admin._apps:
        return
    if FIREBASE_KEY_JSON:
        key_dict = json.loads(FIREBASE_KEY_JSON)
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})
        print(" Login Firebase via FIREBASE_KEY secret.")
    else:
        cred_path = Path(PATH_JSON)
        if not cred_path.exists():
            raise ValueError("❌ FIREBASE_KEY tidak ada dan file JSON lokal tidak ditemukan.")
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})
        print(" Login Firebase via file JSON lokal.")


def fetch_latest_monitoring_state() -> Dict[str, float]:
    hist_data = db.reference("history").order_by_key().limit_to_last(1).get()
    state = {"Daya": 0.0, "Suhu": 25.0, "Arus": 0.0, "Tegangan": 0.0, "waktu_ms": 0}
    if hist_data:
        for _, val in hist_data.items():
            state["Daya"]     = float(val.get("Daya", 0) or 0)
            state["Suhu"]     = float(val.get("Suhu", 25.0) or 25.0)
            state["Arus"]     = float(val.get("Arus", 0) or 0)
            state["Tegangan"] = float(val.get("Tegangan", 0) or 0)
            state["waktu_ms"] = val.get("waktu", 0)
    return state


def fetch_device_state() -> Dict[str, int]:
    """Ambil status perangkat dari log_konfirmasi (1=nyala, 0=mati, -1=tidak ada data)."""
    log_data = db.reference("log_konfirmasi").order_by_key().limit_to_last(1).get()
    # Default -1 = tidak diketahui (sama dengan env training saat conf_ kosong)
    device_state = {"AC": -1, "Magicom": -1, "Waterheater": -1, "TV": -1, "Laptop": -1}
    if log_data:
        for _, val in log_data.items():
            device_state = {
                "AC":          1 if val.get("AC")          else 0,
                "Magicom":     1 if val.get("Magicom")     else 0,
                "Waterheater": 1 if val.get("Waterheater") else 0,
                "TV":          1 if val.get("TV")          else 0,
                "Laptop":      1 if val.get("Laptop")      else 0,
            }
    return device_state


def fetch_budget_context(now: datetime) -> Dict[str, Any]:
    prefs     = db.reference("user_preferences").get() or {}
    dashboard = db.reference("dashboard_info").get() or {}
    history   = db.reference("rekap_harian/history").order_by_key().limit_to_last(30).get() or {}

    monthly_target_rp = float(prefs.get("monthly_budget_rp", DEFAULT_MONTHLY_BUDGET) or DEFAULT_MONTHLY_BUDGET)
    today_cost_rp     = float(dashboard.get("biaya_hari_ini", 0) or 0)
    today_kwh         = float(dashboard.get("kwh_hari_ini", 0) or 0)
    days_in_month     = calendar.monthrange(now.year, now.month)[1]
    day_of_month      = now.day
    daily_target_rp   = monthly_target_rp / days_in_month if days_in_month else monthly_target_rp

    historical_costs = []
    today_key = now.strftime("%Y-%m-%d")
    for key, value in history.items():
        if key == today_key:
            continue
        if isinstance(value, dict):
            cost = float(value.get("biaya_rp", 0) or 0)
            if cost > 0:
                historical_costs.append(cost)

    average_daily_cost_rp       = sum(historical_costs) / len(historical_costs) if historical_costs else today_cost_rp
    projected_monthly_cost_rp   = average_daily_cost_rp * days_in_month
    current_month_run_rate_rp   = (today_cost_rp / max(day_of_month, 1)) * days_in_month if today_cost_rp > 0 else projected_monthly_cost_rp
    reference_daily_cost_rp     = average_daily_cost_rp if average_daily_cost_rp > 0 else daily_target_rp
    daily_saving_rp             = max(reference_daily_cost_rp - today_cost_rp, 0.0)
    daily_saving_pct            = (daily_saving_rp / reference_daily_cost_rp * 100) if reference_daily_cost_rp > 0 else 0.0

    return {
        "monthly_target_rp":           monthly_target_rp,
        "daily_target_rp":             daily_target_rp,
        "today_cost_rp":               today_cost_rp,
        "today_kwh":                   today_kwh,
        "average_daily_cost_rp":       average_daily_cost_rp,
        "projected_monthly_cost_rp":   projected_monthly_cost_rp,
        "current_month_run_rate_rp":   current_month_run_rate_rp,
        "days_in_month":               days_in_month,
        "day_of_month":                day_of_month,
        "daily_saving_rp":             daily_saving_rp,
        "daily_saving_pct":            daily_saving_pct,
    }


def fetch_recent_daya_stats() -> Dict[str, float]:
    history_data = db.reference("history").order_by_key().limit_to_last(120).get() or {}
    daya_values = []
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
        "p95":    float(np.percentile(arr, 95)),
        "mean":   float(np.mean(arr)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE FLAGS & STATE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _device_flags(daya: float) -> Tuple[int, int, int, int, int]:
 
    fa  = 1 if daya >= 316              else 0   # AC kompresor aktif
    fmc = 1 if 280 <= daya <= 315       else 0   # Magicom memasak
    fwh = 1 if 210 <= daya <  280       else 0   # Water Heater
    ftv = 1 if  26 <= daya <= 60        else 0   # TV aktif
    flp = 1 if  12 <= daya <= 48        else 0   # Laptop charging
    return fa, fmc, fwh, ftv, flp


def _prediksi_watt(daya: float, arus: float, tegangan: float) -> float:
    if tegangan > 0 and arus > 0:
        return max(tegangan * CALIBRATION_VOLTAGE_CORRECTION * arus * CALIBRATION_CURRENT_CORRECTION, 0.0)
    return max(daya, 0.0)


def build_state_normalized(
    *,
    daya: float,
    suhu: float,
    arus: float,
    tegangan: float,
    device_state: Dict[str, int],
    now: datetime,
    prediksi_watt: float,
    monthly_target_rp: float,
    budget_terpakai_rp: float,
) -> np.ndarray:
    # Flags: prioritaskan log_konfirmasi, fallback ke threshold daya
    if all(v != -1 for v in device_state.values()):
        fa  = int(bool(device_state.get("AC", 0)))
        fmc = int(bool(device_state.get("Magicom", 0)))
        fwh = int(bool(device_state.get("Waterheater", 0)))
        ftv = int(bool(device_state.get("TV", 0)))
        flp = int(bool(device_state.get("Laptop", 0)))
    else:
        fa, fmc, fwh, ftv, flp = _device_flags(daya)

    # Waktu
    jam        = float(now.hour)
    dow        = float(now.weekday())
    is_weekend = float(1 if now.weekday() >= 5 else 0)
    is_peak    = float(1 if 17 <= now.hour <= 22 else 0)
    days_in_mo = calendar.monthrange(now.year, now.month)[1]
    day_prog   = float(now.day / days_in_mo)
    days_rem   = float(max(days_in_mo - now.day, 0))

    # Normalisasi
    daya_n  = float(np.clip(daya,          0, _DAYA_MAX) / _DAYA_MAX)
    suhu_n  = float(np.clip(suhu,          0, _SUHU_MAX) / _SUHU_MAX)
    jam_n   = float(jam / _JAM_MAX)
    dow_n   = float(dow / _DOW_MAX)
    days_n  = float(np.clip(days_rem,      0, _DAYS_MAX) / _DAYS_MAX)
    pw_n    = float(np.clip(prediksi_watt, 0, _WATT_MAX) / _WATT_MAX)

    # budget_n dinamis: ratio pengeluaran berjalan vs target
    budget_n = float(np.clip(budget_terpakai_rp / max(monthly_target_rp, 1.0), 0.0, 2.0))

    # gap_n: estimasi biaya bulanan dari watt saat ini vs target
    gap_rp = (prediksi_watt / 1000.0) * 24 * TARIF_PER_KWH * 30 - monthly_target_rp
    gap_n  = float(np.clip(gap_rp / max(monthly_target_rp, 1.0), -1.0, 1.0))

    return np.array([
        daya_n, suhu_n,
        fa, fmc, fwh, ftv, flp,
        jam_n, dow_n, is_weekend, is_peak,
        day_prog, days_n, pw_n,
        budget_n, gap_n,
    ], dtype=np.float32)


def sanitize_predicted_watt(
    raw_prediction: float, current_watt: float, stats: Dict[str, float]
) -> Tuple[float, str]:
    pred   = float(raw_prediction)
    reason = "ok"

    if pred < 0:
        pred   = 0.0
        reason = "clamped_negative"

    p95           = max(float(stats.get("p95", 0.0)), current_watt, 100.0)
    dynamic_upper = max(p95 * 1.4, current_watt * 2.5, 300.0)
    if pred > dynamic_upper:
        pred   = dynamic_upper
        reason = "clamped_upper"

    delta     = abs(pred - current_watt)
    tolerance = max(150.0, current_watt * 0.8)
    if delta > tolerance:
        pred   = (0.7 * current_watt) + (0.3 * pred)
        reason = "smoothed_outlier"

    return float(max(pred, 0.0)), reason


# ═══════════════════════════════════════════════════════════════════════════════
# LOADER MODEL RL
# ═══════════════════════════════════════════════════════════════════════════════

def load_rl_model():
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

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

    # Coba model v2 (.pt) dulu
    if os.path.exists(RL_MODEL_PATH) and RL_MODEL_PATH.endswith(".pt"):
        try:
            actor = _Actor().to(DEVICE)
            ckpt  = torch.load(RL_MODEL_PATH, map_location=DEVICE)
            actor.load_state_dict(ckpt.get("actor", ckpt))
            actor.eval()
            print(f" Model RL v2 (custom ActorNetwork) dimuat dari {RL_MODEL_PATH}")
            return actor, "custom_v2"
        except Exception as e:
            print(f"  Gagal muat {RL_MODEL_PATH}: {e}")

    # Fallback ke legacy SB3
    if os.path.exists(LEGACY_RL_MODEL_PATH):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(LEGACY_RL_MODEL_PATH)
            print(f" Model RL legacy SB3 dimuat dari {LEGACY_RL_MODEL_PATH}")
            return model, "ppo_sb3"
        except Exception as e:
            print(f"  Gagal muat legacy {LEGACY_RL_MODEL_PATH}: {e}")

    print("  Tidak ada model RL. Akan pakai rule-based fallback.")
    return None, "none"


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_action(state: np.ndarray, model, model_type: str) -> dict:
    if model_type == "custom_v2":
        with torch.no_grad():
            t     = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # ✅ FIX Bug 5
            dist  = model(t)
            probs = dist.probs.squeeze(0).cpu().numpy()
        action = int(probs.argmax())
    elif model_type == "ppo_sb3":
        action, _ = model.predict(state, deterministic=True)
        action    = int(action)
        probs     = np.zeros(N_ACTIONS); probs[action] = 1.0
    else:
        # Rule-based fallback: pilih aksi berdasarkan gap_n (index 15)
        gap_n  = float(state[15])
        action = 1 if gap_n > 0.3 else 0
        probs  = np.zeros(N_ACTIONS); probs[action] = 1.0

    return {
        "action":        action,
        "label":         ACTION_LABELS[action],
        "reduction_pct": ACTION_REDUCTION[action] * 100,
        "probabilities": {ACTION_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "gap_n":         float(state[15]),
        "budget_n":      float(state[14]),
        "prediksi_watt": float(state[13]) * _WATT_MAX,  # denormalisasi untuk display
    }


def map_rl_action_to_text(action: int) -> str:
    """ FIX Bug 4: mapping aksi 2,3,4 diperbaiki sesuai ACTION_LABELS."""
    if action == 1:
        return "⚠️ Matikan atau naikkan setpoint AC 1–2°C untuk menekan beban puncak."
    if action == 2:
        return "⚠️ Gunakan Magicom seperlunya atau pindahkan ke mode warm saat nasi sudah matang."
    if action == 3:
        return "⚠️ Kurangi durasi Water Heater karena ini salah satu beban terbesar."
    if action == 4:
        return "💡 Matikan TV saat tidak ditonton untuk menjaga konsumsi tetap hemat."
    if action == 5:
        return "💡 Aktifkan mode hemat daya Laptop atau cabut charger saat baterai sudah cukup."
    return " Pola beban saat ini masih aman. Pertahankan kebiasaan hemat hari ini."


def build_budget_recommendation(
    predicted_watt: float,
    device_state: Dict[str, int],
    budget_context: Dict[str, Any],
    rl_action: Optional[int],
) -> Dict[str, Any]:
    daily_target_rp             = float(budget_context["daily_target_rp"])
    monthly_target_rp           = float(budget_context["monthly_target_rp"])
    today_cost_rp               = float(budget_context["today_cost_rp"])
    projected_monthly_cost_rp   = float(budget_context["projected_monthly_cost_rp"])
    current_month_run_rate_rp   = float(budget_context["current_month_run_rate_rp"])
    average_daily_cost_rp       = float(budget_context["average_daily_cost_rp"])
    daily_saving_pct            = float(budget_context["daily_saving_pct"])
    daily_saving_rp             = float(budget_context["daily_saving_rp"])

    predicted_daily_cost_rp  = (max(predicted_watt, 0.0) / 1000.0) * 24 * TARIF_PER_KWH
    projected_reference_rp   = max(projected_monthly_cost_rp, current_month_run_rate_rp)
    over_budget_rp           = max(projected_reference_rp - monthly_target_rp, 0.0)
    target_status            = "on_track" if over_budget_rp <= 0 else "over_budget"

    active_devices   = [name for name, status in device_state.items() if status == 1]
    top_device_hint  = active_devices[0] if active_devices else "beban non-prioritas"

    base_advice = map_rl_action_to_text(rl_action) if rl_action is not None else \
        "✅ Gunakan perangkat seperlunya dan prioritaskan beban yang benar-benar diperlukan."

    if predicted_daily_cost_rp > daily_target_rp or over_budget_rp > 0:
        saving_goal_today_rp = max(today_cost_rp - daily_target_rp, 0.0)
        urgency = "tinggi" if predicted_daily_cost_rp > (daily_target_rp * 1.2) else "sedang"
        budget_advice = (
            f"Prediksi SARIMAX menunjukkan potensi biaya harian sekitar Rp {predicted_daily_cost_rp:,.0f}. "
            f"Agar target bulanan Rp {monthly_target_rp:,.0f} tercapai, usahakan hemat sekitar "
            f"Rp {saving_goal_today_rp:,.0f} hari ini dengan mengurangi pemakaian {top_device_hint}."
        )
    else:
        urgency = "rendah"
        budget_advice = (
            f"Prediksi SARIMAX masih berada dalam batas aman target harian Rp {daily_target_rp:,.0f}. "
            "Pertahankan pola pemakaian saat ini agar target bulanan tetap tercapai."
        )

    saving_badge = (
        f" Hemat {daily_saving_pct:.1f}% dibanding rata-rata harian biasa"
        if daily_saving_pct > 0
        else " Belum ada penghematan signifikan dibanding rata-rata harian"
    )

    return {
        "target_status":            target_status,
        "urgency":                  urgency,
        "ppo_advice":               base_advice,
        "budget_advice":            budget_advice,
        "combined_advice":          f"{base_advice} {budget_advice}",
        "predicted_daily_cost_rp":  round(predicted_daily_cost_rp, 2),
        "over_budget_rp":           round(over_budget_rp, 2),
        "saving_badge":             saving_badge,
        "daily_saving_pct":         round(daily_saving_pct, 2),
        "daily_saving_rp":          round(daily_saving_rp, 2),
        "average_daily_cost_rp":    round(average_daily_cost_rp, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  WORKER INFERENSI: SARIMAX + RL BUDGET ENERGY v2            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # 1. Firebase
    initialize_firebase()

    # 2. Load SARIMAX
    print(" Memuat model SARIMAX...")
    sarimax_path     = SARIMAX_MODEL_PATH if os.path.exists(SARIMAX_MODEL_PATH) else LEGACY_SARIMAX_MODEL_PATH
    sarimax_artifact = load_sarimax_artifact(sarimax_path)
    print(f"    SARIMAX dimuat: {sarimax_path} (type={sarimax_artifact.get('model_type')})")

    # 3. Load RL
    print(" Memuat model RL...")
    model_rl, model_type = load_rl_model()

    # 4. Ambil data Firebase
    now              = datetime.now(TZ)
    monitoring_state = fetch_latest_monitoring_state()
    device_state     = fetch_device_state()
    budget_context   = fetch_budget_context(now)
    recent_stats     = fetch_recent_daya_stats()

    daya     = monitoring_state["Daya"]
    suhu     = monitoring_state["Suhu"]
    arus     = monitoring_state["Arus"]
    tegangan = monitoring_state["Tegangan"]

    print(f"\n Data sensor: Daya={daya}W  Suhu={suhu}°C  Arus={arus}A  Tegangan={tegangan}V")
    print(f" Device state: {device_state}")

    # 5. Forecast SARIMAX
    latest_features = {
        "Suhu":        suhu,
        "AC":          max(device_state.get("AC", 0), 0),
        "Magicom":     max(device_state.get("Magicom", 0), 0),
        "Waterheater": max(device_state.get("Waterheater", 0), 0),
        "Laptop":      max(device_state.get("Laptop", 0), 0),
        "TV":          max(device_state.get("TV", 0), 0),
        "Arus":        arus,
        "Tegangan":    tegangan,
    }

    raw_pred_list = forecast_next_step(
        artifact=sarimax_artifact,
        latest_features=latest_features,
        current_watt=daya,
        steps=6,
    )

    pred_list, reason_list = [], []
    for raw_val in raw_pred_list:
        clean_val, reason = sanitize_predicted_watt(raw_val, daya, recent_stats)
        pred_list.append(round(clean_val, 2))
        reason_list.append(reason)

    prediksi_watt = pred_list[0]
    print(f"📈 Prediksi SARIMAX: {pred_list} (sanitasi: {reason_list[0]})")

    # 6. Bangun state ternormalisasi & jalankan RL
    # Estimasi biaya berjalan bulan ini (untuk budget_n)
    budget_terpakai_rp = float(budget_context.get("current_month_run_rate_rp", 0))

    state = build_state_normalized(
        daya=daya,
        suhu=suhu,
        arus=arus,
        tegangan=tegangan,
        device_state=device_state,
        now=now,
        prediksi_watt=prediksi_watt,
        monthly_target_rp=float(budget_context["monthly_target_rp"]),
        budget_terpakai_rp=budget_terpakai_rp,
    )

    assert len(state) == OBS_DIM, f"State dim salah: {len(state)} != {OBS_DIM}"

    rl_result = predict_action(state, model_rl, model_type)
    rl_action = rl_result["action"]

    print(f"\n RL action: {rl_action} — {rl_result['label']}  (reduksi {rl_result['reduction_pct']:.0f}%)")
    print(f"   gap_n={rl_result['gap_n']:.3f}  budget_n={rl_result['budget_n']:.3f}  pred_watt={rl_result['prediksi_watt']:.1f}W")

    # 7. Rekomendasi teks
    recommendation = build_budget_recommendation(
        predicted_watt=prediksi_watt,
        device_state=device_state,
        budget_context=budget_context,
        rl_action=rl_action,
    )

    print(f" Rekomendasi: {recommendation['combined_advice']}")
    print(f"  Indikator: {recommendation['saving_badge']}")

    # 8. Push ke Firebase
    db.reference("Hasil_AI").set({
        "prediksi_daya_selanjutnya":     round(float(prediksi_watt), 2),
        "prediksi_masa_depan":           pred_list,
        "prediksi_daya_raw":             round(float(raw_pred_list[0]), 2),
        "prediksi_adjustment":           reason_list[0],
        "rekomendasi_rl":                recommendation["combined_advice"],
        "rekomendasi_ppo":               recommendation["ppo_advice"],
        "rekomendasi_budget":            recommendation["budget_advice"],
        "rl_action":                     rl_action,
        "rl_action_label":               rl_result["label"],
        "rl_probabilities":              rl_result["probabilities"],
        "rl_gap_n":                      round(rl_result["gap_n"], 4),
        "rl_budget_n":                   round(rl_result["budget_n"], 4),
        "target_bulanan_rp":             round(float(budget_context["monthly_target_rp"]), 2),
        "target_harian_rp":              round(float(budget_context["daily_target_rp"]), 2),
        "biaya_hari_ini_rp":             round(float(budget_context["today_cost_rp"]), 2),
        "prediksi_biaya_harian_rp":      recommendation["predicted_daily_cost_rp"],
        "proyeksi_bulanan_rp":           round(float(budget_context["current_month_run_rate_rp"]), 2),
        "proyeksi_histori_bulanan_rp":   round(float(budget_context["projected_monthly_cost_rp"]), 2),
        "selisih_target_bulanan_rp":     recommendation["over_budget_rp"],
        "persen_hemat_hari_ini":         recommendation["daily_saving_pct"],
        "nominal_hemat_hari_ini_rp":     recommendation["daily_saving_rp"],
        "rata_rata_biaya_harian_rp":     recommendation["average_daily_cost_rp"],
        "indikator_hemat":               recommendation["saving_badge"],
        "status_target":                 recommendation["target_status"],
        "urgency":                       recommendation["urgency"],
        "sarimax_model_type":            sarimax_artifact.get("model_type", "unknown"),
        "sarimax_model_path":            sarimax_path,
        "waktu_update":                  now.strftime("%Y-%m-%d %H:%M:%S"),
    })

    print("\n Laporan AI berhasil dikirim ke Firebase node 'Hasil_AI'.")


if __name__ == "__main__":
    main()
