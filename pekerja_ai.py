import calendar
import json
import os
from datetime import datetime
from typing import Any, Dict

import firebase_admin
import numpy as np
import pytz
from firebase_admin import credentials, db
from stable_baselines3 import PPO

from sarimax_utils import forecast_next_step, load_sarimax_artifact

print("🤖 Pekerja AI successful...")

DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app",
)
SARIMAX_MODEL_PATH = os.getenv("SARIMAX_MODEL_PATH", "model_ai/sarimax_bundle.pkl")
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "model_ai/model_rl_budget.zip")
LEGACY_SARIMAX_MODEL_PATH = "model_ai/model_sarimax_kairo (1).pkl"
TARIF_PER_KWH = float(os.getenv("TARIF_PER_KWH", "1352.0"))
DEFAULT_MONTHLY_TARGET_RP = float(os.getenv("DEFAULT_MONTHLY_TARGET_RP", "250000"))


def initialize_firebase() -> None:
    firebase_key_json = os.getenv("FIREBASE_KEY")
    if not firebase_key_json:
        raise ValueError("❌ GAGAL: Kunci FIREBASE_KEY tidak ditemukan di environment/secrets!")

    key_dict = json.loads(firebase_key_json)
    cred = credentials.Certificate(key_dict)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})
    print("✅ Berhasil login ke Firebase melalui jalur aman!")


def load_rl_model() -> PPO | None:
    if not os.path.exists(RL_MODEL_PATH):
        print(f"⚠️ Model RL tidak ditemukan di {RL_MODEL_PATH}. Menggunakan policy hemat berbasis aturan.")
        return None
    return PPO.load(RL_MODEL_PATH)


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


def map_rl_action_to_text(action: int) -> str:
    if action == 1:
        return "⚠️ Matikan atau naikkan setpoint AC 1-2°C untuk menekan beban puncak."
    if action == 2:
        return "⚠️ Tunda pemakaian Magicom bila belum mendesak agar konsumsi turun."
    if action == 3:
        return "⚠️ Kurangi durasi Waterheater karena ini salah satu beban terbesar."
    if action == 4:
        return "💡 Matikan TV saat tidak ditonton untuk menjaga konsumsi tetap hemat."
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
        f" Hemat {daily_saving_pct:.1f}% dibanding rata-rata harian biasa"
        if daily_saving_pct > 0
        else " Belum ada penghematan signifikan dibanding rata-rata harian"
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
prediksi_watt = forecast_next_step(sarimax_artifact, latest_features)

rl_action = None
if model_rl is not None:
    gap_to_target = float(budget_context["current_month_run_rate_rp"]) - float(budget_context["monthly_target_rp"])
    state_v2 = np.array(
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
        state = state_v2 if expected_obs_dim >= len(state_v2) else state_v1
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
    f"📊 Daya Aktual: {monitoring_state['Daya']} W | Prediksi SARIMAX: {prediksi_watt:.2f} W | "
    f"Target Bulanan: Rp {float(budget_context['monthly_target_rp']):,.0f}"
)
print(f"🗣️ Rekomendasi Hemat: {recommendation['combined_advice']}")
print(f"💰 Indikator hemat hari ini: {recommendation['saving_badge']}")

db.reference("Hasil_AI").set(
    {
        "prediksi_daya_selanjutnya": round(float(prediksi_watt), 2),
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
