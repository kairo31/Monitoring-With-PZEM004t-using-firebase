
# ⚡ Smart Power Monitoring — PZEM-004T + Firebase + AI

🌐 **Language:** **English** | [Bahasa Indonesia](README.id.md)

A household electricity monitoring system built on an **ESP32 + PZEM-004T** that streams data in real time to **Firebase Realtime Database**, visualizes it on a web dashboard, and adds **AI** for load forecasting (SARIMAX) and energy-saving recommendations that respect a monthly budget (Reinforcement Learning).

---

## ✨ Features

- **Real-time monitoring** — voltage, current, power, energy (kWh), temperature, and humidity.
- **History logging** — saved to `history` every minute with a Firebase server timestamp.
- **Automatic recaps** — daily and hourly recaps (kWh and cost in Rupiah) via GitHub Actions.
- **Load forecasting** — SARIMAX with exogenous variables (temperature and device status).
- **Energy-saving recommendations** — an RL agent picks an action (reduce AC, save on rice cooker, etc.) to keep the bill under the monthly budget.
- **Web dashboard** — hosted on Firebase Hosting, plus a demo page and a simulator.

---

## 🏗️ Architecture

```
┌──────────────┐  Serial2   ┌─────────┐   WiFi    ┌──────────────────────┐
│ PZEM-004T v3 ├───────────►│  ESP32  ├──────────►│ Firebase Realtime DB │
│ DHT22        ├───────────►│         │           │  /Monitoring         │
└──────────────┘            └─────────┘           │  /history            │
                                                  └───────┬──────────────┘
                                                          │
                       GitHub Actions (cron) ◄────────────┤
                       ├─ rekap_harian.py  ──► /rekap_harian, /Kwh_Jam, /dashboard_info
                       └─ pekerja_ai.py    ──► /Hasil_AI  (SARIMAX + RL)
                                                          │
                                                  ┌───────▼──────────────┐
                                                  │ Dashboard (Hosting)  │
                                                  │ public/monitoring.html
                                                  └──────────────────────┘
```

---

## 📁 Project Structure

```
.
├── coding_firebase_monitoring.ino   # ESP32 firmware (PZEM + DHT22 → Firebase)
├── public/                          # Web app (Firebase Hosting)
│   ├── index.html                   # Portal / landing page
│   ├── monitoring.html              # Budget & AI dashboard
│   ├── monitoring_whrl.html         # Simple live monitoring
│   ├── demo.html                    # Demo analyzer (non-live)
│   ├── simulasi.html                # System simulator
│   └── 404.html
├── monitoring_test.html             # Live monitoring test page
├── rekap_harian.py                  # Daily/hourly kWh & cost recap
├── pekerja_ai.py                    # "AI worker": SARIMAX + RL inference → Hasil_AI
├── pekerja_ai_1                     # Alternative AI worker (archive)
├── sarimax_utils.py                 # SARIMAX loader & forecasting helpers
├── sarimax_firebase_train.py        # SARIMAX training from Firebase data
├── model_ai/                        # Model folder (downloaded automatically in CI/cron)
│   ├── pekerja_ai.py                # Shim → imports root pekerja_ai.py
│   └── requirements.txt
├── Train/                           # Training & experiments
│   ├── rl_budget_v2.py              # RL budget environment & agent (v2)
│   ├── rl_budget_collab.py
│   ├── rl_retrain_budget.py         # RL retraining from Firebase data
│   ├── sarimax.py / sarimax_firebase_train.py
│   ├── *.ipynb                      # SARIMAX & RL notebooks
│   └── rl_budget.pt
├── .github/workflows/
│   ├── ai-cron.yml                  # Pipeline every 15 min + manual retraining
│   ├── main.yml                     # "Robot-Rekap" every 5 min
│   └── ci.yml                       # Compile check + model download test
├── firebase.json / .firebaserc      # Firebase Hosting config
└── requirements.txt
```

---

## 🔌 Hardware

| Component | Notes |
|---|---|
| ESP32 | Microcontroller with WiFi |
| PZEM-004T v3.0 | Voltage, current, power, energy sensor |
| DHT22 | Temperature & humidity sensor |

**Wiring**

| Sensor | ESP32 Pin |
|---|---|
| PZEM-004T TX → | GPIO **16** (RX2) |
| PZEM-004T RX ← | GPIO **17** (TX2) |
| DHT22 DATA | GPIO **26** |
| VCC / GND | 5V / GND (per your module) |

> ⚠️ The PZEM-004T connects to mains voltage. Wire it safely, use proper insulation, and switch off the breaker while assembling.

---

## 🚀 Getting Started

### 1. ESP32 Firmware

1. Install the **Arduino IDE** with ESP32 board support.
2. Install these libraries (Library Manager):
   - `Firebase ESP32 Client` (FirebaseESP32)
   - `PZEM004Tv30`
   - `DHT sensor library` (+ Adafruit Unified Sensor)
3. Open `coding_firebase_monitoring.ino` and fill in:
   ```cpp
   #define FIREBASE_HOST "<project-id>-default-rtdb.<region>.firebasedatabase.app"
   #define FIREBASE_AUTH "<database-secret>"
   wifiMulti.addAP("<SSID>", "<PASSWORD>");
   ```
4. Upload to the ESP32 (serial monitor baud rate: `115200`).

### 2. Firebase

1. Create a project in the [Firebase Console](https://console.firebase.google.com/) and enable **Realtime Database**.
2. Create a **Service Account** (Project Settings → Service accounts → Generate new private key) for the Python scripts.
3. Configure Database Rules as needed (do not leave the database open to the public in production).
4. Update the project ID in `.firebaserc`.

### 3. Web Dashboard

```bash
npm install -g firebase-tools
firebase login
firebase deploy --only hosting
```

Fill in the `firebaseConfig` inside the HTML files in `public/` with your own project's configuration.

### 4. AI Pipeline (local)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export FIREBASE_KEY='<service account JSON on a single line>'
export FIREBASE_DB_URL='https://<project-id>-default-rtdb.<region>.firebasedatabase.app'

python rekap_harian.py          # daily & hourly recap
python pekerja_ai.py            # forecast + recommendation → /Hasil_AI
```

> The models (`sarimax_bundle.pkl`, `model_rl_budget_v2.pt`, `sarimax_kwh_jam.pkl`) are **not** stored in the repo. Put them in `model_ai/` or download them from Google Drive (see the workflows).

---

## ⚙️ GitHub Actions

| Workflow | Schedule | Purpose |
|---|---|---|
| `ai-cron.yml` (AI Pipeline Scheduler) | every 15 min + manual | Daily recap, AI inference, optional SARIMAX / RL retraining |
| `main.yml` (Robot-Rekap) | every 5 min + manual | Daily recap + run the AI worker |
| `ci.yml` (CI) | every push / PR | Python compile check + model download test |

**Required secrets** (Settings → Secrets and variables → Actions):

| Secret | Required | Description |
|---|---|---|
| `FIREBASE_KEY` | ✅ | Service account JSON contents |
| `FIREBASE_DB_URL` | ✅ | Realtime Database URL |
| `TARIF_PER_KWH` | – | Electricity tariff per kWh (default `1352`) |
| `DEFAULT_MONTHLY_TARGET_RP` | – | Fallback monthly budget |
| `SARIMAX_DRIVE_URL` / `SARIMAX_DRIVE_FILE_ID` | – | SARIMAX model source (Google Drive) |
| `RL_DRIVE_URL` / `RL_DRIVE_FILE_ID` | – | RL model source (Google Drive) |
| `SARIMAX_KWH_DRIVE_URL` | – | Hourly-kWh SARIMAX model source |

Retraining can be triggered manually via **Actions → AI Pipeline Scheduler → Run workflow** (`retrain_sarimax` / `retrain_rl` options).

---

## 🗄️ Firebase Data Structure

| Node | Written by | Contents |
|---|---|---|
| `/Monitoring` | ESP32 | Live data: `Tegangan`, `Arus`, `Daya`, `Energy`, `Suhu`, `Kelembapan` |
| `/history` | ESP32 | Per-minute history + `waktu` (server timestamp) |
| `/rekap_harian/history/<YYYY-MM-DD>` | `rekap_harian.py` | `kwh_awal`, `penggunaan_kwh`, `biaya_rp`, `total_kwh_akhir` |
| `/Kwh_Jam/<YYYY-MM-DD_HH>` | `rekap_harian.py` | Hourly kWh consumption |
| `/dashboard_info` | `rekap_harian.py` | Summary for the dashboard |
| `/user_preferences` | Dashboard | Monthly budget (`monthly_budget_rp`), etc. |
| `/log_konfirmasi` | Dashboard | Device status (AC, Magicom, Waterheater, TV, Laptop) |
| `/Hasil_AI` | `pekerja_ai.py` | Power forecast & RL action recommendation |
| `/Evaluasi_Model_SARIMAX` | SARIMAX training | Model evaluation metrics |

> Field and node names are kept in Indonesian because the code and dashboard use them as-is.

---

## 🧠 AI Models

### SARIMAX — load forecasting
- Order `(1,1,1)` with seasonal order `(1,1,1,24)` (daily, hourly pattern).
- Exogenous variables: `Suhu` (temperature), `AC`, `Magicom`, `Waterheater`, `Laptop`, `TV`.
- Targets: `Daya` (power), `Kwh_Jam` (hourly kWh), and daily recap.
- Training: `python sarimax_firebase_train.py` (or the notebooks in `Train/`).

### Reinforcement Learning — budget recommendations
- Actor-Critic/PPO-style agent (PyTorch), 16-dimensional observation, 6 actions.
- Recommended actions:

| ID | Action | Estimated reduction |
|---|---|---|
| 0 | No action | 0% |
| 1 | Turn off / reduce AC | 30% |
| 2 | Delay / save on rice cooker (Magicom) | 10% |
| 3 | Turn off / reduce water heater | 20% |
| 4 | Turn off TV | 5% |
| 5 | Laptop power-saving mode | 5% |

- The state includes remaining budget, days left in the month, current power, the SARIMAX forecast, temperature, hour, day of week, and device status.
- Retraining: `python Train/rl_retrain_budget.py`.

> The normalization constants, calibration, and action labels in `pekerja_ai.py` **must match** `Train/rl_budget_v2.py`, otherwise model output will be wrong.

---

## 🔐 Security Notes

- **Never commit credentials.** Move the WiFi SSID/password and `FIREBASE_AUTH` out of `coding_firebase_monitoring.ino` into a separate file (e.g. `secrets.h`, listed in `.gitignore`). If credentials were ever committed, **change the WiFi password and Database Secret**, since they remain in git history.
- Never commit `serviceAccountKey.json`; use GitHub Secrets (`FIREBASE_KEY`).
- Apply Firebase Security Rules so the database can't be read or written by anyone.
- Add `__pycache__/` to `.gitignore` and remove it from the repo.

---

## 🛠️ Tech Stack

ESP32 · Arduino C++ · PZEM-004T · DHT22 · Firebase Realtime Database & Hosting · Python 3.11 · statsmodels / pmdarima (SARIMAX) · PyTorch · Stable-Baselines3 · GitHub Actions

---

