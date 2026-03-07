 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index a2976c9d8fe36472de24fa985cc3020518b920fe..e04c7aa6fd1d816c0ebaa729754707bf43804413 100644
--- a/README.md
+++ b/README.md
@@ -17,25 +17,49 @@
 ### 3. Machine Learning & AI Inference Layer
 
 * **Training Environment:** Jupyter Notebook / Google Colab.
 * **Time-Series Forecasting:** `statsmodels` (SARIMAX) digunakan untuk memprediksi konsumsi daya di masa depan berdasarkan variabel suhu dan *boolean state* perangkat.
 * **Reinforcement Learning:** `stable-baselines3` (PPO - Proximal Policy Optimization) dikonfigurasi dengan *custom Gym Environment* untuk bertindak sebagai *digital controller*. Agen PPO menghitung metrik *reward/penalty* untuk menghasilkan *discrete action* (contoh: "Matikan AC").
 * **Model Serialization:** Pickle (`.pkl`) dan Zip archive untuk *freezing* bobot *Neural Network*.
 
 ### 4. DevOps & Automation (CI/CD / Workflows)
 
 * **Serverless Execution:** GitHub Actions (YAML Workflows).
 * **Cron Job Trigger:** Skrip *inference* Python dieksekusi secara otomatis sebagai *headless task* setiap 5 menit. Skrip ini memuat ulang ( *load*) model dari repositori, membaca *state* Firebase terakhir, menjalankan prediksi, dan melakukan *push update* ke *database*.
 
 ### 5. Frontend / Client-Side (User Interface)
 
 * **Stack:** Vanilla HTML5, CSS3, dan JavaScript (ES6).
 * **State Management:** Menggunakan Firebase Client SDK untuk merekam perubahan data secara asinkron (*event-driven*).
 * **UI/UX:** *Dashboard* interaktif yang melakukan *Direct DOM Manipulation* untuk memperbarui grafik *real-time* dan menampilkan *actionable advice* dari server AI tanpa perlu memuat ulang (*refresh*) halaman web.
 
 ## ⚙️ Execution Workflow
 
 1. **Data Ingestion:** ESP32 mengirimkan metrik daya dan suhu ke *Firebase Backend* setiap 10 detik.
 2. **User Interaction:** Perubahan *state* alat keras (ON/OFF) dicatat via *Frontend Web* ke *database*.
 3. **Automated Inference:** *GitHub Actions Cron Job* memicu *Python script*.
 4. **Decision Matrix:** *RL Agent* mengevaluasi matriks *state* lingkungan saat ini, sementara *SARIMAX* mengalkulasi *predicted load*.
 5. **Real-time Feedback:** Hasil akhir diekspor ke *Firebase*, yang secara instan memicu *event listener* di *Frontend* untuk merender perintah intervensi kepada *user*.
+
+## 📈 SARIMAX Research Pipeline (Skripsi)
+
+Tambahan script modular tersedia di `model_ai/train_sarimax_pipeline.py` untuk training dan analisis time-series dari Firebase dengan fitur:
+- rekonstruksi timestamp untuk data lama tanpa timestamp,
+- visualisasi per tahap preprocessing dan modeling,
+- uji stasioneritas (ADF), dekomposisi musiman, ACF/PACF,
+- validasi parameter dengan `auto_arima`,
+- grid search SARIMAX berbasis AIC,
+- walk-forward validation,
+- forecasting 10 menit, 30 menit, dan 1 jam,
+- penyimpanan artefak model (`.pkl`), metrik (`.json`), dan plot (`.png`).
+
+Contoh eksekusi:
+
+```bash
+python model_ai/train_sarimax_pipeline.py \
+  --database-url "https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app" \
+  --firebase-key-env FIREBASE_KEY \
+  --history-node history \
+  --sampling-interval 10s \
+  --resample-interval 1min \
+  --seasonal-period 12
+```
 
EOF
)
