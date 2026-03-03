🏗️ System Architecture & Tech Stack

 1. Hardware & Edge Layer (IoT Node)

1. Microcontroller Unit (MCU): ESP32 (berperan sebagai *Edge Node* untuk akuisisi data dan transmisi HTTP/MQTT).
* **Sensors:** * `PZEM-004T`: Mengukur parameter kelistrikan/fisika (Voltage, Current, Active Power) secara *real-time*.
* `DHT22`: Mengukur parameter Suhu Indoor sebagai *exogenous variable* untuk SARIMAX.


* **Firmware:** C++ / Arduino IDE.

### 2. Backend & Data Pipeline (BaaS)

* **Database:** Firebase Realtime Database (BaaS - *Backend as a Service*) bertindak sebagai *message broker* dan repositori state (Node: `history`, `log_konfirmasi`, `Prediksi`).
* **External API Integration:** API OpenWheaterMap untuk menarik data *historical* dan *real-time* suhu *outdoor* kota Purwokerto.

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
