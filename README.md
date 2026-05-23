

# Monitoring & Optimasi Energi: PZEM-004T + Firebase + AI Hybrid

Sistem IoT untuk pemantauan beban listrik berbasis PZEM-004T dan Firebase, dilengkapi dengan **Hybrid Forecasting Model (SARIMAX-XGBoost)** untuk optimasi anggaran energi melalui *Reinforcement Learning*.

## 🎯 Deskripsi Proyek

Proyek skripsi ini dirancang untuk menyelesaikan tantangan dalam prediksi beban listrik rumah tangga yang bersifat *intermittent* (sering *idle*). Model statistik linear klasik (SARIMAX) cenderung gagal menangkap lonjakan (*spike*) daya yang tajam. Oleh karena itu, dikembangkan sebuah arsitektur **Hybrid Error Correction** yang mengintegrasikan:

1. **SARIMAX**: Memodelkan komponen linear dan tren musiman.
2. **XGBoost**: Memperbaiki residual (error) untuk menangkap pola *spike* non-linear.
3. **Reinforcement Learning**: Menggunakan hasil prediksi untuk optimasi budget energi secara *real-time*.

## 🏗️ Arsitektur Sistem

```text
Data PZEM-004T ──► Firebase RTDB ──► [Pipeline Hybrid] ──► RL Agent (Budgeting)
                                            │
                                    ┌───────┴───────┐
                                    │  SARIMAX +    │
                                    │   XGBoost     │
                                    └───────────────┘

```

## 📂 Struktur Repositori

* `Train/`: Berisi skrip pelatihan model (`sarimax.py`, `rl_budget_v2.py`).
* `model_ai/`: Modul *deployment* dan konfigurasi AI.
* `public/`: *Dashboard* visualisasi berbasis web (Firebase Hosting).
* `.github/workflows/`: *Pipeline* CI/CD untuk pengujian otomatis.

## 🚀 Fitur Utama

* **Hybrid Forecasting**: Menggabungkan kekuatan statistika SARIMAX dengan fleksibilitas ML XGBoost untuk menekan eror (MAE) pada beban *intermittent*.
* **Automated Data Cleaning**: Fungsi *auto-repair* JSON untuk menangani *crash* data Firebase.
* **CI/CD Integrated**: *Pipeline* GitHub Actions untuk pengujian otomatis setiap ada perubahan kode.
* **Real-time Monitoring**: Integrasi langsung dengan Firebase RTDB.

## 🛠️ Instalasi & Setup

Untuk menjalankan *pipeline* pelatihan di lingkungan lokal:

1. Clone repositori ini:
```bash
git clone https://github.com/kairo31/monitoring-with-pzem004t-using-firebase.git
cd monitoring-with-pzem004t-using-firebase

```


2. Instal dependensi:
```bash
pip install -r requirements.txt

```


3. Jalankan pelatihan model (contoh):
```bash
python Train/sarimax_firebase_train.py

```




## 📜 Lisensi & Kredit

Proyek ini dikembangkan sebagai bagian dari tugas akhir skripsi Program Studi Fisika (Electronics & Instrumentation), Universitas terkait.

