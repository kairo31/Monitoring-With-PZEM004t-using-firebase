import os
import json
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import pickle
from stable_baselines3 import PPO
from datetime import datetime
import pytz

print("🤖 Pekerja AI succesfull...")

# ==========================================
# 1. INISIALISASI FIREBASE (VIA GITHUB SECRETS)
# ==========================================
# Mengambil teks JSON dari Brankas GitHub
firebase_key_json = os.getenv('FIREBASE_KEY')

if not firebase_key_json:
    raise ValueError("❌ GAGAL: Kunci FIREBASE_KEY tidak ditemukan di environment/secrets!")

# Mengubah teks rahasia menjadi format kredensial yang dikenali Firebase
key_dict = json.loads(firebase_key_json)
cred = credentials.Certificate(key_dict)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app'
    })
print("✅ Berhasil login ke Firebase melalui jalur aman!")

# ==========================================
# 2. LOAD OTAK AI (DARI FOLDER 'model')
# ==========================================
print("🧠 Memuat otak SARIMAX dan RL...")

# Perhatikan tambahan 'model/' di depan nama file
# Perhatikan nama foldernya sekarang disesuaikan menjadi 'model_ai/'
with open('model_ai/model_sarimax_kairo.pkl', 'rb') as f:
    sarimax = pickle.load(f)

model_rl = PPO.load("model_ai/model_rl_kairo.zip")
# Perhatikan tambahan 'model/' di depan nama file
model_rl = PPO.load("model/model_rl_kairo.zip")

# ==========================================
# 3. AMBIL DATA TERBARU DARI FIREBASE
# ==========================================
hist_data = db.reference('history').order_by_key().limit_to_last(1).get()
daya_skrg = 0
suhu_skrg = 25.0
if hist_data:
    for key, val in hist_data.items():
        daya_skrg = val.get('Daya', 0)
        suhu_skrg = val.get('Suhu', 25.0)

log_data = db.reference('log_konfirmasi').order_by_key().limit_to_last(1).get()
ac = magicom = wh = tv = laptop = 0
if log_data:
    for key, val in log_data.items():
        ac = 1 if val.get('AC') else 0
        magicom = 1 if val.get('Magicom') else 0
        wh = 1 if val.get('Waterheater') else 0
        tv = 1 if val.get('TV') else 0
        laptop = 1 if val.get('Laptop') else 0

tz = pytz.timezone('Asia/Jakarta')
jam_skrg = datetime.now(tz).hour

# ==========================================
# 4. TUGAS 1: PREDIKSI SARIMAX
# ==========================================
skenario = pd.DataFrame([[suhu_skrg, ac, magicom, wh, laptop, tv]], 
                        columns=['Suhu', 'AC', 'Magicom', 'Waterheater', 'Laptop', 'TV'])
prediksi_watt = sarimax.predict(n_periods=1, X=skenario).iloc[0]

# ==========================================
# 5. TUGAS 2: KEPUTUSAN RL
# ==========================================
kondisi_kamar = np.array([daya_skrg, suhu_skrg, ac, magicom, wh, tv, jam_skrg], dtype=np.float32)
aksi, _ = model_rl.predict(kondisi_kamar, deterministic=True)

rekomendasi = "Aman. Biarkan menyala."
if aksi == 1: rekomendasi = "⚠️ BAHAYA BEBAN PUNCAK: Matikan AC sekarang!"
elif aksi == 2: rekomendasi = "⚠️ BAHAYA BEBAN PUNCAK: Matikan Magicom sekarang!"
elif aksi == 3: rekomendasi = "⚠️ BAHAYA BEBAN PUNCAK: Matikan Waterheater!"
elif aksi == 4: rekomendasi = "💡 PEMBOROSAN TERDETEKSI: Matikan TV (Sedang tidak ditonton/jam tidur)."

print(f"📊 Daya Aktual: {daya_skrg} W | Prediksi 1 Menit ke Depan: {prediksi_watt:.2f} W")
print(f"🗣️ Rekomendasi RL: {rekomendasi}")

# ==========================================
# 6. PUSH HASIL KE FIREBASE
# ==========================================
db.reference('Hasil_AI').set({
    'prediksi_daya_selanjutnya': round(float(prediksi_watt), 2),
    'rekomendasi_rl': rekomendasi,
    'waktu_update': datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
})

print("✅ Laporan berhasil dikirim ke Firebase! Pekerja AI kembali tidur.")


