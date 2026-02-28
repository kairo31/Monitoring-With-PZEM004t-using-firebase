import firebase_admin
from firebase_admin import credentials, db
import pytz
from datetime import datetime

# 1. Inisialisasi Firebase (Gunakan URL asli kamu)
# Pastikan tidak ada spasi di dalam URL
DB_URL = 'https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app/'

if not firebase_admin._apps:
    # File serviceAccountKey.json ini akan dibuat otomatis oleh GitHub Action
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': DB_URL
    })

def jalankan_rekap():
    print("Memulai proses rekap data...")
    try:
        # 2. Ambil Referensi Database Utama
        ref = db.reference('/') 
        data = ref.get()
        
        if data:
            print("Koneksi Sukses! Data PZEM berhasil ditarik.")
            
            # --- LOGIKA REKAP (Contoh sederhana) ---
            # Di sini kamu bisa tambahkan logika perhitungan biaya listrik
            # Misalnya: total_kwh * 1444.7 (tarif 900VA)
            
            # 3. Catat waktu rekap (WIB)
            tz_jkt = pytz.timezone('Asia/Jakarta')
            waktu_rekap = datetime.now(tz_jkt).strftime('%Y-%m-%d %H:%M:%S')
            
            # Update status di Firebase agar dashboard kamu tahu rekap sudah jalan
            db.reference('/status_sistem').update({
                'terakhir_update': waktu_rekap,
                'sumber': 'GitHub Actions (Otomatis)'
            })
            
            print(f" Rekap berhasil diperbarui pada: {waktu_rekap}")
            
        else:
            print("Database kosong. Cek apakah ESP32 kamu sudah kirim data?")

    except Exception as e:
        print(f" Terjadi kesalahan: {e}")

if __name__ == "__main__":
    jalankan_rekap()
