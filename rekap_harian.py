import os
import json
import firebase_admin
from firebase_admin import credentials, db
import pytz
from datetime import datetime

# ==========================================
# 1. KONFIGURASI DATABASE DAN TARIF
# ==========================================
DB_URL = 'https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app/'
TARIF_PER_KWH = 1352.00  # Tarif PLN 900VA Non-Subsidi (R-1/900 VA-RTM)

# Gunakan jalur keamanan tingkat tinggi (GitHub Secrets)
if not firebase_admin._apps:
    firebase_key_json = os.getenv('FIREBASE_KEY')
    if not firebase_key_json:
        raise ValueError("❌ GAGAL: Kunci FIREBASE_KEY tidak ditemukan di environment!")
    
    key_dict = json.loads(firebase_key_json)
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})

def jalankan_rekap():
    print("Memulai perhitungan selisih penggunaan energi...")
    try:
        # ==========================================
        # 2. AMBIL DATA kWh TERBARU (DIET DATA)
        # ==========================================
        # Kita ambil langsung dari '/Monitoring' (bukan dari '/history' yang raksasa)
        ref_sensor = db.reference('/Monitoring')
        data_pzem = ref_sensor.get()
        
        if not data_pzem:
            print("❌ Gagal: Folder /Monitoring kosong atau ESP32 belum mengirim data.")
            return

        # Mengambil nilai Energy dari folder Monitoring
        kwh_sekarang = data_pzem.get('Energy', 0)

        # ==========================================
        # 3. AMBIL DATA KEMARIN (DIET DATA)
        # ==========================================
        ref_history = db.reference('/rekap_harian/history')
        # Gunakan limit_to_last(1) agar tidak menyedot data bertahun-tahun
        last_entry_query = ref_history.order_by_key().limit_to_last(1).get()
        
        kwh_kemarin = 0
        if last_entry_query:
            for key in last_entry_query:
                kwh_kemarin = last_entry_query[key].get('total_kwh_akhir', 0)
        
        # ==========================================
        # 4. HITUNG SELISIH (PENGGUNAAN HARI INI)
        # ==========================================
        if kwh_sekarang >= kwh_kemarin:
            penggunaan_hari_ini = kwh_sekarang - kwh_kemarin
        else:
            penggunaan_hari_ini = kwh_sekarang # Antisipasi jika sensor reset ke 0

        total_biaya = penggunaan_hari_ini * TARIF_PER_KWH

        # ==========================================
        # 5. PENGATURAN WAKTU WIB
        # ==========================================
        tz_jkt = pytz.timezone('Asia/Jakarta')
        waktu_sekarang = datetime.now(tz_jkt)
        tanggal_str = waktu_sekarang.strftime('%Y-%m-%d')
        timestamp_str = waktu_sekarang.strftime('%Y-%m-%d %H:%M:%S')

        # ==========================================
        # 6. SIMPAN HASIL KE FIREBASE
        # ==========================================
        db.reference(f'/rekap_harian/history/{tanggal_str}').set({
            'kwh_awal': round(kwh_kemarin, 4),
            'total_kwh_akhir': round(kwh_sekarang, 4),
            'penggunaan_kwh': round(penggunaan_hari_ini, 4),
            'biaya_rp': round(total_biaya, 2),
            'timestamp': timestamp_str
        })

        db.reference('/dashboard_info').update({
            'kwh_hari_ini': round(penggunaan_hari_ini, 4),
            'biaya_hari_ini': round(total_biaya, 2),
            'terakhir_update': timestamp_str
        })

        print(f"✅ Perhitungan selesai untuk tanggal {tanggal_str}")
        print(f"📊 Hasil: {penggunaan_hari_ini:.4f} kWh | Rp {total_biaya:.2f}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan teknis: {e}")

if __name__ == "__main__":
    jalankan_rekap()
