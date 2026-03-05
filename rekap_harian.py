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
TARIF_PER_KWH = 1352.00  # Tarif PLN 900VA Non-Subsidi

# Gunakan jalur keamanan tingkat tinggi
if not firebase_admin._apps:
    firebase_key_json = os.getenv('FIREBASE_KEY')
    if not firebase_key_json:
        raise ValueError("❌ GAGAL: Kunci FIREBASE_KEY tidak ditemukan!")
    
    key_dict = json.loads(firebase_key_json)
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})

def jalankan_rekap():
    print("Memulai perhitungan selisih penggunaan energi Harian dan Per Jam...")
    try:
        # ==========================================
        # 2. PENGATURAN WAKTU WIB (Penting untuk Nama Folder)
        # ==========================================
        tz_jkt = pytz.timezone('Asia/Jakarta')
        waktu_sekarang = datetime.now(tz_jkt)
        tanggal_str = waktu_sekarang.strftime('%Y-%m-%d')
        jam_str = waktu_sekarang.strftime('%H')
        kunci_jam = f"{tanggal_str}_{jam_str}" # Format: 2024-03-05_14
        timestamp_str = waktu_sekarang.strftime('%Y-%m-%d %H:%M:%S')

        # ==========================================
        # 3. AMBIL DATA kWh TERBARU DARI SENSOR
        # ==========================================
        ref_sensor = db.reference('/Monitoring')
        data_pzem = ref_sensor.get()
        
        if not data_pzem:
            print("❌ Gagal: Folder /Monitoring kosong.")
            return

        kwh_sekarang = data_pzem.get('Energy', 0)

        # ==========================================
        # 4. LOGIKA REKAP HARIAN (Anti-Amnesia)
        # ==========================================
        # Cek apakah rekap hari ini sudah ada?
        hari_ini_ref = db.reference(f'/rekap_harian/history/{tanggal_str}').get()
        
        kwh_awal_hari_ini = 0
        if hari_ini_ref:
            # Jika sudah ada, gunakan nilai awal tadi pagi (jangan diganti)
            kwh_awal_hari_ini = hari_ini_ref.get('kwh_awal', 0)
        else:
            # Jika belum ada (Ganti hari), cari data kwh akhir dari hari kemarin
            last_entry_query = db.reference('/rekap_harian/history').order_by_key().limit_to_last(1).get()
            if last_entry_query:
                for key in last_entry_query:
                    kwh_awal_hari_ini = last_entry_query[key].get('total_kwh_akhir', 0)
            else:
                kwh_awal_hari_ini = kwh_sekarang # Antisipasi jika database benar-benar kosong

        # Hitung selisih
        if kwh_sekarang >= kwh_awal_hari_ini:
            penggunaan_hari_ini = kwh_sekarang - kwh_awal_hari_ini
        else:
            penggunaan_hari_ini = kwh_sekarang # Antisipasi sensor ter-reset
            
        total_biaya = penggunaan_hari_ini * TARIF_PER_KWH

        # ==========================================
        # 5. LOGIKA REKAP PER JAM (Untuk Grafik Wh)
        # ==========================================
        jam_ini_ref = db.reference(f'/Kwh_Jam/{kunci_jam}').get()
        
        kwh_awal_jam_ini = 0
        if jam_ini_ref:
            kwh_awal_jam_ini = jam_ini_ref.get('kwh_awal', 0)
        else:
            # Jika pindah jam baru, ambil kwh terakhir dari jam sebelumnya
            last_jam_query = db.reference('/Kwh_Jam').order_by_key().limit_to_last(1).get()
            if last_jam_query:
                for key in last_jam_query:
                    kwh_awal_jam_ini = last_jam_query[key].get('total_kwh_akhir', 0)
            else:
                kwh_awal_jam_ini = kwh_sekarang

        # Hitung pemakaian jam ini, lalu kalikan 1000 agar menjadi Watt-hour (Wh)
        if kwh_sekarang >= kwh_awal_jam_ini:
            pemakaian_wh_jam_ini = (kwh_sekarang - kwh_awal_jam_ini) * 1000
        else:
            pemakaian_wh_jam_ini = kwh_sekarang * 1000

        # ==========================================
        # 6. SIMPAN SEMUA HASIL KE FIREBASE
        # ==========================================
        # Simpan Data Harian
        db.reference(f'/rekap_harian/history/{tanggal_str}').set({
            'kwh_awal': round(kwh_awal_hari_ini, 4),
            'total_kwh_akhir': round(kwh_sekarang, 4),
            'penggunaan_kwh': round(penggunaan_hari_ini, 4),
            'biaya_rp': round(total_biaya, 2),
            'timestamp': timestamp_str
        })

        # Simpan Data Per Jam
        db.reference(f'/Kwh_Jam/{kunci_jam}').set({
            'jam': jam_str,
            'kwh_awal': round(kwh_awal_jam_ini, 4),
            'total_kwh_akhir': round(kwh_sekarang, 4),
            'wh_pakai': round(pemakaian_wh_jam_ini, 2),
            'timestamp': timestamp_str
        })

        # Update Dashboard
        db.reference('/dashboard_info').update({
            'kwh_hari_ini': round(penggunaan_hari_ini, 4),
            'biaya_hari_ini': round(total_biaya, 2),
            'terakhir_update': timestamp_str
        })

        print(f"✅ Rekap Harian: {penggunaan_hari_ini:.4f} kWh | Rp {total_biaya:.2f}")
        print(f"✅ Rekap Jam ({jam_str}:00): {pemakaian_wh_jam_ini:.2f} Wh")

    except Exception as e:
        print(f"❌ Terjadi kesalahan teknis: {e}")

if __name__ == "__main__":
    jalankan_rekap()
