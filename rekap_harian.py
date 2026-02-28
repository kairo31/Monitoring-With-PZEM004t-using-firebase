import firebase_admin
from firebase_admin import credentials, db
import pytz
from datetime import datetime

# 1. Konfigurasi Database dan Tarif
DB_URL = 'https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app/'
TARIF_PER_KWH = 1352.00  # Tarif PLN 900VA Non-Subsidi (R-1/900 VA-RTM)

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})

def jalankan_rekap():
    print("Memulai perhitungan selisih penggunaan energi...")
    try:
        # 2. Ambil data kWh terbaru dari sensor PZEM
        # Pastikan nama 'total_kwh' sesuai dengan yang dikirimkan ESP32 Anda
        ref_sensor = db.reference('/pzem_data')
        data_pzem = ref_sensor.get()
        
        if not data_pzem:
            print("Gagal: Data sensor tidak ditemukan di database.")
            return

        kwh_sekarang = data_pzem.get('total_kwh', 0)

        # 3. Ambil data terakhir dari riwayat untuk mendapatkan kwh_kemarin
        ref_history = db.reference('/rekap_harian/history')
        # Mengambil 1 data terakhir berdasarkan urutan kunci (tanggal)
        last_entry_query = ref_history.order_by_key().limit_to_last(1).get()
        
        kwh_kemarin = 0
        if last_entry_query:
            for key in last_entry_query:
                kwh_kemarin = last_entry_query[key].get('total_kwh_akhir', 0)
        
        # 4. Hitung penggunaan hari ini (Selisih)
        # Jika kwh_sekarang lebih kecil dari kwh_kemarin, artinya sensor baru saja reset/overflow
        if kwh_sekarang >= kwh_kemarin:
            penggunaan_hari_ini = kwh_sekarang - kwh_kemarin
        else:
            penggunaan_hari_ini = kwh_sekarang

        total_biaya = penggunaan_hari_ini * TARIF_PER_KWH

        # 5. Pengaturan Waktu WIB
        tz_jkt = pytz.timezone('Asia/Jakarta')
        waktu_sekarang = datetime.now(tz_jkt)
        tanggal_str = waktu_sekarang.strftime('%Y-%m-%d')
        timestamp_str = waktu_sekarang.strftime('%Y-%m-%d %H:%M:%S')

        # 6. Simpan hasil perhitungan ke Firebase
        # Simpan ke riwayat harian (untuk data tabel/grafik di tesis)
        db.reference(f'/rekap_harian/history/{tanggal_str}').set({
            'kwh_awal': round(kwh_kemarin, 4),
            'total_kwh_akhir': round(kwh_sekarang, 4),
            'penggunaan_kwh': round(penggunaan_hari_ini, 4),
            'biaya_rp': round(total_biaya, 2),
            'timestamp': timestamp_str
        })

        # Update data terbaru untuk ditampilkan di dashboard utama
        db.reference('/dashboard_info').update({
            'kwh_hari_ini': round(penggunaan_hari_ini, 4),
            'biaya_hari_ini': round(total_biaya, 2),
            'terakhir_update': timestamp_str
        })

        print(f"Perhitungan selesai untuk tanggal {tanggal_str}")
        print(f"Hasil: {penggunaan_hari_ini:.4f} kWh | Rp {total_biaya:.2f}")

    except Exception as e:
        print(f"Terjadi kesalahan teknis: {e}")

if __name__ == "__main__":
    jalankan_rekap()
