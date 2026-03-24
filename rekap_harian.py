import json
import os
from datetime import datetime

import firebase_admin
import pytz
from firebase_admin import credentials, db

# ==========================================
# 1. KONFIGURASI DATABASE DAN TARIF
# ==========================================
DB_URL = 'https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app/'
TARIF_PER_KWH = 1352.00  # Tarif PLN 900VA Non-Subsidi

if not firebase_admin._apps:
    firebase_key_json = os.getenv('FIREBASE_KEY')
    if not firebase_key_json:
        raise ValueError("❌ GAGAL: Kunci FIREBASE_KEY tidak ditemukan!")

    key_dict = json.loads(firebase_key_json)
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})


def hitung_rata_rata_harian(tanggal_hari_ini: str) -> tuple[float, float]:
    history = db.reference('/rekap_harian/history').order_by_key().limit_to_last(30).get() or {}
    biaya_list = []
    kwh_list = []

    for key, value in history.items():
        if key == tanggal_hari_ini or not isinstance(value, dict):
            continue
        biaya = float(value.get('biaya_rp', 0) or 0)
        kwh = float(value.get('penggunaan_kwh', 0) or 0)
        if biaya > 0:
            biaya_list.append(biaya)
        if kwh > 0:
            kwh_list.append(kwh)

    avg_biaya = sum(biaya_list) / len(biaya_list) if biaya_list else 0.0
    avg_kwh = sum(kwh_list) / len(kwh_list) if kwh_list else 0.0
    return avg_biaya, avg_kwh


def jalankan_rekap():
    print("Memulai perhitungan selisih penggunaan energi Harian dan Per Jam...")
    try:
        tz_jkt = pytz.timezone('Asia/Jakarta')
        waktu_sekarang = datetime.now(tz_jkt)
        tanggal_str = waktu_sekarang.strftime('%Y-%m-%d')
        jam_str = waktu_sekarang.strftime('%H')
        kunci_jam = f"{tanggal_str}_{jam_str}"
        timestamp_str = waktu_sekarang.strftime('%Y-%m-%d %H:%M:%S')

        ref_sensor = db.reference('/Monitoring')
        data_pzem = ref_sensor.get()

        if not data_pzem:
            print("❌ Gagal: Folder /Monitoring kosong.")
            return

        kwh_sekarang = float(data_pzem.get('Energy', 0) or 0)

        hari_ini_ref = db.reference(f'/rekap_harian/history/{tanggal_str}').get()

        kwh_awal_hari_ini = 0
        if hari_ini_ref:
            kwh_awal_hari_ini = float(hari_ini_ref.get('kwh_awal', 0) or 0)
        else:
            last_entry_query = db.reference('/rekap_harian/history').order_by_key().limit_to_last(1).get()
            if last_entry_query:
                for key in last_entry_query:
                    kwh_awal_hari_ini = float(last_entry_query[key].get('total_kwh_akhir', 0) or 0)
            else:
                kwh_awal_hari_ini = kwh_sekarang

        penggunaan_hari_ini = (kwh_sekarang - kwh_awal_hari_ini) if kwh_sekarang >= kwh_awal_hari_ini else kwh_sekarang
        total_biaya = penggunaan_hari_ini * TARIF_PER_KWH

        jam_ini_ref = db.reference(f'/Kwh_Jam/{kunci_jam}').get()

        kwh_awal_jam_ini = 0
        if jam_ini_ref:
            kwh_awal_jam_ini = float(jam_ini_ref.get('kwh_awal', 0) or 0)
        else:
            last_jam_query = db.reference('/Kwh_Jam').order_by_key().limit_to_last(1).get()
            if last_jam_query:
                for key in last_jam_query:
                    kwh_awal_jam_ini = float(last_jam_query[key].get('total_kwh_akhir', 0) or 0)
            else:
                kwh_awal_jam_ini = kwh_sekarang

        pemakaian_wh_jam_ini = (kwh_sekarang - kwh_awal_jam_ini) * 1000 if kwh_sekarang >= kwh_awal_jam_ini else kwh_sekarang * 1000
        rata_biaya_harian, rata_kwh_harian = hitung_rata_rata_harian(tanggal_str)
        hemat_rp = max(rata_biaya_harian - total_biaya, 0)
        hemat_pct = (hemat_rp / rata_biaya_harian * 100) if rata_biaya_harian > 0 else 0

        db.reference(f'/rekap_harian/history/{tanggal_str}').set({
            'kwh_awal': round(kwh_awal_hari_ini, 4),
            'total_kwh_akhir': round(kwh_sekarang, 4),
            'penggunaan_kwh': round(penggunaan_hari_ini, 4),
            'biaya_rp': round(total_biaya, 2),
            'hemat_dari_rata_rata_rp': round(hemat_rp, 2),
            'hemat_dari_rata_rata_pct': round(hemat_pct, 2),
            'timestamp': timestamp_str
        })

        db.reference(f'/Kwh_Jam/{kunci_jam}').set({
            'jam': jam_str,
            'kwh_awal': round(kwh_awal_jam_ini, 4),
            'total_kwh_akhir': round(kwh_sekarang, 4),
            'wh_pakai': round(pemakaian_wh_jam_ini, 2),
            'timestamp': timestamp_str
        })

        db.reference('/dashboard_info').update({
            'kwh_hari_ini': round(penggunaan_hari_ini, 4),
            'biaya_hari_ini': round(total_biaya, 2),
            'rata_biaya_harian': round(rata_biaya_harian, 2),
            'rata_kwh_harian': round(rata_kwh_harian, 4),
            'hemat_hari_ini_rp': round(hemat_rp, 2),
            'hemat_hari_ini_pct': round(hemat_pct, 2),
            'terakhir_update': timestamp_str
        })

        print(f"✅ Rekap Harian: {penggunaan_hari_ini:.4f} kWh | Rp {total_biaya:.2f} | Hemat {hemat_pct:.2f}%")
        print(f"✅ Rekap Jam ({jam_str}:00): {pemakaian_wh_jam_ini:.2f} Wh")

    except Exception as e:
        print(f"❌ Terjadi kesalahan teknis: {e}")


if __name__ == "__main__":
    jalankan_rekap()
