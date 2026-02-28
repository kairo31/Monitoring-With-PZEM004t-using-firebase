import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import pytz

# 1. KONEKSI KE FIREBASE
# Sesuaikan path ke file JSON kamu
cred = credentials.Certificate("firebase-project/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app'
})

wib = pytz.timezone('Asia/Jakarta')
TARIF_900VA_NON_SUBSIDI = 1352 # Tarif R-1M per kWh

def proses_rekap():
    print(f"[{datetime.now(wib)}] Menjalankan Robot Rekap Biaya...")
    
    ref_history = db.ref('history')
    data_history = ref_history.get()

    if not data_history:
        print("Folder history kosong.")
        return

    # Ambil data paling baru
    last_key = list(data_history.keys())[-1]
    last_item = data_history[last_key]
    
    # Hitung biaya
    energy = last_item.get('Energy', 0)
    biaya = energy * TARIF_900VA_NON_SUBSIDI
    
    # Update ke Firebase (History & Rekap Harian)
    waktu_skrg = datetime.now(wib).strftime('%H:%M:%S')
    nama_hari = datetime.now(wib).strftime('%A')
    
    # Tambahkan stempel waktu ke history jika belum ada
    if 'waktu' not in last_item:
        ref_history.child(last_key).update({'waktu': waktu_skrg, 'hari': nama_hari})

    # Update Dashboard Harian
    db.ref('Kwh_Harian').child(nama_hari).set({
        'hari': nama_hari,
        'kwh_pakai': round(energy, 4),
        'biaya_rp': round(biaya, 2),
        'last_update': waktu_skrg
    })
    
    print(f"✅ Rekap Selesai: {energy} kWh | Estimasi: Rp {round(biaya, 2)}")

if __name__ == "__main__":
    proses_rekap()