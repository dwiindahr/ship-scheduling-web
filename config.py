# =========================
# CONFIG & WEIGHTS
# config.py
# =========================

# ── Internal data paths ───────────────────────────────────────────
TERMINAL_DATA_PATH = "internal_data/terminal_jamrud.xlsx"
TIDAL_DATA_PATH    = "internal_data/pasang_surut.csv"

# ── Re-berthing ───────────────────────────────────────────────────
REBERTHING_PART_DURATION = 3.0  # jam per part (kecuali Passenger)

# ── Operational config ────────────────────────────────────────────
CONFIG = {
    # Waktu & clearance
    'APPROACH_TIME'       : 1.0,   # jam, waktu pendekatan kapal ke dermaga
    'TIDE_DELTA'          : 1.76,   # meter, selisih tinggi air pasang vs surut
    'UNDER_KEEL_CLEARANCE': 1.0,   # meter, clearance minimum bawah lunas

    # Dermaga & posisi
    'LOA_MARGIN_DISKRIT'  : 5.0,   # meter, margin LOA untuk penempatan diskrit
    'CONTINUOUS_GAP_M'    : 5.0,   # meter, jarak minimum antar kapal (kontinu)
    'MERGE_POS_TOL_M'     : 1.0,   # meter, toleransi merge posisi

    # Re-berthing
    'MAX_REBERTH'         : 2,     # maksimum jumlah re-berthing per kapal

    # Numerik
    'EPSILON'             : 1e-9,  # nilai sangat kecil untuk perbandingan float
}

# ── Weights & penalties ───────────────────────────────────────────
WEIGHTS = {
    'CATEGORY_WEIGHTS'  : {
        'PASSENGER': 2,
        'RORO'     : 1,
        'CARGO'    : 2,
        'OTHER'    : 1,
    },
    'PENALTY_WAIT_SOP'  : 10,   # penalti jika waktu tunggu melebihi threshold
    'WAIT_THRESHOLD'    : 1.5,   # jam, batas waktu tunggu sebelum kena penalti
    'PENALTY_REBERTHING': 10,   # penalti per kejadian re-berthing
}