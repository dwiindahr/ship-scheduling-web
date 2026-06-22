import bisect
import random
import time

import numpy as np
import pandas as pd

from config import CONFIG, WEIGHTS

# =============================================================================
# CATEGORY PRIORITY
# =============================================================================
CATEGORY_PRIORITY = {
    'PASSENGER': 1,
    'RORO': 2,
    'CARGO': 1,
    'OTHER': 2
}


# =============================================================================
# TIDE DEPTH PENALTY
# =============================================================================
def get_tide_depth_penalty(start_dt, end_dt, config):
    def normalize_hour(h):
        return h % 24, int(h // 24)

    low_s1, extra_s1 = normalize_hour(config['LOW_TIDE_1_START_H'])
    low_e1, extra_e1 = normalize_hour(config['LOW_TIDE_1_END_H'])
    low_s2, extra_s2 = normalize_hour(config['LOW_TIDE_2_START_H'])
    low_e2, extra_e2 = normalize_hour(config['LOW_TIDE_2_END_H'])

    if not hasattr(start_dt, 'hour'):
        sh = start_dt % 24
        eh = end_dt % 24
        if eh < sh:
            eh += 24

        def check_overlap(s, e, ls, le):
            if le < ls:
                le += 24
            return s <= le and e >= ls

        if check_overlap(sh, eh, low_s1, low_e1) or check_overlap(sh, eh, low_s2, low_e2):
            return config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA']
        return config['UNDER_KEEL_CLEARANCE']

    day     = start_dt.normalize() - pd.Timedelta(days=1)
    day_end = end_dt.normalize()   + pd.Timedelta(days=1)

    while day <= day_end:
        lt1_start = day + pd.Timedelta(hours=low_s1) + pd.Timedelta(days=extra_s1)
        lt1_end   = day + pd.Timedelta(hours=low_e1) + pd.Timedelta(days=extra_e1)
        if start_dt <= lt1_end and end_dt >= lt1_start:
            return config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA']

        lt2_start = day + pd.Timedelta(hours=low_s2) + pd.Timedelta(days=extra_s2)
        lt2_end   = day + pd.Timedelta(hours=low_e2) + pd.Timedelta(days=extra_e2)
        if start_dt <= lt2_end and end_dt >= lt2_start:
            return config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA']

        day += pd.Timedelta(days=1)

    return config['UNDER_KEEL_CLEARANCE']


# =============================================================================
# SHIP FITS BERTH
# =============================================================================
def ship_fits_berth(ship, berth, config):
    EPS = config.get('EPSILON', 1e-6)

    loa   = float(ship.get('LOA',   0))
    draft = float(ship.get('DRAFT', 0))
    kat   = str(ship.get('KATEGORI', 'OTHER')).strip().upper()
    jenis = str(berth.get('JENIS', 'KONTINU')).strip().upper()

    # ── 1. Cek LOA vs PANJANG dermaga ───────────────────────
    panjang = float(berth.get('PANJANG', 0))
    loa_eff = loa + config.get('LOA_MARGIN_DISKRIT', 0) if jenis == 'DISKRIT' else loa
    if loa_eff > panjang + EPS:
        return False

    # ── 2. Cek DRAFT vs KEDALAMAN (dikurangi UKC) ───────────
    kedalaman = float(berth.get('KEDALAMAN', 0))
    ukc       = float(config.get('UNDER_KEEL_CLEARANCE', 0))
    kdl_eff   = kedalaman - ukc
    if draft > kdl_eff + EPS:
        return False

    # ── 3. Cek kategori via flag kolom dermaga ───────────────
    def _flag(col):
        val = berth.get(col, 0)
        try:
            return int(float(str(val).strip() or 0))
        except Exception:
            return 0

    if kat == 'PASSENGER' and _flag('PASSENGER') != 1:
        return False
    if kat == 'RORO'      and _flag('RORO')      != 1:
        return False
    if kat == 'CARGO'     and _flag('CARGO')      != 1:
        return False
    if kat == 'OTHER'     and _flag('OTHER')      != 1:
        return False
    # Kategori tidak dikenal → perlakukan sebagai OTHER
    if kat not in ('PASSENGER', 'RORO', 'CARGO', 'OTHER') and _flag('OTHER') != 1:
        return False

    return True


# =============================================================================
# BUILD TIDE WINDOWS
# =============================================================================
def build_tide_windows(t_start_horizon: pd.Timestamp,
                       t_end_horizon:   pd.Timestamp,
                       config: dict) -> list:
    low_params = [
        (config['LOW_TIDE_1_START_H'], config['LOW_TIDE_1_END_H']),
        (config['LOW_TIDE_2_START_H'], config['LOW_TIDE_2_END_H']),
    ]
    day     = t_start_horizon.normalize()
    horizon = t_end_horizon + pd.Timedelta(days=32)
    windows = []
    while day <= horizon:
        for s_h, e_h in low_params:
            ws = day + pd.Timedelta(hours=s_h)
            we = day + pd.Timedelta(hours=e_h)
            if s_h >= e_h:
                we += pd.Timedelta(days=1)
            windows.append((ws, we))
        day += pd.Timedelta(days=1)
    windows.sort()
    return windows


# =============================================================================
# EMPTY RESULT
# =============================================================================
def _empty_result(ship_idx):
    return {
        'idx'               : ship_idx,
        'DERMAGA_ASSIGNED'  : None,
        'NAMA_DERMAGA'      : None,
        'MULAI_SANDAR'      : pd.NaT,
        'SELESAI_SANDAR'    : pd.NaT,
        'WAITING_TIME_HOURS': np.nan,
        'POSISI_START_M'    : np.nan,
        'POSISI_END_M'      : np.nan,
        'TIDE_FLAG'         : 0,
        'IS_LATE'           : 0,
        'REBERTHING'        : 0,
    }


# =============================================================================
# NEXT SAFE START
# =============================================================================
def _next_safe_start(t_candidate, svc_td, tide_windows, config):
    if not tide_windows:
        return t_candidate

    deadline = t_candidate + pd.Timedelta(days=30)
    t = t_candidate
    tw_starts = [w[0] for w in tide_windows]

    for _ in range(len(tide_windows) * 2 + 4):
        if t > deadline:
            return None

        t_end = t + svc_td

        lo = bisect.bisect_right(tw_starts, t) - 1
        lo = max(lo, 0)

        overlap_we = None
        for ws, we in tide_windows[lo:]:
            if ws >= t_end:
                break
            if we <= t:
                continue
            if overlap_we is None or we > overlap_we:
                overlap_we = we

        if overlap_we is None:
            return t
        t = overlap_we

    return None


# =============================================================================
# PREPARE DATA
# =============================================================================
def _prepare_data(df_kapal_raw, df_dermaga, config):
    df = df_kapal_raw.copy()
    df['ID_KUNJUNGAN'] = df['ID_KUNJUNGAN'].astype(str)

    if 'ID_KUNJUNGAN' in df.columns and 'BERTH_PART' in df.columns:
        if df.duplicated(subset=['ID_KUNJUNGAN', 'BERTH_PART']).any():
            n = df.duplicated(subset=['ID_KUNJUNGAN', 'BERTH_PART']).sum()
            print(f"[WARNING] {n} baris duplikat (ID_KUNJUNGAN & BERTH_PART) dibuang.")
            df = df.drop_duplicates(subset=['ID_KUNJUNGAN', 'BERTH_PART'], keep='first')

    if 'BERTH_PART' not in df.columns:
        df['BERTH_PART'] = 1
    else:
        df['BERTH_PART'] = pd.to_numeric(df['BERTH_PART'], errors='coerce').fillna(1).astype(int)

    df = df.sort_values(['ID_KUNJUNGAN', 'BERTH_PART']).reset_index(drop=True)

    approach_td = pd.Timedelta(hours=config['APPROACH_TIME'])
    df['MULAI_SANDAR_AWAL'] = df['KEDATANGAN'] + approach_td

    df['_priority'] = df['KATEGORI'].map(CATEGORY_PRIORITY).fillna(99)

    berths       = df_dermaga.to_dict('records')
    id_to_parent = {str(b['ID']): str(b.get('DERMAGA', b['ID'])) for b in berths}
    id_to_nama   = {str(b['ID']): str(b.get('DERMAGA', b['ID'])) for b in berths}
    eligibility  = {
        i: [b for b in berths if ship_fits_berth(ship, b, config)]
        for i, ship in df.iterrows()
    }

    return df, berths, id_to_parent, id_to_nama, eligibility


# =============================================================================
# IS BETTER
# =============================================================================
def _is_better(t_start, berth_id, best, kade_occ, berths_info=None):
    if best is None:
        return True

    best_t_start  = best[0]
    best_berth_id = best[4]

    if t_start < best_t_start:
        return True

    if t_start == best_t_start:
        current_load = len(kade_occ.get(str(berth_id), set()))
        best_load    = len(kade_occ.get(str(best_berth_id), set()))

        if current_load < best_load:
            return True

        if current_load == best_load and berths_info is not None:
            curr_group = berths_info.get(str(berth_id), {}).get('group', '')
            best_group = berths_info.get(str(best_berth_id), {}).get('group', '')
            if curr_group != best_group:
                curr_group_load = sum(
                    len(kade_occ.get(bid, set()))
                    for bid, info in berths_info.items()
                    if info.get('group') == curr_group
                )
                best_group_load = sum(
                    len(kade_occ.get(bid, set()))
                    for bid, info in berths_info.items()
                    if info.get('group') == best_group
                )
                return curr_group_load < best_group_load

    return False


# =============================================================================
# MAKE VARIATION PARAMS
# =============================================================================
def _make_variation_params(rnd):
    strategy = rnd.choice([
        'default',
        'default',
        'selatan_first',
        'least_loaded'
    ])
    return {
        'shuffle_berth_prob'   : rnd.random(),
        'shuffle_ship_in_group': rnd.random() < 0.30,
        'berth_strategy'       : strategy,
    }


# =============================================================================
# EVALUATE SOLUTION METRICS
# =============================================================================
def evaluate_solution_metrics(df_sol):
    assigned   = df_sol[df_sol['DERMAGA_ASSIGNED'].notna()].copy()
    unassigned = int(df_sol['DERMAGA_ASSIGNED'].isna().sum())

    if not assigned.empty:
        total_wait = float(assigned['WAITING_TIME_HOURS'].fillna(0).sum())
        cat_dist   = assigned['KATEGORI'].value_counts().to_dict()
    else:
        total_wait = 0.0
        cat_dist   = {}

    return {
        'num_assigned'    : len(assigned),
        'num_unassigned'  : unassigned,
        'total_wait_hours': round(total_wait, 4),
        'category_dist'   : cat_dist,
    }