import pandas as pd
import numpy as np
from config import CONFIG, WEIGHTS, TERMINAL_DATA_PATH, TIDAL_DATA_PATH, REBERTHING_PART_DURATION


# =========================
# COLUMN MAPPING
# =========================
COLUMN_MAP = {
    # English
    "ship name"         : "NAMA_KAPAL",
    "loa"               : "LOA",
    "draft"             : "DRAFT",
    "arrival time"      : "KEDATANGAN",
    "total service time": "TOTAL_SERVICE_TIME",
    "category"          : "KATEGORI",
    # Indonesian (backward compat)
    "nama kapal"        : "NAMA_KAPAL",
    "kedatangan"        : "KEDATANGAN",
    "kategori"          : "KATEGORI",
}

REQUIRED_COLUMNS = list(dict.fromkeys([
    "NAMA_KAPAL", "LOA", "DRAFT", "KEDATANGAN", "TOTAL_SERVICE_TIME", "KATEGORI"
]))
VALID_CATEGORIES = {"PASSENGER", "RORO", "CARGO", "OTHER"}
MAX_ARRIVAL_DAYS = 3


# =========================
# TIDAL CONFIG
# =========================
def load_tidal_config(tidal_path: str = TIDAL_DATA_PATH) -> list[str]:
    """
    Read pasang_surut.csv, calculate average low tide hours (1st and 2nd),
    then update CONFIG with low tide window periods.

    Returns
    -------
    errors : list of str (empty = success)
    """
    errors = []

    try:
        pasut = pd.read_csv(tidal_path)
    except Exception as e:
        errors.append(f"Failed to read tidal data file: {e}")
        return errors

    if "Waktu" not in pasut.columns or "Jenis" not in pasut.columns:
        errors.append("Tidal data file must contain 'Waktu' and 'Jenis' columns.")
        return errors

    pasut["Waktu"] = pd.to_datetime(pasut["Waktu"])

    # ── Calculate average low tide peak hours ─────────────────────
    pasut["Tanggal"]   = pasut["Waktu"].dt.date
    pasut["Jam_Float"] = pasut["Waktu"].dt.hour + pasut["Waktu"].dt.minute / 60.0

    surut = pasut[pasut["Jenis"].str.upper() == "SURUT"].copy()

    if surut.empty:
        errors.append("No records with Jenis='SURUT' found in tidal data file.")
        return errors

    surut = surut.sort_values("Waktu")
    surut["Urutan"] = surut.groupby("Tanggal").cumcount() + 1

    s1 = surut[surut["Urutan"] == 1]["Jam_Float"]
    s2 = surut[surut["Urutan"] == 2]["Jam_Float"]

    if s1.empty:
        errors.append("No first low tide data found.")
        return errors
    if s2.empty:
        errors.append("No second low tide data found.")
        return errors

    s1_mean = s1.mean()
    s2_mean = s2.mean()

    # ── Calculate low tide duration from High→Low transitions ─────
    pasut_sorted = pasut.sort_values("Waktu").reset_index(drop=True)
    durations_1  = []
    durations_2  = []
    cycle_count  = 0
    last_date    = None

    for i in range(len(pasut_sorted) - 1):
        if (pasut_sorted.loc[i, "Jenis"].strip().capitalize() == "Pasang" and
                pasut_sorted.loc[i + 1, "Jenis"].strip().capitalize() == "Surut"):
            current_date = pasut_sorted.loc[i, "Waktu"].date()
            if current_date != last_date:
                cycle_count = 0
                last_date   = current_date

            dur = pasut_sorted.loc[i + 1, "Waktu"] - pasut_sorted.loc[i, "Waktu"]
            h   = dur.total_seconds() / 3600

            if cycle_count == 0:
                durations_1.append(h)
            else:
                durations_2.append(h)

            cycle_count += 1

    if not durations_1:
        errors.append("No High→Low tide transition (1st) found in tidal data.")
        return errors
    if not durations_2:
        errors.append("No High→Low tide transition (2nd) found in tidal data.")
        return errors

    rata_1 = sum(durations_1) / len(durations_1)
    rata_2 = sum(durations_2) / len(durations_2)

    CONFIG.update({
        "LOW_TIDE_1_START_H": round(float(s1_mean - rata_1 / 2), 2),
        "LOW_TIDE_1_END_H"  : round(float(s1_mean + rata_1 / 2), 2),
        "LOW_TIDE_2_START_H": round(float(s2_mean - rata_2 / 2), 2),
        "LOW_TIDE_2_END_H"  : round(float(s2_mean + rata_2 / 2), 2),
    })

    return errors


def check_arrival_day_span(df: pd.DataFrame, max_days: int = MAX_ARRIVAL_DAYS) -> dict:
    """
    Cek jumlah hari kalender unik pada KEDATANGAN.

    Returns
    -------
    dict dengan keys:
        - exceeds (bool)
        - total_days (int)
        - max_days (int)
        - unique_dates (list of date, terurut)
    """
    unique_dates = sorted(df["KEDATANGAN"].dt.date.unique())
    total_days = len(unique_dates)
    return {
        "exceeds": total_days > max_days,
        "total_days": total_days,
        "max_days": max_days,
        "unique_dates": unique_dates,
    }


def limit_to_n_days(df: pd.DataFrame, n_days: int = MAX_ARRIVAL_DAYS) -> pd.DataFrame:
    """
    Potong df agar hanya berisi kapal dengan KEDATANGAN pada n_days
    hari kalender pertama (berdasarkan tanggal unik terkecil).
    """
    unique_dates = sorted(df["KEDATANGAN"].dt.date.unique())
    keep_dates = set(unique_dates[:n_days])
    result = df[df["KEDATANGAN"].dt.date.isin(keep_dates)].copy()
    result.reset_index(drop=True, inplace=True)
    return result


# =========================
# MAIN PREPROCESSING
# =========================
def preprocess(uploaded_file, scenario: str) -> tuple[pd.DataFrame, list[str], dict]:
    errors = []

    # ── 1. Load tidal config ──────────────────────────────────────
    tidal_errors = load_tidal_config()
    errors.extend(tidal_errors)
    if any("Failed" in e or "No " in e for e in tidal_errors):
        return pd.DataFrame(), errors, {}

    # ── 2. Read file ──────────────────────────────────────────────
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        return pd.DataFrame(), errors + [f"Failed to read Excel file: {e}"], {}

    if df.empty:
        return pd.DataFrame(), errors + ["Excel file is empty."], {}

    # ── 3. Normalize column names ─────────────────────────────────
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    df.rename(columns=COLUMN_MAP, inplace=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return pd.DataFrame(), errors + [
            f"The following columns were not found: {', '.join(missing)}. "
            f"Please ensure the file contains: Ship Name, LOA, Draft, "
            f"Arrival Time, Total Service Time, Category."
        ], {}

    df = df[REQUIRED_COLUMNS].copy()

    # ── 4. Drop fully empty rows ──────────────────────────────────
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 5. Validate & convert data types ──────────────────────────

    # 5a. KEDATANGAN → datetime
    df["KEDATANGAN"] = pd.to_datetime(df["KEDATANGAN"], errors="coerce")
    bad_dt_idx = df[df["KEDATANGAN"].isna()].index.tolist()
    if bad_dt_idx:
        rows = [i + 2 for i in bad_dt_idx]
        errors.append(
            f"Invalid Arrival Time at rows: {rows}. "
            f"Please use a valid date format (e.g. 2024-06-01 08:00)."
        )
        df = df[df["KEDATANGAN"].notna()].copy()

    # 5b. LOA, DRAFT, TOTAL_SERVICE_TIME → float > 0
    col_display = {
        "LOA"               : "LOA",
        "DRAFT"             : "Draft",
        "TOTAL_SERVICE_TIME": "Total Service Time",
    }
    for col in ["LOA", "DRAFT", "TOTAL_SERVICE_TIME"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad_idx = df[df[col].isna() | (df[col] <= 0)].index.tolist()
        if bad_idx:
            rows = [i + 2 for i in bad_idx]
            errors.append(
                f"Invalid {col_display[col]} at rows: {rows}. "
                f"Value must be a positive number."
            )
        df = df[df[col].notna() & (df[col] > 0)].copy()

    # 5c. KATEGORI → uppercase, validate
    df["KATEGORI"] = df["KATEGORI"].astype(str).str.strip().str.upper()
    df["KATEGORI"] = df["KATEGORI"].replace({
        "RO-RO": "RORO",
        "RO RO": "RORO",
        "RO_RO": "RORO",
    })
    invalid_cat = ~df["KATEGORI"].isin(VALID_CATEGORIES)
    if invalid_cat.sum() > 0:
        bad_vals = df.loc[invalid_cat, "KATEGORI"].unique().tolist()
        rows = [i + 2 for i in df[invalid_cat].index.tolist()]
        errors.append(
            f"Invalid Category at rows: {rows}. "
            f"Unrecognized values: {bad_vals}. "
            f"Valid values are: Passenger, RoRo, Cargo, Other."
        )
        df = df[~invalid_cat].copy()

    if df.empty:
        return pd.DataFrame(), errors + ["No valid data remaining after cleaning."], {}

    # ── 6. Normalize strings ──────────────────────────────────────
    df["NAMA_KAPAL"] = df["NAMA_KAPAL"].astype(str).str.strip().str.upper()

    # ── 7. Sort by arrival time ───────────────────────────────────
    df.sort_values("KEDATANGAN", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 7b. Check arrival day span ─────────────────────────────────
    arrival_info = check_arrival_day_span(df, MAX_ARRIVAL_DAYS)

    # ── 8. Add ID_KUNJUNGAN ───────────────────────────────────────
    df.insert(0, "ID_KUNJUNGAN", range(1, len(df) + 1))

    # ── 9. Auto-generate KODE_KAPAL per category ─────────────────
    df["KODE_KAPAL"] = None
    for kategori in df["KATEGORI"].unique():
        subset_idx = df[df["KATEGORI"] == kategori].index
        prefix = kategori[0].upper()
        codes = [f"{prefix}{str(i + 1).zfill(3)}" for i in range(len(subset_idx))]
        df.loc[subset_idx, "KODE_KAPAL"] = codes

    # ── 10. Add scenario columns ──────────────────────────────────
    if scenario == "single":
        df = _add_single_berthing_cols(df)
    elif scenario == "reberthing":
        df = _add_reberthing_cols(df)
    else:
        return pd.DataFrame(), errors + [f"Unknown scenario: '{scenario}'."], {}

    return df, errors, arrival_info


# =========================
# SINGLE BERTHING
# =========================
def _add_single_berthing_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BERTH_PART"] = 1
    df["BERTH_TIME"] = df["TOTAL_SERVICE_TIME"].astype(float)
    return df


# =========================
# RE-BERTHING (SPLIT PARTS)
# =========================
def _add_reberthing_cols(df: pd.DataFrame) -> pd.DataFrame:
    return split_berthing_parts(df, time_col="TOTAL_SERVICE_TIME", part_duration=REBERTHING_PART_DURATION)


def split_berthing_parts(
    df: pd.DataFrame,
    time_col: str = "TOTAL_SERVICE_TIME",
    part_duration: float = 3.0,
) -> pd.DataFrame:
    new_rows = []

    for _, row in df.iterrows():
        t_total  = row[time_col]
        kategori = str(row.get("KATEGORI", "")).strip().upper()

        # Passenger is not split
        if pd.isna(t_total) or t_total <= 0 or kategori == "PASSENGER":
            r = row.to_dict()
            r["BERTH_PART"] = 1
            r["BERTH_TIME"] = float(t_total) if not pd.isna(t_total) else 0.0
            new_rows.append(r)
            continue

        full_parts = int(t_total // part_duration)
        remainder  = round(t_total % part_duration, 6)
        part_idx   = 1

        if full_parts == 0:
            r = row.to_dict()
            r["BERTH_PART"] = 1
            r["BERTH_TIME"] = round(t_total, 6)
            new_rows.append(r)

        elif remainder == 0:
            for _ in range(full_parts):
                r = row.to_dict()
                r["BERTH_PART"] = part_idx
                r["BERTH_TIME"] = part_duration
                new_rows.append(r)
                part_idx += 1

        else:
            for i in range(full_parts):
                r = row.to_dict()
                r["BERTH_PART"] = part_idx
                is_last = (i == full_parts - 1)
                r["BERTH_TIME"] = round(
                    part_duration + remainder if is_last else part_duration, 6
                )
                new_rows.append(r)
                part_idx += 1

    result = pd.DataFrame(new_rows)
    result.reset_index(drop=True, inplace=True)
    return result


def load_dermaga():
    dermaga = pd.read_excel(TERMINAL_DATA_PATH)

    dermaga.columns = (
        dermaga.columns
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
    )

    for col in ["START", "END", "PANJANG", "KEDALAMAN"]:
        dermaga[col] = pd.to_numeric(dermaga[col], errors="coerce")

    return dermaga


# =========================
# BERTH SUITABILITY VALIDATION
# =========================
def _berth_loa_ok(loa: float, berth: dict, config: dict, eps: float) -> bool:
    jenis = str(berth.get("JENIS", "KONTINU")).strip().upper()
    panjang = float(berth.get("PANJANG", 0))
    loa_eff = loa + config.get("LOA_MARGIN_DISKRIT", 0) if jenis == "DISKRIT" else loa
    return loa_eff <= panjang + eps


def _berth_draft_ok_at_high_tide(draft: float, berth: dict, config: dict, eps: float) -> bool:
    kedalaman = float(berth.get("KEDALAMAN", 0))
    ukc = float(config.get("UNDER_KEEL_CLEARANCE", 0))
    kdl_eff_pasang = kedalaman - ukc
    return draft <= kdl_eff_pasang + eps


def _berth_draft_ok_at_low_tide(draft: float, berth: dict, config: dict, eps: float) -> bool:
    kedalaman = float(berth.get("KEDALAMAN", 0))
    ukc = float(config.get("UNDER_KEEL_CLEARANCE", 0))
    tide_delta = float(config.get("TIDE_DELTA", 0))
    kdl_eff_surut = kedalaman - tide_delta - ukc
    return draft <= kdl_eff_surut + eps


def check_no_eligible_berth(df_kapal: pd.DataFrame, df_dermaga: pd.DataFrame, config: dict) -> list[str]:
    eps = float(config.get("EPSILON", 1e-6))
    berths = df_dermaga.to_dict("records")
    errors = []

    for _, ship in df_kapal.iterrows():
        loa      = float(ship.get("LOA", 0))
        draft    = float(ship.get("DRAFT", 0))
        nama     = ship.get("NAMA_KAPAL", "Unknown")
        kategori = str(ship.get("KATEGORI", "")).strip().upper()

        has_eligible = any(
            _berth_category_ok(kategori, berth)             # ← tambah ini
            and _berth_loa_ok(loa, berth, config, eps)
            and _berth_draft_ok_at_high_tide(draft, berth, config, eps)
            for berth in berths
        )

        if not has_eligible:
            errors.append(
                f"Ship '{nama}' (Category: {kategori}, LOA {loa:g} m, Draft {draft:g} m) "
                f"does not fit any berth at Jamrud Terminal, even at high tide. "
                f"Please check the LOA/Draft values or contact the terminal "
                f"for alternative arrangements."
            )

    return errors


def _get_high_tide_windows(config: dict) -> list[float]:
    required = [
        "LOW_TIDE_1_START_H", "LOW_TIDE_1_END_H",
        "LOW_TIDE_2_START_H", "LOW_TIDE_2_END_H",
    ]
    if not all(k in config for k in required):
        return [24.0]  # fallback

    s1_start = config["LOW_TIDE_1_START_H"]
    s1_end   = config["LOW_TIDE_1_END_H"]
    s2_start = config["LOW_TIDE_2_START_H"]
    s2_end   = config["LOW_TIDE_2_END_H"]

    pasang_1 = s1_start - 0.0           # 00:00 → awal surut 1
    pasang_2 = s2_start - s1_end        # akhir surut 1 → awal surut 2
    pasang_3 = (24.0 - s2_end) + s1_start  # akhir surut 2 → surut 1 hari berikut
    #           (sisa malam)     + (pagi hari berikut sebelum surut 1 lagi)

    windows = [w for w in [pasang_1, pasang_2, pasang_3] if w > 0]
    return windows

def _berth_category_ok(kategori: str, berth: dict) -> bool:
    """Cek apakah kategori kapal diizinkan di berth ini."""
    col_map = {
        "PASSENGER": "PASSENGER",
        "RORO"     : "RORO",
        "CARGO"    : "CARGO",
        "OTHER"    : "OTHER",
    }
    col = col_map.get(str(kategori).strip().upper())
    if col is None:
        return False
    return bool(berth.get(col, 0))


def check_single_berth_insufficient_time(
    df_kapal: pd.DataFrame, df_dermaga: pd.DataFrame, config: dict
) -> list[str]:
    eps = float(config.get("EPSILON", 1e-6))
    berths = df_dermaga.to_dict("records")
    high_tide_windows = _get_high_tide_windows(config)
    max_single_pasang = max(high_tide_windows) if high_tide_windows else 24.0
    errors = []

    for _, ship in df_kapal.iterrows():
        loa          = float(ship.get("LOA", 0))
        draft        = float(ship.get("DRAFT", 0))
        service_time = float(ship.get("TOTAL_SERVICE_TIME", 0))
        nama         = ship.get("NAMA_KAPAL", "Unknown")
        kategori     = str(ship.get("KATEGORI", "")).strip().upper()

        safe_at_high_tide = [
            berth for berth in berths
            if _berth_category_ok(kategori, berth)          # ← tambah ini
            and _berth_loa_ok(loa, berth, config, eps)
            and _berth_draft_ok_at_high_tide(draft, berth, config, eps)
        ]

        if not safe_at_high_tide:
            continue

        unsafe_at_low_tide = [
            berth for berth in safe_at_high_tide
            if not _berth_draft_ok_at_low_tide(draft, berth, config, eps)
        ]

        if not unsafe_at_low_tide:
            continue

        # Semua berth eligible hanya aman saat pasang
        if len(unsafe_at_low_tide) != len(safe_at_high_tide):
            continue

        if service_time <= max_single_pasang + eps:
            continue

        nama_dermaga_list = ", ".join(
            b.get("DERMAGA", b.get("ID", "Unknown")) for b in safe_at_high_tide
        )

        s1s = config.get("LOW_TIDE_1_START_H", 0)
        s1e = config.get("LOW_TIDE_1_END_H", 0)
        s2s = config.get("LOW_TIDE_2_START_H", 0)
        s2e = config.get("LOW_TIDE_2_END_H", 0)
        window_labels = [f"00:00–{s1s:.1f}h ({high_tide_windows[0]:.1f} jam)"]
        if len(high_tide_windows) > 1:
            window_labels.append(f"{s1e:.1f}h–{s2s:.1f}h ({high_tide_windows[1]:.1f} jam)")
        if len(high_tide_windows) > 2:
            window_labels.append(f"{s2e:.1f}h–24:00+{s1s:.1f}h ({high_tide_windows[2]:.1f} jam)")

        errors.append(
            f"'{nama}' (Draft {draft:g} m) only fits [{nama_dermaga_list}] at high tide, "
            f"but service time ({service_time:g} hrs) exceeds max tide window "
            f"(~{max_single_pasang:.1f} hrs). Use RE-BERTHING scenario."
        )

    return errors