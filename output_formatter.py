import pandas as pd


def format_output(df: pd.DataFrame, scenario: str) -> pd.DataFrame:

    df = df.copy()

    if 'SEGMENT_NO' not in df.columns:
        df['SEGMENT_NO'] = 1

    if 'FLAG_SANDAR' not in df.columns:
        # S1: STATUS_KUNJUNGAN selalu 'KUNJUNGAN BARU'
        df['FLAG_SANDAR'] = df.get('STATUS_KUNJUNGAN', 'New Call')

    # Normalise FLAG_SANDAR → English
    flag_map = {
        'KUNJUNGAN BARU': 'New Call',
        'SANDAR ULANG'  : 'Re-berthing',
    }
    df['FLAG_SANDAR'] = (
        df['FLAG_SANDAR']
        .astype(str)
        .str.strip()
        .str.upper()
        .map({k.upper(): v for k, v in flag_map.items()})
        .fillna(df['FLAG_SANDAR'])
    )

    # ── 2. Hitung Berth Duration (jam) dari MULAI & SELESAI ────────────────
    if 'MULAI_SANDAR' in df.columns and 'SELESAI_SANDAR' in df.columns:
        mulai   = pd.to_datetime(df['MULAI_SANDAR'],   errors='coerce')
        selesai = pd.to_datetime(df['SELESAI_SANDAR'], errors='coerce')
        df['_BERTH_DURATION_H'] = (
            (selesai - mulai).dt.total_seconds() / 3600.0
        ).round(4)
    else:
        df['_BERTH_DURATION_H'] = None

    # ── 3. Pilih & urutkan kolom ────────────────────────────────────────────
    col_map = {
        'KODE_KAPAL'         : 'Ship Code',
        'NAMA_KAPAL'         : 'Ship Name',
        'KEDATANGAN'         : 'Arrival Time',
        'MULAI_SANDAR'       : 'Berthing Start',
        'SELESAI_SANDAR'     : 'Berthing End',
        'WAITING_TIME_HOURS' : 'Waiting Time (hrs)',
        '_BERTH_DURATION_H'  : 'Berth Duration (hrs)',
        'NAMA_DERMAGA'       : 'Berth Name',
        'POSISI_START_M'     : 'Start Position (m)',
        'POSISI_END_M'       : 'End Position (m)',
        'FLAG_SANDAR'        : 'Remark',
        'TOTAL_SERVICE_TIME' : 'Total Service Time (hrs)',
        'SEGMENT_NO'         : 'Berth Sequence',
        'LOA'                : 'LOA (m)',
        'DRAFT'              : 'Draft (m)',
    }

    # Hanya ambil kolom yang ada di df
    cols_available = [c for c in col_map if c in df.columns]
    df_out = df[cols_available].rename(columns=col_map)

    # ── 4. Format datetime ──────────────────────────────────────────────────
    for col in ['Arrival Time', 'Berthing Start', 'Berthing End']:
        if col in df_out.columns:
            df_out[col] = pd.to_datetime(df_out[col], errors='coerce') \
                            .dt.strftime('%Y-%m-%d %H:%M')

    # ── 5. Berth Sequence selalu int ────────────────────────────────────────
    if 'Berth Sequence' in df_out.columns:
        df_out['Berth Sequence'] = pd.to_numeric(
            df_out['Berth Sequence'], errors='coerce'
        ).fillna(1).astype(int)

    return df_out.reset_index(drop=True)