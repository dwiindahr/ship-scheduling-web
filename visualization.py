import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def plot_berth_allocation_by_category(
    df_result: pd.DataFrame,
    df_dermaga_info: pd.DataFrame,
    config=None
):
    # ======= KOLOM MAPPING =======
    NAME_COL      = 'KODE_KAPAL'
    T_MULAI_COL   = 'MULAI_SANDAR'
    T_SELESAI_COL = 'SELESAI_SANDAR'
    ASSIGN_COL    = 'DERMAGA_ASSIGNED'
    POS_START_COL = 'POSISI_START_M'
    POS_END_COL   = 'POSISI_END_M'
    CATEGORY_COL  = 'KATEGORI'
    FLAG_COL      = 'FLAG_SANDAR'
    DRAFT_COL     = 'DRAFT'

    CATEGORY_COLORS = {
        'PASSENGER': '#1f77b4',
        'RORO'     : '#2ca02c',
        'CARGO'    : '#9467bd',
        'OTHER'    : '#FF0000',
    }
    DEFAULT_COLOR = '#95A5A6'

    # ======= CROP MAX PER DERMAGA =======
    DERMAGA_MAX_CROP = {
        'JAMRUD SELATAN': 400,
    }

    # ======= PREPROCESS =======
    df_plot = df_result.copy()
    for col in [T_MULAI_COL, T_SELESAI_COL]:
        if col in df_plot.columns:
            df_plot[col] = pd.to_datetime(df_plot[col], errors='coerce')

    df_assigned = df_plot[df_plot[POS_START_COL].notna()].copy()

    # ======= BUILD DERMAGA LAYOUT =======
    df_d = df_dermaga_info.copy()
    df_d['START']     = pd.to_numeric(df_d['START'],     errors='coerce').fillna(0.0)
    df_d['END']       = pd.to_numeric(df_d['END'],       errors='coerce').fillna(0.0)
    df_d['KEDALAMAN'] = pd.to_numeric(df_d['KEDALAMAN'], errors='coerce')

    nama_order = list(dict.fromkeys(df_d['DERMAGA'].tolist()))

    dermaga_total_length = {}
    for nama in nama_order:
        sub = df_d[df_d['DERMAGA'] == nama]
        dermaga_total_length[nama] = float(sub['END'].max())

    id_to_nama = df_d.set_index('ID')['DERMAGA'].to_dict()

    # ======= SUB-KADE BOUNDARIES & KEDALAMAN =======
    sub_kade_boundaries = {}
    sub_kade_segments   = {}
    for nama in nama_order:
        sub = df_d[df_d['DERMAGA'] == nama].sort_values('START')
        boundaries = []
        segments   = []
        for _, srow in sub.iterrows():
            s = float(srow['START'])
            e = float(srow['END'])
            k = srow['KEDALAMAN']
            if s not in boundaries:
                boundaries.append(s)
            if e not in boundaries:
                boundaries.append(e)
            segments.append({'start': s, 'end': e, 'kedalaman': k})
        sub_kade_boundaries[nama] = sorted(set(boundaries))
        sub_kade_segments[nama]   = segments

    # ======= CONVERT TIMES =======
    df_assigned['t_mulai_num']   = df_assigned[T_MULAI_COL].apply(mdates.date2num)
    df_assigned['t_selesai_num'] = df_assigned[T_SELESAI_COL].apply(mdates.date2num)
    df_assigned['duration_days'] = df_assigned['t_selesai_num'] - df_assigned['t_mulai_num']
    df_assigned = df_assigned.dropna(subset=['t_mulai_num', 't_selesai_num'])
    df_assigned = df_assigned[df_assigned['duration_days'] > 0].reset_index(drop=True)

    if len(df_assigned) == 0:
        print("❌ No valid ships to plot.")
        return None

    # ======= HITUNG CROP PER DERMAGA =======
    Y_PAD  = 30.0
    STUB_H = 60.0

    dermagas_with_ships = set()
    min_pos_per_dermaga = {}
    max_pos_per_dermaga = {}

    for _, row in df_assigned.iterrows():
        berth_id = row[ASSIGN_COL]
        nama     = id_to_nama.get(berth_id)
        if nama is None:
            nama = id_to_nama.get(str(berth_id))
        if nama is None and str(berth_id).isdigit():
            nama = id_to_nama.get(int(berth_id))
        if nama is None:
            continue

        if pd.isna(row[POS_START_COL]) or pd.isna(row[POS_END_COL]):
            continue

        pos_start = float(row[POS_START_COL])
        pos_end   = float(row[POS_END_COL])

        kade_rows  = df_d[df_d['ID'].astype(str) == str(berth_id)]
        kade_start = float(kade_rows.iloc[0]['START']) if len(kade_rows) > 0 else 0.0

        dermagas_with_ships.add(nama)

        if nama not in min_pos_per_dermaga:
            min_pos_per_dermaga[nama] = kade_start
            max_pos_per_dermaga[nama] = pos_end
        else:
            min_pos_per_dermaga[nama] = min(min_pos_per_dermaga[nama], kade_start)
            max_pos_per_dermaga[nama] = max(max_pos_per_dermaga[nama], pos_end)

    # ======= BUILD Y LAYOUT =======
    y_positions = {}
    y_offsets   = {}
    y_current   = 0.0
    y_ticks     = []
    y_labels    = []
    gap         = 50.0

    for nama in nama_order:
        L_full = dermaga_total_length.get(nama, 200.0)

        if nama in dermagas_with_ships:
            pad = Y_PAD if nama in DERMAGA_MAX_CROP else 0.0
            y_start_crop = max(0.0, min_pos_per_dermaga[nama] - pad)
            y_end_crop = max(
                min(DERMAGA_MAX_CROP.get(nama, L_full), L_full),
                max_pos_per_dermaga[nama] + pad
            )
            L = y_end_crop - y_start_crop
        else:
            y_start_crop = 0.0
            L            = STUB_H

        y_offsets[nama]   = y_start_crop
        y_positions[nama] = (y_current, y_current + L)
        y_ticks.append(y_current + L / 2)
        display_len = DERMAGA_MAX_CROP.get(nama, dermaga_total_length.get(nama, L))
        y_labels.append(f"{nama}\n({display_len:.0f}m)")
        y_current += L + gap

    # ======= HITUNG t_min / t_max =======
    t_min = df_assigned['t_mulai_num'].min()  - 0.2
    t_max = df_assigned['t_selesai_num'].max() + 0.2

    # ======= HELPER: padded rect (pixel-space) =======
    def get_padded_rect(ax, x, y, w, h, pad_pts):
        d0 = ax.transData.inverted().transform(
            ax.transData.transform((x, y)) + np.array([-pad_pts, -pad_pts])
        )
        d1 = ax.transData.inverted().transform(
            ax.transData.transform((x + w, y + h)) + np.array([pad_pts, pad_pts])
        )
        return d0[0], d0[1], d1[0] - d0[0], d1[1] - d0[1]

    # ======= INIT FIGURE =======
    fig_height = max(8, y_current / 90.0)
    fig, ax    = plt.subplots(figsize=(18, fig_height))

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-20, y_current + gap)

    # Paksa draw agar transData tersedia untuk get_padded_rect
    fig.canvas.draw()

    # ======= HW / LW (Pasang Surut) =======
    has_tide_periods = False
    if config is not None:
        raw_lw    = []
        start_day = pd.Timestamp(mdates.num2date(t_min)).tz_localize(None).normalize()
        end_day   = pd.Timestamp(mdates.num2date(t_max)).tz_localize(None).normalize()

        for day in pd.date_range(start_day, end_day, freq='D'):
            raw_lw.append({
                'start': mdates.date2num(day + pd.Timedelta(hours=config['LOW_TIDE_1_START_H'])),
                'end'  : mdates.date2num(day + pd.Timedelta(hours=config['LOW_TIDE_1_END_H']))
            })
            raw_lw.append({
                'start': mdates.date2num(day + pd.Timedelta(hours=config['LOW_TIDE_2_START_H'])),
                'end'  : mdates.date2num(day + pd.Timedelta(hours=config['LOW_TIDE_2_END_H']))
            })

        raw_lw      = sorted(raw_lw, key=lambda x: x['start'])
        all_periods = []
        prev_end    = t_min

        for lw in raw_lw:
            if lw['start'] > prev_end:
                all_periods.append({'label': 'HT',            'start': prev_end,   'end': lw['start'], 'color': 'white'  })
            all_periods.append(    {'label': 'LT (-1,76 m)', 'start': lw['start'], 'end': lw['end'],   'color': '#AED6F1'})
            prev_end = lw['end']

        if prev_end < t_max:
            all_periods.append({'label': 'HT', 'start': prev_end, 'end': t_max, 'color': 'white'})

        has_tide_periods = True

        y_bot, y_top = ax.get_ylim()
        label_y = y_top - (y_top - y_bot) * 0.02

        min_period_width_for_label = (t_max - t_min) * 0.03

        for p in all_periods:
            x0 = max(p['start'], t_min)
            x1 = min(p['end'],   t_max)
            if x1 <= x0:
                continue
            ax.axvspan(x0, x1, color=p['color'], alpha=0.30, zorder=-10)
            if t_min < x0 < t_max:
                ax.axvline(x=x0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

            # Skip label untuk periode yang terlalu sempit (label akan
            # ditulis berdempetan/tumpang tindih kalau dipaksakan).
            if (x1 - x0) < min_period_width_for_label:
                continue

            label_x = (x0 + x1) / 2
            ax.text(label_x, label_y, p['label'],
                    ha='center', va='top', fontsize=9, fontweight='bold', color='black', zorder=20,
                    clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=p['color'], edgecolor='black', alpha=0.9))

    # ======= BAND DERMAGA =======
    for nama, (ystart, yend) in y_positions.items():
        ax.axhspan(ystart, yend, alpha=0.08, color='lightgray', zorder=0)
        ax.axhline(y=ystart, color='black', linewidth=1.2, zorder=1)
        ax.axhline(y=yend,   color='black', linewidth=1.2, zorder=1)

        offset = y_offsets.get(nama, 0.0)
        for boundary in sub_kade_boundaries.get(nama, []):
            y_boundary = ystart + (boundary - offset)
            if ystart < y_boundary < yend:
                ax.axhline(y=y_boundary, color='black', linewidth=1.0,
                           linestyle='--', alpha=0.6, zorder=2)

    # ======= LABEL BOUNDARY METER + KEDALAMAN =======
    for nama, (ystart, yend) in y_positions.items():
        offset = y_offsets.get(nama, 0.0)

        for boundary in sub_kade_boundaries.get(nama, []):
            y_boundary = ystart + (boundary - offset)
            if ystart < y_boundary < yend:
                ax.text(t_min + 0.005, y_boundary,
                        f'{boundary:.0f}m',
                        ha='left', va='bottom', fontsize=7, color='black',
                        fontstyle='italic', zorder=3)

        for seg in sub_kade_segments.get(nama, []):
            k = seg['kedalaman']
            if pd.isna(k):
                continue

            vis_start = max(seg['start'], offset)
            vis_end   = min(seg['end'], offset + (yend - ystart))
            if vis_end <= vis_start:
                continue

            seg_mid_y = ystart + ((vis_start + vis_end) / 2 - offset)
            ax.text(
                t_min + 0.01, seg_mid_y,
                f"≤{k:.1f}m",
                ha='left', va='center', fontsize=7,
                color='#1a5276', fontweight='bold', fontstyle='italic', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#D6EAF8',
                          edgecolor='#2980B9', alpha=0.85, linewidth=0.8)
            )

    # ======= GAMBAR KAPAL =======
    min_width_days_for_inner_label = 0.04
    min_height_m_for_inner_label   = 15.0
    categories_used  = set()
    has_sandar_ulang = False

    for _, row in df_assigned.iterrows():
        berth_id = row[ASSIGN_COL]
        nama     = id_to_nama.get(berth_id)
        if nama is None:
            nama = id_to_nama.get(str(berth_id))
        if nama is None and str(berth_id).isdigit():
            nama = id_to_nama.get(int(berth_id))
        if nama is None or nama not in y_positions:
            continue

        ystart, yend  = y_positions[nama]
        offset        = y_offsets.get(nama, 0.0)
        berth_vis_len = yend - ystart

        pos_start   = float(row[POS_START_COL])
        pos_end     = float(row[POS_END_COL]) if not pd.isna(row[POS_END_COL]) else pos_start
        height_ship = pos_end - pos_start

        pos_vis = pos_start - offset
        pos_vis = max(0.0, min(pos_vis, berth_vis_len))
        height_ship = min(height_ship, berth_vis_len - pos_vis)

        if height_ship <= 0:
            continue

        y_ship     = ystart + pos_vis
        x_start    = row['t_mulai_num']
        width_days = row['duration_days']

        category = str(row[CATEGORY_COL]).upper().strip()
        color    = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        categories_used.add(category)

        # Highlight Re-berthing
        is_sandar_ulang = str(row.get(FLAG_COL, '')).upper() == 'SANDAR ULANG'
        if is_sandar_ulang:
            has_sandar_ulang = True
            yx, yy, yw, yh = get_padded_rect(ax, x_start, y_ship, width_days, height_ship, pad_pts=-2)
            ax.add_patch(Rectangle((yx, yy), yw, yh,
                                   facecolor='none', edgecolor='yellow',
                                   linewidth=1.5, linestyle='-', zorder=5))

        ax.add_patch(Rectangle((x_start, y_ship), width_days, height_ship,
                                facecolor=color, edgecolor='black',
                                linewidth=0.8, alpha=0.85, zorder=4))

        # Label kapal + draft
        ship_name  = str(row[NAME_COL])
        draft_val  = row.get(DRAFT_COL, None)
        label_text = (
            f"{ship_name}\nd={float(draft_val):.1f}m"
            if (draft_val is not None and not pd.isna(draft_val))
            else ship_name
        )

        if (width_days >= min_width_days_for_inner_label) and (height_ship >= min_height_m_for_inner_label):
            ax.text(x_start + width_days / 2, y_ship + height_ship / 2, label_text,
                    ha='center', va='center', fontsize=7, color='black',
                    weight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                              edgecolor='gray', linewidth=0.5))
        else:
            ax.text(x_start + width_days + 0.01, y_ship + height_ship / 2, label_text,
                    ha='left', va='center', fontsize=7, color='black',
                    weight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                              edgecolor='gray', linewidth=0.5))

    # ======= FORMATTING =======
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())   # AutoDateLocator lebih aman di Streamlit
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlabel('Time', fontsize=13, fontweight='bold')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Berth (Length in meters) | Berthing Position', fontsize=13, fontweight='bold')

    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax.grid(axis='y', linestyle=':',  alpha=0.2, zorder=0)

    # ======= LEGEND =======
    legend_elements = [
        mpatches.Patch(facecolor=CATEGORY_COLORS.get(cat, DEFAULT_COLOR),
                       edgecolor='black', label=cat, linewidth=0.8)
        for cat in sorted(categories_used)
    ]

    if has_sandar_ulang:
        legend_elements.append(mpatches.Patch(
            facecolor='gray', edgecolor='yellow',
            linewidth=2.5, linestyle='-', label='Sandar Ulang'))

    if has_tide_periods:
        legend_elements.append(
            mpatches.Patch(facecolor='white', edgecolor='black', linewidth=0.8,
                           label='HT (High Tide)')
        )
        legend_elements.append(
            mpatches.Patch(facecolor='#AED6F1', edgecolor='black', linewidth=0.8,
                           label='LT (Low Tide)')
        )

    # Garis batas kade
    legend_elements.append(
        Line2D([0], [0], color='black', linewidth=1.0, linestyle='--',
               alpha=0.6, label='Batas Kade')
    )

    if legend_elements:
        ax.legend(handles=legend_elements,
                  loc='upper left', bbox_to_anchor=(1.01, 1),
                  fontsize=9, title="Legend", title_fontsize=10,
                  framealpha=0.95, edgecolor='black')

    plt.title('Ship Berthing Schedule Visualization at Jamrud Terminal',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig