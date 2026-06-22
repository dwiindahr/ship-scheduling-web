import bisect
import random
import time

import numpy as np
import pandas as pd

from config import CONFIG, WEIGHTS
from optimizer_utils import (
    get_tide_depth_penalty,
    ship_fits_berth,
    build_tide_windows,
    _empty_result,
    _next_safe_start,
    _prepare_data,
    _make_variation_params,
    CATEGORY_PRIORITY,
)

def _occ_add(ends_ns, occs, occ):
    occ = dict(occ)
    occ['_start_ns'] = pd.Timestamp(occ['start']).value
    occ['_end_ns']   = pd.Timestamp(occ['end']).value

    val = occ['_end_ns']
    idx = bisect.bisect_left(ends_ns, val)
    ends_ns.insert(idx, val)
    occs.insert(idx, occ)


def _occ_active(ends_ns, occs, t_start, t_end):
    ts_start = pd.Timestamp(t_start).value
    ts_end   = pd.Timestamp(t_end).value
    lo       = bisect.bisect_right(ends_ns, ts_start)
    result   = []
    for occ in occs[lo:]:
        start_ns = occ.get('_start_ns', pd.Timestamp(occ['start']).value)
        if start_ns < ts_end:
            result.append(occ)
    return result


def _build_util_cache(berths, id_to_parent, parent_ends, parent_occs, kade_capacities):
    cache = {}

    for b in berths:
        bid    = str(b['ID'])
        parent = id_to_parent[bid]

        occs_bid = [
            o for o in parent_occs[parent]
            if str(o.get('bid', bid)) == bid
        ]

        beban_kunjungan = len({
            str(o.get('kid'))
            for o in occs_bid
            if o.get('kid')
        })

        beban_part = len(occs_bid)

        durasi_total = 0.0
        for o in occs_bid:
            durasi_total += (
                pd.Timestamp(o['end']) - pd.Timestamp(o['start'])
            ).total_seconds() / 3600.0

        utilisasi = float(durasi_total)

        cache[(parent, bid)] = (
            utilisasi,
            beban_kunjungan,
            beban_part,
        )

    return cache

def _build_output_df2(df, results):
    df_out = df.copy()
    df_res = pd.DataFrame(results)

    if df_res.empty:
        return df_out

    df_res = df_res.set_index('idx')

    cols_to_add = [
        'DERMAGA_ASSIGNED',
        'NAMA_DERMAGA',
        'MULAI_SANDAR',
        'SELESAI_SANDAR',
        'WAITING_TIME_HOURS',
        'POSISI_START_M',
        'POSISI_END_M',
        'TIDE_FLAG',
        'IS_LATE',
        'REBERTHING',
        'SEGMENT_NO_PLAN',
    ]

    for col in cols_to_add:
        if col in df_res.columns:
            df_out[col] = df_res[col]

    return df_out


def generate_sol2d_s2(df_sol):
    is_merged = 'BERTH_PART_LIST' in df_sol.columns

    bids = sorted(df_sol['DERMAGA_ASSIGNED'].dropna().unique())
    format_a = {}
    format_b = {}

    for bid in bids:
        rows = df_sol[df_sol['DERMAGA_ASSIGNED'] == bid].copy()
        if rows.empty:
            continue

        rows_sorted = rows.sort_values('MULAI_SANDAR')

        seen_a = {}
        for _, row in rows_sorted.iterrows():
            kid = str(row['ID_KUNJUNGAN'])
            if kid not in seen_a:
                seen_a[kid] = None
        format_a[bid] = list(seen_a.keys())

        seen_b = {}
        for _, row in rows_sorted.iterrows():
            if is_merged:
                parts = row['BERTH_PART_LIST']
                if not isinstance(parts, list):
                    parts = [row.get('BERTH_PART', 1)]
                for part in parts:
                    key = f"{row['ID_KUNJUNGAN']}_{int(part)}"
                    if key not in seen_b:
                        seen_b[key] = None
            else:
                key = f"{row['ID_KUNJUNGAN']}_{int(row['BERTH_PART'])}"
                if key not in seen_b:
                    seen_b[key] = None
        format_b[bid] = list(seen_b.keys())

    return {'format_a': format_a, 'format_b': format_b}

def _build_run_params_s2(rnd, vparams, df, eligibility):
    groups = {}
    for i in df.index:
        groups.setdefault(int(df.loc[i, '_priority']), []).append(i)

    ship_order = []
    for p in sorted(groups):
        grp = groups[p][:]

        if vparams['shuffle_ship_in_group']:
            sub_groups = {}
            for i in grp:
                kid = str(df.loc[i, 'ID_KUNJUNGAN']) if 'ID_KUNJUNGAN' in df.columns else str(i)
                sub_groups.setdefault(kid, []).append(i)

            for kid in sub_groups:
                sub_groups[kid].sort(key=lambda i: int(df.loc[i, 'BERTH_PART']))

            kunjungan_keys = list(sub_groups.keys())
            rnd.shuffle(kunjungan_keys)

            for kid in kunjungan_keys:
                ship_order.extend(sub_groups[kid])
        else:
            grp.sort(key=lambda i: (
                df.loc[i, 'KEDATANGAN'],
                int(df.loc[i, 'BERTH_PART']),
            ))
            ship_order.extend(grp)

    berth_order_per_ship = {}
    for i in df.index:
        elig = eligibility.get(i, [])[:]
        if rnd.random() < vparams['shuffle_berth_prob']:
            rnd.shuffle(elig)
        berth_order_per_ship[i] = elig

    return ship_order, berth_order_per_ship


def _compute_kade_capacities(berths, horizon_hours=96.0):
    cap = {}
    for b in berths:
        bid     = str(b['ID'])
        panjang = max(float(b['END']) - float(b['START']), 1.0)
        cap[bid] = panjang * horizon_hours
    return cap


def find_slot_ch2(ship, berth, parent_occupancy, config,
                  allowed_starts=None, preferred_pos=None,
                  strict_preferred=False):

    EPS        = float(config.get('EPSILON', 1e-6))
    safe       = float(config.get('CONTINUOUS_GAP_M', 0.0))
    jenis      = str(berth['JENIS']).upper().strip()
    loa        = float(ship['LOA'])
    draft      = float(ship['DRAFT'])
    kdl        = float(berth['KEDALAMAN'])
    earliest   = pd.Timestamp(ship['MULAI_SANDAR_AWAL']).round('s')
    service    = float(ship['BERTH_TIME'])
    svc_td     = pd.Timedelta(hours=service)
    kade_start = float(berth['START'])
    kade_end   = float(berth['END'])
    step_td    = pd.Timedelta(minutes=int(config.get('TIME_STEP_MINUTES', 30)))
    max_iter   = int(config.get('MAX_SLOT_ITER', 500))
    ukc        = float(config['UNDER_KEEL_CLEARANCE'])
    tide_delta = float(config['TIDE_DELTA'])

    if isinstance(parent_occupancy, tuple):
        ends_ns, occs = parent_occupancy

        def _active_occ(t_start, t_end):
            return _occ_active(ends_ns, occs, t_start, t_end)

        all_occ = occs
    else:
        def _active_occ(t_start, t_end):
            return [
                occ for occ in parent_occupancy
                if pd.Timestamp(occ['start']) < t_end
                and pd.Timestamp(occ['end']) > t_start
            ]
        all_occ = parent_occupancy

    _depth_cache: dict = {}

    def _penalty_and_flag(t_start, t_end):
        key = (pd.Timestamp(t_start).value, pd.Timestamp(t_end).value)
        if key not in _depth_cache:
            penalty  = float(get_tide_depth_penalty(t_start, t_end, config))
            low_pen  = ukc + tide_delta
            tf       = 1 if abs(penalty - low_pen) < EPS else 0
            kdl_real = kdl - ukc - tf * tide_delta
            ok       = draft <= kdl_real + EPS
            _depth_cache[key] = (ok, tf)
        return _depth_cache[key]

    def _depth_ok(t_start, t_end):
        return _penalty_and_flag(t_start, t_end)[0]

    def _tide_flag(t_start, t_end):
        return _penalty_and_flag(t_start, t_end)[1]

    def _build_pos_candidates(active_sorted):
        """
        Kandidat posisi untuk kade KONTINU.

        Prinsip:
        - Tetap isi dari START dulu.
        - Kalau ada kapal dari ID dermaga sebelah dalam parent yang sama,
          safe gap tetap dihormati.
        - Gap di-clip ke range kade ini: [kade_start, kade_end].
        """

        candidates = []

        # Jika tidak ada occupancy aktif, isi dari START dulu
        if not active_sorted:
            candidates.extend([
                kade_start,
                kade_end - loa,
            ])
        else:
            blocked = []

            for occ in active_sorted:
                occ_sp = float(occ['start_pos'])
                occ_ep = float(occ['end_pos'])

                block_start = occ_sp - safe
                block_end   = occ_ep + safe

                # Kalau block sama sekali di luar range kade ini, skip
                if block_end <= kade_start + EPS:
                    continue
                if block_start >= kade_end - EPS:
                    continue

                block_start = max(kade_start, block_start)
                block_end   = min(kade_end, block_end)

                if block_end > block_start + EPS:
                    blocked.append((block_start, block_end))

            blocked = sorted(blocked, key=lambda x: x[0])

            merged_blocked = []

            for bs, be in blocked:
                if not merged_blocked:
                    merged_blocked.append([bs, be])
                else:
                    last_bs, last_be = merged_blocked[-1]
                    if bs <= last_be + EPS:
                        merged_blocked[-1][1] = max(last_be, be)
                    else:
                        merged_blocked.append([bs, be])

            gaps = []
            cursor = kade_start

            for bs, be in merged_blocked:
                if bs - cursor >= loa - EPS:
                    gaps.append((cursor, bs))
                cursor = max(cursor, be)

            if kade_end - cursor >= loa - EPS:
                gaps.append((cursor, kade_end))

            for g_start, g_end in gaps:
                left_pos = g_start
                right_pos = g_end - loa
                center_pos = g_start + (g_end - g_start - loa) / 2

                candidates.extend([
                    left_pos,
                    center_pos,
                    right_pos,
                ])

        if preferred_pos is not None and not pd.isna(preferred_pos):
            pf = float(preferred_pos)
            end_pf = pf + loa

            if pf >= kade_start - EPS and end_pf <= kade_end + EPS:
                if strict_preferred:
                    return [pf]
                candidates = [pf] + candidates

        clean = []
        seen = set()

        for p in candidates:
            p = round(float(p), 6)

            if p in seen:
                continue

            if p < kade_start - EPS:
                continue

            if p + loa > kade_end + EPS:
                continue

            seen.add(p)
            clean.append(p)

        return clean

    def _try_at(t_start):
        t_start = pd.Timestamp(t_start).round('s')
        t_end   = (t_start + svc_td).round('s')

        if t_start < earliest:
            return None

        if not _depth_ok(t_start, t_end):
            return None

        active = _active_occ(t_start, t_end)

        if jenis == 'DISKRIT':
            pos     = kade_start + float(config.get('LOA_MARGIN_DISKRIT', 0))
            end_pos = pos + loa

            if end_pos > kade_end + EPS or active:
                return None

            return (t_start, t_end, round(pos, 2), round(end_pos, 2))

        active_sorted  = sorted(active, key=lambda x: float(x['start_pos']))
        pos_candidates = _build_pos_candidates(active_sorted)

        feasible_positions = []

        for pos in pos_candidates:
            pos     = float(pos)
            end_pos = pos + loa

            if end_pos > kade_end + EPS:
                continue

            conflict = any(
                pos < float(occ['end_pos']) + safe - EPS
                and end_pos + safe > float(occ['start_pos']) + EPS
                for occ in active_sorted
            )

            if conflict:
                continue

            if preferred_pos is not None and not pd.isna(preferred_pos):
                score = abs(pos - float(preferred_pos))
            else:
                # Isi dari START dulu
                score = pos

            feasible_positions.append((score, pos, end_pos))

        if feasible_positions:
            feasible_positions.sort(key=lambda x: x[0])
            _, pos, end_pos = feasible_positions[0]
            return (t_start, t_end, round(pos, 2), round(end_pos, 2))

        return None

    candidate_starts = set()

    if allowed_starts is not None:
        for t in allowed_starts:
            t = pd.Timestamp(t).round('s')
            if t >= earliest:
                candidate_starts.add(t)

    candidate_starts.add(earliest)

    for occ in all_occ:
        occ_end = pd.Timestamp(occ['end']).round('s')
        if occ_end >= earliest:
            candidate_starts.add(occ_end)

    t_cursor = min(candidate_starts)

    for _ in range(max_iter):
        slot = _try_at(t_cursor)

        if slot is not None:
            return slot

        t_end_try = t_cursor + svc_td
        active    = _active_occ(t_cursor, t_end_try)

        conflict_ends = [
            pd.Timestamp(occ['end'])
            for occ in active
            if pd.Timestamp(occ['end']) > t_cursor
        ]

        if conflict_ends:
            t_cursor = max(conflict_ends).round('s')
        else:
            t_cursor = (t_cursor + step_td).round('s')

    return None

def _run_single_s2(df, berths, id_to_parent, id_to_nama, eligibility,
                   config, ship_order, berth_order_per_ship,
                   max_reberth=2):

    max_segments = int(max_reberth) + 1
    EPS        = float(config.get('EPSILON', 1e-6))
    safe       = float(config.get('CONTINUOUS_GAP_M', 0.0))
    ukc        = float(config['UNDER_KEEL_CLEARANCE'])
    tide_delta = float(config['TIDE_DELTA'])


    kid_first_pos = {}
    for pos, ship_idx in enumerate(ship_order):
        kid = str(df.loc[ship_idx, 'ID_KUNJUNGAN'])
        if kid not in kid_first_pos:
            kid_first_pos[kid] = pos

    kid_groups = {}
    for ship_idx in ship_order:
        kid = str(df.loc[ship_idx, 'ID_KUNJUNGAN'])
        kid_groups.setdefault(kid, []).append(ship_idx)
    for kid in kid_groups:
        kid_groups[kid].sort(key=lambda i: int(df.loc[i, 'BERTH_PART']))

    ordered_kids = sorted(kid_groups.keys(), key=lambda k: kid_first_pos[k])
    ship_order = []
    for kid in ordered_kids:
        ship_order.extend(kid_groups[kid])

    approach_td = pd.Timedelta(hours=float(config['APPROACH_TIME']))
    reberth_gap = approach_td + approach_td

    unique_parents = set(id_to_parent.values())

    # [OPT-1] dua list paralel per parent (pakai id_to_parent asli)
    parent_ends = {p: [] for p in unique_parents}
    parent_occs = {p: [] for p in unique_parents}

    kade_usage_counter   = {str(b['ID']): set() for b in berths}
    kade_kunjungan_count = {str(b['ID']): set() for b in berths}

    results     = []
    results_map = {}   # [OPT-2] {ship_idx: record dict}

    kunjungan_segments = {}
    berth_map          = {str(b['ID']): b for b in berths}
    kade_capacities    = _compute_kade_capacities(berths)

    # ── helper: cek kedalaman ─────────────────────────────────────────────────
    _kedalaman_cache: dict = {}

    def _kedalaman_ok(draft, kdl, t_start, t_end, merged_start=None):
        check_start = pd.Timestamp(merged_start if merged_start else t_start).round('s')
        check_end   = pd.Timestamp(t_end).round('s')
        key = (float(draft), float(kdl), check_start.value, check_end.value)
        if key not in _kedalaman_cache:
            penalty    = float(get_tide_depth_penalty(check_start, check_end, config))
            low_pen    = ukc + tide_delta
            tide_flag  = 1 if abs(penalty - low_pen) < EPS else 0
            kdl_actual = float(kdl) - ukc - tide_flag * tide_delta
            ok         = float(draft) <= kdl_actual + EPS
            _kedalaman_cache[key] = (ok, tide_flag)
        return _kedalaman_cache[key]

    def _try_fixed_pos_exact(ship, berth, parent, fixed_pos, earliest, merged_start=None):
        jenis      = str(berth['JENIS']).upper().strip()
        loa        = float(ship['LOA'])
        draft      = float(ship['DRAFT'])
        kdl        = float(berth['KEDALAMAN'])
        kade_start = float(berth['START'])
        kade_end   = float(berth['END'])
        svc_td     = pd.Timedelta(hours=float(ship['BERTH_TIME']))

        t_start = max(
            pd.Timestamp(earliest).round('s'),
            pd.Timestamp(ship['MULAI_SANDAR_AWAL']).round('s'),
        )
        t_end = (t_start + svc_td).round('s')

        if jenis == 'DISKRIT':
            pos = kade_start + float(config.get('LOA_MARGIN_DISKRIT', 0))
        else:
            if fixed_pos is None or pd.isna(fixed_pos):
                return None
            pos = float(fixed_pos)

        end_pos = pos + loa
        if pos < kade_start - EPS or end_pos > kade_end + EPS:
            return None

        span_start = pd.Timestamp(merged_start if merged_start else t_start).round('s')
        depth_ok, _ = _kedalaman_ok(draft, kdl, t_start, t_end, merged_start=span_start)
        if not depth_ok:
            return None

        occ_list = _occ_active(parent_ends[parent], parent_occs[parent], span_start, t_end)
        kid_ship = str(ship.get('ID_KUNJUNGAN', ''))

        for occ in occ_list:
            time_overlap = (
                pd.Timestamp(occ['start']) < t_end
                and pd.Timestamp(occ['end']) > span_start
            )
            if not time_overlap:
                continue
            if str(occ.get('kid', '')) == kid_ship:
                continue
            if jenis == 'DISKRIT':
                return None
            if (pos < float(occ['end_pos']) + safe - EPS
                    and end_pos + safe > float(occ['start_pos']) + EPS):
                return None

        return (t_start, t_end, round(pos, 2), round(end_pos, 2))

    # ── helper: local repack kontinu ──────────────────────────────────────────
    def _try_local_repack_continuous(parent, berth, span_start, span_end,
                                     fixed_kid, fixed_pos, fixed_loa):
        jenis = str(berth['JENIS']).upper().strip()
        if jenis != 'KONTINU':
            return None

        kade_start = float(berth['START'])
        kade_end   = float(berth['END'])
        span_start = pd.Timestamp(span_start).round('s')
        span_end   = pd.Timestamp(span_end).round('s')
        fixed_pos  = float(fixed_pos)
        fixed_end  = fixed_pos + float(fixed_loa)

        if fixed_pos < kade_start - EPS or fixed_end > kade_end + EPS:
            return None

        fixed_block = {
            'idx'      : '__FIXED__',
            'start'    : span_start,
            'end'      : span_end,
            'start_pos': fixed_pos,
            'end_pos'  : fixed_end,
        }

        def _time_overlap(a, b):
            return (
                pd.Timestamp(a['start']) < pd.Timestamp(b['end'])
                and pd.Timestamp(a['end']) > pd.Timestamp(b['start'])
            )

        def _pos_overlap(a_start, a_end, b_start, b_end):
            return (
                float(a_start) < float(b_end) + safe - EPS
                and float(a_end) + safe > float(b_start) + EPS
            )

        candidates = _occ_active(parent_ends[parent], parent_occs[parent], span_start, span_end)

        active_other = []
        for occ in candidates:
            if str(occ.get('kid', '')) == str(fixed_kid):
                continue
            if occ.get('idx') is None:
                return None
            occ_sp = float(occ['start_pos'])
            occ_ep = float(occ['end_pos'])
            active_other.append({
                'idx'    : occ.get('idx'),
                'kid'    : occ.get('kid'),
                'start'  : pd.Timestamp(occ['start']).round('s'),
                'end'    : pd.Timestamp(occ['end']).round('s'),
                'loa'    : occ_ep - occ_sp,
                'old_pos': occ_sp,
                'old_end': occ_ep,
            })

        blockers  = []
        obstacles = []
        for item in active_other:
            cand_occ = {
                'start'    : item['start'],
                'end'      : item['end'],
                'start_pos': item['old_pos'],
                'end_pos'  : item['old_end'],
            }
            if (_time_overlap(cand_occ, fixed_block)
                    and _pos_overlap(item['old_pos'], item['old_end'], fixed_pos, fixed_end)):
                blockers.append(item)
            else:
                obstacles.append({
                    'idx'      : item['idx'],
                    'start'    : item['start'],
                    'end'      : item['end'],
                    'start_pos': item['old_pos'],
                    'end_pos'  : item['old_end'],
                })

        if not blockers:
            return {}

        max_blockers = 0
        if len(blockers) > max_blockers:
            return None

        placed     = [fixed_block] + obstacles
        assignment = {}
        blockers   = sorted(blockers, key=lambda x: (-float(x['loa']), float(x['old_pos'])))

        def _can_place(item, pos):
            pos     = float(pos)
            end_pos = pos + float(item['loa'])
            if pos < kade_start - EPS or end_pos > kade_end + EPS:
                return False
            cand = {
                'start'    : item['start'],
                'end'      : item['end'],
                'start_pos': pos,
                'end_pos'  : end_pos,
            }
            for pl in placed:
                if not _time_overlap(cand, pl):
                    continue
                if _pos_overlap(pos, end_pos, float(pl['start_pos']), float(pl['end_pos'])):
                    return False
            return True

        def _candidate_positions(item):
            loa_item = float(item['loa'])
            old_pos  = float(item['old_pos'])
            cands    = [old_pos, kade_start, kade_end - loa_item]
            for pl in placed:
                cands.append(float(pl['end_pos']) + safe)
                cands.append(float(pl['start_pos']) - safe - loa_item)
            seen, clean = set(), []
            for p in cands:
                p = round(float(p), 6)
                if p in seen or p < kade_start - EPS or p + loa_item > kade_end + EPS:
                    continue
                seen.add(p)
                clean.append(p)
            clean.sort(key=lambda p: abs(float(p) - old_pos))
            return clean

        def _backtrack(i):
            if i >= len(blockers):
                return True
            item = blockers[i]
            for pos in _candidate_positions(item):
                if not _can_place(item, pos):
                    continue
                end_pos = pos + float(item['loa'])
                placed.append({
                    'idx'      : item['idx'],
                    'start'    : item['start'],
                    'end'      : item['end'],
                    'start_pos': pos,
                    'end_pos'  : end_pos,
                })
                assignment[item['idx']] = (pos, end_pos)
                if _backtrack(i + 1):
                    return True
                placed.pop()
                assignment.pop(item['idx'], None)
            return False

        return assignment if _backtrack(0) else None

    # ── helper: skor posisi ───────────────────────────────────────────────────
    def _slot_position_score(slot, berth, parent):
        jenis = str(berth['JENIS']).upper().strip()
        if slot is None:
            return 10**18
        t_start, t_end, sp, ep = slot
        sp, ep     = float(sp), float(ep)
        kade_start = float(berth['START'])
        kade_end   = float(berth['END'])
        if jenis == 'DISKRIT':
            return 0.0
        active = _occ_active(parent_ends[parent], parent_occs[parent], t_start, t_end)
        if not active:
            return abs(sp - kade_start)
        active_sorted = sorted(active, key=lambda x: float(x['start_pos']))
        left_gap  = sp - kade_start
        right_gap = kade_end - ep
        for occ in active_sorted:
            occ_sp = float(occ['start_pos'])
            occ_ep = float(occ['end_pos'])
            if occ_ep <= sp:
                left_gap  = min(left_gap, sp - occ_ep)
            if occ_sp >= ep:
                right_gap = min(right_gap, occ_sp - ep)
        return float(min(abs(left_gap), abs(right_gap)) + abs(sp - kade_start) * 0.01)

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    for ship_idx in ship_order:
        ship         = df.loc[ship_idx].copy()
        id_kunjungan = str(ship.get('ID_KUNJUNGAN', ''))
        berth_part   = int(ship.get('BERTH_PART', 1))

        ordered = berth_order_per_ship.get(ship_idx, eligibility.get(ship_idx, []))
        if not ordered:
            raise ValueError(
                f"ID_KUNJUNGAN={id_kunjungan}, "
                f"BERTH_PART={berth_part} tidak punya dermaga eligible."
            )

        segments = kunjungan_segments.get(id_kunjungan, [])

        found_slot             = None
        found_bid              = None
        found_parent           = None
        used_seg_idx           = None
        reposition_segment_pos = None
        repack_other_positions = {}

        # ── TAHAP 1: lanjut segment terakhir ─────────────────────────────────
        if segments:
            last_seg_idx  = len(segments) - 1
            last_seg      = segments[last_seg_idx]
            seg_bid       = str(last_seg['bid'])
            seg_parent    = last_seg['parent']
            seg_berth     = berth_map.get(seg_bid)
            seg_pos       = last_seg.get('pos')
            eligible_last = any(
                str(b['ID']) == seg_bid for b in (eligibility.get(ship_idx) or [])
            )

            if seg_berth is not None and eligible_last:
                exact_start = pd.Timestamp(last_seg['selesai']).round('s')
                ship_cont   = ship.copy()
                ship_cont['MULAI_SANDAR_AWAL'] = exact_start

                fit_last = ship_fits_berth(ship_cont, seg_berth, config)

                if fit_last:
                    slot = _try_fixed_pos_exact(
                        ship_cont, seg_berth, seg_parent,
                        seg_pos, exact_start, merged_start=last_seg['mulai'],
                    )
                else:
                    slot = None

                if slot is not None and pd.Timestamp(slot[0]).round('s') == exact_start:
                    found_slot   = slot
                    found_bid    = seg_bid
                    found_parent = seg_parent
                    used_seg_idx = last_seg_idx

                if (
                    fit_last
                    and found_slot is None
                    and str(seg_berth['JENIS']).upper().strip() == 'KONTINU'
                ):
                    ship_loa       = float(ship_cont['LOA'])
                    t_start_repack = exact_start
                    t_end_repack   = (
                        t_start_repack + pd.Timedelta(hours=float(ship_cont['BERTH_TIME']))
                    ).round('s')
                    depth_ok, _ = _kedalaman_ok(
                        float(ship_cont['DRAFT']), float(seg_berth['KEDALAMAN']),
                        t_start_repack, t_end_repack, merged_start=last_seg['mulai'],
                    )

                    if depth_ok:
                        kade_start = float(seg_berth['START'])
                        kade_end   = float(seg_berth['END'])
                        loa_now    = float(ship_cont['LOA'])

                        cand_positions = [float(seg_pos), kade_start, kade_end - loa_now]

                        for occ in _occ_active(
                            parent_ends[seg_parent], parent_occs[seg_parent],
                            last_seg['mulai'], t_end_repack,
                        ):
                            cand_positions.extend([
                                float(occ['start_pos']),
                                float(occ['end_pos']) + safe,
                                float(occ['start_pos']) - safe - loa_now,
                            ])

                        seen_pos, clean_positions = set(), []
                        for p in cand_positions:
                            p = round(float(p), 6)
                            if p in seen_pos or p < kade_start - EPS or p + loa_now > kade_end + EPS:
                                continue
                            seen_pos.add(p)
                            clean_positions.append(p)
                        clean_positions.sort(key=lambda p: abs(float(p) - float(seg_pos)))

                        for cand_pos in clean_positions:
                            ok_reposition = True
                            for prev_idx in last_seg.get('parts', []):
                                prev_ship  = df.loc[prev_idx].copy()
                                old_res    = results_map.get(prev_idx)
                                if old_res is None:
                                    ok_reposition = False
                                    break
                                old_start  = pd.Timestamp(old_res['MULAI_SANDAR']).round('s')
                                old_end    = pd.Timestamp(old_res['SELESAI_SANDAR']).round('s')
                                prev_loa   = float(prev_ship['LOA'])
                                prev_draft = float(prev_ship['DRAFT'])
                                if (cand_pos < kade_start - EPS
                                        or cand_pos + prev_loa > kade_end + EPS):
                                    ok_reposition = False
                                    break
                                depth_prev_ok, _ = _kedalaman_ok(
                                    prev_draft, float(seg_berth['KEDALAMAN']),
                                    old_start, old_end, merged_start=last_seg['mulai'],
                                )
                                if not depth_prev_ok:
                                    ok_reposition = False
                                    break

                            if not ok_reposition:
                                continue

                            repack_result = _try_local_repack_continuous(
                                parent=seg_parent, berth=seg_berth,
                                span_start=last_seg['mulai'], span_end=t_end_repack,
                                fixed_kid=id_kunjungan, fixed_pos=cand_pos,
                                fixed_loa=ship_loa,
                            )
                            if repack_result is None:
                                continue

                            found_slot = (
                                t_start_repack, t_end_repack,
                                round(cand_pos, 2), round(cand_pos + ship_loa, 2),
                            )
                            found_bid              = seg_bid
                            found_parent           = seg_parent
                            used_seg_idx           = last_seg_idx
                            reposition_segment_pos = cand_pos
                            repack_other_positions = repack_result
                            break

        # ── TAHAP 2: buka segment baru ────────────────────────────────────────
        if found_slot is None and len(segments) < max_segments:
            ship_new = ship.copy()
            if segments:
                ship_new['MULAI_SANDAR_AWAL'] = max(
                    pd.Timestamp(ship_new['MULAI_SANDAR_AWAL']).round('s'),
                    pd.Timestamp(segments[-1]['selesai']).round('s') + reberth_gap,
                )

            util_cache = _build_util_cache(
                berths, id_to_parent, parent_ends, parent_occs, kade_capacities
            )

            new_seg_candidates = []
            last_bid      = str(segments[-1]['bid']) if segments else None
            last_pos      = float(segments[-1]['pos']) if segments else None
            existing_bids = {str(seg['bid']) for seg in segments}
            kid_now       = str(ship['ID_KUNJUNGAN'])
            part_now      = int(ship['BERTH_PART'])

            sisa_part_idx = (
                df[
                    (df['ID_KUNJUNGAN'].astype(str) == kid_now)
                    & (df['BERTH_PART'].astype(int) > part_now)
                ]
                .sort_values('BERTH_PART').index.tolist()
            )

            has_remaining_parts = bool(sisa_part_idx)
            sisa_part_idx = sisa_part_idx[:1]

            wajib_sisa_lanjut = (
                len(segments) == max_segments - 1
                and has_remaining_parts
            )

            for urutan, berth in enumerate(ordered):
                if not ship_fits_berth(ship_new, berth, config):
                    continue
                bid    = str(berth['ID'])
                parent = id_to_parent[bid]   # pakai id_to_parent asli untuk occupancy

                pocc = (parent_ends[parent], parent_occs[parent])

                if segments and bid == last_bid:
                    slot = find_slot_ch2(
                        ship_new, berth, pocc, config,
                        preferred_pos=last_pos, strict_preferred=False,
                    )
                    prefer_penalty = 1
                else:
                    slot = find_slot_ch2(ship_new, berth, pocc, config)
                    prefer_penalty = 2 if bid not in existing_bids else 3

                if slot is None:
                    continue

                t_start = pd.Timestamp(slot[0]).round('s')

                # Lookup util_cache pakai (bid, bid) karena id_to_parent_util[bid] == bid
                utilisasi, beban_kunjungan, beban_part = util_cache.get(
                    (parent, bid), (0.0, 0, 0)
                )

                pos_score   = _slot_position_score(slot, berth, parent)
                skor_lanjut = 0
                boleh_pilih = True

                temp_ends = list(parent_ends[parent])
                temp_occs = list(parent_occs[parent])

                ts, te, tsp, tep = slot
                temp_new = {
                    'start'    : pd.Timestamp(ts).round('s'),
                    'end'      : pd.Timestamp(te).round('s'),
                    'start_pos': float(tsp),
                    'end_pos'  : float(tep),
                    'kid'      : kid_now,
                    'part'     : part_now,
                }
                _occ_add(temp_ends, temp_occs, temp_new)

                temp_last_end  = pd.Timestamp(te).round('s')
                temp_seg_start = pd.Timestamp(ts).round('s')
                temp_fixed_pos = float(tsp)

                for next_idx in sisa_part_idx:
                    ship_next = df.loc[next_idx].copy()

                    old_ends = parent_ends[parent]
                    old_occs = parent_occs[parent]
                    parent_ends[parent] = temp_ends
                    parent_occs[parent] = temp_occs

                    try:
                        slot_next = _try_fixed_pos_exact(
                            ship_next, berth, parent,
                            temp_fixed_pos, temp_last_end,
                            merged_start=temp_seg_start,
                        )
                    finally:
                        parent_ends[parent] = old_ends
                        parent_occs[parent] = old_occs

                    if slot_next is not None and pd.Timestamp(slot_next[0]).round('s') != temp_last_end:
                        slot_next = None

                    if slot_next is None:
                        if wajib_sisa_lanjut:
                            boleh_pilih = False
                        break

                    ns, ne, nsp, nep = slot_next
                    next_new = {
                        'start'    : pd.Timestamp(ns).round('s'),
                        'end'      : pd.Timestamp(ne).round('s'),
                        'start_pos': float(nsp),
                        'end_pos'  : float(nep),
                        'kid'      : kid_now,
                        'part'     : int(ship_next['BERTH_PART']),
                    }
                    _occ_add(temp_ends, temp_occs, next_new)
                    temp_last_end = pd.Timestamp(ne).round('s')
                    skor_lanjut  += 1

                if not boleh_pilih:
                    continue

                panjang_bid = max(float(berth['END']) - float(berth['START']), 1.0)

                panjang_scale = max(panjang_bid ** 0.5, 1.0)

                beban_kunjungan_ratio = beban_kunjungan / panjang_scale
                beban_part_ratio      = beban_part / panjang_scale
                utilisasi_ratio       = utilisasi / panjang_scale

                t_bucket = pd.Timestamp(t_start).floor('2h')

                new_seg_candidates.append((
                    t_bucket,               # 1. jaga waktu tetap dalam kelompok 2 jam
                    beban_kunjungan_ratio,  # 2. ratakan beban proporsional panjang dermaga
                    beban_part_ratio,       # 3. ratakan part proporsional panjang dermaga
                    utilisasi_ratio,        # 4. ratakan durasi proporsional panjang dermaga
                    t_start,                # 5. baru pilih waktu aktual tercepat
                    prefer_penalty,
                    -skor_lanjut,
                    pos_score,
                    urutan,
                    bid,
                    parent,
                    slot,
                ))
            if new_seg_candidates:
                new_seg_candidates.sort(key=lambda x: x[:9])
                _, _, _, _, _, _, _, _, _, found_bid, found_parent, found_slot = new_seg_candidates[0]

        # ── tidak ketemu ──────────────────────────────────────────────────────
        if found_slot is None:
            raise ValueError(
                f"TIDAK FEASIBLE: "
                f"ID_KUNJUNGAN={id_kunjungan}, "
                f"BERTH_PART={berth_part}, "
                f"max_reberth={max_reberth}. "
            )

        if id_kunjungan and used_seg_idx is None and len(segments) >= max_segments:
            raise ValueError(
                f"TIDAK FEASIBLE: ID_KUNJUNGAN={id_kunjungan}, "
                f"BERTH_PART={berth_part} butuh segment baru, "
                f"tapi sudah max segment={max_segments}."
            )

        # ── COMMIT ────────────────────────────────────────────────────────────
        t_start, t_end, sp, ep = found_slot
        t_start = pd.Timestamp(t_start).round('s')
        t_end   = pd.Timestamp(t_end).round('s')
        sp, ep  = float(sp), float(ep)

        new_occ = {
            'start'    : t_start,
            'end'      : t_end,
            'start_pos': sp,
            'end_pos'  : ep,
            'kid'      : id_kunjungan,
            'part'     : berth_part,
            'idx'      : ship_idx,
            'bid'      : found_bid,
        }
        _occ_add(parent_ends[found_parent], parent_occs[found_parent], new_occ)

        kade_usage_counter[found_bid].add(id_kunjungan)
        if id_kunjungan:
            kade_kunjungan_count[found_bid].add(id_kunjungan)

            if used_seg_idx is not None:
                if reposition_segment_pos is not None:
                    new_pos = float(reposition_segment_pos)
                    segments[used_seg_idx]['pos'] = new_pos
                    for old_idx in segments[used_seg_idx].get('parts', []):
                        old_loa = float(df.loc[old_idx, 'LOA'])
                        r = results_map.get(old_idx)
                        if r is not None:
                            r['POSISI_START_M'] = round(new_pos, 2)
                            r['POSISI_END_M']   = round(new_pos + old_loa, 2)
                        for occ in parent_occs[found_parent]:
                            if occ.get('idx') == old_idx:
                                occ['start_pos'] = new_pos
                                occ['end_pos']   = new_pos + old_loa

                if repack_other_positions:
                    for other_idx, (np_, nep_) in repack_other_positions.items():
                        r = results_map.get(other_idx)
                        if r is not None:
                            r['POSISI_START_M'] = round(float(np_), 2)
                            r['POSISI_END_M']   = round(float(nep_), 2)
                        for occ in parent_occs[found_parent]:
                            if occ.get('idx') == other_idx:
                                occ['start_pos'] = float(np_)
                                occ['end_pos']   = float(nep_)

                segments[used_seg_idx]['selesai'] = t_end
                segments[used_seg_idx].setdefault('parts', []).append(ship_idx)
                segment_no_plan = used_seg_idx + 1

            else:
                segments.append({
                    'bid'    : found_bid,
                    'parent' : found_parent,
                    'mulai'  : t_start,
                    'selesai': t_end,
                    'pos'    : sp,
                    'parts'  : [ship_idx],
                })
                segment_no_plan = len(segments)

            kunjungan_segments[id_kunjungan] = segments
        else:
            segment_no_plan = 1

        if segment_no_plan > max_segments:
            raise ValueError(
                f"TIDAK FEASIBLE: ID_KUNJUNGAN={id_kunjungan}, "
                f"SEGMENT_NO_PLAN={segment_no_plan} > max_segments={max_segments}."
            )

        flag_start = (
            kunjungan_segments[id_kunjungan][used_seg_idx]['mulai']
            if used_seg_idx is not None else t_start
        )
        _, tf = _kedalaman_ok(
            float(ship['DRAFT']),
            float(berth_map[found_bid]['KEDALAMAN']),
            flag_start, t_end,
        )

        if segments and used_seg_idx is None:
            prev_end  = pd.Timestamp(segments[-1]['selesai'])
            waiting_h = (t_start - (prev_end + reberth_gap)).total_seconds() / 3600
        elif used_seg_idx is not None and used_seg_idx > 0:
            prev_end  = pd.Timestamp(
                kunjungan_segments[id_kunjungan][used_seg_idx - 1]['selesai']
            )
            waiting_h = (t_start - (prev_end + reberth_gap)).total_seconds() / 3600
        else:
            waiting_h = (
                t_start - pd.Timestamp(df.loc[ship_idx, 'KEDATANGAN']) - approach_td
            ).total_seconds() / 3600

        record = {
            'idx'                : ship_idx,
            'DERMAGA_ASSIGNED'   : found_bid,
            'NAMA_DERMAGA'       : id_to_nama[found_bid],
            'MULAI_SANDAR'       : t_start,
            'SELESAI_SANDAR'     : t_end,
            'WAITING_TIME_HOURS' : round(max(waiting_h, 0.0), 4),
            'POSISI_START_M'     : round(sp, 2),
            'POSISI_END_M'       : round(ep, 2),
            'TIDE_FLAG'          : tf,
            'IS_LATE'            : 0,
            'REBERTHING'         : max(0, segment_no_plan - 1),
            'SEGMENT_NO_PLAN'    : segment_no_plan,
        }
        results.append(record)
        results_map[ship_idx] = record

    return results

def _merge_parts_s2(df_sol, config, max_reberth=2):

    max_segments = int(max_reberth) + 1
    EPS = float(config.get('EPSILON', 1e-6))
    time_tol = pd.Timedelta(seconds=1)

    df = df_sol.copy()

    if df.empty:
        return df

    # Pastikan kolom waktu datetime
    if 'MULAI_SANDAR' in df.columns:
        df['MULAI_SANDAR'] = pd.to_datetime(df['MULAI_SANDAR'], errors='coerce')
    if 'SELESAI_SANDAR' in df.columns:
        df['SELESAI_SANDAR'] = pd.to_datetime(df['SELESAI_SANDAR'], errors='coerce')

    assigned = df[df['DERMAGA_ASSIGNED'].notna()].copy()
    unassigned = df[df['DERMAGA_ASSIGNED'].isna()].copy()

    if assigned.empty:
        merged_rows = []
        for _, row in unassigned.iterrows():
            r = row.to_dict()
            r['BERTH_PART_LIST'] = [int(row.get('BERTH_PART', 1))]
            r['SEGMENT_ID'] = -1
            r['SEGMENT_NO'] = -1
            r['SEGMENT_NO_PLAN'] = -1
            r['REBERTHING'] = 0
            r['FLAG_SANDAR'] = None
            merged_rows.append(r)
        return pd.DataFrame(merged_rows).reset_index(drop=True)

    # Pastikan SEGMENT_NO_PLAN ada
    if 'SEGMENT_NO_PLAN' not in assigned.columns:
        assigned['SEGMENT_NO_PLAN'] = 1

    assigned['SEGMENT_NO_PLAN'] = pd.to_numeric(
        assigned['SEGMENT_NO_PLAN'],
        errors='coerce'
    ).fillna(1).astype(int).clip(lower=1, upper=max_segments)

    # Pastikan BERTH_PART numerik
    if 'BERTH_PART' not in assigned.columns:
        assigned['BERTH_PART'] = 1

    assigned['BERTH_PART'] = pd.to_numeric(
        assigned['BERTH_PART'],
        errors='coerce'
    ).fillna(1).astype(int)

    merged_rows = []
    seg_id_global = 0

    assigned = assigned.sort_values([
        'ID_KUNJUNGAN',
        'SEGMENT_NO_PLAN',
        'MULAI_SANDAR',
        'BERTH_PART'
    ]).copy()

    def _make_single_part_row(row, seg_no_plan):
        """
        Jadikan 1 berth part sebagai 1 segment sendiri.
        Dipakai kalau merge antar part tidak aman.
        """
        nonlocal seg_id_global

        seg_id_global += 1

        r = row.to_dict()

        mulai = pd.Timestamp(r['MULAI_SANDAR']).round('s')
        selesai = pd.Timestamp(r['SELESAI_SANDAR']).round('s')

        r['MULAI_SANDAR'] = mulai
        r['SELESAI_SANDAR'] = selesai
        r['BERTH_TIME'] = (selesai - mulai).total_seconds() / 3600.0
        r['SERVICE_TIME_SUM'] = float(row.get('BERTH_TIME', r['BERTH_TIME']))

        r['BERTH_PART_LIST'] = [int(row.get('BERTH_PART', 1))]
        r['SEGMENT_ID'] = seg_id_global
        r['SEGMENT_NO'] = int(seg_no_plan)
        r['SEGMENT_NO_PLAN'] = int(seg_no_plan)
        r['REBERTHING'] = max(0, int(seg_no_plan) - 1)
        r['FLAG_SANDAR'] = (
            'KUNJUNGAN BARU' if int(seg_no_plan) == 1 else 'SANDAR ULANG'
        )

        if 'TIDE_FLAG' in row.index:
            r['TIDE_FLAG'] = int(pd.to_numeric(row.get('TIDE_FLAG', 0), errors='coerce') or 0)
        else:
            r['TIDE_FLAG'] = 0

        return r

    for (kid, seg_no_plan), seg_grp in assigned.groupby(
        ['ID_KUNJUNGAN', 'SEGMENT_NO_PLAN'],
        sort=False
    ):
        seg_no_plan = int(seg_no_plan)

        if seg_no_plan > max_segments:
            continue

        seg_grp = seg_grp.sort_values(['MULAI_SANDAR', 'BERTH_PART']).copy()

        # Kalau cuma 1 part, langsung jadikan 1 segment
        if len(seg_grp) == 1:
            merged_rows.append(_make_single_part_row(seg_grp.iloc[0], seg_no_plan))
            continue

        # ============================================================
        # CEK APAKAH AMAN DI-MERGE
        # ============================================================
        can_merge = True

        # 1. Semua part dalam segment harus di dermaga yang sama
        berths_in_seg = seg_grp['DERMAGA_ASSIGNED'].astype(str).dropna().unique()
        if len(berths_in_seg) != 1:
            can_merge = False

        # 2. Semua part harus posisi sama / konsisten
        if can_merge:
            sp0 = float(seg_grp.iloc[0]['POSISI_START_M'])
            ep0 = float(seg_grp.iloc[0]['POSISI_END_M'])

            for _, rr in seg_grp.iterrows():
                sp = float(rr['POSISI_START_M'])
                ep = float(rr['POSISI_END_M'])

                if abs(sp - sp0) > EPS or abs(ep - ep0) > EPS:
                    can_merge = False
                    break

        # 3. Waktu antar part harus benar-benar nyambung
        #    Part berikutnya harus mulai tepat setelah part sebelumnya selesai.
        if can_merge:
            prev_end = None

            for _, rr in seg_grp.iterrows():
                st = pd.Timestamp(rr['MULAI_SANDAR']).round('s')
                en = pd.Timestamp(rr['SELESAI_SANDAR']).round('s')

                if pd.isna(st) or pd.isna(en) or en <= st:
                    can_merge = False
                    break

                if prev_end is not None:
                    # Harus kontigu. Kalau ada gap besar, jangan merge.
                    if abs((st - prev_end).total_seconds()) > time_tol.total_seconds():
                        can_merge = False
                        break

                prev_end = en

        # Kalau tidak aman, jangan paksa jadi blok panjang
        if not can_merge:
            for _, rr in seg_grp.iterrows():
                merged_rows.append(_make_single_part_row(rr, seg_no_plan))
            continue

        # ============================================================
        # MERGE AMAN: part sama dermaga, sama posisi, dan waktunya kontigu
        # ============================================================
        seg_id_global += 1

        first = seg_grp.iloc[0].to_dict()

        mulai = pd.Timestamp(seg_grp['MULAI_SANDAR'].min()).round('s')
        selesai = pd.Timestamp(seg_grp['SELESAI_SANDAR'].max()).round('s')

        first['MULAI_SANDAR'] = mulai
        first['SELESAI_SANDAR'] = selesai
        first['BERTH_TIME'] = (selesai - mulai).total_seconds() / 3600.0
        first['SERVICE_TIME_SUM'] = float(
            pd.to_numeric(seg_grp['BERTH_TIME'], errors='coerce').fillna(0).sum()
        )

        first['BERTH_PART_LIST'] = seg_grp['BERTH_PART'].astype(int).tolist()
        first['SEGMENT_ID'] = seg_id_global
        first['SEGMENT_NO'] = seg_no_plan
        first['SEGMENT_NO_PLAN'] = seg_no_plan

        if 'TIDE_FLAG' in seg_grp.columns:
            first['TIDE_FLAG'] = int(
                pd.to_numeric(seg_grp['TIDE_FLAG'], errors='coerce')
                .fillna(0)
                .astype(int)
                .max()
            )
        else:
            first['TIDE_FLAG'] = 0

        first['REBERTHING'] = max(0, seg_no_plan - 1)
        first['FLAG_SANDAR'] = (
            'KUNJUNGAN BARU' if seg_no_plan == 1 else 'SANDAR ULANG'
        )

        # Posisi aman karena sudah dicek semua part posisinya sama
        first['POSISI_START_M'] = round(float(seg_grp.iloc[0]['POSISI_START_M']), 2)
        first['POSISI_END_M'] = round(float(seg_grp.iloc[0]['POSISI_END_M']), 2)

        merged_rows.append(first)

    # Tambahkan unassigned
    for _, row in unassigned.iterrows():
        r = row.to_dict()
        r['BERTH_PART_LIST'] = [int(row.get('BERTH_PART', 1))]
        r['SEGMENT_ID'] = -1
        r['SEGMENT_NO'] = -1
        r['SEGMENT_NO_PLAN'] = -1
        r['REBERTHING'] = 0
        r['FLAG_SANDAR'] = None
        merged_rows.append(r)

    return pd.DataFrame(merged_rows).reset_index(drop=True)


def _recalculate_s2(df_merged, df_kapal, berths, id_to_parent, id_to_nama,
                    config, weights, max_reberth=2):

    df_final    = df_merged.copy()
    approach_td = pd.Timedelta(hours=float(config['APPROACH_TIME']))
    reberth_gap = approach_td + approach_td

    sop_threshold_h = float(weights['WAIT_THRESHOLD'])
    max_segments    = int(max_reberth) + 1

    kedatangan_map = {
        str(r['ID_KUNJUNGAN']): pd.Timestamp(r['KEDATANGAN'])
        for _, r in df_kapal.iterrows()
    }

    assigned_idx = df_final[df_final['DERMAGA_ASSIGNED'].notna()].index
    updates = {}  # {idx: {col: val}}

    for kid, grp in df_final.loc[assigned_idx].groupby('ID_KUNJUNGAN', sort=False):
        grp = grp.sort_values('MULAI_SANDAR').copy()

        if len(grp) > max_segments:
            extra_idx = grp.iloc[max_segments:].index
            for idx in extra_idx:
                updates[idx] = {
                    'DERMAGA_ASSIGNED': None,
                    'NAMA_DERMAGA': None,
                    'FLAG_SANDAR': None,
                    'REBERTHING': 0,
                }
            continue

        prev_selesai = None

        for local_no, idx in enumerate(grp.index, start=1):
            row = df_final.loc[idx]

            mulai   = pd.Timestamp(row['MULAI_SANDAR']).round('s')
            selesai = pd.Timestamp(row['SELESAI_SANDAR']).round('s')

            kedatangan = kedatangan_map.get(
                str(kid),
                pd.Timestamp(row['KEDATANGAN']),
            )

            if local_no == 1:
                wait_h = max(
                    0.0,
                    (mulai - (kedatangan + approach_td)).total_seconds() / 3600.0,
                )
                earliest = kedatangan + approach_td
            else:
                wait_h = max(
                    0.0,
                    (mulai - (prev_selesai + reberth_gap)).total_seconds() / 3600.0,
                )
                earliest = prev_selesai + reberth_gap

            bid = str(row['DERMAGA_ASSIGNED'])
            tide_flag = 0
            if 'TIDE_FLAG' in df_final.columns and pd.notna(row.get('TIDE_FLAG')):
                tide_flag = int(row['TIDE_FLAG'])

            updates[idx] = {
                'MULAI_SANDAR_AWAL' : earliest,
                'WAITING_TIME_HOURS': round(wait_h, 4),
                'TIDE_FLAG'         : tide_flag,
                'IS_LATE'           : int(wait_h > sop_threshold_h),
                'SEGMENT_NO'        : local_no,
                'SEGMENT_NO_PLAN'   : local_no,
                'REBERTHING'        : max(0, local_no - 1),
                'FLAG_SANDAR'       : 'KUNJUNGAN BARU' if local_no == 1 else 'SANDAR ULANG',
                'NAMA_DERMAGA'      : id_to_nama.get(bid, row.get('NAMA_DERMAGA')),
            }

            prev_selesai = selesai

    # Bulk update sekaligus
    for idx, cols in updates.items():
        for col, val in cols.items():
            df_final.at[idx, col] = val

    return df_final.reset_index(drop=True)


def _validate_s2(df_merged, config, berths=None, max_reberth=2):
    EPS    = config.get('EPSILON', 1e-6)
    safe   = config.get('CONTINUOUS_GAP_M', 0)
    ukc    = float(config['UNDER_KEEL_CLEARANCE'])
    tid    = float(config['TIDE_DELTA'])
    issues = []

    assigned = df_merged[df_merged['DERMAGA_ASSIGNED'].notna()].copy()

    if 'MULAI_SANDAR_AWAL' in assigned.columns:
        bad_time = assigned[
            pd.to_datetime(assigned['MULAI_SANDAR']) <
            pd.to_datetime(assigned['MULAI_SANDAR_AWAL']) - pd.Timedelta(seconds=1)
        ]
        for _, row in bad_time.iterrows():
            issues.append(
                f"ID_KUNJUNGAN={row['ID_KUNJUNGAN']}: mulai={row['MULAI_SANDAR']} "
                f"< earliest={row['MULAI_SANDAR_AWAL']}"
            )

    if berths is not None:
        berth_map = {str(b['ID']): b for b in berths}

        for _, row in assigned.iterrows():
            bid = str(row['DERMAGA_ASSIGNED'])
            berth = berth_map.get(bid)

            if berth is None:
                issues.append(
                    f"ID_KUNJUNGAN={row['ID_KUNJUNGAN']}: dermaga {bid} tidak ditemukan"
                )
                continue

            # 1. Cek eligibility dasar: kategori, LOA, draft vs kedalaman-UKC
            if not ship_fits_berth(row, berth, config):
                issues.append(
                    f"ID_KUNJUNGAN={row['ID_KUNJUNGAN']}: tidak eligible di dermaga {bid}"
                )

            # 2. Cek kedalaman aktual saat surut
            draft = float(row['DRAFT'])
            tide_flag = int(row.get('TIDE_FLAG', 0)) if pd.notna(row.get('TIDE_FLAG', 0)) else 0

            kdl_actual = (
                float(berth['KEDALAMAN'])
                - ukc
                - tide_flag * tid
            )

            if draft > kdl_actual + EPS:
                issues.append(
                    f"ID_KUNJUNGAN={row['ID_KUNJUNGAN']}: kandas di {bid}, "
                    f"draft={draft}, kdl_aktual={round(kdl_actual, 3)}, tide_flag={tide_flag}"
                )

    bad = assigned[assigned['SELESAI_SANDAR'] <= assigned['MULAI_SANDAR']]
    for _, row in bad.iterrows():
        issues.append(f"SEGMENT_ID={row['SEGMENT_ID']} {row['ID_KUNJUNGAN']}: SELESAI <= MULAI")

    approach_td = pd.Timedelta(hours=float(config['APPROACH_TIME']))
    reberth_gap = approach_td + approach_td

    for kid, grp in assigned.groupby('ID_KUNJUNGAN'):
        grp   = grp.sort_values('MULAI_SANDAR').reset_index(drop=True)
        n_seg = len(grp)
        if n_seg > max_reberth + 1:
            issues.append(f"ID_KUNJUNGAN={kid}: {n_seg} segment > batas {max_reberth + 1}")
        for i in range(n_seg):
            curr     = grp.iloc[i]
            earliest = (
                pd.Timestamp(curr['KEDATANGAN']) + approach_td
                if i == 0
                else pd.Timestamp(grp.iloc[i - 1]['SELESAI_SANDAR']) + reberth_gap
            )
            if pd.Timestamp(curr['MULAI_SANDAR']) < earliest - pd.Timedelta(seconds=1):
                issues.append(
                    f"ID_KUNJUNGAN={kid}: mulai={curr['MULAI_SANDAR']} < earliest={earliest}"
                )

    for bid, grp in assigned.groupby('DERMAGA_ASSIGNED'):
        events = grp.sort_values('MULAI_SANDAR').reset_index(drop=True)
        active = []
        for _, row in events.iterrows():
            mulai_ns   = pd.Timestamp(row['MULAI_SANDAR']).value
            selesai_ns = pd.Timestamp(row['SELESAI_SANDAR']).value
            sp         = float(row['POSISI_START_M'])
            ep         = float(row['POSISI_END_M'])
            seg_id     = row['SEGMENT_ID']
            kid        = row['ID_KUNJUNGAN']

            active = [a for a in active if a[0] > mulai_ns]

            for (a_end_ns, a_sp, a_ep, a_seg, a_kid) in active:
                # Skip KID yang sama — reberth di posisi sama secara waktu
                # sudah terpisah, tidak perlu dicek overlap posisi
                if str(a_kid) == str(kid):
                    continue
                if sp < a_ep + safe - EPS and ep + safe > a_sp + EPS:
                    issues.append(
                        f"Overlap kade {bid}: "
                        f"SEG {seg_id} ({kid}) vs SEG {a_seg} ({a_kid})"
                    )

            active.append((selesai_ns, sp, ep, seg_id, kid))

    return issues


def _count_parts(rows: pd.DataFrame) -> int:
    if rows is None or rows.empty:
        return 0

    if 'BERTH_PART_LIST' in rows.columns:
        total = 0

        for x in rows['BERTH_PART_LIST']:
            if isinstance(x, list):
                total += len(x)

            elif isinstance(x, tuple):
                total += len(x)

            elif isinstance(x, np.ndarray):
                total += len(x)

            elif pd.isna(x):
                total += 1

            elif isinstance(x, str):
                s = x.strip()

                # handle string seperti "[1, 2]"
                if s.startswith('[') and s.endswith(']'):
                    inner = s[1:-1].strip()
                    if inner == '':
                        total += 0
                    else:
                        total += len([
                            p for p in inner.split(',')
                            if p.strip() != ''
                        ])
                else:
                    total += 1

            else:
                total += 1

        return int(total)

    return int(len(rows))


def _evaluate_metrics_s2(df_final: pd.DataFrame) -> dict:
    assigned_mask   = df_final['DERMAGA_ASSIGNED'].notna()
    assigned_rows   = df_final[assigned_mask]
    unassigned_rows = df_final[~assigned_mask]

    num_assigned   = _count_parts(assigned_rows)
    num_unassigned = _count_parts(unassigned_rows)

    total_wait = (
        float(assigned_rows['WAITING_TIME_HOURS'].sum())
        if 'WAITING_TIME_HOURS' in assigned_rows.columns else 0.0
    )

    return {
        'num_assigned'    : num_assigned,
        'num_unassigned'  : num_unassigned,
        'total_wait_hours': round(total_wait, 4),
    }


def run_ch2(df_kapal_raw, df_dermaga,
            population_size=10,
            base_seed=None,
            max_reberth=2,
            verbose=True):

    target_valid = population_size 
    max_attempts = target_valid * 100

    base_seed = int(base_seed) if base_seed is not None else random.randint(0, 2**30 - 1)

    df, berths, id_to_parent, id_to_nama, eligibility = _prepare_data(
        df_kapal_raw, df_dermaga, CONFIG
    )

    solutions    = []
    metrics_rows = []

    valid_count = 0
    attempt     = 0
    start_time  = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"POPULATION CONSTRUCTIVE HEURISTIC — CH2")
        print(f"Target solusi valid: {target_valid} | Max attempt: {max_attempts}")
        print(f"Seed: {base_seed}")
        print(f"{'='*60}")

    while valid_count < target_valid and attempt < max_attempts:
        attempt += 1

        g = valid_count // population_size
        p = valid_count % population_size

        seed = (base_seed + attempt) & 0xFFFFFFFF
        rnd  = random.Random(seed)

        vparams = _make_variation_params(rnd)

        ship_order, berth_order_per_ship = _build_run_params_s2(
            rnd, vparams, df, eligibility
        )

        try:
            results = _run_single_s2(
                df, berths, id_to_parent, id_to_nama, eligibility,
                CONFIG, ship_order, berth_order_per_ship,
                max_reberth=max_reberth,
            )
        except ValueError:
            continue

        df_sol = _build_output_df2(df, results)

        df_merged = _merge_parts_s2(df_sol, CONFIG, max_reberth=max_reberth)

        df_final = _recalculate_s2(
            df_merged, df, berths, id_to_parent, id_to_nama,
            CONFIG, WEIGHTS, max_reberth=max_reberth,
        )

        issues = _validate_s2(df_final, CONFIG, berths=berths, max_reberth=max_reberth)

        # ── [FIX-2] Hard check assigned — robust BERTH_PART_LIST counting ─────
        total_part_raw      = len(df)
        assigned_tmp        = df_final[df_final['DERMAGA_ASSIGNED'].notna()].copy()
        total_part_assigned = _count_parts(assigned_tmp)

        if total_part_assigned != total_part_raw:
            issues.append(
                f"Berth part belum semua assigned: {total_part_assigned}/{total_part_raw}"
            )

        # ── Hard check max reberth ─────────────────────────────────────────────
        max_segments   = int(max_reberth) + 1
        assigned_check = df_final[df_final['DERMAGA_ASSIGNED'].notna()].copy()

        if not assigned_check.empty:
            seg_count = (
                assigned_check
                .groupby('ID_KUNJUNGAN')['SEGMENT_NO']
                .nunique()
            )
            for kid, nseg in seg_count[seg_count > max_segments].items():
                issues.append(
                    f"ID_KUNJUNGAN={kid}: {nseg} segment > batas {max_segments}"
                )

        if issues:
            continue

        metrics     = _evaluate_metrics_s2(df_final)
        sequence_2d = generate_sol2d_s2(df_final)

        solutions.append({
            'individual'      : p + 1,
            'seed'            : seed,
            'df_schedule'     : df_final,
            'df_sol_raw'      : df_sol,
            'metrics'         : metrics,
            'sequence_2d'     : sequence_2d,
            'variation_params': vparams,
        })

        metrics_rows.append({
            'individual'        : p + 1,
            'seed'              : seed,
            'num_assigned'      : metrics['num_assigned'],
            'num_unassigned'    : metrics['num_unassigned'],
            'total_wait_hours'  : metrics['total_wait_hours'],
            'shuffle_berth_prob': round(vparams['shuffle_berth_prob'], 4),
            'shuffle_ship'      : vparams['shuffle_ship_in_group'],
        })

        valid_count += 1

        log_every = max(1, target_valid // 10)
        if verbose and (valid_count % log_every == 0 or valid_count == target_valid):
            elapsed = time.time() - start_time
            print(
                f"  Valid {valid_count:>4}/{target_valid} | "
                f"Attempt {attempt:>4}/{max_attempts} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    metrics_df = pd.DataFrame(metrics_rows)

    if verbose:
        print(f"\n{'='*60}")
        print(f"SELESAI — {len(solutions)}/{attempt} attempt menghasilkan solusi valid")
        print(f"{'='*60}\n")

    return solutions, metrics_df

def fitness_s2(df_final, weights, approach_td, config, max_reberth=2):
    cat_weights     = weights['CATEGORY_WEIGHTS']
    sop_multiplier  = float(weights['PENALTY_WAIT_SOP'])
    sop_threshold_h = float(weights['WAIT_THRESHOLD'])
    penalty_reberth = float(weights['PENALTY_REBERTHING'])

    reberth_gap = approach_td + approach_td

    assigned = df_final[df_final['DERMAGA_ASSIGNED'].notna()].copy()

    if assigned.empty:
        return float('inf'), 0.0, 0, 0, 0

    term_wait     = 0.0
    term_sop      = 0.0
    term_reberth  = 0.0
    total_wait_h  = 0.0
    n_late        = 0
    total_reberth = 0

    for kid, grp in assigned.groupby('ID_KUNJUNGAN'):
        grp = grp.sort_values('MULAI_SANDAR').reset_index(drop=True)

        first    = grp.iloc[0]
        kategori = str(first.get('KATEGORI')).upper().strip()
        cat_w    = float(cat_weights.get(kategori))

        kedatangan = pd.Timestamp(first['KEDATANGAN'])

        for i, row in grp.iterrows():
            mulai = pd.Timestamp(row['MULAI_SANDAR'])

            if i == 0:
                w = (mulai - (kedatangan + approach_td)).total_seconds() / 3600.0
            else:
                prev_selesai = pd.Timestamp(grp.loc[i - 1, 'SELESAI_SANDAR'])
                earliest_ok  = prev_selesai + reberth_gap
                w = (mulai - earliest_ok).total_seconds() / 3600.0

            w = max(0.0, w)

            late_seg = int(w > sop_threshold_h)  # tetap dihitung untuk reporting

            # ====== GANTI: penalti SOP proporsional terhadap excess ======
            excess_h = max(0.0, w - sop_threshold_h)
            term_wait   += cat_w * w
            term_sop    += cat_w * sop_multiplier * excess_h
            # ===============================================================

            total_wait_h += w
            n_late       += late_seg  # tetap jumlah segmen yang late

        # reberth tetap per kunjungan
        if 'SEGMENT_NO' in grp.columns and grp['SEGMENT_NO'].notna().any():
            n_segment = int(pd.to_numeric(grp['SEGMENT_NO'], errors='coerce').nunique())
        elif 'SEGMENT_NO_PLAN' in grp.columns and grp['SEGMENT_NO_PLAN'].notna().any():
            n_segment = int(pd.to_numeric(grp['SEGMENT_NO_PLAN'], errors='coerce').nunique())
        else:
            n_segment = len(grp)

        reberth_count = max(0, n_segment - 1)

        if reberth_count > max_reberth:
            raise ValueError(
                f"ID_KUNJUNGAN={kid}: reberth={reberth_count} > max_reberth={max_reberth}"
            )

        term_reberth  += cat_w * penalty_reberth * reberth_count
        total_reberth += reberth_count

    fitness = term_wait + term_sop + term_reberth

    if 'BERTH_PART_LIST' in assigned.columns:
        n_assigned = int(
            assigned['BERTH_PART_LIST'].apply(
                lambda x: len(x) if isinstance(x, list) else 1
            ).sum()
        )
    else:
        n_assigned = int(len(assigned))

    return float(fitness), float(total_wait_h), n_assigned, int(n_late), int(total_reberth)


# ---------------------------------------------------------------------------
# Evaluasi langsung dari df_schedule CH (tanpa run ulang simulator)
# ---------------------------------------------------------------------------
def _evaluate_ch_final_s2(df_final, df_raw, weights, config, max_reberth,
                           schedule_2d=None):
    approach_td = pd.Timedelta(hours=float(config['APPROACH_TIME']))

    n_raw = len(df_raw)

    if 'BERTH_PART_LIST' in df_final.columns:
        n_final_parts = int(
            df_final['BERTH_PART_LIST'].apply(
                lambda x: len(x) if isinstance(x, list) else 1
            ).sum()
        )
    else:
        n_final_parts = len(df_final)

    if n_final_parts != n_raw:
        raise ValueError(f"Part tidak lengkap: {n_final_parts}/{n_raw}")

    fit, wait, n_ass, n_late, n_rb = fitness_s2(
        df_final    = df_final,
        weights     = weights,
        approach_td = approach_td,
        config      = config,
        max_reberth = max_reberth,
    )

    return fit, wait, n_ass, n_late, n_rb, df_final, schedule_2d



# ---------------------------------------------------------------------------
# HELPER: deep copy kromosom
# ---------------------------------------------------------------------------
def _dc2(sched):
    return {k: v[:] for k, v in sched.items()}


# ---------------------------------------------------------------------------
# HELPER: validasi kromosom format_b
# ---------------------------------------------------------------------------
def _val2(sched, all_parts, part_elig=None):
    placed = [s for v in sched.values() for s in v]

    if len(placed) != len(all_parts):
        return False, f"jumlah {len(placed)} vs {len(all_parts)}"

    if len(set(placed)) != len(placed):
        return False, "duplikat"

    if set(placed) != set(all_parts):
        return False, f"hilang: {set(all_parts) - set(placed)}"

    if part_elig is not None:
        for bid, parts in sched.items():
            if bid == '_unassigned':
                continue

            for p in parts:
                elig = {str(x) for x in part_elig.get(p, [])}
                if str(bid) not in elig:
                    return False, f"{p} tidak eligible di dermaga {bid}"

    return True, "ok"

# ---------------------------------------------------------------------------
# HELPER: repair kromosom
# ---------------------------------------------------------------------------
def _repair2(sched, all_parts, part_elig, rng):
    seen = set()
    new_s = {k: [] for k in sched}
    new_s.setdefault('_unassigned', [])

    for bid, parts in sched.items():
        for p in parts:
            if p in seen:
                continue

            if bid != '_unassigned':
                elig = {str(x) for x in part_elig.get(p, [])}
                if str(bid) not in elig:
                    continue

            seen.add(p)
            new_s.setdefault(bid, []).append(p)

    for p in set(all_parts) - seen:
        elig = [str(b) for b in part_elig.get(p, []) if str(b) in new_s]
        bid = rng.choice(elig) if elig else '_unassigned'
        new_s.setdefault(bid, []).append(p)

    return new_s


# ---------------------------------------------------------------------------
# HELPER: df_final -> kromosom format_b
# ---------------------------------------------------------------------------
def _df_to_2d_s2(df_sol, all_bid, df_raw=None):
    sched = {bid: [] for bid in all_bid}
    sched['_unassigned'] = []

    # Bangun lookup part valid per KID dari df_raw
    valid_parts_per_kid = {}
    if df_raw is not None:
        for _, row in df_raw.iterrows():
            kid = str(row['ID_KUNJUNGAN'])
            pt  = int(row['BERTH_PART'])
            valid_parts_per_kid.setdefault(kid, set()).add(pt)

    assigned   = df_sol[df_sol['DERMAGA_ASSIGNED'].notna()].copy()
    unassigned = df_sol[df_sol['DERMAGA_ASSIGNED'].isna()]

    if 'MULAI_SANDAR' in assigned.columns:
        assigned = assigned.sort_values(['DERMAGA_ASSIGNED', 'MULAI_SANDAR'])

    for _, row in assigned.iterrows():
        bid       = str(row['DERMAGA_ASSIGNED'])
        kid       = str(row['ID_KUNJUNGAN'])
        part_list = row.get('BERTH_PART_LIST', None)
        parts     = part_list if isinstance(part_list, list) \
                    else [int(row.get('BERTH_PART', 1))]

        if valid_parts_per_kid:
            valid = valid_parts_per_kid.get(kid, set())
            parts = [pt for pt in parts if int(pt) in valid]
            if not parts:
                parts = [1]

        for pt in parts:
            key = f"{kid}_{int(pt)}"
            if bid in sched:
                sched[bid].append(key)
            else:
                sched['_unassigned'].append(key)

    for _, row in unassigned.iterrows():
        kid       = str(row['ID_KUNJUNGAN'])
        part_list = row.get('BERTH_PART_LIST', None)
        parts     = part_list if isinstance(part_list, list) \
                    else [int(row.get('BERTH_PART', 1))]

        if valid_parts_per_kid:
            valid = valid_parts_per_kid.get(kid, set())
            parts = [pt for pt in parts if int(pt) in valid]
            if not parts:
                parts = [1]

        for pt in parts:
            sched['_unassigned'].append(f"{kid}_{int(pt)}")

    return sched


# ---------------------------------------------------------------------------
# kromosom format_b -> ship_order & berth_order_per_ship
# ---------------------------------------------------------------------------
def _2d_to_run_params_s2(schedule_2d, df, eligibility):
    part_key_to_idx = {}
    idx_to_home_bid = {}

    for idx, row in df.iterrows():
        key = f"{row['ID_KUNJUNGAN']}_{int(row['BERTH_PART'])}"
        part_key_to_idx[key] = idx

    ship_order = []
    seen_idx   = set()

    for bid in sorted(k for k in schedule_2d if k != '_unassigned'):
        for part_key in schedule_2d[bid]:
            idx = part_key_to_idx.get(part_key)
            if idx is not None:
                idx_to_home_bid[idx] = str(bid)
                if idx not in seen_idx:
                    ship_order.append(idx)
                    seen_idx.add(idx)

    for part_key in schedule_2d.get('_unassigned', []):
        idx = part_key_to_idx.get(part_key)
        if idx is not None and idx not in seen_idx:
            ship_order.append(idx)
            seen_idx.add(idx)

    for idx in df.index:
        if idx not in seen_idx:
            ship_order.append(idx)
            seen_idx.add(idx)

    berth_order_per_ship = {}

    for idx in ship_order:
        elig     = eligibility.get(idx, [])[:]
        home_bid = idx_to_home_bid.get(idx)

        if home_bid is not None:
            primary = [b for b in elig if str(b['ID']) == str(home_bid)]
            berth_order_per_ship[idx] = primary
        else:
            berth_order_per_ship[idx] = elig

    return ship_order, berth_order_per_ship


# ============================================================
# FIND SLOT LB2
# ============================================================

def _depth_ok_lb2(t_start, t_end, draft, berth, config):
    EPS = float(config.get('EPSILON', 1e-6))
    kdl = float(berth['KEDALAMAN'])
    ukc = float(config['UNDER_KEEL_CLEARANCE'])
    tide_delta = float(config['TIDE_DELTA'])
    penalty = float(get_tide_depth_penalty(t_start, t_end, CONFIG))
    low_pen = ukc + tide_delta
    tide_flag = 1 if abs(penalty - low_pen) < EPS else 0
    kedalaman_aktual = kdl - ukc - tide_flag * tide_delta
    return float(draft) <= kedalaman_aktual + EPS, tide_flag, kedalaman_aktual

def _merge_intervals_lb2(intervals, EPS):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _free_gaps_lb2(active_sorted, kade_start, kade_end, safe, EPS):
    blocked = []
    for occ in active_sorted:
        bs = max(kade_start, float(occ['start_pos']) - safe)
        be = min(kade_end, float(occ['end_pos']) + safe)
        if bs < be - EPS:
            blocked.append([bs, be])
    blocked = _merge_intervals_lb2(blocked, EPS)
    gaps = []
    cursor = kade_start
    for bs, be in blocked:
        if bs - cursor > EPS:
            gaps.append((cursor, bs))
        cursor = max(cursor, be)
    if kade_end - cursor > EPS:
        gaps.append((cursor, kade_end))
    return gaps, blocked

def _candidate_positions_lb2(gap_start, gap_end, loa, EPS, preferred_pos=None):
    gap_len  = gap_end - gap_start
    if gap_len < loa - EPS:
        return []

    p_left   = gap_start
    p_right  = gap_end - loa
    p_center = gap_start + (gap_len - loa) / 2.0

    candidates = []

    if preferred_pos is not None and not pd.isna(preferred_pos):
        p_pref = float(preferred_pos)
        if p_pref >= gap_start - EPS and p_pref + loa <= gap_end + EPS:
            candidates.append(p_pref)

    if gap_len >= 2.0 * loa - EPS:
        candidates.append(p_center)

    candidates.append(p_left)

    if abs(p_right - p_left) > EPS:
        candidates.append(p_right)

    clean = []
    seen  = set()
    for p in candidates:
        p   = round(float(p), 6)
        key = round(p, 4)
        if key in seen:
            continue
        seen.add(key)
        clean.append(p)
    return clean

def _score_position_lb2(pos, loa, gap_start, gap_end,
                        preferred_pos=None, avg_loa_hint=None, EPS=1e-6,
                        is_reberth=False):
    end_pos     = pos + loa
    left_space  = max(0.0, pos - gap_start)
    right_space = max(0.0, gap_end - end_pos)

    if avg_loa_hint is None or pd.isna(avg_loa_hint):
        avg_loa_hint = loa

    frag_penalty = 0.0
    if left_space > EPS and left_space < avg_loa_hint - EPS:
        frag_penalty += avg_loa_hint - left_space
    if right_space > EPS and right_space < avg_loa_hint - EPS:
        frag_penalty += avg_loa_hint - right_space

    usable_sides = int(left_space >= avg_loa_hint - EPS) + int(right_space >= avg_loa_hint - EPS)
    if left_space > EPS and right_space > EPS and usable_sides == 0:
        frag_penalty += avg_loa_hint

    max_remaining = max(left_space, right_space)
    min_remaining = min(left_space, right_space)

    pref_penalty = 0.0
    if preferred_pos is not None and not pd.isna(preferred_pos):
        pref_penalty = abs(pos - float(preferred_pos))
    if is_reberth and preferred_pos is not None and not pd.isna(preferred_pos):
        if abs(float(pos) - float(preferred_pos)) <= EPS:
            return -1e12

    pref_weight = 5.0 if is_reberth else 2.0

    score = (
        10.0 * frag_penalty
        + pref_weight * pref_penalty
        + 0.10 * min_remaining
        - 1.0  * max_remaining
    )
    return float(score)

def find_slot_lb2(ship, berth, parent_occupancy, config,
                  allowed_starts=None, preferred_pos=None,
                  strict_preferred=False,
                  is_reberth=False):

    EPS      = float(config.get('EPSILON', 1e-6))
    safe     = float(config.get('CONTINUOUS_GAP_M', 0.0))
    jenis    = str(berth['JENIS']).upper().strip()
    loa      = float(ship['LOA'])
    draft    = float(ship['DRAFT'])
    earliest = pd.Timestamp(ship['MULAI_SANDAR_AWAL']).round('s')
    svc_td   = pd.Timedelta(hours=float(ship['BERTH_TIME']))
    kade_start   = float(berth['START'])
    kade_end     = float(berth['END'])
    step_td      = pd.Timedelta(minutes=int(config.get('TIME_STEP_MINUTES', 30)))
    avg_loa_hint = float(ship.get('_AVG_LOA_HINT', loa))

    max_horizon = earliest + pd.Timedelta(hours=float(config.get('MAX_WAIT_HOURS', 240)))

    occ_sorted = sorted(
        parent_occupancy,
        key=lambda o: (pd.Timestamp(o['start']).value, pd.Timestamp(o['end']).value)
    )

    def _active_occ(t_start, t_end):
        ts = pd.Timestamp(t_start)
        te = pd.Timestamp(t_end)
        return [
            o for o in occ_sorted
            if pd.Timestamp(o['start']) < te and pd.Timestamp(o['end']) > ts
        ]

    _depth_cache: dict = {}

    def _depth_ok_cached(t_start, t_end):
        key = (pd.Timestamp(t_start).value, pd.Timestamp(t_end).value)
        if key not in _depth_cache:
            ok, tf, _ = _depth_ok_lb2(t_start, t_end, draft, berth, CONFIG)
            _depth_cache[key] = (ok, tf)
        return _depth_cache[key]

    # ------------------------------------------------------------------
    # tempatkan kapal mulai t_start
    # ------------------------------------------------------------------
    def _try_at(t_start):
        t_start = pd.Timestamp(t_start).round('s')
        t_end   = (t_start + svc_td).round('s')
        if t_start < earliest:
            return None

        ok, _ = _depth_ok_cached(t_start, t_end)
        if not ok:
            return None

        active = _active_occ(t_start, t_end)

        if jenis == 'DISKRIT':
            pos     = kade_start + float(config.get('LOA_MARGIN_DISKRIT', 0.0))
            end_pos = pos + loa
            if end_pos > kade_end + EPS or active:
                return None
            return (t_start, t_end, round(pos, 2), round(end_pos, 2))

        active_sorted = sorted(active, key=lambda x: float(x['start_pos']))
        gaps, _       = _free_gaps_lb2(active_sorted, kade_start, kade_end, safe, EPS)

        if not gaps:
            return None

        best_slot  = None
        best_score = float('inf')

        for gap_start, gap_end in gaps:
            if gap_end - gap_start < loa - EPS:
                continue

            if strict_preferred:
                if preferred_pos is None or pd.isna(preferred_pos):
                    continue
                p_pref = float(preferred_pos)
                if not (p_pref >= gap_start - EPS and p_pref + loa <= gap_end + EPS):
                    continue
                pos_candidates = [p_pref]
            else:
                pos_candidates = _candidate_positions_lb2(
                    gap_start, gap_end, loa, EPS, preferred_pos
                )

            for pos in pos_candidates:
                pos     = round(float(pos), 6)
                end_pos = pos + loa
                if pos < kade_start - EPS or end_pos > kade_end + EPS:
                    continue
                conflict = any(
                    pos < float(occ['end_pos']) + safe - EPS
                    and end_pos + safe > float(occ['start_pos']) + EPS
                    for occ in active_sorted
                )
                if conflict:
                    continue
                score = _score_position_lb2(
                    pos, loa, gap_start, gap_end,
                    preferred_pos, avg_loa_hint, EPS,
                    is_reberth=is_reberth
                )
                if score < best_score - EPS:
                    best_score = score
                    best_slot  = (t_start, t_end, round(pos, 2), round(end_pos, 2))

        return best_slot

    candidate_starts = {earliest}

    if allowed_starts is not None:
        for t in allowed_starts:
            t = pd.Timestamp(t).round('s')
            if t >= earliest:
                candidate_starts.add(t)

    for occ in occ_sorted:
        occ_end = pd.Timestamp(occ['end']).round('s')
        if earliest <= occ_end <= max_horizon:
            candidate_starts.add(occ_end)

    candidate_starts_list = sorted(candidate_starts)

    t_cursor      = candidate_starts_list[0] if candidate_starts_list else earliest
    safety_max    = int(config.get('MAX_SLOT_ITER', 2000))
    safety_count  = 0
    cs_ptr        = 0

    while t_cursor <= max_horizon and safety_count < safety_max:
        safety_count += 1

        slot = _try_at(t_cursor)
        if slot is not None:
            return slot

        t_end_try = t_cursor + svc_td

        ok, _ = _depth_ok_cached(t_cursor, t_end_try)
        if not ok:
            cs_ptr = bisect.bisect_right(candidate_starts_list, t_cursor)
            if cs_ptr < len(candidate_starts_list):
                t_next = candidate_starts_list[cs_ptr]
            else:
                t_next = t_cursor + step_td
            t_cursor = t_next.round('s')
            continue

        active = _active_occ(t_cursor, t_end_try)
        conflict_ends = [
            pd.Timestamp(occ['end']).round('s')
            for occ in active
            if pd.Timestamp(occ['end']).round('s') > t_cursor
        ]

        if conflict_ends:
            next_by_conflict = min(conflict_ends)
        else:
            next_by_conflict = None

        cs_ptr_next = bisect.bisect_right(candidate_starts_list, t_cursor)
        next_by_cand = (
            candidate_starts_list[cs_ptr_next]
            if cs_ptr_next < len(candidate_starts_list)
            else None
        )

        next_step = (t_cursor + step_td).round('s')

        candidates_next = [next_step]
        if next_by_conflict is not None and next_by_conflict > t_cursor:
            candidates_next.append(next_by_conflict)
        if next_by_cand is not None and next_by_cand > t_cursor:
            candidates_next.append(next_by_cand)

        t_cursor = min(candidates_next)

    return None

def _build_parent_occ_from_df(df_unchanged, id_to_parent, berths, config,
                               exclude_kids=None):
    EPS        = float(config.get('EPSILON', 1e-6))
    safe       = float(config.get('CONTINUOUS_GAP_M', 0.0))
    ukc        = float(config['UNDER_KEEL_CLEARANCE'])
    tide_delta = float(config['TIDE_DELTA'])

    if exclude_kids is None:
        _exclude = set()
    else:
        _exclude = {str(k) for k in exclude_kids}

    berth_map      = {str(b['ID']): b for b in berths}
    unique_parents = set(id_to_parent.values())
    parent_occ     = {p: [] for p in unique_parents}

    assigned = df_unchanged[df_unchanged['DERMAGA_ASSIGNED'].notna()].copy()

    for _, row in assigned.iterrows():
        kid = str(row['ID_KUNJUNGAN'])

        if kid in _exclude:
            continue

        bid    = str(row['DERMAGA_ASSIGNED'])
        parent = id_to_parent.get(bid)
        if parent is None:
            continue

        berth_info = berth_map.get(bid)
        if berth_info is None:
            continue

        t_start = pd.Timestamp(row['MULAI_SANDAR']).round('s')
        t_end   = pd.Timestamp(row['SELESAI_SANDAR']).round('s')
        sp      = float(row['POSISI_START_M'])
        ep      = float(row['POSISI_END_M'])

        kade_start = float(berth_info['START'])
        kade_end   = float(berth_info['END'])
        if sp < kade_start - EPS or ep > kade_end + EPS:
            continue

        draft     = float(row.get('DRAFT', 0.0))
        kdl       = float(berth_info['KEDALAMAN'])
        penalty   = float(get_tide_depth_penalty(t_start, t_end, config))
        low_pen   = ukc + tide_delta
        tide_flag = 1 if abs(penalty - low_pen) < EPS else 0
        eff_kdl   = kdl - ukc - tide_flag * tide_delta
        if draft > eff_kdl + EPS:
            continue

        jenis    = str(berth_info.get('JENIS', 'KONTINU')).upper().strip()
        existing = parent_occ[parent]
        collision = False

        for occ in existing:
            if t_start >= pd.Timestamp(occ['end']) or t_end <= pd.Timestamp(occ['start']):
                continue
            if jenis == 'DISKRIT':
                if str(occ['bid']) == bid:
                    collision = True
                    break
            else:
                occ_sp = float(occ['start_pos'])
                occ_ep = float(occ['end_pos'])
                if sp < occ_ep + safe - EPS and ep > occ_sp - safe + EPS:
                    collision = True
                    break

        if collision:
            continue

        parent_occ[parent].append({
            'start'    : t_start,
            'end'      : t_end,
            'start_pos': sp,
            'end_pos'  : ep,
            'kid'      : kid,
            'part'     : 1,
            'bid'      : bid,
            'idx'      : None,
        })

    return parent_occ

# =============================================================================
# _run_partial_s2_lb2
# =============================================================================
def _run_partial_s2_lb2(df_sub, df_unchanged, berths, id_to_parent, id_to_nama,
                         eligibility, config, schedule_2d, max_reberth=2,
                         affected_kids=None):
    valid_part_keys = {
        f"{row['ID_KUNJUNGAN']}_{int(row['BERTH_PART'])}"
        for _, row in df_sub.iterrows()
    }

    clean_2d = {}
    for bid, parts in schedule_2d.items():
        clean_2d[bid] = [p for p in parts if p in valid_part_keys]

    placed = {p for parts in clean_2d.values() for p in parts}
    for pk in sorted(valid_part_keys - placed):
        clean_2d.setdefault('_unassigned', []).append(pk)

    schedule_2d = clean_2d
    parent_occ_seed = _build_parent_occ_from_df(
        df_unchanged,
        id_to_parent,
        berths,
        CONFIG,
        exclude_kids=affected_kids,
    )

    kade_capacities = _compute_kade_capacities(berths)

    df_sub2 = df_sub.copy()
    if 'LOA' in df_sub2.columns:
        all_loa = pd.concat([
            pd.to_numeric(df_sub['LOA'], errors='coerce'),
            pd.to_numeric(df_unchanged['LOA'], errors='coerce')
            if 'LOA' in df_unchanged.columns
            else pd.Series(dtype=float)
        ]).dropna()
        avg_loa = float(all_loa.mean()) if not all_loa.empty else float(df_sub2['LOA'].mean())
        if not pd.isna(avg_loa):
            df_sub2['_AVG_LOA_HINT'] = avg_loa

    elig_sub = {i: eligibility[i] for i in df_sub.index if i in eligibility}

    ship_order_sub, berth_order_sub = _2d_to_run_params_s2(
        schedule_2d, df_sub2, elig_sub
    )
    valid_idx       = set(df_sub2.index)
    ship_order_sub  = [i for i in ship_order_sub if i in valid_idx]
    berth_order_sub = {
        i: berth_order_sub[i]
        for i in ship_order_sub
        if i in berth_order_sub
    }

    results = _run_single_s2_with_parent_occ(
        df                   = df_sub2,
        berths               = berths,
        id_to_parent         = id_to_parent,
        id_to_nama           = id_to_nama,
        eligibility          = elig_sub,
        config               = config,
        ship_order           = ship_order_sub,
        berth_order_per_ship = berth_order_sub,
        max_reberth          = max_reberth,
        parent_occ_seed      = parent_occ_seed,
        kade_capacities      = kade_capacities,
    )

    return results

# ---------------------------------------------------------------------------
# EVALUASI KROMOSOM S2 — partial re-sim
# ---------------------------------------------------------------------------
def _full_evaluate_s2(schedule_2d, df, berths, id_to_parent, id_to_nama,
                       eligibility, config, weights, max_reberth, calc_fitness):
    ship_order, berth_order_per_ship = _2d_to_run_params_s2(
        schedule_2d, df, eligibility
    )

    kade_capacities = _compute_kade_capacities(berths)

    df2 = df.copy()
    if 'LOA' in df2.columns:
        df2['_AVG_LOA_HINT'] = float(df2['LOA'].mean())

    results = _run_single_s2_with_parent_occ(
        df                   = df2,
        berths               = berths,
        id_to_parent         = id_to_parent,
        id_to_nama           = id_to_nama,
        eligibility          = eligibility,
        config               = config,
        ship_order           = ship_order,
        berth_order_per_ship = berth_order_per_ship,
        max_reberth          = max_reberth,
        parent_occ_seed      = None,
        kade_capacities      = kade_capacities,
    )

    df_sol = _build_output_df2(df2, results)

    if 'SEGMENT_NO_PLAN' not in df_sol.columns:
        seg_map = {r['idx']: r.get('SEGMENT_NO_PLAN') for r in results}
        df_sol['SEGMENT_NO_PLAN'] = df_sol.index.map(seg_map)

    df_sol['SEGMENT_NO_PLAN'] = (
        pd.to_numeric(df_sol['SEGMENT_NO_PLAN'], errors='coerce')
        .fillna(1).astype(int)
    )

    df_merged = _merge_parts_s2(df_sol, CONFIG, max_reberth=max_reberth)

    issues = _validate_s2(df_merged, CONFIG, berths=berths, max_reberth=max_reberth)
    if issues:
        return None

    df_final = _recalculate_s2(
        df_merged, df, berths, id_to_parent, id_to_nama,
        CONFIG, weights, max_reberth=max_reberth,
    )

    issues2 = _validate_s2(df_final, CONFIG, berths=berths, max_reberth=max_reberth)
    if issues2:
        return None

    fit, wait, n_ass, n_late, n_rb = calc_fitness(df_final)
    return (fit, wait, n_ass, n_late, n_rb, df_final, schedule_2d)

def _evaluate_s2(schedule_2d, df, berths, id_to_parent, id_to_nama,
                 eligibility, config, weights, max_reberth,
                 prev_result=None, affected_bids=None):
    approach_td = pd.Timedelta(hours=float(config['APPROACH_TIME']))

    def _calc_fitness(df_final):
        return fitness_s2(
            df_final    = df_final,
            weights     = weights,
            approach_td = approach_td,
            config      = config,
            max_reberth = max_reberth,
        )

    out = None

    try:
        # ============================================================
        # SAFETY: prev_result wajib ada untuk partial evaluation
        # ============================================================
        if prev_result is None or len(prev_result) < 6 or prev_result[5] is None:
            raise ValueError("partial gagal: prev_result tidak ada")


        if affected_bids is None or len(affected_bids) == 0:
            raise ValueError("partial gagal: affected_bids kosong")

        prev_df_final = prev_result[5].copy()
        col_bid = 'DERMAGA_ASSIGNED'

        # ============================================================
        # 1. Ambil bid/kade yang terdampak mutasi
        # ============================================================
        affected_str = {str(b) for b in affected_bids if b is not None}

        if not affected_str:
            raise ValueError("partial gagal: affected_str kosong")

        # ============================================================
        # 2. Naikkan level affected dari bid ke parent
        # ============================================================
        affected_parents = {
            id_to_parent[str(b)]
            for b in affected_str
            if str(b) in id_to_parent
        }

        if not affected_parents:
            raise ValueError("partial gagal: affected_parents kosong")

        affected_str = {
            str(bid)
            for bid, parent in id_to_parent.items()
            if parent in affected_parents
        }

        if not affected_str:
            raise ValueError("partial gagal: affected_str kosong")

        # ============================================================
        # 3. Kumpulkan SEMUA kapal yang harus di-resim
        # ============================================================
        affected_kids = set()

        mask_affected_prev = (
            prev_df_final[col_bid].notna()
            & prev_df_final[col_bid].astype(str).isin(affected_str)
        )
        prev_affected_kids = prev_df_final.loc[
            mask_affected_prev,
            'ID_KUNJUNGAN'
        ].astype(str).unique()
        affected_kids.update(prev_affected_kids)

        for bid in affected_str:
            for pk in schedule_2d.get(str(bid), []):
                kid = '_'.join(str(pk).split('_')[:-1])
                if kid:
                    affected_kids.add(kid)

            prev_in_bid = prev_df_final[
                prev_df_final[col_bid].notna()
                & (prev_df_final[col_bid].astype(str) == str(bid))
            ]
            for kid in prev_in_bid['ID_KUNJUNGAN'].astype(str).unique():
                affected_kids.add(kid)

        if not affected_kids:
            raise ValueError("partial gagal: affected_kids kosong")

        kids_to_check = set(affected_kids)
        for _chain_iter in range(3):   # max 3 iterasi, cukup untuk reberth=2
            newly_added = set()

            for kid in kids_to_check:
                kid_rows = prev_df_final[
                    prev_df_final['ID_KUNJUNGAN'].astype(str) == kid
                ]
                # Cek semua parent yang pernah ditempati KID ini
                for bid_k in kid_rows[col_bid].dropna().astype(str).unique():
                    parent_k = id_to_parent.get(str(bid_k))
                    if parent_k is None:
                        continue

                    if parent_k in affected_parents:
                        # Sudah tercakup — lewati
                        continue

                    affected_parents.add(parent_k)

                    new_bids_in_parent = {
                        str(b_id)
                        for b_id, p in id_to_parent.items()
                        if p == parent_k
                    }
                    affected_str.update(new_bids_in_parent)

                    sibling_mask = (
                        prev_df_final[col_bid].notna()
                        & prev_df_final[col_bid].astype(str).isin(new_bids_in_parent)
                    )
                    siblings = prev_df_final.loc[
                        sibling_mask, 'ID_KUNJUNGAN'
                    ].astype(str).unique()
                    newly_added.update(siblings)

                    for b_id in new_bids_in_parent:
                        for pk in schedule_2d.get(str(b_id), []):
                            k2 = '_'.join(str(pk).split('_')[:-1])
                            if k2:
                                newly_added.add(k2)

            truly_new = newly_added - affected_kids
            if not truly_new:
                break

            affected_kids.update(truly_new)
            kids_to_check = truly_new

        # ============================================================
        # 4. Split data
        # ============================================================
        df_unchanged = prev_df_final[
            ~prev_df_final['ID_KUNJUNGAN'].astype(str).isin(affected_kids)
        ].copy()

        df_sub = df[
            df['ID_KUNJUNGAN'].astype(str).isin(affected_kids)
        ].copy()

        if df_sub.empty:
            raise ValueError("partial gagal: df_sub kosong")

        # ============================================================
        # 5. Safety check
        # ============================================================
        unchanged_affected = df_unchanged[
            df_unchanged[col_bid].notna()
            & df_unchanged[col_bid].astype(str).isin(affected_str)
        ]

        if not unchanged_affected.empty:
            raise ValueError("partial gagal: unchanged masih ada di affected_str")

        # ============================================================
        # 6. Run partial reschedule
        # ============================================================
        results_sub = _run_partial_s2_lb2(
            df_sub        = df_sub,
            df_unchanged  = df_unchanged,
            berths        = berths,
            id_to_parent  = id_to_parent,
            id_to_nama    = id_to_nama,
            eligibility   = eligibility,
            config        = config,
            schedule_2d   = schedule_2d,
            max_reberth   = max_reberth,
            affected_kids = affected_kids,
        )

        df_sol_sub = _build_output_df2(df_sub, results_sub)

        # ============================================================
        # 7. Pastikan SEGMENT_NO_PLAN ada
        # ============================================================
        if 'SEGMENT_NO_PLAN' not in df_sol_sub.columns:
            seg_map = {
                r['idx']: r.get('SEGMENT_NO_PLAN')
                for r in results_sub
            }
            df_sol_sub['SEGMENT_NO_PLAN'] = df_sol_sub.index.map(seg_map)

        df_sol_sub['SEGMENT_NO_PLAN'] = pd.to_numeric(
            df_sol_sub['SEGMENT_NO_PLAN'],
            errors='coerce'
        ).fillna(1).astype(int)

        # ============================================================
        # 8. Merge hasil berth part
        # ============================================================
        df_merged_sub = _merge_parts_s2(
            df_sol_sub,
            CONFIG,
            max_reberth=max_reberth
        )

        df_combined = pd.concat(
            [df_unchanged, df_merged_sub],
            ignore_index=True
        )

        issues_before_recalc = _validate_s2(
            df_combined,
            CONFIG,
            berths=berths,
            max_reberth=max_reberth
        )

        if issues_before_recalc:
            raise ValueError("partial gagal: validate sebelum recalc gagal")

        df_final = _recalculate_s2(
            df_combined,
            df,
            berths,
            id_to_parent,
            id_to_nama,
            CONFIG,
            WEIGHTS,
            max_reberth=max_reberth,
        )

        issues_after_recalc = _validate_s2(
            df_final,
            CONFIG,
            berths=berths,
            max_reberth=max_reberth
        )

        if issues_after_recalc:
            raise ValueError("partial gagal: validate setelah recalc gagal")

        # ============================================================
        # 12. Hitung fitness
        # ============================================================
        fit, wait, n_ass, n_late, n_rb = _calc_fitness(df_final)

        out = (
            fit,
            wait,
            n_ass,
            n_late,
            n_rb,
            df_final,
            schedule_2d
        )

    except Exception:
        out = None

    if out is not None:
        return out

    # Fallback full re-simulation
    try:
        approach_td = pd.Timedelta(hours=float(config['APPROACH_TIME']))

        def _calc_fitness(df_final):
            return fitness_s2(
                df_final    = df_final,
                weights     = weights,
                approach_td = approach_td,
                config      = config,
                max_reberth = max_reberth,
            )

        out = _full_evaluate_s2(
            schedule_2d  = schedule_2d,
            df           = df,
            berths       = berths,
            id_to_parent = id_to_parent,
            id_to_nama   = id_to_nama,
            eligibility  = eligibility,
            config       = config,
            weights      = weights,
            max_reberth  = max_reberth,
            calc_fitness = _calc_fitness,
        )
    except Exception:
        return None

    return out

# ===========================================================================
# OPERATOR MUTASI S2
# ===========================================================================

def _kid_from_part(part_key):
    return '_'.join(part_key.split('_')[:-1])


_ETA_CACHE_S2 = {}
def _eta_of_kid(kid, df_ships):
    kid = str(kid)
    cache_key = (id(df_ships), kid)

    if cache_key in _ETA_CACHE_S2:
        return _ETA_CACHE_S2[cache_key]

    rows = df_ships[df_ships['ID_KUNJUNGAN'].astype(str) == kid]

    if rows.empty:
        val = pd.Timestamp('2099-01-01')
    else:
        val = pd.Timestamp(rows['KEDATANGAN'].iloc[0])

    _ETA_CACHE_S2[cache_key] = val
    return val

def _would_exceed_max_reberth_after_move(schedule_2d, part_key,
                                         from_bid, to_bid, max_reberth):
    kid       = _kid_from_part(part_key)
    used_bids = {
        bid for bid, parts in schedule_2d.items()
        if bid != '_unassigned'
        and any(_kid_from_part(p) == kid for p in parts)
    }
    used_after = set(used_bids)
    parts_left_in_from = [
        p for p in schedule_2d.get(from_bid, [])
        if p != part_key and _kid_from_part(p) == kid
    ]
    if not parts_left_in_from:
        used_after.discard(from_bid)
    used_after.add(to_bid)
    return max(0, len(used_after) - 1) > max_reberth

# ===========================================================================
# reberth count per kid dari schedule_2d
# ===========================================================================

def _reberthing_count_from_schedule(kid, schedule_2d):
    bids = [
        bid for bid, parts in schedule_2d.items()
        if bid != '_unassigned'
        and any(_kid_from_part(p) == kid for p in parts)
    ]
    return max(0, len(set(bids)) - 1)


def _reberth_count_of_part(part_key, schedule_2d):
    """
    Shortcut: reberth count dari kid pemilik part_key.
    """
    return _reberthing_count_from_schedule(_kid_from_part(part_key), schedule_2d)


# ===========================================================================
# _pick_part
# ===========================================================================
def _pick_part(schedule_2d, rng, prev_result=None, weights=None,
               allow_reberth2=False):

    pool = [
        (part_key, bid)
        for bid, parts in schedule_2d.items()
        if bid != '_unassigned'
        for part_key in parts
    ]

    if not pool:
        return None, None

    # Semua kapal boleh dipilih termasuk reberth=2
    eligible_pool = pool

    if rng.random() < 0.25:
        return rng.choice(eligible_pool)

    if prev_result is None or len(prev_result) < 6:
        return rng.choice(eligible_pool)

    df_final = prev_result[5]
    if df_final is None or df_final.empty:
        return rng.choice(eligible_pool)

    df_ass = df_final[df_final['DERMAGA_ASSIGNED'].notna()].copy()
    if df_ass.empty:
        return rng.choice(eligible_pool)

    sop_threshold = 1.5
    if weights is not None and 'WAIT_THRESHOLD' in weights:
        sop_threshold = float(weights['WAIT_THRESHOLD'])

    kade_density = {}
    for bid, parts in schedule_2d.items():
        if bid != '_unassigned':
            kade_density[bid] = len(parts)
    max_density = max(kade_density.values()) if kade_density else 1

    kid_score = {}
    for kid, grp in df_ass.groupby('ID_KUNJUNGAN'):
        kid = str(kid)
        wait_total = float(grp['WAITING_TIME_HOURS'].sum()) \
            if 'WAITING_TIME_HOURS' in grp.columns else 0.0
        rb = _reberthing_count_from_schedule(kid, schedule_2d)
        n_late_segs = sum(
            1 for _, r in grp.iterrows()
            if float(r.get('WAITING_TIME_HOURS', 0.0)) > sop_threshold
        )
        kategori = str(grp['KATEGORI'].iloc[0]).upper().strip()
        cat_w = float(weights['CATEGORY_WEIGHTS'].get(kategori, 1.0)) if weights else 1.0

        score = (
            cat_w * max(wait_total, 0.0) +
            cat_w * 15.0 * n_late_segs
            + 10.0 * rb
            + 1e-6
        )
        kid_score[kid] = score

    scored_pool = []
    for part_key, bid in eligible_pool:
        kid = _kid_from_part(part_key)
        base_score = kid_score.get(str(kid), 1e-6)
        density = kade_density.get(bid, 0)
        density_bonus = 1.0 + (density / max_density)
        scored_pool.append((base_score * density_bonus, part_key, bid))

    total = sum(s for s, _, _ in scored_pool)
    if total <= 0:
        return rng.choice(eligible_pool)

    r = rng.random() * total
    acc = 0.0
    for score, part_key, bid in scored_pool:
        acc += score
        if acc >= r:
            return part_key, bid

    _, part_key, bid = scored_pool[-1]
    return part_key, bid



# ===========================================================================
# op_flip_s2 — REVISED
# ===========================================================================

def op_flip_s2(schedule_2d, chosen_part, home_bid, rng, **_):
    new_2d = _dc2(schedule_2d)
    lst    = new_2d[home_bid]
    n      = len(lst)
    if n < 2:
        return new_2d, set()

    i = lst.index(chosen_part)

    # Pilih posisi lain secara random — semua posisi boleh
    candidates = [k for k in range(n) if k != i]
    if not candidates:
        return new_2d, set()

    j      = rng.choice(candidates)
    lo, hi = (i, j) if i < j else (j, i)

    if lo >= hi:
        return new_2d, set()

    lst[lo:hi + 1] = lst[lo:hi + 1][::-1]
    return new_2d, {home_bid}

# ===========================================================================
# op_move_push_s2
# ===========================================================================
def op_move_push_s2(schedule_2d, chosen_part, home_bid, rng, **_):
    new_2d = _dc2(schedule_2d)
    lst    = new_2d[home_bid]
    n      = len(lst)

    if n < 2:
        return new_2d, set()

    i = lst.index(chosen_part)

    right_positions = [p for p in range(i + 1, n)]

    if not right_positions:
        return new_2d, set()

    push_positions = [i + 1] if i + 1 < n else []

    if rng.random() < 0.7 and push_positions:
        target_pos_before = rng.choice(push_positions)
    else:
        target_pos_before = rng.choice(right_positions)

    lst.remove(chosen_part)

    insert_pos = target_pos_before

    insert_pos = max(0, min(insert_pos, len(lst)))
    lst.insert(insert_pos, chosen_part)

    return new_2d, {home_bid}

# ===========================================================================
# op_2swap_same_s2
# ===========================================================================
def op_2swap_same_s2(schedule_2d, chosen_part, home_bid, rng, **_):
    new_2d = _dc2(schedule_2d)

    home_bid = str(home_bid)

    if home_bid not in new_2d:
        return new_2d, set()

    if chosen_part not in new_2d.get(home_bid, []):
        return new_2d, set()

    lst = new_2d[home_bid]
    n   = len(lst)

    # Minimal A B C D supaya bisa swap AB dan CD
    if n < 4:
        return new_2d, set()

    pos = lst.index(chosen_part)

    # Anchor yang boleh dipilih adalah range tengah:
    # A B C D E F -> anchor valid = B C D E
    valid_anchor = list(range(1, n - 1))

    if pos not in valid_anchor:
        return new_2d, set()

    valid_pairs = []

    for a in valid_anchor:
        for b in valid_anchor:
            if a >= b:
                continue

            # blok kanan = [b, b+1], jadi b+1 harus ada
            if b + 1 >= n:
                continue

            # chosen_part harus menjadi salah satu anchor yang dipilih
            if pos not in (a, b):
                continue

            # blok kiri  = [a-1, a]
            # blok kanan = [b, b+1]
            # Dengan a < b, blok tidak overlap.
            valid_pairs.append((a, b))

    if not valid_pairs:
        return new_2d, set()

    a, b = rng.choice(valid_pairs)

    left_block  = lst[a - 1:a + 1]
    right_block = lst[b:b + 2]

    # Contoh:
    # A B C D E F
    # a = index C, b = index E
    # left_block  = B C
    # right_block = E F
    # hasil       = A E F D B C
    lst[a - 1:a + 1] = right_block
    lst[b:b + 2]     = left_block

    return new_2d, {home_bid}

# ===========================================================================
# ship_fits_berth_s2 — adaptor S2 + cache
# part_key: "ID_KUNJUNGAN_BERTH_PART"
# bid     : ID dermaga
# ===========================================================================

_SHIP_FITS_BERTH_S2_CACHE = {}

def _part_no_from_key(part_key):
    return int(str(part_key).split('_')[-1])


def ship_fits_berth_s2(part_key, bid, df_ships, berth_map, config):
    cache_key = (id(df_ships), id(berth_map), str(part_key), str(bid))

    if cache_key in _SHIP_FITS_BERTH_S2_CACHE:
        return _SHIP_FITS_BERTH_S2_CACHE[cache_key]

    berth = berth_map.get(str(bid))
    if berth is None:
        _SHIP_FITS_BERTH_S2_CACHE[cache_key] = False
        return False

    kid = _kid_from_part(part_key)
    part_no = _part_no_from_key(part_key)

    rows = df_ships[
        (df_ships['ID_KUNJUNGAN'].astype(str) == str(kid))
        & (df_ships['BERTH_PART'].astype(int) == int(part_no))
    ]

    if rows.empty:
        _SHIP_FITS_BERTH_S2_CACHE[cache_key] = False
        return False

    ship = rows.iloc[0]

    ok = bool(ship_fits_berth(ship, berth, config))

    _SHIP_FITS_BERTH_S2_CACHE[cache_key] = ok
    return ok


# ===========================================================================
# op_interchange_s2
# ===========================================================================
def op_interchange_s2(schedule_2d, chosen_part, home_bid, rng,
                      df_ships, part_elig, max_reberth=2,
                      df_final=None, berth_map=None, config=None, **_):

    new_2d = _dc2(schedule_2d)

    home_bid = str(home_bid)

    if home_bid not in new_2d:
        return new_2d, set()

    if chosen_part not in new_2d.get(home_bid, []):
        return new_2d, set()

    kid    = _kid_from_part(chosen_part)
    rb_cnt = _reberth_count_of_part(chosen_part, schedule_2d)

    eta_chosen = _eta_of_kid(kid, df_ships)

    # ============================================================
    # OPSI 1: interchange dalam dermaga yang sama
    # Artinya swap chosen_part dengan 1 part lain di home_bid.
    # Contoh:
    # A B C D E, chosen=C, pair=E
    # hasil = A B E D C
    # ============================================================
    if rng.random() < 0.5:
        lst = new_2d[home_bid]

        if len(lst) >= 2:
            same_parts = [p for p in lst if p != chosen_part]

            if same_parts:
                scored_same = []

                for p in same_parts:
                    eta_p = _eta_of_kid(_kid_from_part(p), df_ships)
                    delta = abs((eta_chosen - eta_p).total_seconds())
                    scored_same.append((delta, p))

                scored_same.sort(key=lambda x: x[0])
                top_n = max(1, len(scored_same) // 3)

                pair_part = rng.choice(scored_same[:top_n])[1]

                i = lst.index(chosen_part)
                j = lst.index(pair_part)

                lst[i], lst[j] = lst[j], lst[i]

                return new_2d, {home_bid}

    # ============================================================
    # OPSI 2: interchange antar dermaga
    # Artinya swap chosen_part dengan 1 part dari dermaga target.
    # ============================================================
    elig_bids = [str(b) for b in part_elig.get(chosen_part, [])]

    other_bids = [
        b for b in elig_bids
        if b != home_bid
        and b in new_2d
        and b != '_unassigned'
        and new_2d.get(b, [])
        and not _would_exceed_max_reberth_after_move(
            schedule_2d, chosen_part, home_bid, b, max_reberth
        )
        and (
            berth_map is None or config is None
            or ship_fits_berth_s2(chosen_part, b, df_ships, berth_map, config)
        )
    ]

    # Kalau reberth sudah max, jangan membuka dermaga baru.
    # Hanya boleh ke dermaga yang sudah pernah dipakai oleh KID ini.
    if rb_cnt >= max_reberth:
        used_bids = {
            str(bid)
            for bid, parts in schedule_2d.items()
            if bid != '_unassigned'
            and str(bid) != home_bid
            and any(_kid_from_part(p) == kid for p in parts)
        }
        other_bids = [b for b in other_bids if b in used_bids]

    if not other_bids:
        return new_2d, set()

    valid_pairs = []

    for target_bid in other_bids:
        target_lst = new_2d.get(target_bid, [])

        for p in target_lst:
            # Calon pair p harus bisa balik ke home_bid
            if home_bid not in {str(b) for b in part_elig.get(p, [])}:
                continue

            if _would_exceed_max_reberth_after_move(
                schedule_2d, p, target_bid, home_bid, max_reberth
            ):
                continue

            if berth_map is not None and config is not None:
                if not ship_fits_berth_s2(p, home_bid, df_ships, berth_map, config):
                    continue

            # Kalau pair juga sudah max reberth, tetap harus dicek oleh fungsi di atas.
            eta_p = _eta_of_kid(_kid_from_part(p), df_ships)
            delta = abs((eta_chosen - eta_p).total_seconds())

            valid_pairs.append((delta, target_bid, p))

    if not valid_pairs:
        return new_2d, set()

    valid_pairs.sort(key=lambda x: x[0])
    top_n = max(1, len(valid_pairs) // 3)

    _, target_bid, pair_part = rng.choice(valid_pairs[:top_n])

    i = new_2d[home_bid].index(chosen_part)
    j = new_2d[target_bid].index(pair_part)

    new_2d[home_bid][i]     = pair_part
    new_2d[target_bid][j]   = chosen_part

    return new_2d, {home_bid, target_bid}


# ===========================================================================
# op_2swap_inter_s2 — perketat cek fisik dua arah
# ===========================================================================
def op_2swap_inter_s2(schedule_2d, chosen_part, home_bid, rng,
                      df_ships, part_elig, max_reberth=2,
                      df_final=None, berth_map=None, config=None, **_):

    new_2d = _dc2(schedule_2d)

    home_bid = str(home_bid)

    if home_bid not in new_2d:
        return new_2d, set()

    if chosen_part not in new_2d.get(home_bid, []):
        return new_2d, set()

    home_lst = new_2d[home_bid]
    n_home   = len(home_lst)

    # Butuh minimal 2 part di dermaga asal
    if n_home < 2:
        return new_2d, set()

    i = home_lst.index(chosen_part)

    # Untuk A B C D:
    # i=0 -> A B
    # i=1 -> B C
    # i=2 -> C D
    # i=3 -> tidak valid, karena tidak ada part setelah D
    if i + 1 >= n_home:
        return new_2d, set()

    home_block = home_lst[i:i + 2]

    # Kalau salah satu part dalam home_block sudah reberth >= max,
    # jangan dipaksa swap antar dermaga.
    for hp in home_block:
        if _reberth_count_of_part(hp, schedule_2d) >= max_reberth:
            return new_2d, set()

    # Target bid harus bisa menerima SEMUA part dalam home_block
    candidate_bids = None

    for hp in home_block:
        hp_elig = {
            str(b)
            for b in part_elig.get(hp, [])
            if str(b) in new_2d
            and str(b) != home_bid
            and str(b) != '_unassigned'
            and len(new_2d[str(b)]) >= 2
            and not _would_exceed_max_reberth_after_move(
                new_2d, hp, home_bid, str(b), max_reberth
            )
            and (
                berth_map is None or config is None
                or ship_fits_berth_s2(hp, str(b), df_ships, berth_map, config)
            )
        }

        if candidate_bids is None:
            candidate_bids = hp_elig
        else:
            candidate_bids = candidate_bids.intersection(hp_elig)

    if not candidate_bids:
        return new_2d, set()

    other_bids = list(candidate_bids)

    # Skor target bid berdasarkan kedekatan ETA rata-rata
    home_eta_avg = sum(
        _eta_of_kid(_kid_from_part(p), df_ships).timestamp()
        for p in home_block
    ) / len(home_block)

    scored_bids = []

    for bid in other_bids:
        target_lst = new_2d[bid]

        # Cari semua blok 2 part di target
        # Misal E F G H:
        # start 0 -> E F
        # start 1 -> F G
        # start 2 -> G H
        valid_target_blocks = []

        for j in range(0, len(target_lst) - 1):
            target_block = target_lst[j:j + 2]

            ok = True

            # Semua part target_block harus bisa pindah balik ke home_bid
            for tp in target_block:
                if _reberth_count_of_part(tp, schedule_2d) >= max_reberth:
                    ok = False
                    break

                if home_bid not in {str(b) for b in part_elig.get(tp, [])}:
                    ok = False
                    break

                if _would_exceed_max_reberth_after_move(
                    new_2d, tp, bid, home_bid, max_reberth
                ):
                    ok = False
                    break

                if berth_map is not None and config is not None:
                    if not ship_fits_berth_s2(tp, home_bid, df_ships, berth_map, config):
                        ok = False
                        break

            if not ok:
                continue

            target_eta_avg = sum(
                _eta_of_kid(_kid_from_part(p), df_ships).timestamp()
                for p in target_block
            ) / len(target_block)

            delta = abs(target_eta_avg - home_eta_avg)

            valid_target_blocks.append((delta, j, target_block))

        if valid_target_blocks:
            valid_target_blocks.sort(key=lambda x: x[0])
            best_delta, best_j, best_block = valid_target_blocks[0]
            scored_bids.append((best_delta, bid, best_j, best_block))

    if not scored_bids:
        return new_2d, set()

    scored_bids.sort(key=lambda x: x[0])
    top_n = max(1, len(scored_bids) // 3)

    _, target_bid, j, target_block = rng.choice(scored_bids[:top_n])

    # Swap 2 part vs 2 part
    new_2d[home_bid][i:i + 2]       = target_block
    new_2d[target_bid][j:j + 2]     = home_block

    return new_2d, {home_bid, target_bid}

# ===========================================================================
# _apply_op_s2
# ===========================================================================
def _apply_op_s2(schedule_2d, rng, df_ships, part_elig,
                 max_reberth=2, ls_mode=False, pop_fits=None,
                 prev_result=None, weights=None,
                 df_final=None, berth_map=None, config=None):

    chosen_part, home_bid = _pick_part(
        schedule_2d,
        rng,
        prev_result    = prev_result,
        weights        = weights,
        allow_reberth2 = False,
    )

    if chosen_part is None:
        return _dc2(schedule_2d), set()

    if home_bid is None or home_bid == '_unassigned':
        return _dc2(schedule_2d), set()

    if chosen_part not in schedule_2d.get(home_bid, []):
        return _dc2(schedule_2d), set()

    parent_key = tuple((k, tuple(v)) for k, v in sorted(schedule_2d.items()))

    rb_cnt  = _reberth_count_of_part(chosen_part, schedule_2d)
    home_lst = schedule_2d.get(home_bid, [])
    n_kade  = len(home_lst)

    # Posisi chosen_part di dermaga asal
    try:
        pos = home_lst.index(chosen_part)
    except ValueError:
        return _dc2(schedule_2d), set()

    # ============================================================
    # Boundary feasibility per operator
    # ============================================================

    # flip: butuh minimal 2 part, chosen boleh di mana saja
    can_flip = n_kade >= 2

    # move_push: hanya ke kanan
    # A B C D -> valid A/B/C, invalid D
    can_move_push = n_kade >= 2 and pos + 1 < n_kade

    # interchange same-berth: swap 1 vs 1, chosen boleh di mana saja
    can_interchange_same = n_kade >= 2

    # 2swap_same:
    # A B C D E F -> chosen/anchor valid B C D E
    can_2swap_same = n_kade >= 4 and 0 < pos < n_kade - 1

    # 2swap_inter:
    # A B C D -> valid A/B/C, invalid D
    can_2swap_inter_home = n_kade >= 2 and pos + 1 < n_kade

    # ============================================================
    # Cek dermaga lain untuk operator antar dermaga
    # ============================================================
    elig_other = []

    if rb_cnt < max_reberth:
        elig_other = [
            b for b in part_elig.get(chosen_part, [])
            if str(b) != str(home_bid)
            and str(b) in schedule_2d
            and str(b) != '_unassigned'
            and not _would_exceed_max_reberth_after_move(
                schedule_2d, chosen_part, home_bid, str(b), max_reberth
            )
            and (
                berth_map is None or config is None
                or ship_fits_berth_s2(chosen_part, str(b), df_ships, berth_map, config)
            )
        ]

    # ============================================================
    # Pilih operator: mutasi biasa vs local search
    # ============================================================
    valid_ops = []

    if not ls_mode:
        # ========================================================
        # MODE MUTASI BIASA / OFFSPRING
        # Tetap eksploratif: semua operator feasible boleh
        # ========================================================
        if can_flip:
            valid_ops.append(1)

        if can_move_push:
            valid_ops.append(3)

        if can_2swap_same:
            valid_ops.append(4)

        if can_interchange_same or elig_other:
            valid_ops.append(2)

        if rb_cnt < max_reberth and elig_other and can_2swap_inter_home:
            valid_ops.append(5)

    else:
        # ========================================================
        # MODE LOCAL SEARCH
        # HANYA pakai:
        # 2 = interchange
        # 4 = 2swap_same
        # 5 = 2swap_inter
        # ========================================================

        # 2swap_same: dalam dermaga, anchor tengah
        if can_2swap_same:
            valid_ops.append(4)

        # 2swap_inter: antar dermaga, chosen + kanan, belum max reberth
        if rb_cnt < max_reberth and elig_other and can_2swap_inter_home:
            valid_ops.append(5)

        # interchange: swap 1 vs 1, bisa same-berth atau inter-berth
        if can_interchange_same or elig_other:
            valid_ops.append(2)

    if not valid_ops:
        return _dc2(schedule_2d), set()

    op_id = rng.choice(valid_ops)

    kwargs = dict(
        schedule_2d = schedule_2d,
        chosen_part = chosen_part,
        home_bid    = home_bid,
        rng         = rng,
        df_ships    = df_ships,
        part_elig   = part_elig,
        max_reberth = max_reberth,
        df_final    = df_final,
        berth_map   = berth_map,
        config      = config,
    )

    if op_id == 1:
        child, aff = op_flip_s2(**kwargs)
    elif op_id == 2:
        child, aff = op_interchange_s2(**kwargs)
    elif op_id == 3:
        child, aff = op_move_push_s2(**kwargs)
    elif op_id == 4:
        child, aff = op_2swap_same_s2(**kwargs)
    elif op_id == 5:
        child, aff = op_2swap_inter_s2(**kwargs)
    else:
        child, aff = _dc2(schedule_2d), set()

    # Rapikan urutan part per KID dalam setiap dermaga
    for bid in list(child.keys()):
        if bid == '_unassigned':
            continue

        parts = child.get(bid, [])
        if len(parts) <= 1:
            continue

        kid_to_positions = {}
        for ppos, part_key in enumerate(parts):
            k = _kid_from_part(part_key)
            kid_to_positions.setdefault(k, []).append(ppos)

        for k, positions in kid_to_positions.items():
            if len(positions) <= 1:
                continue

            same_parts = [parts[ppos] for ppos in positions]
            same_parts_sorted = sorted(
                same_parts,
                key=lambda p: int(str(p).split('_')[-1])
            )

            for ppos, part_key_sorted in zip(positions, same_parts_sorted):
                parts[ppos] = part_key_sorted

        child[bid] = parts

    ok, msg = _val2(child, set(part_elig.keys()), part_elig=part_elig)

    if not ok:
        return _dc2(schedule_2d), set()

    child_key = tuple((k, tuple(v)) for k, v in sorted(child.items()))

    if child_key == parent_key:
        return _dc2(schedule_2d), set()

    if not aff:
        aff = {
            bid for bid in set(schedule_2d.keys()) | set(child.keys())
            if schedule_2d.get(bid, []) != child.get(bid, [])
        }

    return child, aff


# ===========================================================================
# _run_single_s2_with_parent_occ
# ===========================================================================
def _run_single_s2_with_parent_occ(df, berths, id_to_parent, id_to_nama, eligibility,
                                    config, ship_order, berth_order_per_ship,
                                    max_reberth=2,
                                    parent_occ_seed=None,
                                    kade_capacities=None):

    max_segments = int(max_reberth) + 1

    EPS        = float(config.get('EPSILON', 1e-6))
    safe       = float(config.get('CONTINUOUS_GAP_M', 0.0))
    ukc        = float(config['UNDER_KEEL_CLEARANCE'])
    tide_delta = float(config['TIDE_DELTA'])

    approach_td = pd.Timedelta(hours=float(config['APPROACH_TIME']))
    reberth_gap = approach_td + approach_td

    unique_parents = set(id_to_parent.values())

    parent_ends = {p: [] for p in unique_parents}
    parent_occs = {p: [] for p in unique_parents}

    if parent_occ_seed is not None:
        for p in unique_parents:
            for occ in parent_occ_seed.get(p, []):
                _occ_add(parent_ends[p], parent_occs[p], occ)

    berth_map = {str(b['ID']): b for b in berths}

    if kade_capacities is None:
        kade_capacities = _compute_kade_capacities(berths)

    kid_first_pos = {}
    for pos, ship_idx in enumerate(ship_order):
        kid = str(df.loc[ship_idx, 'ID_KUNJUNGAN'])
        if kid not in kid_first_pos:
            kid_first_pos[kid] = pos

    kid_groups = {}
    for ship_idx in ship_order:
        kid = str(df.loc[ship_idx, 'ID_KUNJUNGAN'])
        kid_groups.setdefault(kid, []).append(ship_idx)
    for kid in kid_groups:
        kid_groups[kid].sort(key=lambda i: int(df.loc[i, 'BERTH_PART']))

    ordered_kids     = sorted(kid_groups.keys(), key=lambda k: kid_first_pos[k])
    ship_order_fixed = []
    for kid in ordered_kids:
        ship_order_fixed.extend(kid_groups[kid])
    ship_order = ship_order_fixed

    results            = []
    kunjungan_segments = {}

    kid_occ_map = {}
    for p in unique_parents:
        for occ in parent_occs[p]:
            kid_occ = str(occ.get('kid', ''))
            if not kid_occ:
                continue
            kid_occ_map.setdefault(kid_occ, []).append(occ)

    for kid_occ, occ_list in kid_occ_map.items():
        occ_list.sort(key=lambda o: pd.Timestamp(o['start']))
        segments_seed = []
        seg_no = 0
        for occ in occ_list:
            t_start = pd.Timestamp(occ['start']).round('s')
            t_end   = pd.Timestamp(occ['end']).round('s')
            sp      = float(occ['start_pos'])
            ep      = float(occ['end_pos'])
            bid     = str(occ.get('bid', ''))
            if (
                segments_seed
                and bid == segments_seed[-1]['bid']
                and abs((t_start - segments_seed[-1]['t_end']).total_seconds()) <= 1
                and abs(sp - segments_seed[-1]['start_pos']) <= EPS
            ):
                segments_seed[-1]['t_end']   = t_end
                segments_seed[-1]['end_pos'] = ep
                segments_seed[-1]['part']    = int(occ.get('part', 1))
            else:
                seg_no += 1
                segments_seed.append({
                    't_start'        : t_start,
                    't_end'          : t_end,
                    'start_pos'      : sp,
                    'end_pos'        : ep,
                    'bid'            : bid,
                    'part'           : int(occ.get('part', 1)),
                    'segment_no_plan': seg_no,
                })
        kunjungan_segments[kid_occ] = segments_seed

    def _kedalaman_ok(draft, kdl, t_start, t_end):
        penalty   = float(get_tide_depth_penalty(t_start, t_end, config))
        low_pen   = ukc + tide_delta
        tide_flag = 1 if abs(penalty - low_pen) < EPS else 0
        eff_kdl   = float(kdl) - ukc - tide_flag * tide_delta
        return float(draft) <= eff_kdl + EPS, tide_flag

    def _calc_avg_loa_remaining(current_pos):
        remaining = ship_order[current_pos + 1:]
        if not remaining:
            return None
        loas = [float(df.loc[i, 'LOA']) for i in remaining if i in df.index]
        return float(sum(loas) / len(loas)) if loas else None

    # ============================================================
    # Main loop
    # ============================================================
    for ship_pos, ship_idx in enumerate(ship_order):
        row = df.loc[ship_idx]

        kid          = str(row['ID_KUNJUNGAN'])
        loa          = float(row['LOA'])
        draft        = float(row['DRAFT'])
        part_no      = int(row['BERTH_PART'])
        earliest_raw = pd.Timestamp(row['MULAI_SANDAR_AWAL']).round('s')

        elig_berths = berth_order_per_ship.get(ship_idx, eligibility.get(ship_idx, []))
        if not elig_berths:
            raise ValueError(
                f"LB2 partial tidak feasible: ID_KUNJUNGAN={kid}, "
                f"BERTH_PART={part_no}, tidak ada eligible berth"
            )

        last_seg      = None
        n_segs_so_far = 0
        if kid in kunjungan_segments and kunjungan_segments[kid]:
            last_seg      = kunjungan_segments[kid][-1]
            n_segs_so_far = len(kunjungan_segments[kid])

        avg_loa_remaining = _calc_avg_loa_remaining(ship_pos)
        placed            = False

        for berth in elig_berths:
            bid    = str(berth['ID'])
            parent = id_to_parent.get(bid)
            if parent is None:
                continue

            occ_list_parent = list(parent_occs[parent])

            last_bid           = str(last_seg['bid']) if last_seg is not None else None
            same_berth_as_last = (part_no > 1 and last_seg is not None and last_bid == bid)

            # ----------------------------------------------------------------
            # MODE A: coba lanjut mepet (hanya jika berth sama dengan last_seg)
            # ----------------------------------------------------------------
            if same_berth_as_last:
                earliest_A    = max(earliest_raw, pd.Timestamp(last_seg['t_end']).round('s'))
                preferred_pos = float(last_seg['start_pos'])
                earliest_A = max(
                    earliest_raw,
                    pd.Timestamp(last_seg['t_end']).round('s')
                )

                ship_dict_A = row.to_dict()
                ship_dict_A['MULAI_SANDAR_AWAL'] = earliest_A
                ship_dict_A['_AVG_LOA_HINT']     = avg_loa_remaining if avg_loa_remaining else loa

                slot_A = find_slot_lb2(
                    ship             = ship_dict_A,
                    berth            = berth,
                    parent_occupancy = occ_list_parent,
                    config           = config,
                    preferred_pos    = preferred_pos,
                    strict_preferred = True,
                    is_reberth       = False,
                )

                if slot_A is not None:
                    ts_A, te_A, sp_A, ep_A = slot_A
                    mepet_waktu  = abs((pd.Timestamp(ts_A).round('s') - earliest_A).total_seconds()) <= 1
                    mepet_posisi = abs(float(sp_A) - preferred_pos) <= EPS
                    if mepet_waktu and mepet_posisi:
                        ok_depth, tide_flag = _kedalaman_ok(draft, float(berth['KEDALAMAN']), ts_A, te_A)
                        if ok_depth:
                            # Commit MODE A — lanjut mepet di berth yang sama
                            new_occ = {
                                'start'    : pd.Timestamp(ts_A).round('s'),
                                'end'      : pd.Timestamp(te_A).round('s'),
                                'start_pos': float(sp_A),
                                'end_pos'  : float(ep_A),
                                'kid'      : kid,
                                'part'     : part_no,
                                'bid'      : bid,
                                'idx'      : ship_idx,
                            }
                            _occ_add(parent_ends[parent], parent_occs[parent], new_occ)

                            segment_no_plan = int(last_seg.get('segment_no_plan', n_segs_so_far))
                            last_seg['t_end']   = pd.Timestamp(te_A).round('s')
                            last_seg['end_pos'] = float(ep_A)
                            last_seg['part']    = part_no

                            results.append({
                                'idx'                : ship_idx,
                                'DERMAGA_ASSIGNED'   : bid,
                                'NAMA_DERMAGA'       : id_to_nama.get(bid, bid),
                                'MULAI_SANDAR'       : pd.Timestamp(ts_A).round('s'),
                                'SELESAI_SANDAR'     : pd.Timestamp(te_A).round('s'),
                                'WAITING_TIME_HOURS' : 0.0,
                                'POSISI_START_M'     : round(float(sp_A), 2),
                                'POSISI_END_M'       : round(float(ep_A), 2),
                                'TIDE_FLAG'          : int(tide_flag),
                                'IS_LATE'            : 0,
                                'REBERTHING'         : max(0, segment_no_plan - 1),
                                'SEGMENT_NO_PLAN'    : segment_no_plan,
                            })
                            placed = True
                            break

            # ----------------------------------------------------------------
            # MODE B: sandar ulang (segment baru)
            # Dicapai dalam dua kondisi:
            #   1. same_berth_as_last == False  → berth berbeda dari last_seg
            #   2. same_berth_as_last == True tapi MODE A gagal (fallback)
            # ----------------------------------------------------------------
            if n_segs_so_far >= max_segments:
                # Sudah maksimal segmen — tidak bisa tambah segment baru
                # di berth manapun, skip berth ini
                continue

            if part_no > 1 and last_seg is not None:
                earliest_B = max(
                    earliest_raw,
                    pd.Timestamp(last_seg['t_end']).round('s') + reberth_gap,
                )
            else:
                earliest_B = earliest_raw

            ship_dict_B = row.to_dict()
            ship_dict_B['MULAI_SANDAR_AWAL'] = earliest_B
            ship_dict_B['_AVG_LOA_HINT']     = avg_loa_remaining if avg_loa_remaining else loa

            same_parent_as_last = (
                last_seg is not None
                and last_bid is not None
                and id_to_parent.get(str(last_bid)) == parent
            )

            pref_B = float(last_seg['start_pos']) if same_parent_as_last else None

            slot_B = find_slot_lb2(
                ship             = ship_dict_B,
                berth            = berth,
                parent_occupancy = occ_list_parent,
                config           = config,
                preferred_pos    = pref_B,
                strict_preferred = False,
                is_reberth       = (part_no > 1),
            )

            if slot_B is None:
                continue

            ts_B, te_B, sp_B, ep_B = slot_B

            ok_depth, tide_flag = _kedalaman_ok(draft, float(berth['KEDALAMAN']), ts_B, te_B)
            if not ok_depth:
                continue

            # Commit MODE B
            new_occ = {
                'start'    : pd.Timestamp(ts_B).round('s'),
                'end'      : pd.Timestamp(te_B).round('s'),
                'start_pos': float(sp_B),
                'end_pos'  : float(ep_B),
                'kid'      : kid,
                'part'     : part_no,
                'bid'      : bid,
                'idx'      : ship_idx,
            }
            _occ_add(parent_ends[parent], parent_occs[parent], new_occ)

            segment_no_plan = n_segs_so_far + 1

            if segment_no_plan > max_segments:
                raise ValueError(
                    f"LB2 partial tidak feasible: ID_KUNJUNGAN={kid}, "
                    f"BERTH_PART={part_no}, "
                    f"segment={segment_no_plan} > max_segments={max_segments}"
                )

            kunjungan_segments.setdefault(kid, []).append({
                't_start'        : pd.Timestamp(ts_B).round('s'),
                't_end'          : pd.Timestamp(te_B).round('s'),
                'start_pos'      : float(sp_B),
                'end_pos'        : float(ep_B),
                'bid'            : bid,
                'part'           : part_no,
                'segment_no_plan': segment_no_plan,
            })

            # BUG 2 FIX: update n_segs_so_far setelah commit MODE B
            n_segs_so_far = segment_no_plan

            results.append({
                'idx'                : ship_idx,
                'DERMAGA_ASSIGNED'   : bid,
                'NAMA_DERMAGA'       : id_to_nama.get(bid, bid),
                'MULAI_SANDAR'       : pd.Timestamp(ts_B).round('s'),
                'SELESAI_SANDAR'     : pd.Timestamp(te_B).round('s'),
                'WAITING_TIME_HOURS' : 0.0,
                'POSISI_START_M'     : round(float(sp_B), 2),
                'POSISI_END_M'       : round(float(ep_B), 2),
                'TIDE_FLAG'          : int(tide_flag),
                'IS_LATE'            : 0,
                'REBERTHING'         : max(0, segment_no_plan - 1),
                'SEGMENT_NO_PLAN'    : segment_no_plan,
            })
            placed = True
            break

        if not placed:
            raise ValueError(
                f"LB2 partial tidak feasible: ID_KUNJUNGAN={kid}, "
                f"BERTH_PART={part_no}, no slot found"
            )

    return results

# ===========================================================================
# LoveBirdOptimizerS2
# ===========================================================================
class LoveBirdOptimizerS2:

    def __init__(self, df_kapal_raw, df_dermaga, config=None, weights=None,
             max_reberth=2, population_size=10,
             max_generations=30, seed=42):

        self.config      = config if config is not None else CONFIG
        self.weights     = weights if weights is not None else WEIGHTS
        self.max_reberth = max_reberth
        self.pop_size    = population_size
        self.max_gen     = max_generations
        self._rng        = random.Random(seed)
        np.random.seed(seed)

        self.approach_td = pd.Timedelta(hours=float(CONFIG['APPROACH_TIME']))

        self.df, self.berths, self.id_to_parent, self.id_to_nama, self.eligibility = \
            _prepare_data(df_kapal_raw, df_dermaga, CONFIG)

        self.all_bid   = sorted({str(b['ID']) for b in self.berths})
        self.berth_map = {str(b['ID']): b for b in self.berths}

        self.all_parts = set()
        for idx, row in self.df.iterrows():
            self.all_parts.add(f"{row['ID_KUNJUNGAN']}_{int(row['BERTH_PART'])}")

        self.part_elig = {}
        for idx, row in self.df.iterrows():
            key  = f"{row['ID_KUNJUNGAN']}_{int(row['BERTH_PART'])}"
            elig = [str(b['ID']) for b in self.eligibility.get(idx, [])]
            self.part_elig[key] = elig

        self.n_elites    = max(1, int(round(population_size * 0.10)))
        self.n_offspring = population_size
        self.n_best_off  = population_size - self.n_elites
        self.ls_iters    = max(1, max_generations // 2)
        self._arch_size  = max(self.n_elites, int(round(population_size * 0.20)))
        self.n_kunjungan = int(self.df['ID_KUNJUNGAN'].nunique())

        self._global_archive = []
        self.history         = []

    def _eval(self, sched, prev_result=None, affected_bids=None):
        return _evaluate_s2(
            schedule_2d   = sched,
            df            = self.df,
            berths        = self.berths,
            id_to_parent  = self.id_to_parent,
            id_to_nama    = self.id_to_nama,
            eligibility   = self.eligibility,
            config        = self.config,
            weights       = self.weights,
            max_reberth   = self.max_reberth,
            prev_result   = prev_result,
            affected_bids = affected_bids,
        )

    def _update_archive(self, candidates):
        merged = {}
        for s in self._global_archive + candidates:
            if s is None:
                continue
            key = tuple((k, tuple(v)) for k, v in sorted(s[6].items()))
            if key not in merged or s[0] < merged[key][0]:
                merged[key] = s
        self._global_archive = sorted(
            merged.values(), key=lambda x: x[0]
        )[:self._arch_size]

    def _roulette(self, pop_fits):
        scores = np.array([1.0 / (f + 1e-6) for f in pop_fits])
        probs  = scores / scores.sum()
        return int(np.random.choice(len(pop_fits), p=probs))

    def optimize(self, initial_solutions):
        global _ETA_CACHE_S2, _SHIP_FITS_BERTH_S2_CACHE

        _ETA_CACHE_S2.clear()
        _SHIP_FITS_BERTH_S2_CACHE.clear()
        t_start_total = time.perf_counter()
        n_total = len(self.all_parts)

        print("=" * 70)
        print("FASE 1: Membangun populasi awal S2")
        print("=" * 70)

        population = []
        pop_fits   = []
        pop_waits  = []
        pop_ass    = []
        pop_late   = []
        pop_rb     = []
        pop_dfs    = []
        pop_prev   = []

        # ============================================================
        # FASE 1 — POPULASI AWAL
        # ============================================================
        for i, sol in enumerate(initial_solutions[:self.pop_size]):
            df_final = sol.get('df_schedule')

            if df_final is None:
                continue

            try:
                sched_2d = _df_to_2d_s2(df_final, self.all_bid, df_raw=self.df)

                ok, msg = _val2(
                    sched_2d,
                    self.all_parts,
                    part_elig=self.part_elig
                )

                if not ok:
                    sched_2d = _repair2(
                        sched_2d,
                        self.all_parts,
                        self.part_elig,
                        self._rng
                    )

                res = _evaluate_ch_final_s2(
                    df_final     = df_final,
                    df_raw       = self.df,
                    weights      = self.weights,
                    config       = self.config,
                    max_reberth  = self.max_reberth,
                    schedule_2d  = sched_2d,
                )

                fit, wait, n_ass, n_late, n_rb, df_f, sched = res

                population.append(sched_2d)
                pop_fits.append(fit)
                pop_waits.append(wait)
                pop_ass.append(n_ass)
                pop_late.append(n_late)
                pop_rb.append(n_rb)
                pop_dfs.append(df_f)
                pop_prev.append(res)

            except Exception as e:
                print(f"[INIT SKIP] solusi-{i}: {type(e).__name__}: {e}")
                continue

        if not population:
            raise ValueError("Populasi awal kosong. Cek initial_solutions / df_schedule.")

        # ============================================================
        # GBEST AWAL
        # ============================================================
        gbest_idx  = int(np.argmin(pop_fits))
        gbest_fit  = pop_fits[gbest_idx]
        gbest_wait = pop_waits[gbest_idx]
        gbest_ass  = pop_ass[gbest_idx]
        gbest_late = pop_late[gbest_idx]
        gbest_rb   = pop_rb[gbest_idx]
        gbest_df   = pop_dfs[gbest_idx]
        gbest_sch  = _dc2(population[gbest_idx])
        gbest_res  = pop_prev[gbest_idx]

        init_candidates = list(zip(
            pop_fits,
            pop_waits,
            pop_ass,
            pop_late,
            pop_rb,
            pop_dfs,
            population
        ))
        self._update_archive(init_candidates)

        print(
            f"\nGBest awal: Fit={gbest_fit:,.4f} | "
            f"Wait={gbest_wait:.4f}h | "
            f"Late={gbest_late} | "
            f"Reberth={gbest_rb} | "
            f"Assigned={gbest_ass}/{n_total}"
        )

        print("\n" + "=" * 70)
        print(f"FASE 2: Optimasi S2 ({self.max_gen} generasi)")
        print("=" * 70)

        # ============================================================
        # FASE 2 — OPTIMASI
        # ============================================================
        for gen in range(self.max_gen):
            t_gen = time.perf_counter()

            # ========================================================
            # ELITISM: population + archive
            # ========================================================
            archive_scheds = [s[6] for s in self._global_archive]
            archive_fits   = [s[0] for s in self._global_archive]
            archive_waits  = [s[1] for s in self._global_archive]
            archive_ass    = [s[2] for s in self._global_archive]
            archive_late   = [s[3] for s in self._global_archive]
            archive_rb     = [s[4] for s in self._global_archive]
            archive_dfs    = [s[5] for s in self._global_archive]

            pool_scheds = population + archive_scheds
            pool_fits   = pop_fits   + archive_fits
            pool_waits  = pop_waits  + archive_waits
            pool_ass    = pop_ass    + archive_ass
            pool_late   = pop_late   + archive_late
            pool_rb     = pop_rb     + archive_rb
            pool_dfs    = pop_dfs    + archive_dfs

            elite_indices = list(np.argsort(pool_fits)[:self.n_elites])

            elite_scheds = [_dc2(pool_scheds[i]) for i in elite_indices]
            elite_fits   = [pool_fits[i] for i in elite_indices]
            elite_waits  = [pool_waits[i] for i in elite_indices]
            elite_ass    = [pool_ass[i] for i in elite_indices]
            elite_late   = [pool_late[i] for i in elite_indices]
            elite_rb     = [pool_rb[i] for i in elite_indices]
            elite_dfs    = [pool_dfs[i] for i in elite_indices]

            elite_prev = []
            for i in elite_indices:
                elite_prev.append((
                    pool_fits[i],
                    pool_waits[i],
                    pool_ass[i],
                    pool_late[i],
                    pool_rb[i],
                    pool_dfs[i],
                    pool_scheds[i],
                ))

            # ========================================================
            # OFFSPRING
            # ========================================================
            off_scheds = []
            off_fits   = []
            off_waits  = []
            off_ass    = []
            off_late   = []
            off_rb     = []
            off_dfs    = []
            off_prev   = []

            for _ in range(self.n_offspring):
                p_idx = self._roulette(pop_fits)
                parent = population[p_idx]
                parent_res = pop_prev[p_idx]

                child, aff = _apply_op_s2(
                    schedule_2d = parent,
                    rng         = self._rng,
                    df_ships    = self.df,
                    part_elig   = self.part_elig,
                    max_reberth = self.max_reberth,
                    ls_mode     = False,
                    pop_fits    = pop_fits,
                    prev_result = parent_res,
                    weights     = self.weights,
                    df_final    = parent_res[5] if parent_res is not None else None,
                    berth_map   = self.berth_map,   # <-- TAMBAH INI
                    config      = self.config,
                )

                # ----------------------------
                # CEK NO-OP
                # ----------------------------
                parent_key = tuple((k, tuple(v)) for k, v in sorted(parent.items()))
                child_key  = tuple((k, tuple(v)) for k, v in sorted(child.items()))

                if not aff or child_key == parent_key:
                    continue


                # ----------------------------
                # VALIDASI CHILD
                # ----------------------------
                ok, msg = _val2(
                    child,
                    self.all_parts,
                    part_elig=self.part_elig
                )

                if not ok:
                    child = _repair2(
                        child,
                        self.all_parts,
                        self.part_elig,
                        self._rng
                    )

                    aff = {
                        bid for bid in set(parent.keys()) | set(child.keys())
                        if parent.get(bid, []) != child.get(bid, [])
                    }

                    child_key = tuple((k, tuple(v)) for k, v in sorted(child.items()))

                    if not aff or child_key == parent_key:
                        continue

                # ----------------------------
                # EVALUASI CHILD
                # ----------------------------
                res = self._eval(
                    child,
                    prev_result=parent_res,
                    affected_bids=aff
                )

                if res is None:
                    continue


                fit, wait, n_ass, n_late, n_rb, df_f, sched = res

                off_scheds.append(child)
                off_fits.append(fit)
                off_waits.append(wait)
                off_ass.append(n_ass)
                off_late.append(n_late)
                off_rb.append(n_rb)
                off_dfs.append(df_f)
                off_prev.append(res)

            # ========================================================
            # JIKA SEMUA OFFSPRING GAGAL
            # ========================================================
            if not off_fits:
                print(
                    f"Gen {gen + 1:3d}: [ALL OFFSPRING FAILED] | "
                    f"Fit={gbest_fit:,.4f} | "
                    f"Wait={gbest_wait:.4f}h | "
                    f"Late={gbest_late} | "
                    f"Reberth={gbest_rb} | "
                    f"Assigned={gbest_ass}/{n_total}"
                )

                new_pop       = list(elite_scheds)
                new_pop_fits  = list(elite_fits)
                new_pop_waits = list(elite_waits)
                new_pop_ass   = list(elite_ass)
                new_pop_late  = list(elite_late)
                new_pop_rb    = list(elite_rb)
                new_pop_dfs   = list(elite_dfs)
                new_pop_prev  = list(elite_prev)

                n_elite = len(new_pop)
                i_pad   = 0
                while len(new_pop) < self.pop_size:
                    idx_pad = i_pad % n_elite
                    new_pop.append(_dc2(elite_scheds[idx_pad]))
                    new_pop_fits.append(elite_fits[idx_pad])
                    new_pop_waits.append(elite_waits[idx_pad])
                    new_pop_ass.append(elite_ass[idx_pad])
                    new_pop_late.append(elite_late[idx_pad])
                    new_pop_rb.append(elite_rb[idx_pad])
                    new_pop_dfs.append(elite_dfs[idx_pad])
                    new_pop_prev.append(elite_prev[idx_pad])
                    i_pad += 1

                population = new_pop
                pop_fits   = new_pop_fits
                pop_waits  = new_pop_waits
                pop_ass    = new_pop_ass
                pop_late   = new_pop_late
                pop_rb     = new_pop_rb
                pop_dfs    = new_pop_dfs
                pop_prev   = new_pop_prev

                continue

            # ========================================================
            # UPDATE ARCHIVE DARI OFFSPRING
            # ========================================================
            off_candidates = list(zip(
                off_fits,
                off_waits,
                off_ass,
                off_late,
                off_rb,
                off_dfs,
                off_scheds
            ))
            self._update_archive(off_candidates)

            # ========================================================
            # PILIH BEST OFFSPRING
            # ========================================================
            best_off_idx = list(np.argsort(off_fits)[:self.n_best_off])

            new_off_scheds = [off_scheds[i] for i in best_off_idx]
            new_off_fits   = [off_fits[i] for i in best_off_idx]
            new_off_waits  = [off_waits[i] for i in best_off_idx]
            new_off_ass    = [off_ass[i] for i in best_off_idx]
            new_off_late   = [off_late[i] for i in best_off_idx]
            new_off_rb     = [off_rb[i] for i in best_off_idx]
            new_off_dfs    = [off_dfs[i] for i in best_off_idx]
            new_off_prev   = [off_prev[i] for i in best_off_idx]

            curr_idx  = int(np.argmin(new_off_fits))
            curr_fit  = new_off_fits[curr_idx]
            curr_wait = new_off_waits[curr_idx]
            curr_ass  = new_off_ass[curr_idx]
            curr_late = new_off_late[curr_idx]
            curr_rb   = new_off_rb[curr_idx]
            curr_df   = new_off_dfs[curr_idx]
            curr_sch  = new_off_scheds[curr_idx]
            curr_res  = new_off_prev[curr_idx]

            status = "-> No change"

            # ========================================================
            # UPDATE GBEST
            # ========================================================
            if curr_fit < gbest_fit:
                gbest_fit  = curr_fit
                gbest_wait = curr_wait
                gbest_ass  = curr_ass
                gbest_late = curr_late
                gbest_rb   = curr_rb
                gbest_df   = curr_df
                gbest_sch  = _dc2(curr_sch)
                gbest_res  = curr_res
                status = "** GBest UPDATED"

            else:
                # ====================================================
                # LOCAL SEARCH DI GBEST
                # ====================================================
                ls_improved = False

                for ls_iter in range(self.ls_iters):
                    cand, aff = _apply_op_s2(
                        schedule_2d = gbest_sch,
                        rng         = self._rng,
                        df_ships    = self.df,
                        part_elig   = self.part_elig,
                        max_reberth = self.max_reberth,
                        ls_mode     = True,
                        pop_fits    = pop_fits,
                        prev_result = gbest_res,
                        weights     = self.weights,
                        df_final    = gbest_df,
                        berth_map   = self.berth_map,   # <-- TAMBAH INI
                        config      = self.config,
                    )

                    parent_key = tuple((k, tuple(v)) for k, v in sorted(gbest_sch.items()))
                    cand_key   = tuple((k, tuple(v)) for k, v in sorted(cand.items()))

                    if not aff or cand_key == parent_key:
                        continue

                    res_ls = self._eval(
                        cand,
                        prev_result=gbest_res,
                        affected_bids=aff
                    )

                    if res_ls is None:
                        continue

                    fit_ls, wait_ls, ass_ls, late_ls, rb_ls, df_ls, sched_ls = res_ls

                    if fit_ls < gbest_fit:
                        gbest_fit  = fit_ls
                        gbest_wait = wait_ls
                        gbest_ass  = ass_ls
                        gbest_late = late_ls
                        gbest_rb   = rb_ls
                        gbest_df   = df_ls
                        gbest_sch  = _dc2(cand)
                        gbest_res  = res_ls

                        self._update_archive([
                            (
                                fit_ls,
                                wait_ls,
                                ass_ls,
                                late_ls,
                                rb_ls,
                                df_ls,
                                cand
                            )
                        ])

                        ls_improved = True
                        status = f">> LS[{ls_iter + 1}] SUCCESS"
                        break

                if not ls_improved:
                    status = "-- LS no improv "

            # ========================================================
            # POPULASI BARU
            # ========================================================
            population = elite_scheds + new_off_scheds
            pop_fits   = elite_fits   + new_off_fits
            pop_waits  = elite_waits  + new_off_waits
            pop_ass    = elite_ass    + new_off_ass
            pop_late   = elite_late   + new_off_late
            pop_rb     = elite_rb     + new_off_rb
            pop_dfs    = elite_dfs    + new_off_dfs
            pop_prev   = elite_prev   + new_off_prev

            # Kalau populasi melebihi ukuran, potong
            if len(population) > self.pop_size:
                idx_keep = list(np.argsort(pop_fits)[:self.pop_size])

                population = [population[i] for i in idx_keep]
                pop_fits   = [pop_fits[i] for i in idx_keep]
                pop_waits  = [pop_waits[i] for i in idx_keep]
                pop_ass    = [pop_ass[i] for i in idx_keep]
                pop_late   = [pop_late[i] for i in idx_keep]
                pop_rb     = [pop_rb[i] for i in idx_keep]
                pop_dfs    = [pop_dfs[i] for i in idx_keep]
                pop_prev   = [pop_prev[i] for i in idx_keep]

            if len(population) < self.pop_size:
                n_elite = len(elite_scheds)
                i_pad   = 0
                while len(population) < self.pop_size:
                    idx_pad = i_pad % n_elite
                    population.append(_dc2(elite_scheds[idx_pad]))
                    pop_fits.append(elite_fits[idx_pad])
                    pop_waits.append(elite_waits[idx_pad])
                    pop_ass.append(elite_ass[idx_pad])
                    pop_late.append(elite_late[idx_pad])
                    pop_rb.append(elite_rb[idx_pad])
                    pop_dfs.append(elite_dfs[idx_pad])
                    pop_prev.append(elite_prev[idx_pad])
                    i_pad += 1

            gen_time = time.perf_counter() - t_gen

            self.history.append({
                'generation'  : gen + 1,
                'fitness'     : gbest_fit,
                'wait_hours'  : gbest_wait,
                'n_late'      : gbest_late,
                'n_reberth'   : gbest_rb,
                'assigned'    : gbest_ass,
                'gen_time_s'  : round(gen_time, 3),
            })

            print(
                f"Gen {gen + 1:3d}: {status} | "
                f"Fit={gbest_fit:,.4f} | "
                f"Wait={gbest_wait:.4f}h | "
                f"Late={gbest_late} | "
                f"Reberth={gbest_rb} | "
                f"Assigned={gbest_ass}/{n_total} | "
                f"Time={gen_time:.2f}s"
            )

        # ============================================================
        # FINAL OUTPUT
        # ============================================================
        total_time_s = time.perf_counter() - t_start_total
        total_time_m = total_time_s / 60.0

        print("\n" + "=" * 70)
        print("SUMMARY OPTIMASI — Skenario 2")
        print("=" * 70)
        print(f"Fitness Total       : {gbest_fit:,.4f}")
        print(f"Total Waktu Tunggu  : {gbest_wait:.4f} jam")
        print(f"Kapal Late          : {gbest_late}")
        print(f"Total Reberth       : {gbest_rb}")
        print(f"Assigned            : {gbest_ass}/{n_total}")
        print(f"Running Time        : {total_time_s:.2f} detik ({total_time_m:.2f} menit)")
        print("=" * 70)

        return {
            'df_schedule'    : gbest_df,
            'schedule_2d'    : gbest_sch,
            'fitness'        : gbest_fit,
            'total_wait'     : gbest_wait,
            'n_late'         : gbest_late,
            'total_reberth'  : gbest_rb,
            'assigned'       : gbest_ass,
            'running_time_s' : round(total_time_s, 3),
            'running_time_m' : round(total_time_m, 4),
        }, pd.DataFrame(self.history), self.history, None


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
def run_love_bird_s2(df_kapal_raw, df_dermaga, initial_solutions,
                     **kwargs):
    population_size = kwargs.get('population_size', None)
    if population_size is None:
        population_size = len(initial_solutions)

    optimizer = LoveBirdOptimizerS2(
        df_kapal_raw    = df_kapal_raw,
        df_dermaga      = df_dermaga,
        max_reberth     = int(CONFIG.get('MAX_REBERTH', 2)),
        population_size = population_size,
        max_generations = kwargs.get('max_generations', 30),
        seed            = kwargs.get('seed', 42),
    )
    return optimizer.optimize(initial_solutions)