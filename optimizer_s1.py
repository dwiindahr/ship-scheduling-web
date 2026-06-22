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
    _is_better,
    _make_variation_params,
    evaluate_solution_metrics,
    CATEGORY_PRIORITY,
)


# =============================================================================
# FITNESS S1
# =============================================================================
def fitness_s1(df_sol, weights, approach_td):
    cat_weights     = weights['CATEGORY_WEIGHTS']
    sop_multiplier  = weights['PENALTY_WAIT_SOP']
    sop_threshold_h = weights['WAIT_THRESHOLD']

    if 'WAITING_TIME_HOURS' in df_sol.columns:
        wait_h = pd.to_numeric(df_sol['WAITING_TIME_HOURS'],
                               errors='coerce').fillna(0).clip(lower=0)
    else:
        earliest = df_sol['KEDATANGAN'] + approach_td
        wait_h   = ((df_sol['MULAI_SANDAR'] - earliest)
                    .dt.total_seconds() / 3600.0).clip(lower=0)

    cat_w = df_sol['KATEGORI'].map(
        lambda c: float(cat_weights.get(str(c), 1.0))
    )

    term_wait  = float((cat_w * wait_h).sum())
    late_flag  = (wait_h > sop_threshold_h).astype(float)

    excess_h   = (wait_h - sop_threshold_h).clip(lower=0)
    term_sop   = float((cat_w * sop_multiplier * excess_h).sum())

    fitness      = term_wait + term_sop
    n_assigned   = int(df_sol['DERMAGA_ASSIGNED'].notna().sum())
    n_late       = int(late_flag.sum())
    total_wait_h = float(wait_h.sum())

    return fitness, n_assigned, n_late, total_wait_h


# =============================================================================
# HELPERS S1
# =============================================================================
def _tide_safe(t_start, t_end, draft, berth, config):
    kdl        = berth['KEDALAMAN']
    penalty    = get_tide_depth_penalty(t_start, t_end, config)
    safe_depth = kdl - penalty
    return draft <= safe_depth + config['EPSILON']


def _ship_obj_contribution(wait_h, w_cat, sop_multiplier, sop_threshold_h):
    term1 = w_cat * wait_h
    excess_h = max(0.0, wait_h - sop_threshold_h)
    term2 = w_cat * sop_multiplier * excess_h
    return term1 + term2


def _skip_tide_window(t_start, t_end, svc_td, draft, berth, config,
                      tide_skip_cache=None):
    low_s1  = config['LOW_TIDE_1_START_H']
    low_e1  = config['LOW_TIDE_1_END_H']
    low_s2  = config['LOW_TIDE_2_START_H']
    low_e2  = config['LOW_TIDE_2_END_H']
    kdl     = berth['KEDALAMAN']
    low_pen = config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA']
    EPS     = config['EPSILON']

    if draft <= kdl - low_pen + EPS:
        return t_start, t_end

    cache_key = None
    if tide_skip_cache is not None:
        cache_key = (draft, berth['ID'], t_start)
        if cache_key in tide_skip_cache:
            return tide_skip_cache[cache_key]

    day     = t_start.normalize() - pd.Timedelta(days=1)
    horizon = t_start + pd.Timedelta(days=30)
    best_skip = None

    while day <= horizon:
        for s_h, e_h in [(low_s1, low_e1), (low_s2, low_e2)]:
            lt_start = day + pd.Timedelta(hours=s_h)
            lt_end   = day + pd.Timedelta(hours=e_h)
            if s_h >= e_h:
                lt_end += pd.Timedelta(days=1)
            if t_start < lt_end and t_end > lt_start:
                if best_skip is None or lt_end < best_skip:
                    best_skip = lt_end
        day += pd.Timedelta(days=1)
        if best_skip is not None and day > best_skip:
            break

    result = (None, None) if best_skip is None else (best_skip, best_skip + svc_td)

    if tide_skip_cache is not None and cache_key is not None:
        tide_skip_cache[cache_key] = result

    return result


def _build_merged_blocked(active_pos, kade_start, kade_end, safe, EPS):
    blocked = []
    for o in active_pos:
        b_start = o['start_pos'] - safe
        b_end   = o['end_pos']   + safe
        c_start = max(b_start, kade_start)
        c_end   = min(b_end,   kade_end)
        if c_start < c_end - EPS:
            blocked.append([c_start, c_end])

    blocked.sort(key=lambda x: x[0])
    merged = []
    for bs, be in blocked:
        if merged and bs <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], be)
        else:
            merged.append([bs, be])
    return merged


def _active_occ(occ_by_start, occ_by_end, starts_list, ends_list, t_start, t_end):
    if not occ_by_start:
        return []
    lo = bisect.bisect_right(ends_list, t_start)
    hi = bisect.bisect_left(starts_list, t_end)
    if lo >= len(occ_by_end) or hi == 0:
        return []
    right_objs = set(map(id, occ_by_end[lo:]))
    return [o for o in occ_by_start[:hi] if id(o) in right_objs]


def _score_position(pos, loa, kade_start, kade_end, merged_blocked, EPS,
                    avg_loa_hint=None):
    end_pos   = pos + loa
    all_blocks = sorted(merged_blocked + [[pos, end_pos]], key=lambda x: x[0])

    merged = []
    for bs, be in all_blocks:
        if merged and bs <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], be)
        else:
            merged.append([bs, be])

    free_segments = []
    cursor = kade_start
    for bs, be in merged:
        gap = bs - cursor
        if gap > EPS:
            free_segments.append(gap)
        cursor = max(cursor, be)
    tail = kade_end - cursor
    if tail > EPS:
        free_segments.append(tail)

    if not free_segments:
        total_free = 0.0
        max_free   = 0.0
    else:
        total_free = sum(free_segments)
        max_free   = max(free_segments)

    frag_penalty = 0.0
    if avg_loa_hint is not None and len(free_segments) >= 2:
        usable = sum(1 for s in free_segments if s >= avg_loa_hint - EPS)
        if usable == 0:
            frag_penalty = -avg_loa_hint

    primary  = max_free + frag_penalty
    tiebreak = total_free

    return primary, tiebreak


def _gap_position_candidates(gap_start, gap_end, loa, EPS):
    gap_len = gap_end - gap_start
    if gap_len < loa - EPS:
        return []

    candidates = []
    p_center = round(gap_start + (gap_len - loa) / 2.0, 6)
    p_left   = round(gap_start, 6)
    p_right  = round(gap_end - loa, 6)

    if gap_len >= 2.0 * loa - EPS:
        candidates.append(p_center)

    candidates.append(p_left)

    if abs(p_right - p_left) > EPS:
        candidates.append(p_right)

    seen = set()
    result = []
    for p in candidates:
        key = round(p, 4)
        if key not in seen:
            seen.add(key)
            result.append(p)

    return result


# =============================================================================
# BUILD OUTPUT S1
# =============================================================================
def _build_output_df1(df, results):
    res_df = pd.DataFrame(results).set_index('idx')
    out    = df.copy()
    out_cols = [
        'DERMAGA_ASSIGNED', 'NAMA_DERMAGA', 'MULAI_SANDAR', 'SELESAI_SANDAR',
        'WAITING_TIME_HOURS', 'POSISI_START_M', 'POSISI_END_M', 'TIDE_FLAG',
    ]
    for col in out_cols:
        out[col] = res_df[col]
    drop_cols = [c for c in ['_priority', 'MULAI_SANDAR_AWAL', '_id_kunjungan']
                 if c in out.columns]
    out = out.drop(columns=drop_cols)
    return out


def generate_sol2d_s1(df_sol, ship_identifier='ID_KUNJUNGAN'):
    assigned = df_sol[df_sol['DERMAGA_ASSIGNED'].notna()].copy()
    if assigned.empty:
        return {}

    assigned = assigned.sort_values(['DERMAGA_ASSIGNED', 'MULAI_SANDAR', 'POSISI_START_M'])

    result = {}
    for bid, group in assigned.groupby('DERMAGA_ASSIGNED'):
        result[str(bid)] = group[ship_identifier].astype(str).tolist()

    return result


# =============================================================================
# FIND SLOT CH1
# =============================================================================
def find_slot_ch1(ship, berth, parent_occupancy, config, tide_windows=None):
    EPS        = config['EPSILON']
    safe       = config['CONTINUOUS_GAP_M']
    berth_type = berth['JENIS']
    loa        = ship['LOA']
    draft      = ship['DRAFT']
    depth      = berth['KEDALAMAN']
    service    = ship['BERTH_TIME']
    earliest   = ship['MULAI_SANDAR_AWAL']

    limit_start = float(berth['START'])
    limit_end   = float(berth['END'])
    svc_td      = pd.Timedelta(hours=service)

    low_penalty       = config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA']
    draft_always_safe = (draft <= depth - low_penalty + EPS)

    candidate_set = {earliest} | {
        occ['end'] for occ in parent_occupancy if occ['end'] > earliest
    }

    if not draft_always_safe and tide_windows is not None:
        tw_starts   = [w[0] for w in tide_windows]
        lo_tw       = bisect.bisect_left(tw_starts, earliest)
        horizon_cand = earliest + pd.Timedelta(days=7)
        for ws, we in tide_windows[lo_tw:]:
            if ws >= horizon_cand:
                break
            if we > earliest:
                candidate_set.add(we)

    candidate_starts = sorted(candidate_set)

    for t_raw in candidate_starts:
        if draft_always_safe:
            t_start = t_raw
            t_end   = t_start + svc_td
        else:
            if tide_windows is not None:
                t_safe = _next_safe_start(t_raw, svc_td, tide_windows, config)
            if t_safe is None:
                continue
            t_start = max(t_safe, earliest)
            t_end   = t_start + svc_td

        active = [occ for occ in parent_occupancy
                  if occ['start'] < t_end and occ['end'] > t_start]

        if berth_type == 'DISKRIT':
            start_pos = limit_start + config['LOA_MARGIN_DISKRIT']
            end_pos   = start_pos + loa

            if end_pos > limit_end + EPS:
                continue

            conflict = any(
                (start_pos < occ['end_pos'] + safe - EPS) and
                (end_pos + safe > occ['start_pos'] + EPS)
                for occ in active
            )

            if not conflict:
                return (t_start, t_end, start_pos, end_pos)

        else:
            pos_candidates = sorted({limit_start} | {
                occ['end_pos'] + safe for occ in active
            })

            for p_start in pos_candidates:
                s_pos = max(p_start, limit_start)
                e_pos = s_pos + loa

                if e_pos > limit_end + EPS:
                    break

                conflict = any(
                    (s_pos < occ['end_pos'] + safe - EPS) and
                    (e_pos + safe > occ['start_pos'] + EPS)
                    for occ in active
                )

                if not conflict:
                    return (t_start, t_end, s_pos, e_pos)

    return None


# =============================================================================
# FIND SLOT LB1
# =============================================================================
def find_slot_lb1(ship, berth, parent_occupancy, dermaga_occ, config,
                  WEIGHTS=None, w_cat=None, sop_multiplier=None,
                  sop_threshold_h=None, approach_td=None,
                  tide_skip_cache=None, tide_windows=None, avg_loa_hint=None):

    EPS        = config['EPSILON']
    safe       = config['CONTINUOUS_GAP_M']
    jenis      = berth['JENIS']
    loa        = ship['LOA']
    draft      = ship['DRAFT']
    service    = ship['BERTH_TIME']
    earliest   = ship['MULAI_SANDAR_AWAL']
    kade_start = berth['START']
    kade_end   = berth['END']
    svc_td     = pd.Timedelta(hours=service)

    if approach_td     is None: approach_td     = pd.Timedelta(hours=config['APPROACH_TIME'])
    if w_cat           is None: w_cat           = WEIGHTS['CATEGORY_WEIGHTS']
    if sop_multiplier  is None: sop_multiplier  = WEIGHTS['PENALTY_WAIT_SOP']
    if sop_threshold_h is None: sop_threshold_h = WEIGHTS['WAIT_THRESHOLD']

    kdl               = berth['KEDALAMAN']
    low_penalty       = config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA']
    draft_always_safe = (draft <= kdl - low_penalty + EPS)

    def _wait_h(t):
        return max(0.0, (t - earliest).total_seconds() / 3600.0)

    occ_sorted_start  = sorted(parent_occupancy, key=lambda o: o['start'])
    occ_sorted_end    = sorted(parent_occupancy, key=lambda o: o['end'])
    derm_sorted_start = sorted(dermaga_occ,      key=lambda o: o['start'])
    derm_sorted_end   = sorted(dermaga_occ,      key=lambda o: o['end'])

    occ_starts_list  = [o['start'] for o in occ_sorted_start]
    occ_ends_list    = [o['end']   for o in occ_sorted_end]
    derm_starts_list = [o['start'] for o in derm_sorted_start]
    derm_ends_list   = [o['end']   for o in derm_sorted_end]

    candidate_set = {earliest}

    if jenis == 'DISKRIT':
        for o in occ_sorted_start:
            if o['end'] > earliest:
                candidate_set.add(o['end'])
    else:
        for o in derm_sorted_start:
            if o['end'] > earliest:
                candidate_set.add(o['end'])

    if not draft_always_safe and tide_windows is not None:
        tw_starts    = [w[0] for w in tide_windows]
        lo_tw        = bisect.bisect_left(tw_starts, earliest)
        horizon_cand = earliest + pd.Timedelta(days=14)
        for ws, we in tide_windows[lo_tw:]:
            if ws >= horizon_cand:
                break
            if we > earliest:
                candidate_set.add(we)

    candidate_starts = sorted(candidate_set)

    best_slot    = None
    best_obj     = float('inf')
    best_primary = -float('inf')
    best_tbreak  = -float('inf')

    _last_t_key    = None
    _last_active_p = None
    _last_active_d = None
    _last_merged   = None

    for t_raw in candidate_starts:
        tide_shifted = False

        if draft_always_safe:
            t_start = t_raw
            t_end   = t_start + svc_td
        else:
            if tide_windows is not None:
                t_safe = _next_safe_start(t_raw, svc_td, tide_windows, config)
            if t_safe is None:
                continue

            t_start = max(t_safe, earliest)
            t_end   = t_start + svc_td

            if t_start > t_raw + pd.Timedelta(seconds=1):
                tide_shifted = True

        obj = _ship_obj_contribution(
            _wait_h(t_start), w_cat, sop_multiplier, sop_threshold_h)

        if obj > best_obj + EPS and not tide_shifted:
            break
        if tide_shifted and obj > best_obj * 2 + EPS:
            break

        t_key = (t_start, t_end)
        if t_key != _last_t_key:
            _last_t_key    = t_key
            _last_active_p = _active_occ(
                occ_sorted_start, occ_sorted_end,
                occ_starts_list, occ_ends_list, t_start, t_end)
            _last_active_d = _active_occ(
                derm_sorted_start, derm_sorted_end,
                derm_starts_list, derm_ends_list, t_start, t_end)
            _last_merged = None

        active_parent  = _last_active_p
        active_dermaga = _last_active_d

        if jenis == 'DISKRIT':
            if active_parent:
                continue

            pos     = kade_start + config.get('LOA_MARGIN_DISKRIT', 0)
            end_pos = pos + loa
            if end_pos > kade_end + EPS:
                continue

            if obj < best_obj - EPS:
                best_obj  = obj
                best_slot = (t_start, t_end, pos, end_pos)
                if draft_always_safe:
                    return best_slot

        else:
            active_pos = sorted(active_dermaga, key=lambda o: o['start_pos'])

            if _last_merged is None:
                _last_merged = _build_merged_blocked(
                    active_pos, kade_start, kade_end, safe, EPS)
            merged_blocked = _last_merged

            gaps = []
            cursor = kade_start
            for bs, be in merged_blocked:
                if bs - cursor >= loa - EPS:
                    gaps.append((cursor, bs))
                cursor = max(cursor, be)
            if kade_end - cursor >= loa - EPS:
                gaps.append((cursor, kade_end))

            obj_improved = (obj < best_obj - EPS)

            for gap_start, gap_end in gaps:
                pos_candidates = _gap_position_candidates(gap_start, gap_end, loa, EPS)

                for pos in pos_candidates:
                    end_pos = round(pos + loa, 6)

                    if pos < kade_start - EPS or end_pos > kade_end + EPS:
                        continue

                    primary, tbreak = _score_position(
                        pos, loa, kade_start, kade_end,
                        merged_blocked, EPS, avg_loa_hint=avg_loa_hint)

                    if obj_improved or (
                        abs(obj - best_obj) <= EPS and (
                            primary > best_primary + EPS or (
                                abs(primary - best_primary) <= EPS and
                                tbreak > best_tbreak + EPS))):
                        best_obj     = obj
                        best_primary = primary
                        best_tbreak  = tbreak
                        best_slot    = (t_start, t_end, pos, end_pos)
                        obj_improved = False

    return best_slot


# =============================================================================
# SIMULATE SCHEDULE S1
# =============================================================================
def simulate_schedule(df_ships, berths, ship_order, eligibility, id_to_nama,
                      config, approach_td, bid_to_group, all_groups,
                      initial_berth_count,
                      cat_weights=None, sop_multiplier=None, sop_threshold_h=None,
                      affected_bids=None, prev_result=None, new_schedule_2d=None,
                      tide_skip_cache=None, tide_windows=None):

    if cat_weights     is None: cat_weights     = WEIGHTS['CATEGORY_WEIGHTS']
    if sop_multiplier  is None: sop_multiplier  = WEIGHTS['PENALTY_WAIT_SOP']
    if sop_threshold_h is None: sop_threshold_h = WEIGHTS['WAIT_THRESHOLD']
    if tide_skip_cache is None: tide_skip_cache = {}

    berth_occ   = {str(b['ID']): [] for b in berths}
    berth_count = initial_berth_count.copy()
    results     = []
    ships_to_resimulate = set(ship_order)
    EPS = config.get('EPSILON', 1e-6)

    if tide_windows is None:
        if not df_ships.empty:
            t_min = df_ships['MULAI_SANDAR_AWAL'].min()
            t_max = (df_ships['MULAI_SANDAR_AWAL'].max()
                     + pd.Timedelta(hours=float(df_ships['BERTH_TIME'].max())))
        else:
            t_min = pd.Timestamp.now()
            t_max = t_min + pd.Timedelta(days=30)
        tide_windows = build_tide_windows(t_min, t_max, CONFIG)

    avg_loa_global = float(df_ships['LOA'].mean()) if not df_ships.empty else None

    if affected_bids is not None and prev_result is not None and new_schedule_2d is not None:
        prev_df = prev_result.get('df_schedule')
        prev_2d = prev_result.get('schedule_2d')

        if prev_df is not None and prev_2d is not None:
            unaffected_bids = set(new_schedule_2d.keys()) - affected_bids
            unaffected_bids.discard('_unassigned')

            for bid in unaffected_bids:
                if new_schedule_2d.get(bid, []) == prev_2d.get(bid, []):
                    berth_info = next((b for b in berths if str(b['ID']) == bid), None)
                    if not berth_info:
                        continue

                    for ship_idx in new_schedule_2d[bid]:
                        if ship_idx not in prev_df.index:
                            continue
                        old_row   = prev_df.loc[ship_idx]
                        ship_data = df_ships.loc[ship_idx]

                        if not ship_fits_berth(ship_data, berth_info, config):
                            continue
                        if pd.isna(old_row['MULAI_SANDAR']):
                            continue

                        mulai_sandar_awal = ship_data['KEDATANGAN'] + approach_td
                        t_start_old = old_row['MULAI_SANDAR']
                        t_end_old   = old_row['SELESAI_SANDAR']

                        if t_start_old < mulai_sandar_awal - pd.Timedelta(seconds=EPS):
                            continue

                        sp_koreksi = old_row['POSISI_START_M']
                        ep_koreksi = round(sp_koreksi + ship_data['LOA'], 6)

                        if (sp_koreksi < berth_info['START'] - EPS or
                                ep_koreksi > berth_info['END'] + EPS):
                            continue

                        if not _tide_safe(t_start_old, t_end_old,
                                          ship_data['DRAFT'], berth_info, config):
                            continue

                        group = bid_to_group[bid]
                        dermaga_occ_f1 = [
                            o for b_id, b_name in bid_to_group.items()
                            if b_name == group
                            for o in berth_occ[b_id]
                        ]

                        is_collision = False
                        safe_gap = (config.get('CONTINUOUS_GAP_M', 5)
                                    if berth_info['JENIS'] == 'KONTINU' else 0)
                        for o in dermaga_occ_f1:
                            if t_start_old < o['end'] and t_end_old > o['start']:
                                if berth_info['JENIS'] == 'DISKRIT':
                                    if bid == o.get('bid'):
                                        is_collision = True
                                        break
                                else:
                                    if (sp_koreksi < o['end_pos']   + safe_gap - EPS and
                                            ep_koreksi > o['start_pos'] - safe_gap + EPS):
                                        is_collision = True
                                        break
                        if is_collision:
                            continue

                        new_occ = {
                            'bid': bid, 'start': t_start_old, 'end': t_end_old,
                            'start_pos': sp_koreksi, 'end_pos': ep_koreksi
                        }
                        bisect.insort(berth_occ[bid], new_occ, key=lambda x: x['start'])
                        berth_count[bid] += 1

                        wait_h_f1 = (
                            (t_start_old - mulai_sandar_awal).total_seconds() / 3600.0
                        )
                        results.append({
                            'idx': ship_idx,
                            'DERMAGA_ASSIGNED': bid,
                            'NAMA_DERMAGA': id_to_nama[bid],
                            'MULAI_SANDAR': t_start_old,
                            'SELESAI_SANDAR': t_end_old,
                            'WAITING_TIME_HOURS': round(max(wait_h_f1, 0.0), 4),
                            'POSISI_START_M': sp_koreksi,
                            'POSISI_END_M': ep_koreksi,
                            'TIDE_FLAG': old_row['TIDE_FLAG'],
                            'IS_LATE': int(wait_h_f1 > sop_threshold_h),
                            'REBERTHING': 0
                        })
                        ships_to_resimulate.discard(ship_idx)

    for ship_idx in ship_order:
        if ship_idx not in ships_to_resimulate:
            continue

        ship  = df_ships.loc[ship_idx]
        w_cat = float(cat_weights.get(str(ship.get('KATEGORI', 'OTHER')), 1.0))

        assigned_bid = next(
            (bid for bid, s_list in new_schedule_2d.items()
             if ship_idx in s_list and bid != '_unassigned'),
            None
        )

        eligible_berths = [b for b in berths if ship_fits_berth(ship, b, config)]
        if assigned_bid:
            ordered_full = [b for b in eligible_berths if str(b['ID']) == str(assigned_bid)]
        else:
            ordered_full = eligible_berths

        if not ordered_full:
            results.append(_empty_result(ship_idx))
            continue

        best, best_obj_val = None, float('inf')
        dermaga_occ_cache  = {}

        for berth in ordered_full:
            bid   = str(berth['ID'])
            group = bid_to_group[bid]

            if group not in dermaga_occ_cache:
                dermaga_occ_cache[group] = [
                    o for b_id, b_name in bid_to_group.items()
                    if b_name == group
                    for o in berth_occ[b_id]
                ]
            dermaga_occ = dermaga_occ_cache[group]

            slot = find_slot_lb1(
                ship, berth, berth_occ[bid], dermaga_occ, config, WEIGHTS,
                w_cat=w_cat, sop_multiplier=sop_multiplier,
                sop_threshold_h=sop_threshold_h, approach_td=approach_td,
                tide_skip_cache=tide_skip_cache, tide_windows=tide_windows,
                avg_loa_hint=avg_loa_global,
            )
            if slot:
                t_s, t_e, sp, ep = slot
                wait_h  = max(0.0, (
                    t_s - (ship['KEDATANGAN'] + approach_td)
                ).total_seconds() / 3600.0)
                obj_val = _ship_obj_contribution(
                    wait_h, w_cat, sop_multiplier, sop_threshold_h)
                if best is None or obj_val < best_obj_val:
                    best, best_obj_val = (t_s, t_e, sp, ep, bid), obj_val

        if best is None:
            results.append(_empty_result(ship_idx))
            continue

        t_s, t_e, sp, ep, bid = best
        penalty = get_tide_depth_penalty(t_s, t_e, config)
        wait_h  = (t_s - ship['KEDATANGAN'] - approach_td).total_seconds() / 3600.0

        new_occ = {'bid': bid, 'start': t_s, 'end': t_e, 'start_pos': sp, 'end_pos': ep}
        bisect.insort(berth_occ[bid], new_occ, key=lambda x: x['start'])
        berth_count[bid] += 1

        results.append({
            'idx': ship_idx,
            'DERMAGA_ASSIGNED': bid,
            'NAMA_DERMAGA': id_to_nama[bid],
            'MULAI_SANDAR': t_s.round('s'),
            'SELESAI_SANDAR': t_e.round('s'),
            'WAITING_TIME_HOURS': round(max(wait_h, 0.0), 4),
            'POSISI_START_M': round(sp, 2),
            'POSISI_END_M': round(ep, 2),
            'TIDE_FLAG': 1 if abs(
                penalty - (config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA'])
            ) < EPS else 0,
            'IS_LATE': int(wait_h > sop_threshold_h),
            'REBERTHING': 0
        })

    res_df = pd.DataFrame(results).set_index('idx')
    out    = df_ships.copy()
    for col in res_df.columns:
        out[col] = res_df[col]

    if 'MULAI_SANDAR_AWAL' in out.columns:
        out.drop(columns=['MULAI_SANDAR_AWAL'], inplace=True)
    out['STATUS_KUNJUNGAN'] = 'KUNJUNGAN BARU'
    out['REBERTHING']       = 0
    return out


# =============================================================================
# OPERATOR MUTASI S1
# =============================================================================
def _deep_copy_2d(schedule_2d):
    return {k: v[:] for k, v in schedule_2d.items()}


def validate_2d(schedule_2d, all_ship_indices, eligibility=None):
    all_placed = [s for ships in schedule_2d.values() for s in ships]

    if len(all_placed) != len(all_ship_indices):
        return False, f"Jumlah kapal: {len(all_placed)} vs {len(all_ship_indices)}"
    if len(set(all_placed)) != len(all_placed):
        return False, "Ada duplikat kapal"
    if set(all_placed) != set(all_ship_indices):
        return False, f"Kapal hilang: {set(all_ship_indices) - set(all_placed)}"

    if eligibility is not None:
        for bid, ships in schedule_2d.items():
            if bid == '_unassigned':
                continue
            for s in ships:
                elig_bids = {str(b['ID']) for b in eligibility.get(s, [])}
                if str(bid) not in elig_bids:
                    return False, f"Kapal {s} tidak eligible di dermaga {bid}"

    return True, "Valid"


def repair_2d(schedule_2d, all_ship_indices, eligibility, rng):
    seen = set()
    new_sched = {k: [] for k in schedule_2d}
    new_sched.setdefault('_unassigned', [])

    for bid, ships in schedule_2d.items():
        for s in ships:
            if s in seen:
                continue
            if bid != '_unassigned':
                elig_bids = {str(b['ID']) for b in eligibility.get(s, [])}
                if str(bid) not in elig_bids:
                    continue
            seen.add(s)
            new_sched.setdefault(bid, []).append(s)

    for s in set(all_ship_indices) - seen:
        elig = [str(b['ID']) for b in eligibility.get(s, []) if str(b['ID']) in new_sched]
        if elig:
            bid = rng.choice(elig)
            new_sched.setdefault(bid, []).append(s)
        else:
            new_sched.setdefault('_unassigned', []).append(s)

    return new_sched


def op_flip(schedule_2d, chosen_ship, rng, df_ships, eligibility, berth_map):
    new_2d = _deep_copy_2d(schedule_2d)
    home_bid = next((bid for bid, ships in new_2d.items() if chosen_ship in ships), None)
    if home_bid is None or home_bid == '_unassigned':
        return new_2d, set()
    home_list = new_2d[home_bid]
    if len(home_list) < 2:
        return new_2d, set()
    i = home_list.index(chosen_ship)
    other_indices = [k for k in range(len(home_list)) if k != i]
    j  = rng.choice(other_indices)
    lo, hi = (i, j) if i < j else (j, i)
    home_list[lo:hi+1] = home_list[lo:hi+1][::-1]
    return new_2d, {home_bid}


def op_interchange(schedule_2d, chosen_ship, rng, df_ships, eligibility, berth_map,
                   force_inter=False):
    new_2d   = _deep_copy_2d(schedule_2d)
    home_bid = next((bid for bid, ships in new_2d.items() if chosen_ship in ships), None)
    if home_bid is None:
        return new_2d, set()

    elig_chosen = {str(b['ID']) for b in eligibility.get(chosen_ship, [])}
    mode = 'inter' if (force_inter or home_bid == '_unassigned') else rng.choice(['same', 'inter'])
    eta_chosen = df_ships.loc[chosen_ship, 'KEDATANGAN']

    if mode == 'same':
        same_ships = [s for s in new_2d[home_bid] if s != chosen_ship]
        if not same_ships:
            return new_2d, set()
        best_internal = sorted(
            [(abs((eta_chosen - df_ships.loc[s, 'KEDATANGAN']).total_seconds() / 3600.0), s)
             for s in same_ships]
        )
        top_n = max(1, len(best_internal) // 3)
        pair_ship = rng.choice(best_internal[:top_n])[1]
        idx1 = new_2d[home_bid].index(chosen_ship)
        idx2 = new_2d[home_bid].index(pair_ship)
        new_2d[home_bid][idx1], new_2d[home_bid][idx2] = new_2d[home_bid][idx2], new_2d[home_bid][idx1]
        return new_2d, {home_bid}

    else:
        all_valid_pairs = []
        for bid, ships in new_2d.items():
            if bid == home_bid or bid == '_unassigned' or not ships:
                continue
            if bid not in elig_chosen:
                continue
            for s in ships:
                if home_bid == '_unassigned' or home_bid in {str(b['ID']) for b in eligibility.get(s, [])}:
                    eta_s = df_ships.loc[s, 'KEDATANGAN']
                    delta_h = abs((eta_chosen - eta_s).total_seconds() / 3600.0)
                    all_valid_pairs.append((delta_h, bid, s))

        if not all_valid_pairs:
            return new_2d, set()

        all_valid_pairs.sort(key=lambda x: x[0])
        top_n = max(3, len(all_valid_pairs) // 5)
        _, target_bid, pair_ship = rng.choice(all_valid_pairs[:top_n])

        idx_chosen = new_2d[home_bid].index(chosen_ship)
        idx_pair   = new_2d[target_bid].index(pair_ship)
        new_2d[home_bid][idx_chosen] = pair_ship
        new_2d[target_bid][idx_pair] = chosen_ship

        return new_2d, {home_bid, target_bid}


def op_move_and_push(schedule_2d, chosen_ship, rng, df_ships, eligibility, berth_map):
    new_2d   = _deep_copy_2d(schedule_2d)
    home_bid = next((bid for bid, ships in new_2d.items() if chosen_ship in ships), None)
    if home_bid is None or home_bid == '_unassigned':
        return new_2d, set()
    kade_ships = new_2d[home_bid]
    n = len(kade_ships)
    if n < 2:
        return new_2d, set()
    i = kade_ships.index(chosen_ship)
    right_positions = list(range(i + 1, n))
    if not right_positions:
        return new_2d, set()
    target_pos = rng.choice(right_positions)
    kade_ships.pop(i)
    insert_pos = max(0, min(target_pos, len(kade_ships)))
    kade_ships.insert(insert_pos, chosen_ship)
    return new_2d, {home_bid}


def op_2swap_same(schedule_2d, chosen_ship, rng, df_ships, eligibility, berth_map):
    new_2d   = _deep_copy_2d(schedule_2d)
    home_bid = next((bid for bid, ships in new_2d.items() if chosen_ship in ships), None)
    if home_bid is None or home_bid == '_unassigned':
        return new_2d, set()
    kade_ships = new_2d[home_bid]
    n = len(kade_ships)
    if n < 4:
        return new_2d, set()
    pos          = kade_ships.index(chosen_ship)
    valid_anchor = list(range(1, n - 1))
    if pos not in valid_anchor:
        return new_2d, set()
    valid_pairs = [
        (a, b) for a in valid_anchor for b in valid_anchor
        if a < b and b + 1 < n and pos in (a, b)
    ]
    if not valid_pairs:
        return new_2d, set()
    a, b = rng.choice(valid_pairs)
    kade_ships[a - 1:a + 1], kade_ships[b:b + 2] = kade_ships[b:b + 2], kade_ships[a - 1:a + 1]
    return new_2d, {home_bid}


def op_2swap_inter(schedule_2d, chosen_ship, rng, df_ships, eligibility, berth_map):
    new_2d   = _deep_copy_2d(schedule_2d)
    home_bid = next((bid for bid, ships in new_2d.items() if chosen_ship in ships), None)
    if home_bid is None or home_bid == '_unassigned':
        return new_2d, set()
    kade_A = new_2d[home_bid]
    nA     = len(kade_A)
    if nA < 2:
        return new_2d, set()
    iA = kade_A.index(chosen_ship)
    if iA + 1 >= nA:
        return new_2d, set()
    shipA0, shipA1 = kade_A[iA], kade_A[iA + 1]
    elig_A0 = {str(b['ID']) for b in eligibility.get(shipA0, [])}
    elig_A1 = {str(b['ID']) for b in eligibility.get(shipA1, [])}
    valid_bids_for_A = elig_A0.intersection(elig_A1)
    eta_A0 = df_ships.loc[shipA0, 'KEDATANGAN']
    eta_A1 = df_ships.loc[shipA1, 'KEDATANGAN']
    all_valid_swaps = []
    for bid, ships in new_2d.items():
        if bid == home_bid or bid == '_unassigned' or bid not in valid_bids_for_A:
            continue
        nB = len(ships)
        if nB < 2:
            continue
        for jB in range(nB - 1):
            shipB0, shipB1 = ships[jB], ships[jB + 1]
            elig_B0 = {str(b['ID']) for b in eligibility.get(shipB0, [])}
            elig_B1 = {str(b['ID']) for b in eligibility.get(shipB1, [])}
            if home_bid not in elig_B0 or home_bid not in elig_B1:
                continue
            delta = (
                abs((eta_A0 - df_ships.loc[shipB0, 'KEDATANGAN']).total_seconds() / 3600.0) +
                abs((eta_A1 - df_ships.loc[shipB1, 'KEDATANGAN']).total_seconds() / 3600.0)
            )
            all_valid_swaps.append((delta, bid, jB))
    if not all_valid_swaps:
        return new_2d, set()
    all_valid_swaps.sort(key=lambda x: x[0])
    top_n = max(3, len(all_valid_swaps) // 5)
    _, target_bid, jB = rng.choice(all_valid_swaps[:top_n])
    kade_B = new_2d[target_bid]
    a0, a1 = kade_A[iA], kade_A[iA + 1]
    b0, b1 = kade_B[jB], kade_B[jB + 1]
    kade_A[iA], kade_A[iA + 1] = b0, b1
    kade_B[jB], kade_B[jB + 1] = a0, a1
    return new_2d, {home_bid, target_bid}


# =============================================================================
# BUILD RUN PARAMS S1
# =============================================================================
def _build_run_params_s1(rnd, vparams, df, eligibility, berths=None):
    groups = {}
    for i in df.index:
        groups.setdefault(int(df.loc[i, '_priority']), []).append(i)

    ship_order = []
    for p in sorted(groups):
        grp = groups[p][:]
        if vparams['shuffle_ship_in_group']:
            rnd.shuffle(grp)
        ship_order.extend(grp)

    berth_order_per_ship = {}
    for i in df.index:
        elig     = eligibility.get(i, [])[:]
        strategy = vparams.get('berth_strategy', 'default')

        if strategy == 'selatan_first':
            selatan = [b for b in elig if 'SELATAN' in str(b.get('DERMAGA', '')).upper()]
            utara   = [b for b in elig if 'SELATAN' not in str(b.get('DERMAGA', '')).upper()]
            elig    = selatan + utara
        elif strategy == 'least_loaded':
            elig = sorted(elig, key=lambda b: b['END'] - b['START'], reverse=True)
        elif rnd.random() < vparams['shuffle_berth_prob']:
            rnd.shuffle(elig)

        berth_order_per_ship[i] = elig

    return ship_order, berth_order_per_ship


# =============================================================================
# RUN SINGLE S1
# =============================================================================
def _run_single_s1(df, berths, id_to_parent, id_to_nama, eligibility,
                   config, weights, ship_order, berth_order_per_ship,
                   tide_windows=None):
    approach_td    = pd.Timedelta(hours=config['APPROACH_TIME'])
    unique_parents = set(id_to_parent.values())
    parent_occ     = {p: [] for p in unique_parents}
    kade_usage_counter = {str(b['ID']): set() for b in berths}
    results = []

    for ship_id in ship_order:
        ship = df.loc[ship_id]

        ordered_candidates = berth_order_per_ship.get(ship_id, [])
        valid_candidates   = [b for b in ordered_candidates if b in eligibility[ship_id]]

        if not valid_candidates:
            results.append(_empty_result(ship_id))
            continue

        best = None
        for berth in valid_candidates:
            berth_id = str(berth['ID'])
            parent   = id_to_parent[berth_id]
            slot     = find_slot_ch1(ship, berth, parent_occ[parent], config,
                                     tide_windows=tide_windows)
            if slot:
                t_start, t_end, s_pos, e_pos = slot
                if _is_better(t_start, berth_id, best, kade_usage_counter):
                    best = (t_start, t_end, s_pos, e_pos, berth_id, parent)

        if best is None:
            results.append(_empty_result(ship_id))
            continue

        t_start, t_end, start_pos, end_pos, berth_id, parent = best
        kade_usage_counter[berth_id].add(ship_id)

        new_occ = {
            'bid'      : berth_id,
            'start'    : t_start,
            'end'      : t_end,
            'start_pos': start_pos,
            'end_pos'  : end_pos,
            'ship_id'  : ship_id,
        }
        keys = [o['start_pos'] for o in parent_occ[parent]]
        idx  = bisect.bisect_left(keys, start_pos)
        parent_occ[parent].insert(idx, new_occ)

        penalty  = get_tide_depth_penalty(t_start, t_end, config)
        waiting_h = (t_start - ship['KEDATANGAN'] - approach_td).total_seconds() / 3600

        results.append({
            'idx'               : ship_id,
            'DERMAGA_ASSIGNED'  : berth_id,
            'NAMA_DERMAGA'      : id_to_nama[berth_id],
            'MULAI_SANDAR'      : t_start.round('s'),
            'SELESAI_SANDAR'    : t_end.round('s'),
            'WAITING_TIME_HOURS': round(max(waiting_h, 0.0), 4),
            'POSISI_START_M'    : round(start_pos, 2),
            'POSISI_END_M'      : round(end_pos, 2),
            'TIDE_FLAG'         : 1 if abs(penalty - (config['UNDER_KEEL_CLEARANCE'] + config['TIDE_DELTA'])) < config['EPSILON'] else 0,
            'IS_LATE'           : int(waiting_h > weights['WAIT_THRESHOLD']),
            'REBERTHING'        : 0,
        })
    return results


# =============================================================================
# RUN CH1
# =============================================================================
def run_ch1(df_kapal_raw, df_dermaga,
            population_size=10,
            base_seed=None,
            verbose=True):

    total_runs = population_size
    base_seed  = int(base_seed) if base_seed is not None else random.randint(0, 2**30 - 1)

    df, berths, id_to_parent, id_to_nama, eligibility = _prepare_data(
        df_kapal_raw, df_dermaga, CONFIG
    )
    t_min = df['MULAI_SANDAR_AWAL'].min()
    t_max = df['MULAI_SANDAR_AWAL'].max() + pd.Timedelta(hours=float(df['BERTH_TIME'].max()))
    tide_windows = build_tide_windows(t_min, t_max, CONFIG)

    solutions    = []
    metrics_rows = []
    run_idx      = 0
    start_time   = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"POPULATION CONSTRUCTIVE HEURISTIC")
        print(f"Populasi: {population_size} | Total run: {total_runs} | Seed: {base_seed}")
        print(f"{'='*60}")

    for p in range(population_size):
        run_idx += 1
        seed = (base_seed + p) & 0xFFFFFFFF
        rnd  = random.Random(seed)

        vparams = _make_variation_params(rnd)
        ship_order, berth_order_per_ship = _build_run_params_s1(
            rnd, vparams, df, eligibility, berths=berths
        )

        results = _run_single_s1(
            df, berths, id_to_parent, id_to_nama, eligibility,
            CONFIG, WEIGHTS, ship_order, berth_order_per_ship, tide_windows=tide_windows
        )
        df_sol     = _build_output_df1(df, results)
        metrics    = evaluate_solution_metrics(df_sol)
        sequence_2d = generate_sol2d_s1(df_sol)

        solutions.append({
            'individual'      : p + 1,
            'seed'            : seed,
            'df_schedule'     : df_sol,
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

        log_every = max(1, total_runs // 10)
        if verbose and (run_idx % log_every == 0 or run_idx == total_runs):
            elapsed = time.time() - start_time
            print(f"  Run {run_idx:>4}/{total_runs}  | Elapsed: {elapsed:.1f}s")

    metrics_df = pd.DataFrame(metrics_rows)

    if verbose:
        print(f"\n{'='*60}")
        print(f"SELESAI — {run_idx} solusi dihasilkan")
        print(f"{'='*60}\n")

    return solutions, metrics_df


# =============================================================================
# LOVEBIRD OPTIMIZER S1
# =============================================================================
class LoveBirdOptimizer2D:
    def __init__(self, df_kapal, df_dermaga, config=None, population_size=20,
             max_generations=50, weights=None, seed=42):
        self.config  = CONFIG
        self.weights = WEIGHTS
        self.approach_time = self.config['APPROACH_TIME']
        self._approach_td  = pd.Timedelta(hours=self.approach_time)

        self.cat_weights     = self.weights['CATEGORY_WEIGHTS']
        self.sop_multiplier  = float(self.weights['PENALTY_WAIT_SOP'])
        self.sop_threshold_h = float(self.weights['WAIT_THRESHOLD'])

        self.df_kapal = df_kapal.copy()
        self.df_kapal['ID_KUNJUNGAN'] = self.df_kapal['ID_KUNJUNGAN'].astype(str)
        self.df_kapal['BERTH_PART']   = pd.to_numeric(
            self.df_kapal.get('BERTH_PART', 1), errors='coerce'
        ).fillna(1).astype(int)
        self.df_kapal = self.df_kapal.sort_values(
            ['ID_KUNJUNGAN', 'BERTH_PART']
        ).reset_index(drop=True)
        self.df_kapal['MULAI_SANDAR_AWAL'] = (
            self.df_kapal['KEDATANGAN'] + self._approach_td)

        self.all_ship_indices = self.df_kapal.index.tolist()

        berths_raw      = df_dermaga.copy().to_dict('records')
        self.berths     = berths_raw
        self.all_bid    = sorted({str(b['ID']) for b in berths_raw})
        self.id_to_nama = {str(b['ID']): str(b.get('DERMAGA', b['ID'])) for b in berths_raw}
        self.berth_map  = {str(b['ID']): b for b in berths_raw}

        self._bid_to_group     = {str(b['ID']): str(b.get('DERMAGA', b['ID'])) for b in berths_raw}
        self._all_groups       = set(self._bid_to_group.values())
        self._init_berth_count = {str(b['ID']): 0 for b in berths_raw}

        self.eligibility = {
            i: [b for b in berths_raw if ship_fits_berth(row, b, CONFIG)]
            for i, row in self.df_kapal.iterrows()
        }

        self.population_size = population_size
        self.max_generations = max_generations
        self.n_elites    = max(1, int(round(population_size * 0.10)))
        self.n_offspring = population_size - self.n_elites
        self.n_best_off  = max(1, int(round(population_size * 0.90)))
        self.ls_iters    = max(1, max_generations // 2)

        self.fitness_cache   = {}
        self.history         = []
        self._archive_size   = max(self.n_elites, int(round(population_size * 0.20)))
        self._global_archive = []

        self._rng = random.Random(seed)
        np.random.seed(seed)

        t_min = self.df_kapal['MULAI_SANDAR_AWAL'].min()
        t_max = (self.df_kapal['MULAI_SANDAR_AWAL'].max()
                 + pd.Timedelta(hours=float(self.df_kapal['BERTH_TIME'].max())))
        self._tide_windows = build_tide_windows(t_min, t_max, CONFIG)

    def _schedule_2d_to_id_kunjungan(self, schedule_2d):
        out = {}
        for bid, ships in schedule_2d.items():
            out[bid] = []
            for ship_idx in ships:
                if ship_idx in self.df_kapal.index:
                    kid = str(self.df_kapal.loc[ship_idx, 'ID_KUNJUNGAN'])
                    out[bid].append(kid)
                else:
                    out[bid].append(str(ship_idx))
        return out

    def _evaluate(self, schedule_2d, affected_bids=None, from_df=None, prev_result=None):
        key = tuple((k, tuple(v)) for k, v in sorted(schedule_2d.items()))
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        if from_df is not None:
            df_sol = from_df.copy()
        else:
            ship_order = []
            for bid in sorted(k for k in schedule_2d if k != '_unassigned'):
                ship_order.extend(schedule_2d[bid])
            ship_order.extend(schedule_2d.get('_unassigned', []))

            df_sol = simulate_schedule(
                self.df_kapal, self.berths, ship_order,
                self.eligibility, self.id_to_nama, self.config,
                self._approach_td, self._bid_to_group, self._all_groups,
                self._init_berth_count,
                cat_weights=self.cat_weights,
                sop_multiplier=self.sop_multiplier,
                sop_threshold_h=self.sop_threshold_h,
                affected_bids=affected_bids,
                prev_result=prev_result,
                new_schedule_2d=schedule_2d,
                tide_skip_cache={},
                tide_windows=self._tide_windows,
            )

        fit, n_ass, n_late, total_wait_h = fitness_s1(
            df_sol, self.weights, self._approach_td
        )

        if n_ass < len(self.all_ship_indices):
            fit = float('inf')

        if 'IS_LATE' not in df_sol.columns:
            df_sol['IS_LATE'] = (df_sol['WAITING_TIME_HOURS'] > self.sop_threshold_h).astype(int)
        if 'REBERTHING' not in df_sol.columns:
            df_sol['REBERTHING'] = 0

        result = (fit, total_wait_h, n_ass, n_late, df_sol, schedule_2d)
        self.fitness_cache[key] = result
        return result

    def _pick_ship(self, schedule_2d):
        pool = [(s, len(ships))
                for bid, ships in schedule_2d.items()
                if bid != '_unassigned'
                for s in ships]
        if not pool:
            unass = schedule_2d.get('_unassigned', [])
            if not unass:
                return None, 0
            return self._rng.choice(unass), 0
        chosen, n_kade = self._rng.choice(pool)
        return chosen, n_kade

    def _apply_op(self, schedule_2d):
        chosen, n_kade = self._pick_ship(schedule_2d)
        if chosen is None:
            return _deep_copy_2d(schedule_2d), set()

        home_bid = next((bid for bid, ships in schedule_2d.items() if chosen in ships), None)
        if home_bid is None:
            return _deep_copy_2d(schedule_2d), set()

        if home_bid == '_unassigned':
            args  = (schedule_2d, chosen, self._rng, self.df_kapal, self.eligibility, self.berth_map)
            child, aff = op_interchange(*args, force_inter=True)
            ok, _ = validate_2d(child, self.all_ship_indices, self.eligibility)
            if not ok:
                child = repair_2d(child, self.all_ship_indices, self.eligibility, self._rng)
                aff   = set(child.keys())
            return child, aff

        home_lst = schedule_2d.get(home_bid, [])
        n_kade   = len(home_lst)

        try:
            pos = home_lst.index(chosen)
        except ValueError:
            return _deep_copy_2d(schedule_2d), set()

        valid_ops = []
        if n_kade >= 2:
            valid_ops.extend([1, 2])
        if n_kade >= 2 and pos + 1 < n_kade:
            valid_ops.append(3)
        if n_kade >= 4 and 0 < pos < n_kade - 1:
            valid_ops.append(4)
        if n_kade >= 2 and pos + 1 < n_kade:
            valid_ops.append(5)

        if not valid_ops:
            return _deep_copy_2d(schedule_2d), set()

        op_id = self._rng.choice(valid_ops)
        args  = (schedule_2d, chosen, self._rng, self.df_kapal, self.eligibility, self.berth_map)

        if op_id == 1:   child, aff = op_flip(*args)
        elif op_id == 2: child, aff = op_interchange(*args, force_inter=(n_kade == 1))
        elif op_id == 3: child, aff = op_move_and_push(*args)
        elif op_id == 4: child, aff = op_2swap_same(*args)
        elif op_id == 5: child, aff = op_2swap_inter(*args)
        else:            child, aff = _deep_copy_2d(schedule_2d), set()

        ok, _ = validate_2d(child, self.all_ship_indices, self.eligibility)
        if not ok:
            child = repair_2d(child, self.all_ship_indices, self.eligibility, self._rng)
            aff   = set(child.keys())

        return child, aff

    def _apply_ls_op(self, schedule_2d, pop_fits=None):
        chosen, _ = self._pick_ship(schedule_2d)
        if chosen is None:
            return _deep_copy_2d(schedule_2d), set()

        home_bid = next((bid for bid, ships in schedule_2d.items() if chosen in ships), None)
        if home_bid is None or home_bid == '_unassigned':
            return _deep_copy_2d(schedule_2d), set()

        home_lst = schedule_2d.get(home_bid, [])
        n_kade   = len(home_lst)
        if n_kade < 2:
            return _deep_copy_2d(schedule_2d), set()

        try:
            pos = home_lst.index(chosen)
        except ValueError:
            return _deep_copy_2d(schedule_2d), set()

        valid_ops = []
        if n_kade >= 4 and 0 < pos < n_kade - 1:
            valid_ops.append(4)
        if n_kade >= 2 and pos + 1 < n_kade:
            valid_ops.append(5)
        if n_kade >= 2:
            valid_ops.append(2)

        if not valid_ops:
            return _deep_copy_2d(schedule_2d), set()

        op_id = self._rng.choice(valid_ops)
        args  = (schedule_2d, chosen, self._rng, self.df_kapal, self.eligibility, self.berth_map)

        if op_id == 2:   child, aff = op_interchange(*args)
        elif op_id == 4: child, aff = op_2swap_same(*args)
        elif op_id == 5: child, aff = op_2swap_inter(*args)
        else:            child, aff = _deep_copy_2d(schedule_2d), set()

        ok, _ = validate_2d(child, self.all_ship_indices, self.eligibility)
        if not ok:
            child = repair_2d(child, self.all_ship_indices, self.eligibility, self._rng)
            aff   = set(child.keys())

        return child, aff

    def _roulette(self, pop_fits, eps=1e-6):
        scores = np.array([1.0 / (f + eps) for f in pop_fits])
        probs  = scores / scores.sum()
        return int(np.random.choice(len(pop_fits), p=probs))

    def _sol1_to_2d(self, sol):
        df_sol = sol.get('df_schedule')
        if df_sol is None:
            return {bid: [] for bid in self.all_bid + ['_unassigned']}

        sched_2d   = {bid: [] for bid in self.all_bid}
        sched_2d['_unassigned'] = []

        assigned   = df_sol[df_sol['DERMAGA_ASSIGNED'].notna()].copy()
        unassigned = df_sol[df_sol['DERMAGA_ASSIGNED'].isna()]

        if 'MULAI_SANDAR' in assigned.columns:
            assigned = assigned.sort_values(['DERMAGA_ASSIGNED', 'MULAI_SANDAR'])

        for _, row in assigned.iterrows():
            bid = str(row['DERMAGA_ASSIGNED'])
            try:
                bid = str(int(float(bid)))
            except (ValueError, OverflowError):
                pass
            ship_id = row.name
            if bid in sched_2d:
                sched_2d[bid].append(ship_id)
            else:
                sched_2d['_unassigned'].append(row.name)

        for _, row in unassigned.iterrows():
            sched_2d['_unassigned'].append(row.name)

        return sched_2d

    def _update_archive(self, candidates):
        merged_dict = {}
        for s in self._global_archive + candidates:
            key = tuple((k, tuple(v)) for k, v in sorted(s[5].items()))
            if key not in merged_dict or s[0] < merged_dict[key][0]:
                merged_dict[key] = s
        self._global_archive = sorted(merged_dict.values(), key=lambda x: x[0])[:self._archive_size]

    def optimize(self, initial_solutions):
        t_start_total = time.perf_counter()
        n_total = len(self.all_ship_indices)

        print("=" * 60)
        print("FASE 1: Membangun populasi awal dari initial solutions")
        print("=" * 60)

        population   = []
        pop_fits     = []
        pop_waits    = []
        pop_assigned = []
        pop_n_late   = []
        pop_dfs      = []

        for i, sol in enumerate(initial_solutions[:self.population_size]):
            df_sol = sol.get('df_schedule')
            if df_sol is None:
                continue

            if len(df_sol) != len(self.all_ship_indices):
                print(f"[WARNING] Solusi ke-{i}: jumlah kapal {len(df_sol)} "
                    f"!= {len(self.all_ship_indices)}")

            assigned_ids = df_sol['DERMAGA_ASSIGNED'].dropna().astype(str).unique()
            unknown_ids  = set(assigned_ids) - set(self.all_bid)
            if unknown_ids:
                print(f"[WARNING] Solusi ke-{i}: DERMAGA_ASSIGNED berisi "
                    f"nilai tidak dikenal: {unknown_ids}")
                print(f"          Kemungkinan berisi NAMA bukan ID. "
                    f"Pastikan CH output pakai berth_id bukan nama.")

            sched_2d = self._sol1_to_2d(sol)
            ok, _    = validate_2d(sched_2d, self.all_ship_indices, self.eligibility)

            if not ok:
                sched_2d = repair_2d(sched_2d, self.all_ship_indices, self.eligibility, self._rng)
                fit, total_wait_h, n_ass, n_late, df_sol, sched = self._evaluate(sched_2d, from_df=None)
            else:
                fit, total_wait_h, n_ass, n_late, df_sol, sched = self._evaluate(sched_2d, from_df=df_sol)

            population.append(sched_2d)
            pop_fits.append(fit)
            pop_waits.append(total_wait_h)
            pop_assigned.append(n_ass)
            pop_n_late.append(n_late)
            pop_dfs.append(df_sol)

        gbest_idx  = int(np.argmin(pop_fits))
        gbest_fit  = pop_fits[gbest_idx]
        gbest_sch  = _deep_copy_2d(population[gbest_idx])
        gbest_wait = pop_waits[gbest_idx]
        gbest_ass  = pop_assigned[gbest_idx]
        gbest_late = pop_n_late[gbest_idx]
        gbest_df   = pop_dfs[gbest_idx]

        init_candidates = list(zip(pop_fits, pop_waits, pop_assigned, pop_n_late, pop_dfs, population))
        self._update_archive(init_candidates)

        print(f"\nGBest awal: Fit={gbest_fit:,.2f} | Wait={gbest_wait:.2f}h | "
              f"Late={gbest_late}/{n_total} | Assigned={gbest_ass}/{n_total}")
        print(f"\n{'='*60}")
        print(f"FASE 2: Optimasi ({self.max_generations} generasi)")
        print(f"{'='*60}")

        for gen in range(self.max_generations):
            t_gen = time.perf_counter()

            archive_scheds   = [s[5] for s in self._global_archive]
            archive_fits     = [s[0] for s in self._global_archive]
            archive_waits    = [s[1] for s in self._global_archive]
            archive_assigned = [s[2] for s in self._global_archive]
            archive_n_late   = [s[3] for s in self._global_archive]
            archive_dfs      = [s[4] for s in self._global_archive]

            pool_scheds   = population   + archive_scheds
            pool_fits     = pop_fits     + archive_fits
            pool_waits    = pop_waits    + archive_waits
            pool_assigned = pop_assigned + archive_assigned
            pool_n_late   = pop_n_late   + archive_n_late
            pool_dfs      = pop_dfs      + archive_dfs

            elite_indices  = list(np.argsort(pool_fits)[:self.n_elites])
            elite_scheds   = [_deep_copy_2d(pool_scheds[i]) for i in elite_indices]
            elite_fits     = [pool_fits[i]     for i in elite_indices]
            elite_waits    = [pool_waits[i]    for i in elite_indices]
            elite_assigned = [pool_assigned[i] for i in elite_indices]
            elite_n_late   = [pool_n_late[i]   for i in elite_indices]
            elite_dfs      = [pool_dfs[i]      for i in elite_indices]

            off_scheds, off_fits, off_waits = [], [], []
            off_assigned, off_n_late, off_dfs = [], [], []

            for _ in range(self.n_offspring):
                p_idx  = self._roulette(pop_fits)
                parent = population[p_idx]
                child, aff = self._apply_op(parent)

                parent_key = tuple((k, tuple(v)) for k, v in sorted(parent.items()))
                cached_parent = self.fitness_cache.get(parent_key)
                prev_result_data = None
                if cached_parent:
                    prev_result_data = {'df_schedule': cached_parent[4], 'schedule_2d': cached_parent[5]}

                fit, total_wait_h, n_ass, n_late, df_sol, sched = self._evaluate(
                    child, affected_bids=aff, prev_result=prev_result_data)

                off_scheds.append(child)
                off_fits.append(fit)
                off_waits.append(total_wait_h)
                off_assigned.append(n_ass)
                off_n_late.append(n_late)
                off_dfs.append(df_sol)

            off_candidates = list(zip(off_fits, off_waits, off_assigned, off_n_late, off_dfs, off_scheds))
            self._update_archive(off_candidates)

            best_off_idx     = list(np.argsort(off_fits))[:self.n_best_off]
            new_off_scheds   = [off_scheds[i]   for i in best_off_idx]
            new_off_fits     = [off_fits[i]     for i in best_off_idx]
            new_off_waits    = [off_waits[i]    for i in best_off_idx]
            new_off_assigned = [off_assigned[i] for i in best_off_idx]
            new_off_n_late   = [off_n_late[i]   for i in best_off_idx]
            new_off_dfs      = [off_dfs[i]      for i in best_off_idx]

            curr_idx  = int(np.argmin(new_off_fits))
            curr_fit  = new_off_fits[curr_idx]
            curr_wait = new_off_waits[curr_idx]
            curr_ass  = new_off_assigned[curr_idx]
            curr_late = new_off_n_late[curr_idx]
            curr_df   = new_off_dfs[curr_idx]
            curr_sch  = new_off_scheds[curr_idx]
            status    = "-> No change   "

            if curr_fit < gbest_fit:
                gbest_fit  = curr_fit
                gbest_sch  = _deep_copy_2d(curr_sch)
                gbest_wait = curr_wait
                gbest_ass  = curr_ass
                gbest_late = curr_late
                gbest_df   = curr_df
                status = "** GBest UPDATED"
            else:
                ls_improved = False
                for ls_iter in range(self.ls_iters):
                    cand, aff = self._apply_ls_op(gbest_sch, pop_fits=pop_fits)
                    prev_result_ls = {'df_schedule': gbest_df, 'schedule_2d': gbest_sch}
                    fit, total_wait_h, n_ass, n_late, df_sol, sched = self._evaluate(
                        cand, aff, prev_result=prev_result_ls)

                    if fit < gbest_fit:
                        gbest_fit  = fit
                        gbest_sch  = cand
                        gbest_wait = total_wait_h
                        gbest_ass  = n_ass
                        gbest_late = n_late
                        gbest_df   = df_sol
                        ls_improved = True
                        status = f">> LS [{ls_iter+1:3d}] SUCCESS"
                        self._update_archive([(fit, total_wait_h, n_ass, n_late, df_sol, cand)])
                        break
                if not ls_improved:
                    status = "-- LS no improv"

            population   = elite_scheds   + new_off_scheds
            pop_fits     = elite_fits     + new_off_fits
            pop_waits    = elite_waits    + new_off_waits
            pop_assigned = elite_assigned + new_off_assigned
            pop_n_late   = elite_n_late   + new_off_n_late
            pop_dfs      = elite_dfs      + new_off_dfs

            if len(population) > self.population_size:
                idx_keep     = list(np.argsort(pop_fits)[:self.population_size])
                population   = [population[i]   for i in idx_keep]
                pop_fits     = [pop_fits[i]      for i in idx_keep]
                pop_waits    = [pop_waits[i]     for i in idx_keep]
                pop_assigned = [pop_assigned[i]  for i in idx_keep]
                pop_n_late   = [pop_n_late[i]    for i in idx_keep]
                pop_dfs      = [pop_dfs[i]       for i in idx_keep]

            if len(population) < self.population_size:
                n_elite = len(elite_scheds)
                i_pad   = 0
                while len(population) < self.population_size:
                    idx_pad = i_pad % n_elite
                    population.append(_deep_copy_2d(elite_scheds[idx_pad]))
                    pop_fits.append(elite_fits[idx_pad])
                    pop_waits.append(elite_waits[idx_pad])
                    pop_assigned.append(elite_assigned[idx_pad])
                    pop_n_late.append(elite_n_late[idx_pad])
                    pop_dfs.append(elite_dfs[idx_pad])
                    i_pad += 1

            gen_time = time.perf_counter() - t_gen
            self.history.append({
                'generation': gen + 1, 'fitness': gbest_fit,
                'wait_hours': gbest_wait, 'n_late': gbest_late,
                'assigned': gbest_ass, 'gen_time_s': round(gen_time, 3),
            })
            print(f"Gen {gen+1:3d}: {status} | Fit={gbest_fit:,.2f} | "
                  f"Wait={gbest_wait:.2f}h | Late={gbest_late}/{n_total} | "
                  f"Assigned={gbest_ass}/{n_total} | Time={gen_time:.2f}s")

        total_time_s = time.perf_counter() - t_start_total
        total_time_m = total_time_s / 60
        final_2d     = self._schedule_2d_to_id_kunjungan(gbest_sch)

        print(f"\n{'='*60}")
        print(f"   SUMMARY OPTIMASI — Skenario 1: Sekali Sandar")
        print(f"{'='*60}")
        print(f"  Status              : SELESAI")
        print(f"  Fitness Total       : {gbest_fit:,.4f}")
        print(f"  Total Waktu Tunggu  : {gbest_wait:.4f} jam")
        print(f"  Kapal > SOP (late)  : {gbest_late}/{n_total}")
        print(f"  Assigned            : {gbest_ass}/{n_total}")
        print(f"  Re-berthing         : 0")
        print(f"  Running Time        : {total_time_s:.2f} detik ({total_time_m:.2f} menit)")
        print(f"{'='*60}\n")

        return {
            'df_schedule'    : gbest_df,
            'schedule_2d'    : final_2d,
            'fitness'        : gbest_fit,
            'total_wait'     : gbest_wait,
            'n_late'         : gbest_late,
            'assigned'       : gbest_ass,
            'running_time_s' : round(total_time_s, 3),
            'running_time_m' : round(total_time_m, 4),
        }, pd.DataFrame(self.history), self.history, None


# =============================================================================
# WRAPPER
# =============================================================================
def run_love_bird_optimization(df_kapal, df_dermaga, initial_solutions, **kwargs):
    optimizer = LoveBirdOptimizer2D(
        df_kapal        = df_kapal,
        df_dermaga      = df_dermaga,
        population_size = kwargs.get('population_size', 20),
        max_generations = kwargs.get('max_generations', 50),
        seed            = kwargs.get('seed', 42),
    )
    return optimizer.optimize(initial_solutions)