# -*- coding: utf-8 -*-
"""
Step 2 - Segment-level aggregation (daily files, chunked, gz supported)

Input : second_YYYY_M_D.csv or second_YYYY_M_D.csv.gz (one day per file)
Output: segment_YYYY_M_D.csv.gz (one row per segment_id)

UPDATE (only change):
- Hard accel/brake counts are computed as "time-consecutive-2-seconds events"
  ignoring NaN rows by collapsing to 1 row per second (gps_timestamp) first.

Definition (event-run counting):
- hard_accel_event_count: number of continuous runs with length>=2 seconds where accel >= +2.5 (mph/s)
- hard_brake_event_count: number of continuous runs with length>=2 seconds where accel <= -2.5 (mph/s)
- hard_event_count = hard_accel_event_count + hard_brake_event_count

Everything else remains unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# ====== USER SETTINGS ======
INPUT_DIR = r"D:\Hurricane Paper"          # <-- change if needed
OUT_SUBDIR = "_segment_level"
CHUNK_SIZE = 1_000_000                      # adjust if needed

# Threshold in the SAME unit as accel column (your second-level shows accel_mphps = mph/s)
HARD_A_THRESHOLD = 2.5                      # accel >= +2.5 OR accel <= -2.5 -> hard counts

# Kept (used for eventtype103/201 duration capping)
CONSEC_MAX_GAP_SEC = 1.5

# Column name candidates (auto-detect)
SEGMENT_COL_CANDIDATES = ["segment_id", "segmentId", "segmentID"]
TIME_COL_CANDIDATES = ["gps_timestamp", "timestamp", "time", "datetime", "gps_time"]
LAT_COL_CANDIDATES = ["gps_lat", "latitude", "lat"]
LON_COL_CANDIDATES = ["gps_lon", "longitude", "lon", "lng"]
SPEED_COL_CANDIDATES = ["gps_speed", "speed"]
ACCEL_COL_CANDIDATES = ["accel", "acceleration", "gps_accel", "accel_mphps"]
TURN_COL_CANDIDATES = ["turning_delta", "heading_change", "turn_delta", "turning_change", "turning_change_deg", "turning_change_deg"]
COUNTY_COL_CANDIDATES = ["fl_counties", "fl_countie", "county_code"]
FUNC_CLASS_COL_CANDIDATES = ["func_class", "functional_class", "fclass", "fc"]

EVENT103_COL = "eventtype103"
EVENT201_COL = "eventtype201"


def pick_col(columns: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if required:
        raise KeyError(f"Cannot find required column. Tried: {candidates}. Existing: {columns[:30]} ...")
    return None


def ensure_time_seconds(s: pd.Series) -> pd.Series:
    """
    Convert timestamp column to numeric seconds for diff computations.
    Supports:
      - numeric epoch seconds/ms
      - ISO datetime strings
    Returns float seconds.
    """
    if pd.api.types.is_numeric_dtype(s):
        med = float(np.nanmedian(s.to_numpy(dtype=float)))
        if med > 1e12:   # ms epoch
            return s.astype(float) / 1000.0
        return s.astype(float)

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.view("int64") / 1e9


def mode_or_first(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return np.nan
    m = s2.mode()
    if not m.empty:
        return m.iloc[0]
    return s2.iloc[0]


def deterministic_mode(s: pd.Series):
    """
    Return deterministic mode:
    - unique mode -> mode value
    - tie -> choose smallest numeric value if all numeric, otherwise lexicographically smallest string
    - all missing -> NaN
    """
    s2 = s.dropna()
    if s2.empty:
        return np.nan

    vc = s2.value_counts(dropna=True)
    if vc.empty:
        return np.nan

    max_count = vc.iloc[0]
    modes = vc[vc == max_count].index.tolist()
    if len(modes) == 1:
        return modes[0]

    modes_series = pd.Series(modes)
    modes_num = pd.to_numeric(modes_series, errors="coerce")
    if modes_num.notna().all():
        min_idx = int(modes_num.to_numpy(dtype=float).argmin())
        return modes[min_idx]

    return min(str(x) for x in modes)


def polyline_distance_mile(lat: pd.Series, lon: pd.Series) -> float:
    """Compute trajectory polyline length (sum of adjacent-point Haversine distances) in miles."""
    lat_arr = pd.to_numeric(lat, errors="coerce").to_numpy(dtype=float)
    lon_arr = pd.to_numeric(lon, errors="coerce").to_numpy(dtype=float)

    if len(lat_arr) < 2 or len(lon_arr) < 2:
        return 0.0

    lat1 = np.radians(lat_arr[:-1])
    lon1 = np.radians(lon_arr[:-1])
    lat2 = np.radians(lat_arr[1:])
    lon2 = np.radians(lon_arr[1:])

    valid = np.isfinite(lat1) & np.isfinite(lon1) & np.isfinite(lat2) & np.isfinite(lon2)
    if not np.any(valid):
        return 0.0

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    earth_radius_m = 6_371_000.0
    meters = earth_radius_m * c
    meters = np.where(valid, meters, 0.0)

    return float(meters.sum() / 1609.344)


def count_events_and_duration(flag: np.ndarray, tsec: np.ndarray, consec_max_gap: float) -> Tuple[int, float]:
    """
    flag: 0/1 array for a segment, in time order
    tsec: seconds array (float), same length, in time order
    event_count: number of 0->1 transitions (treat gaps > consec_max_gap as a break)
    duration_sum: sum of dt where flag==1 (dt capped by consec_max_gap to avoid huge gaps)
    """
    n = len(flag)
    if n == 0:
        return 0, 0.0

    dt = np.diff(tsec, prepend=np.nan)
    dt[0] = 1.0
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 1.0)
    dt = np.minimum(dt, consec_max_gap)

    gap_break = (np.diff(tsec, prepend=tsec[0]) > consec_max_gap)

    prev = np.roll(flag, 1)
    prev[0] = 0
    starts = (flag == 1) & ((prev == 0) | gap_break)
    event_count = int(starts.sum())

    duration_sum = float(dt[flag == 1].sum())
    return event_count, duration_sum


def _count_consecutive2_runs_by_second(
    ts: pd.Series,
    accel: pd.Series,
    threshold: float,
    is_brake: bool
) -> Tuple[int, float]:
    """
    Count number of continuous event-runs (length>=2 seconds) in time, ignoring NaN rows:
      1) collapse to 1 record per second (same timestamp) using last() after sorting
      2) drop NaN accel
      3) define over-threshold mask
      4) require dt==1 between consecutive kept seconds to be considered consecutive
      5) count run starts where a run reaches length>=2 seconds

    Returns (event-run count, duration_sec), where duration only sums runs with length>=2.
    """
    if ts.empty:
        return 0, 0.0

    x = pd.DataFrame({"ts": ts, "acc": pd.to_numeric(accel, errors="coerce")}).copy()
    x["ts"] = pd.to_datetime(x["ts"], errors="coerce", utc=True)
    x = x.dropna(subset=["ts"]).sort_values("ts")

    # Collapse to 1 row per second (timestamp) to avoid dt=0 duplicates breaking continuity
    x = x.groupby("ts", as_index=False)["acc"].last()

    # Keep only valid accel values
    x = x.dropna(subset=["acc"])
    if x.empty:
        return 0, 0.0

    dt = x["ts"].diff().dt.total_seconds()
    if is_brake:
        over = (x["acc"] <= -threshold)
    else:
        over = (x["acc"] >= threshold)

    event_count = 0
    duration_sec = 0.0
    run_len = 0

    for i in range(len(x)):
        is_over = bool(over.iloc[i])
        is_consecutive = bool(i > 0 and dt.iloc[i] == 1)

        if is_over:
            if i > 0 and bool(over.iloc[i - 1]) and is_consecutive:
                run_len += 1
            else:
                if run_len >= 2:
                    event_count += 1
                    duration_sec += float(run_len)
                run_len = 1
        else:
            if run_len >= 2:
                event_count += 1
                duration_sec += float(run_len)
            run_len = 0

    if run_len >= 2:
        event_count += 1
        duration_sec += float(run_len)

    return event_count, duration_sec


def summarize_one_segment(df: pd.DataFrame,
                          seg_col: str, time_col: str,
                          lat_col: str, lon_col: str,
                          speed_col: str,
                          accel_col: Optional[str],
                          turn_col: Optional[str],
                          county_col: Optional[str],
                          func_class_col: Optional[str]) -> Dict:
    df = df.sort_values(time_col, kind="mergesort")
    n = len(df)

    out: Dict = {}
    seg_id = df.iloc[0][seg_col]
    out[seg_col] = seg_id
    out["n_points"] = n

    tsec = ensure_time_seconds(df[time_col]).to_numpy(dtype=float)
    out["start_time"] = df.iloc[0][time_col]
    out["end_time"] = df.iloc[-1][time_col]

    if n >= 2 and np.isfinite(tsec[0]) and np.isfinite(tsec[-1]):
        out["duration_sec"] = float(max(tsec[-1] - tsec[0], 0.0))
        dt = np.diff(tsec)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        out["dt_mean"] = float(dt.mean()) if dt.size else np.nan
        out["dt_std"] = float(dt.std(ddof=0)) if dt.size else np.nan
    else:
        out["duration_sec"] = np.nan
        out["dt_mean"] = np.nan
        out["dt_std"] = np.nan

    if county_col is not None:
        out[county_col] = mode_or_first(df[county_col])

    if func_class_col is not None and func_class_col in df.columns:
        out["func_class"] = deterministic_mode(df[func_class_col])
    else:
        out["func_class"] = np.nan

    mid_idx = n // 2
    out["start_lat"] = df.iloc[0][lat_col]
    out["start_lon"] = df.iloc[0][lon_col]
    out["median_lat"] = df.iloc[mid_idx][lat_col]
    out["median_lon"] = df.iloc[mid_idx][lon_col]
    out["end_lat"] = df.iloc[-1][lat_col]
    out["end_lon"] = df.iloc[-1][lon_col]
    out["segment_distance_mile"] = polyline_distance_mile(df[lat_col], df[lon_col])

    sp = pd.to_numeric(df[speed_col], errors="coerce")
    out["speed_mean"] = float(sp.mean()) if sp.notna().any() else np.nan
    out["speed_median"] = float(sp.median()) if sp.notna().any() else np.nan
    out["speed_std"] = float(sp.std(ddof=0)) if sp.notna().any() else np.nan
    out["speed_min"] = float(sp.min()) if sp.notna().any() else np.nan
    out["speed_max"] = float(sp.max()) if sp.notna().any() else np.nan

    # accel stats + hard event-run counts (time-consecutive 2 seconds)
    if accel_col is not None and accel_col in df.columns:
        ac = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=float)
        ac_valid = ac[np.isfinite(ac)]

        out["accel_mean"] = float(np.mean(ac_valid)) if ac_valid.size else np.nan
        out["accel_std"] = float(np.std(ac_valid, ddof=0)) if ac_valid.size else np.nan
        out["accel_abs_mean"] = float(np.mean(np.abs(ac_valid))) if ac_valid.size else np.nan
        out["accel_abs_sum"] = float(np.sum(np.abs(ac_valid))) if ac_valid.size else np.nan

        # UPDATED: time-consecutive-2-seconds event-runs, ignoring NaN rows via per-second collapse
        hard_accel_runs, hard_accel_duration = _count_consecutive2_runs_by_second(
            ts=df[time_col],
            accel=df[accel_col],
            threshold=HARD_A_THRESHOLD,
            is_brake=False
        )
        hard_brake_runs, hard_brake_duration = _count_consecutive2_runs_by_second(
            ts=df[time_col],
            accel=df[accel_col],
            threshold=HARD_A_THRESHOLD,
            is_brake=True
        )
        out["hard_accel_event_count"] = hard_accel_runs
        out["hard_brake_event_count"] = hard_brake_runs
        out["hard_event_count"] = hard_accel_runs + hard_brake_runs
        out["hard_accel_duration_sec"] = hard_accel_duration
        out["hard_brake_duration_sec"] = hard_brake_duration
    else:
        out["accel_mean"] = np.nan
        out["accel_std"] = np.nan
        out["accel_abs_mean"] = np.nan
        out["accel_abs_sum"] = np.nan
        out["hard_accel_event_count"] = 0
        out["hard_brake_event_count"] = 0
        out["hard_event_count"] = 0
        out["hard_accel_duration_sec"] = 0.0
        out["hard_brake_duration_sec"] = 0.0

    # turning stats
    if turn_col is not None and turn_col in df.columns:
        td = pd.to_numeric(df[turn_col], errors="coerce").to_numpy(dtype=float)
        td_valid = td[np.isfinite(td)]
        out["turn_abs_mean"] = float(np.mean(np.abs(td_valid))) if td_valid.size else np.nan
        out["turn_std"] = float(np.std(td_valid, ddof=0)) if td_valid.size else np.nan
        out["turn_abs_sum"] = float(np.sum(np.abs(td_valid))) if td_valid.size else np.nan
    else:
        out["turn_abs_mean"] = np.nan
        out["turn_std"] = np.nan
        out["turn_abs_sum"] = np.nan

    # eventtype103 / 201: event_count + duration_sum
    tsec_for_event = np.where(np.isfinite(tsec), tsec, np.arange(n, dtype=float))

    for col, prefix in [(EVENT103_COL, "event103"), (EVENT201_COL, "event201")]:
        if col in df.columns:
            flag = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy()
            flag = (flag != 0).astype(int)

            cnt, dur = count_events_and_duration(flag, tsec_for_event, CONSEC_MAX_GAP_SEC)
            out[f"{prefix}_event_count"] = cnt
            out[f"{prefix}_duration_sec"] = dur
        else:
            out[f"{prefix}_event_count"] = 0
            out[f"{prefix}_duration_sec"] = 0.0

    return out


def extract_date_tag(filename: str) -> str:
    m = re.search(r"second_(\d{4}_\d{1,2}_\d{1,2})", filename)
    return m.group(1) if m else "unknown_date"


def process_one_file(path: Path, out_dir: Path) -> None:
    print(f"\n[Processing] {path.name}")

    peek = pd.read_csv(path, nrows=5, compression="infer")
    cols = list(peek.columns)

    seg_col = pick_col(cols, SEGMENT_COL_CANDIDATES, required=True)
    time_col = pick_col(cols, TIME_COL_CANDIDATES, required=True)
    lat_col = pick_col(cols, LAT_COL_CANDIDATES, required=True)
    lon_col = pick_col(cols, LON_COL_CANDIDATES, required=True)
    speed_col = pick_col(cols, SPEED_COL_CANDIDATES, required=True)
    accel_col = pick_col(cols, ACCEL_COL_CANDIDATES, required=False)
    turn_col = pick_col(cols, TURN_COL_CANDIDATES, required=False)
    county_col = pick_col(cols, COUNTY_COL_CANDIDATES, required=False)
    func_class_col = pick_col(cols, FUNC_CLASS_COL_CANDIDATES, required=False)

    date_tag = extract_date_tag(path.name)
    out_path = out_dir / f"segment_{date_tag}.csv.gz"

    if out_path.exists():
        out_path.unlink()

    carry = None
    first_write = True

    reader = pd.read_csv(
        path,
        compression="infer",
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for i, chunk in enumerate(reader, start=1):
        if carry is not None and not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        chunk = chunk.sort_values([seg_col, time_col], kind="mergesort")

        last_seg = chunk[seg_col].iloc[-1]
        is_last = chunk[seg_col] == last_seg
        carry = chunk.loc[is_last].copy()
        chunk_main = chunk.loc[~is_last].copy()

        summaries = []
        for _, g in chunk_main.groupby(seg_col, sort=False):
            summaries.append(
                summarize_one_segment(
                    g, seg_col, time_col, lat_col, lon_col,
                    speed_col, accel_col, turn_col, county_col, func_class_col
                )
            )

        if summaries:
            out_df = pd.DataFrame(summaries)
            out_df["date_tag"] = date_tag

            out_df.to_csv(
                out_path,
                index=False,
                compression="gzip",
                mode="wt" if first_write else "at",
                header=first_write
            )
            first_write = False

        print(f"  chunk {i} done. wrote {len(summaries)} segments. carry rows={len(carry)}")

    if carry is not None and not carry.empty:
        summaries = []
        for _, g in carry.groupby(seg_col, sort=False):
            summaries.append(
                summarize_one_segment(
                    g, seg_col, time_col, lat_col, lon_col,
                    speed_col, accel_col, turn_col, county_col, func_class_col
                )
            )
        out_df = pd.DataFrame(summaries)
        out_df["date_tag"] = date_tag
        out_df.to_csv(
            out_path,
            index=False,
            compression="gzip",
            mode="wt" if first_write else "at",
            header=first_write
        )
        print(f"  finalize carry. wrote {len(summaries)} segments.")

    print(f"[OK] Output => {out_path}")


def main():
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        raise SystemExit(f"[Exit] INPUT_DIR does not exist: {in_dir}")

    out_dir = in_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.iterdir()
                    if p.is_file() and re.match(r"second_\d{4}_\d{1,2}_\d{1,2}\.csv(\.gz)?$", p.name, re.I)])

    if not files:
        raise SystemExit(f"[Exit] No files like second_YYYY_M_D.csv(.gz) found in: {in_dir}")

    print(f"[Found] {len(files)} files.")
    for p in files:
        process_one_file(p, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
