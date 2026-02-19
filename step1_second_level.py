# -*- coding: utf-8 -*-
"""
Step 2 - Segment-level aggregation (daily files, chunked, gz supported)

Input : second_YYYY_M_D.csv or second_YYYY_M_D.csv.gz (one day per file)
Output: segment_YYYY_M_D.csv.gz (one row per segment_id)

Included:
- fl_counties (mode)
- func_class (mode)
- start/median/end lat lon
- speed stats: mean/median/std/min/max
- accel stats: mean/std/abs_mean/abs_sum
- hard accel/brake events: consecutive >=2 samples with |a|>2.5 (m/s^2)
- turning_delta stats: abs_mean/std/abs_sum (if column exists)
- eventtype103/201: event_count (0->1) and duration_sum (sum dt when event==1)
- dt_mean/dt_std for QC
"""

from __future__ import annotations

import os
import re
import gzip
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# ====== USER SETTINGS ======
INPUT_DIR = r"D:\Hurricane Paper"          # <-- change if needed
OUT_SUBDIR = "_segment_level"
CHUNK_SIZE = 1_000_000                      # adjust if needed
HARD_A_THRESHOLD = 2.5                      # |a| > 2.5
MIN_CONSEC_POINTS = 2                       # consecutive >=2 points
# If timestamps are not exactly 1s apart, we treat <=1.5s gap as "consecutive"
CONSEC_MAX_GAP_SEC = 1.5

# Column name candidates (auto-detect)
SEGMENT_COL_CANDIDATES = ["segment_id", "segmentId", "segmentID"]
TIME_COL_CANDIDATES = ["gps_timestamp", "timestamp", "time", "datetime", "gps_time"]
LAT_COL_CANDIDATES = ["gps_lat", "latitude", "lat"]
LON_COL_CANDIDATES = ["gps_lon", "longitude", "lon", "lng"]
SPEED_COL_CANDIDATES = ["gps_speed", "speed"]
ACCEL_COL_CANDIDATES = ["accel", "acceleration", "gps_accel"]
TURN_COL_CANDIDATES = ["turning_delta", "heading_change", "turn_delta", "turning_change"]
COUNTY_COL_CANDIDATES = ["fl_counties", "fl_countie", "county_code"]
FUNC_CLASS_COL_CANDIDATES = ["func_class", "functional_class", "fclass"]
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
        # If looks like milliseconds, convert to seconds
        med = float(np.nanmedian(s.to_numpy(dtype=float)))
        if med > 1e12:   # ms epoch
            return s.astype(float) / 1000.0
        return s.astype(float)

    # Try parse datetime strings
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    # convert to unix seconds
    return dt.view("int64") / 1e9


def mode_or_first(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return np.nan
    m = s2.mode()
    if not m.empty:
        return m.iloc[0]
    return s2.iloc[0]


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

    # dt for each point (dt[i] = t[i]-t[i-1]), first dt fallback to 1
    dt = np.diff(tsec, prepend=np.nan)
    # fallback
    dt[0] = 1.0
    # treat negative / nan
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 1.0)
    # cap dt to avoid counting large gaps as "duration"
    dt = np.minimum(dt, consec_max_gap)

    # break points where gap too large: we treat as new "start"
    gap_break = (np.diff(tsec, prepend=tsec[0]) > consec_max_gap)
    # event starts: flag==1 and (prev==0 OR gap_break)
    prev = np.roll(flag, 1)
    prev[0] = 0
    starts = (flag == 1) & ((prev == 0) | gap_break)
    event_count = int(starts.sum())

    duration_sum = float(dt[flag == 1].sum())
    return event_count, duration_sum


def count_hard_events(accel: np.ndarray, tsec: np.ndarray, threshold: float,
                      min_consec_points: int, consec_max_gap: float) -> Tuple[int, int]:
    """
    Hard accel / brake events:
      - accel event: accel > +threshold for >=min_consec_points consecutive samples
      - brake event: accel < -threshold for >=min_consec_points consecutive samples
    Consecutive means time gap <= consec_max_gap.
    Returns: (hard_accel_event_count, hard_brake_event_count)
    """
    n = len(accel)
    if n == 0:
        return 0, 0

    # Define "consecutive" based on time gaps
    gaps = np.diff(tsec, prepend=tsec[0])
    is_consec = gaps <= consec_max_gap
    is_consec[0] = True

    def events_for_mask(mask: np.ndarray) -> int:
        # mask: True where condition holds (accel > thr or accel < -thr)
        # break segment where either mask False or not consecutive
        # find runs of True with consecutive time steps
        run_len = 0
        count = 0
        for i in range(n):
            if mask[i] and is_consec[i]:
                run_len += 1
            elif mask[i] and (not is_consec[i]):
                # gap break starts new run
                if run_len >= min_consec_points:
                    count += 1
                run_len = 1
            else:
                if run_len >= min_consec_points:
                    count += 1
                run_len = 0
        if run_len >= min_consec_points:
            count += 1
        return count

    hard_accel = events_for_mask(accel > threshold)
    hard_brake = events_for_mask(accel < -threshold)
    return hard_accel, hard_brake


def summarize_one_segment(df: pd.DataFrame,
                          seg_col: str, time_col: str,
                          lat_col: str, lon_col: str,
                          speed_col: str,
                          accel_col: Optional[str],
                          turn_col: Optional[str],
                          county_col: Optional[str],
                          func_class_col: Optional[str]) -> Dict:
    # Ensure sorted by time
    df = df.sort_values(time_col, kind="mergesort")
    n = len(df)

    out = {}
    seg_id = df.iloc[0][seg_col]
    out[seg_col] = seg_id
    out["n_points"] = n

    # time in seconds
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

    # county (mode)
    if county_col is not None:
        out[county_col] = mode_or_first(df[county_col])

    # functional class (mode)
    if func_class_col is not None:
        out[func_class_col] = mode_or_first(df[func_class_col])

    # coordinates: start / median / end
    mid_idx = n // 2
    out["start_lat"] = df.iloc[0][lat_col]
    out["start_lon"] = df.iloc[0][lon_col]
    out["median_lat"] = df.iloc[mid_idx][lat_col]
    out["median_lon"] = df.iloc[mid_idx][lon_col]
    out["end_lat"] = df.iloc[-1][lat_col]
    out["end_lon"] = df.iloc[-1][lon_col]

    # speed stats
    sp = pd.to_numeric(df[speed_col], errors="coerce")
    out["speed_mean"] = float(sp.mean()) if sp.notna().any() else np.nan
    out["speed_median"] = float(sp.median()) if sp.notna().any() else np.nan
    out["speed_std"] = float(sp.std(ddof=0)) if sp.notna().any() else np.nan
    out["speed_min"] = float(sp.min()) if sp.notna().any() else np.nan
    out["speed_max"] = float(sp.max()) if sp.notna().any() else np.nan

    # accel stats + hard events
    if accel_col is not None and accel_col in df.columns:
        ac = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=float)
        ac_valid = ac[np.isfinite(ac)]
        out["accel_mean"] = float(np.mean(ac_valid)) if ac_valid.size else np.nan
        out["accel_std"] = float(np.std(ac_valid)) if ac_valid.size else np.nan
        out["accel_abs_mean"] = float(np.mean(np.abs(ac_valid))) if ac_valid.size else np.nan
        out["accel_abs_sum"] = float(np.sum(np.abs(ac_valid))) if ac_valid.size else np.nan

        # hard events (needs time order)
        hard_accel_events, hard_brake_events = count_hard_events(
            accel=np.where(np.isfinite(ac), ac, 0.0),
            tsec=np.where(np.isfinite(tsec), tsec, np.arange(n, dtype=float)),
            threshold=HARD_A_THRESHOLD,
            min_consec_points=MIN_CONSEC_POINTS,
            consec_max_gap=CONSEC_MAX_GAP_SEC
        )
        out["hard_accel_event_count"] = hard_accel_events
        out["hard_brake_event_count"] = hard_brake_events
        out["hard_event_count"] = hard_accel_events + hard_brake_events
    else:
        out["accel_mean"] = np.nan
        out["accel_std"] = np.nan
        out["accel_abs_mean"] = np.nan
        out["accel_abs_sum"] = np.nan
        out["hard_accel_event_count"] = 0
        out["hard_brake_event_count"] = 0
        out["hard_event_count"] = 0

    # turning stats
    if turn_col is not None and turn_col in df.columns:
        td = pd.to_numeric(df[turn_col], errors="coerce").to_numpy(dtype=float)
        td_valid = td[np.isfinite(td)]
        out["turn_abs_mean"] = float(np.mean(np.abs(td_valid))) if td_valid.size else np.nan
        out["turn_std"] = float(np.std(td_valid)) if td_valid.size else np.nan
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
    """
    Extract date like 2022_9_19 from second_2022_9_19.csv.gz
    """
    m = re.search(r"second_(\d{4}_\d{1,2}_\d{1,2})", filename)
    return m.group(1) if m else "unknown_date"


def process_one_file(path: Path, out_dir: Path) -> None:
    print(f"\n[Processing] {path.name}")

    # Peek header to detect columns
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

    # If exists, overwrite
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

        # Ensure minimal sorting to make segment_id blocks contiguous.
        # Your data is already sorted by segment_id time series, but we enforce a stable sort.
        chunk = chunk.sort_values([seg_col, time_col], kind="mergesort")

        # Keep last segment_id as carry to avoid cross-chunk split
        last_seg = chunk[seg_col].iloc[-1]
        is_last = chunk[seg_col] == last_seg
        carry = chunk.loc[is_last].copy()
        chunk_main = chunk.loc[~is_last].copy()

        # Summarize each segment in this chunk_main
        summaries = []
        for seg_id, g in chunk_main.groupby(seg_col, sort=False):
            summaries.append(
                summarize_one_segment(
                    g, seg_col, time_col, lat_col, lon_col,
                    speed_col, accel_col, turn_col, county_col, func_class_col
                )
            )

        if summaries:
            out_df = pd.DataFrame(summaries)
            # add date
            out_df["date_tag"] = date_tag

            # write append
            out_df.to_csv(
                out_path,
                index=False,
                compression="gzip",
                mode="wt" if first_write else "at",
                header=first_write
            )
            first_write = False

        print(f"  chunk {i} done. wrote {len(summaries)} segments. carry rows={len(carry)}")

    # finalize carry
    if carry is not None and not carry.empty:
        summaries = []
        for seg_id, g in carry.groupby(seg_col, sort=False):
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

    # Find daily second-level files
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
