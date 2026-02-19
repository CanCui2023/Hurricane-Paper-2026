# -*- coding: utf-8 -*-
"""
Step 3 - County-Day aggregation (from segment-level outputs)

Goal (UPDATED):
- We only keep COUNT-based metrics for acceleration/deceleration events:
    hard_accel_event_count
    hard_brake_event_count
    hard_event_count
- Optionally compute per-hour rates using total_duration_sec.

Input : segment_YYYY_M_D.csv.gz in _segment_level folder
Output: county_day_hard_events.csv (or per-day files)

Everything else (directory scanning, reading, merging county name later) stays simple.
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np

# ===== USER SETTINGS =====
INPUT_DIR = r"D:\Hurricane Paper"
SEG_SUBDIR = "_segment_level"
OUT_SUBDIR = "_county_day"

# required columns from segment-level
COUNTY_COL = "fl_counties"
DUR_COL = "duration_sec"

HARD_ACCEL_COL = "hard_accel_event_count"
HARD_BRAKE_COL = "hard_brake_event_count"
HARD_EVENT_COL = "hard_event_count"

# ---------- helpers ----------
def extract_date_tag(filename: str) -> str:
    m = re.search(r"segment_(\d{4}_\d{1,2}_\d{1,2})", filename)
    return m.group(1) if m else "unknown_date"


def safe_get(df: pd.DataFrame, col: str, default=0):
    return df[col] if col in df.columns else default


def main():
    base = Path(INPUT_DIR)
    seg_dir = base / SEG_SUBDIR
    out_dir = base / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in seg_dir.iterdir()
                    if p.is_file() and re.match(r"segment_\d{4}_\d{1,2}_\d{1,2}\.csv(\.gz)?$", p.name, re.I)])

    if not files:
        raise SystemExit(f"[Exit] No segment_YYYY_M_D.csv(.gz) found in: {seg_dir}")

    all_days = []

    for fp in files:
        date_tag = extract_date_tag(fp.name)
        df = pd.read_csv(fp, compression="infer", low_memory=False)

        # --- basic column checks ---
        missing = [c for c in [COUNTY_COL, DUR_COL] if c not in df.columns]
        if missing:
            raise SystemExit(f"[Exit] Missing required columns {missing} in {fp.name}")

        # if hard_event_count not present, reconstruct from accel+brake
        if HARD_EVENT_COL not in df.columns:
            if HARD_ACCEL_COL in df.columns and HARD_BRAKE_COL in df.columns:
                df[HARD_EVENT_COL] = df[HARD_ACCEL_COL].fillna(0) + df[HARD_BRAKE_COL].fillna(0)
            else:
                # if none exists, create zeros to avoid crash
                df[HARD_ACCEL_COL] = 0
                df[HARD_BRAKE_COL] = 0
                df[HARD_EVENT_COL] = 0

        # ensure numeric
        df[DUR_COL] = pd.to_numeric(df[DUR_COL], errors="coerce").fillna(0)
        df[HARD_ACCEL_COL] = pd.to_numeric(df.get(HARD_ACCEL_COL, 0), errors="coerce").fillna(0)
        df[HARD_BRAKE_COL] = pd.to_numeric(df.get(HARD_BRAKE_COL, 0), errors="coerce").fillna(0)
        df[HARD_EVENT_COL] = pd.to_numeric(df.get(HARD_EVENT_COL, 0), errors="coerce").fillna(0)

        # group by county
        g = df.groupby(COUNTY_COL, as_index=False).agg(
            total_duration_sec=(DUR_COL, "sum"),
            hard_accel_event_count=(HARD_ACCEL_COL, "sum"),
            hard_brake_event_count=(HARD_BRAKE_COL, "sum"),
            hard_event_count=(HARD_EVENT_COL, "sum"),
            n_segments=("segment_id", "nunique") if "segment_id" in df.columns else (COUNTY_COL, "size"),
        )

        g["date"] = date_tag

        # per-hour rates (optional but useful)
        denom = g["total_duration_sec"].replace(0, np.nan)
        g["hard_event_rate_per_hour"] = g["hard_event_count"] / denom * 3600
        g["hard_accel_rate_per_hour"] = g["hard_accel_event_count"] / denom * 3600
        g["hard_brake_rate_per_hour"] = g["hard_brake_event_count"] / denom * 3600

        all_days.append(g)

        print(f"[OK] {fp.name} -> county rows: {len(g)}")

    out = pd.concat(all_days, ignore_index=True)

    out_fp = out_dir / "county_day_hard_events.csv"
    out.to_csv(out_fp, index=False, encoding="utf-8-sig")
    print(f"\n[Done] Output => {out_fp}")


if __name__ == "__main__":
    main()
