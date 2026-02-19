# step1_second_level.py
# Batch compute second-level acceleration and turning change for daily .gz CSVs
# Output: one "second_YYYY_M_D.csv.gz" per input day in the same folder.

import os
import glob
import pandas as pd
import numpy as np

# ========= USER CONFIG =========
INPUT_DIR = r"D:\Hurricane Paper"     # <- 改成你的文件夹
FILE_GLOB = "merged_*.gz"            # daily files
CHUNK_SIZE = 1_000_000               # adjust if needed
# ===============================

# Required columns (based on your sample)
COL_TRIP = "trip_id"
COL_SEG  = "segment_id"
COL_T    = "gps_timestamp"
COL_SPD  = "gps_speed"
COL_HDG  = "gps_heading"

OUT_PREFIX = "second_"

def circular_diff_deg(curr, prev):
    """
    Circular difference for headings in degrees.
    Result in [-180, 180).
    """
    # works with numpy arrays / pandas Series
    return ((curr - prev + 180) % 360) - 180

def process_one_file(in_path: str) -> str:
    in_name = os.path.basename(in_path)

    # build output name: merged_2022_9_19.gz -> second_2022_9_19.csv.gz
    # if your naming differs, this still keeps the date-ish part after 'merged_'
    out_name = in_name.replace("merged_", OUT_PREFIX)
    out_name = out_name.replace(".gz", ".csv.gz")
    out_path = os.path.join(os.path.dirname(in_path), out_name)

    print(f"\n[START] {in_name}")
    print(f"  -> OUT: {out_name}")

    # We'll carry the last row info per (trip_id, segment_id) across chunks
    # This is necessary if a group boundary happens to split across chunks.
    last_state = {}  # key -> (last_time, last_speed, last_heading)

    first_write = True

    reader = pd.read_csv(
        in_path,
        compression="gzip",
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk_idx, df in enumerate(reader, start=1):
        # Ensure required columns exist
        missing = [c for c in [COL_TRIP, COL_SEG, COL_T, COL_SPD, COL_HDG] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {in_name}: {missing}")

        # Parse timestamps (UTC ISO like 2022-08-10T18:31:06Z)
        # Using utc=True ensures consistent dt computation
        t = pd.to_datetime(df[COL_T], utc=True, errors="coerce")

        # Pre-allocate new columns
        accel = np.full(len(df), np.nan, dtype="float64")
        turn  = np.full(len(df), np.nan, dtype="float64")

        # We compute diffs within each (trip_id, segment_id) group in this chunk
        # But we also need to stitch the first row of each group with last_state from previous chunk.
        # Grouping (trip_id, segment_id) is safest and matches your intended boundary.
        keys = list(zip(df[COL_TRIP].astype(str), df[COL_SEG].astype(str)))

        # We will process in original row order (your data is already sorted by segment time series)
        # Keep a local "prev" cache for this chunk (more efficient than groupby for huge data).
        local_prev = {}

        spd = pd.to_numeric(df[COL_SPD], errors="coerce").to_numpy()
        hdg = pd.to_numeric(df[COL_HDG], errors="coerce").to_numpy()

        for i, k in enumerate(keys):
            ti = t.iat[i]

            # choose previous record for this key:
            #   1) if we've seen it earlier in this chunk -> local_prev
            #   2) else if it exists from previous chunk -> last_state
            #   3) else -> no previous, keep NaN
            if k in local_prev:
                prev_t, prev_spd, prev_hdg = local_prev[k]
            elif k in last_state:
                prev_t, prev_spd, prev_hdg = last_state[k]
            else:
                prev_t, prev_spd, prev_hdg = (pd.NaT, np.nan, np.nan)

            # Update local_prev for next rows
            local_prev[k] = (ti, spd[i], hdg[i])

            # If no valid previous time, skip
            if pd.isna(prev_t) or pd.isna(ti):
                continue

            dt = (ti - prev_t).total_seconds()
            if dt is None or dt <= 0:
                continue

            # acceleration: mph per second
            if not (np.isnan(spd[i]) or np.isnan(prev_spd)):
                accel[i] = (spd[i] - prev_spd) / dt

            # turning change: circular delta heading in degrees (not rate)
            if not (np.isnan(hdg[i]) or np.isnan(prev_hdg)):
                turn[i] = circular_diff_deg(hdg[i], prev_hdg)

        # attach new columns (keep all original columns)
        df["accel_mphps"] = accel
        df["turning_change_deg"] = turn

        # Write output (append by chunks)
        df.to_csv(
            out_path,
            index=False,
            compression="gzip",
            mode="wt" if first_write else "at",
            header=first_write
        )
        first_write = False

        # Update last_state with the last record per key seen in this chunk
        # local_prev holds last seen values for each key in this chunk
        last_state = local_prev

        print(f"  chunk {chunk_idx}: rows={len(df):,} written")

    print(f"[DONE] {in_name} -> {out_name}")
    return out_path

def main():
    pattern = os.path.join(INPUT_DIR, FILE_GLOB)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    print(f"Found {len(files)} files.")
    for f in files:
        process_one_file(f)

if __name__ == "__main__":
    main()
