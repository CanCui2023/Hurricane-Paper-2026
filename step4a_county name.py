# -*- coding: utf-8 -*-
import pandas as pd

# ===== Paths =====
COUNTY_DAY_PATH = r"D:\Hurricane Paper\_county_day\county_day_hard_events.csv"
COUNTY_XLSX_PATH = r"D:\Hurricane Paper\_county_day\County_Extract_short.xlsx"
OUT_PATH = r"D:\Hurricane Paper\_county_day\county_day_5metrics_named.csv"

# ===== Read =====
county_day = pd.read_csv(COUNTY_DAY_PATH)

# Excel: if you have multiple sheets, set sheet_name explicitly, e.g. sheet_name=0 or "Sheet1"
county_map = pd.read_excel(COUNTY_XLSX_PATH, sheet_name=0)

print("county_day columns:", list(county_day.columns))
print("county_map columns:", list(county_map.columns))

# ===== Normalize key column name =====
# sometimes Excel has 'fl_countie' or 'FL_COUNTIES' - normalize to 'fl_counties'
def normalize_fl_counties(df):
    cols = {c.lower(): c for c in df.columns}
    for cand in ["fl_counties", "fl_countie", "flcounty", "county_code"]:
        if cand in cols:
            src = cols[cand]
            if src != "fl_counties":
                df = df.rename(columns={src: "fl_counties"})
            return df
    raise KeyError("Cannot find a fl_counties-like column in this file.")

county_day = normalize_fl_counties(county_day)
county_map = normalize_fl_counties(county_map)

# ===== Force key to integer (very important) =====
county_day["fl_counties"] = pd.to_numeric(county_day["fl_counties"], errors="coerce").astype("Int64")
county_map["fl_counties"] = pd.to_numeric(county_map["fl_counties"], errors="coerce").astype("Int64")

# drop rows with missing key in map
county_map = county_map[county_map["fl_counties"].notna()].copy()

# ===== Find county name column automatically =====
# You can hardcode if you know it, e.g. NAME / County / county_name
possible_name_cols = [
    "county_name", "county", "name", "namelsad", "county_nam", "cntyname", "countyname"
]
lower_cols = {c.lower(): c for c in county_map.columns}
name_col = None
for cand in possible_name_cols:
    if cand in lower_cols:
        name_col = lower_cols[cand]
        break

# If not found, print example and stop
if name_col is None:
    print("\n[ERROR] Cannot auto-detect county name column in County_Extract_short.xlsx.")
    print("Please tell me which column is the county name (from the list above).")
    raise SystemExit(1)

print(f"\n[INFO] Using county name column: {name_col}")

# ===== Keep only key + name, drop duplicates =====
county_map_clean = county_map[["fl_counties", name_col]].dropna().copy()
county_map_clean = county_map_clean.drop_duplicates(subset=["fl_counties"])
county_map_clean = county_map_clean.rename(columns={name_col: "county_name"})

# ===== Merge (left join) =====
merged = county_day.merge(county_map_clean, on="fl_counties", how="left")

# ===== Diagnostics =====
missing = merged["county_name"].isna().sum()
total = len(merged)
print(f"\n[CHECK] Missing county_name after merge: {missing} / {total}")

if missing > 0:
    missing_codes = (merged.loc[merged["county_name"].isna(), "fl_counties"]
                     .dropna().astype(int).unique().tolist())
    missing_codes_sorted = sorted(missing_codes)
    print("[CHECK] Missing fl_counties codes (first 30):", missing_codes_sorted[:30])

    # also show what codes exist in map
    map_codes = set(county_map_clean["fl_counties"].dropna().astype(int).tolist())
    day_codes = set(merged["fl_counties"].dropna().astype(int).tolist())
    print("[CHECK] Codes in county_day but not in map (first 30):", sorted(list(day_codes - map_codes))[:30])

# ===== Save =====
merged.to_csv(OUT_PATH, index=False)
print("\n[OK] Saved:", OUT_PATH)
