# -*- coding: utf-8 -*-
"""
Step 4B (FINAL) - Plot county-day results (Hard events + Phone use)

Figures produced for EACH metric (Hard + Phone):
  Fig1: Daily trend (Evac vs All) with 95% CI + hurricane day markers
  Fig2: Evac counties small multiples (county name as panel title)
  Fig3: Heatmap (Evac counties x day) with county names on y-axis
  Fig4: Evac counties ranking bar chart (During - Pre)

During hurricane dates: 2022-09-28 and 2022-09-29
"""

from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1) INPUT / OUTPUT
# =========================
INPUT_CSV = r"D:\Hurricane Paper\_county_day\county_day_5metrics_named.csv"  # <-- change if needed
OUT_DIR = Path(r"D:\Hurricane Paper\_figures_step4b")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2) Evac counties (use COUNTY CODE)
# =========================
EVAC_COUNTY_CODES = {
    67,  # Charlotte
    12,  # Citrus
    50,  # Clay
    44,  # Collier
    1,   # Hernando
    66,  # Hillsborough
    32,  # Lee
    38,  # Levy
    17,  # Manatee
    22,  # Pasco
    20,  # Pinellas
    46,  # Sarasota
    14,  # St. Johns
}


# =========================
# 3) Date settings
# =========================
DURING_DATES = {pd.Timestamp("2022-09-28"), pd.Timestamp("2022-09-29")}
STORM_START = pd.Timestamp("2022-09-28")


# =========================
# 4) Helpers
# =========================
def parse_date_any(x) -> pd.Timestamp:
    s = str(x)
    # support "2022_9_28"
    if "_" in s and s.count("_") == 2:
        y, m, d = s.split("_")
        try:
            return pd.Timestamp(int(y), int(m), int(d))
        except Exception:
            return pd.NaT
    # support "9/28/2022" or "2022-09-28"
    return pd.to_datetime(s, errors="coerce")


def pick_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def auto_find_metric(columns, preferred_candidates, keyword_fallback=None):
    """
    1) Use first existing in preferred_candidates.
    2) If not found and keyword_fallback is provided, search columns by keywords.
    """
    m = pick_first_existing(columns, preferred_candidates)
    if m is not None:
        return m

    if keyword_fallback:
        # keyword_fallback example: ["phone", "rate"] or ["phone", "count"]
        keys = [k.lower() for k in keyword_fallback]
        for c in columns:
            lc = c.lower()
            if all(k in lc for k in keys):
                return c
    return None


def daily_mean_ci(dfin: pd.DataFrame, metric: str, label: str) -> pd.DataFrame:
    g = (dfin.groupby("date_dt")[metric]
         .agg(["mean", "count", "std"])
         .reset_index())
    g["se"] = g["std"] / np.sqrt(g["count"].replace(0, np.nan))
    g["ci95"] = 1.96 * g["se"]
    g["group"] = label
    return g


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)


def make_county_label(dfin: pd.DataFrame, name_col: str, code_col: str) -> pd.Series:
    """
    Prefer readable county names; append code only if needed for uniqueness.
    """
    lab = dfin[name_col].astype(str).str.strip()
    if lab.nunique() < dfin[code_col].nunique():
        lab = lab + "_" + dfin[code_col].astype(str)
    return lab


def plot_metric_set(df: pd.DataFrame, metric: str, metric_title: str, code_col: str, name_col: str):
    """
    Create 4 figures for one metric.
    """
    # ---- Build evac flag + period ----
    d = df.copy()
    d["is_evac"] = d[code_col].isin(EVAC_COUNTY_CODES)
    d["period3"] = np.where(d["date_dt"].isin(DURING_DATES), "During",
                     np.where(d["date_dt"] < STORM_START, "Pre", "After"))
    d["period3"] = pd.Categorical(d["period3"], categories=["Pre", "During", "After"], ordered=True)

    # =========================
    # FIG 1: Trend (Evac vs All)
    # =========================
    stats_all = daily_mean_ci(d, metric, "All counties")
    stats_evac = daily_mean_ci(d[d["is_evac"]], metric, "Evacuation counties")
    stats = pd.concat([stats_all, stats_evac], ignore_index=True)

    plt.figure(figsize=(10, 4))
    for grp, sub in stats.groupby("group"):
        sub = sub.sort_values("date_dt")
        plt.plot(sub["date_dt"], sub["mean"], label=grp)
        plt.fill_between(sub["date_dt"],
                         sub["mean"] - sub["ci95"],
                         sub["mean"] + sub["ci95"],
                         alpha=0.15)

    for dmark in sorted(DURING_DATES):
        plt.axvline(dmark, linestyle="--", linewidth=1)

    plt.xlabel("Date")
    plt.ylabel(metric_title)
    plt.title(f"Daily trend: {metric_title} (Evacuation vs All counties)\nDuring: 2022-09-28 & 2022-09-29")
    plt.legend()
    plt.tight_layout()
    out1 = OUT_DIR / f"Fig1_trend_evac_vs_all__{sanitize_filename(metric)}.png"
    plt.savefig(out1, dpi=200)
    plt.show()

    # =========================
    # FIG 2: Evac small multiples
    # =========================
    evac = d[d["is_evac"]].copy()
    if evac.empty:
        print(f"[WARN] No evac rows found for metric={metric}. Check county codes/merge.")
        return

    evac["county_label"] = make_county_label(evac, name_col, code_col)

    county_list = (evac[["county_label"]]
                   .drop_duplicates()
                   .sort_values("county_label")["county_label"]
                   .tolist())

    n = len(county_list)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(12, 3*nrows))
    for i, lab in enumerate(county_list, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        sub = evac[evac["county_label"] == lab].sort_values("date_dt")
        ax.plot(sub["date_dt"], sub[metric])
        for dmark in sorted(DURING_DATES):
            ax.axvline(dmark, linestyle="--", linewidth=1)
        ax.set_title(lab, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel(metric_title)

    fig.suptitle(f"Evacuation counties: {metric_title} over time (each panel = one county)", y=1.02)
    fig.tight_layout()
    out2 = OUT_DIR / f"Fig2_evac_small_multiples__{sanitize_filename(metric)}.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.show()

    # =========================
    # FIG 3: Heatmap (Evac x day)
    # =========================
    heat = evac.pivot_table(index="county_label", columns="date_dt", values=metric, aggfunc="mean")
    heat = heat.sort_index()

    plt.figure(figsize=(12, max(4, 0.35 * heat.shape[0])))
    plt.imshow(heat.values, aspect="auto")  # no explicit colors set
    plt.yticks(np.arange(heat.shape[0]), heat.index, fontsize=9)
    plt.xticks(np.arange(heat.shape[1]), [d.strftime("%m-%d") for d in heat.columns], rotation=45)
    plt.title(f"Heatmap: Evacuation counties × day ({metric_title})")
    plt.xlabel("Date")
    plt.ylabel("County")
    plt.tight_layout()
    out3 = OUT_DIR / f"Fig3_heatmap_evac__{sanitize_filename(metric)}.png"
    plt.savefig(out3, dpi=200)
    plt.show()

    # =========================
    # FIG 4: Ranking (During - Pre), evac only
    # =========================
    pre_mean = (evac[evac["period3"] == "Pre"]
                .groupby("county_label")[metric].mean())
    during_mean = (evac[evac["period3"] == "During"]
                   .groupby("county_label")[metric].mean())

    delta = (during_mean - pre_mean).dropna().sort_values()

    if delta.empty:
        print(f"[WARN] Delta (During-Pre) is empty for metric={metric}. "
              f"Maybe missing Pre or During rows.")
    else:
        plt.figure(figsize=(10, max(4, 0.35 * len(delta))))
        plt.barh(delta.index, delta.values)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel(f"Δ {metric_title} (During − Pre)")
        plt.ylabel("County (Evacuation)")
        plt.title(f"Evacuation counties ranking: During (9/28–9/29) minus Pre\nMetric: {metric_title}")
        plt.tight_layout()
        out4 = OUT_DIR / f"Fig4_evac_ranking_delta_during_minus_pre__{sanitize_filename(metric)}.png"
        plt.savefig(out4, dpi=200)
        plt.show()

    print(f"[Saved]\n  {out1}\n  {out2}\n  {out3}\n  {out4 if not delta.empty else '(Fig4 skipped: empty delta)'}\n")


# =========================
# 5) Main
# =========================
df = pd.read_csv(INPUT_CSV)

# detect columns
code_col = auto_find_metric(df.columns, ["fl_counties", "county_code", "county_fips", "fips", "county_id"])
if code_col is None:
    raise SystemExit(f"Cannot find county code column. Columns: {df.columns.tolist()}")

name_col = auto_find_metric(df.columns, ["county_name", "county-name", "county_named", "county", "fl_counties_name"])
if name_col is None:
    # fallback: any column containing 'county' but not code_col
    for c in df.columns:
        if ("county" in c.lower()) and (c != code_col):
            name_col = c
            break
if name_col is None:
    raise SystemExit(f"Cannot find county name column. Columns: {df.columns.tolist()}")

date_col = auto_find_metric(df.columns, ["date_dt", "date", "date_tag", "day"], keyword_fallback=["date"])
if date_col is None:
    raise SystemExit(f"Cannot find date column. Columns: {df.columns.tolist()}")

# parse/standardize date
df["date_dt"] = df[date_col].apply(parse_date_any)
df = df.dropna(subset=["date_dt"]).copy()
df["date_dt"] = pd.to_datetime(df["date_dt"]).dt.normalize()

# standardize code
df[code_col] = pd.to_numeric(df[code_col], errors="coerce")
df = df.dropna(subset=[code_col]).copy()
df[code_col] = df[code_col].astype(int)

# -------------------------
# Choose metrics (Hard + Phone)
# -------------------------
# HARD metric: prefer rate per hour, else count
hard_metric = auto_find_metric(
    df.columns,
    preferred_candidates=["hard_event_rate_per_hour", "hard_event_rate", "hard_event_count"],
    keyword_fallback=["hard", "rate"]
)
if hard_metric is None:
    hard_metric = auto_find_metric(df.columns, ["hard_event_count"], keyword_fallback=["hard", "count"])

# PHONE metric: prefer rate per hour, else count-like
phone_metric = auto_find_metric(
    df.columns,
    preferred_candidates=["phone_use_rate_per_hour", "phone_rate_per_hour", "phone_use_rate", "phone_event_rate_per_hour"],
    keyword_fallback=["phone", "rate"]
)
if phone_metric is None:
    # common count names
    phone_metric = auto_find_metric(
        df.columns,
        preferred_candidates=["phone_use_count", "phone_event_count", "phone_time_count", "phone_use_events", "phone_usage_count"],
        keyword_fallback=["phone", "count"]
    )

print("[Info] detected columns:")
print("  code_col :", code_col)
print("  name_col :", name_col)
print("  date_col :", date_col)
print("  hard_metric :", hard_metric)
print("  phone_metric:", phone_metric)
print("  rows:", len(df), "unique counties:", df[name_col].nunique(), "unique days:", df["date_dt"].nunique())

# Validate metrics exist
metric_jobs = []
if hard_metric and hard_metric in df.columns:
    metric_jobs.append(("HARD", hard_metric))
else:
    print("[WARN] Hard metric not found. Check your CSV column names.")

if phone_metric and phone_metric in df.columns:
    metric_jobs.append(("PHONE", phone_metric))
else:
    print("[WARN] Phone metric not found. Check your CSV column names (phone...).")

if not metric_jobs:
    raise SystemExit("No metrics found to plot. Please check the CSV column names.")

# numeric cast + plot
for tag, mcol in metric_jobs:
    df[mcol] = pd.to_numeric(df[mcol], errors="coerce")
    dplot = df.dropna(subset=[mcol]).copy()

    # Friendly y-axis title
    if "rate" in mcol.lower():
        mtitle = f"{tag}: rate per hour ({mcol})"
    else:
        mtitle = f"{tag}: count ({mcol})"

    print(f"\n[Plotting] {tag} using metric column: {mcol}")
    plot_metric_set(dplot, metric=mcol, metric_title=mtitle, code_col=code_col, name_col=name_col)

print(f"\n[Done] Figures saved to: {OUT_DIR}")
