"""
Quarterly household inflation expectation moments from the BoE/Ipsos
Inflation Attitudes Survey microdata, at three levels of censoring.

    _cens   question q2 only. The open top bin is "5% or more", collapsed to a
            single midpoint. This is the published series.
    _dec    de-censored using the survey's own follow-ups (q2a from 2008Q3,
            q2a4 from 2022Q2, q2a2 from 2009Q1), so the open bin narrows to the
            finest split available in that wave.
    _h10    de-censored and harmonised. From 2022Q2 the survey's top bin
            narrows again to "15% or more", which would put a mechanical break
            in disagreement at the start of the 2022 episode. This column
            re-pools everyone above 10% into one open bin across the whole
            2008Q3-2025Q4 sample.

Run on q2 alone the procedure reproduces the Bank's published moments across
all 94 waves, which checks the weighting and binning rather than the
de-censoring itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

# Configuration -- EDIT THESE PATHS for your local machine / VS Code project

RAW_XLSX          = Path("individual-responses.xlsx")   # raw IAS microdata
RAW_SHEET         = "Dataset"

VALIDATION_XLSX   = Path("Constructed_quarterly_moments_households.xlsx")
VALIDATION_SHEET  = "Quarterly_Moments"

OUTPUT_DIR        = Path("./output")
OUTPUT_XLSX       = OUTPUT_DIR / "ias_moments_combined.xlsx"

# Columns to pull from the raw microdata. Positions are NOT assumed --
# extraction is done by HEADER NAME, not column index, so it is robust to
# the exact column ordering in the workbook.
WANTED_COLS = ["yyyyqq", "weight", "q2", "q2a", "q2a4", "q2a2",
               "q2a_agg", "q2a_agg1"]


# 1. Extraction (replaces the separate ias_extract.csv step)

def extract_raw(path: Path, sheet: str, wanted: list[str]) -> pd.DataFrame:
    """
    Stream the 'Dataset' sheet of the raw IAS workbook and pull only the
    columns needed, by header name. Avoids loading the full (large) sheet
    into memory via pandas.read_excel.
    """
    print(f"[extract] Opening {path} (sheet '{sheet}') ...")
    wb = load_workbook(path, read_only=True)
    ws = wb[sheet]
    it = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(it)]

    missing = [c for c in wanted if c not in header]
    if missing:
        raise ValueError(
            f"Columns not found in '{sheet}': {missing}. "
            f"Check the raw file's header names haven't changed."
        )
    idx = [header.index(c) for c in wanted]

    rows = []
    for r in it:
        if r[idx[0]] is None:  # yyyyqq missing -> skip
            continue
        rows.append([r[i] if i < len(r) else None for i in idx])

    df = pd.DataFrame(rows, columns=wanted)
    for c in wanted:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"[extract] {len(df):,} rows, {df['yyyyqq'].nunique()} waves")
    return df


# 2. Bin table (union of both source scripts, validated to agree on overlap)
# rank: (lo, hi, midpoint_specA, midpoint_specB, is_open)
# Spec A = conservative open-bin midpoint (edge + 1.0)
# Spec B = more extreme open-bin midpoint (edge + 3.0)
BINS = {
    0:  (-np.inf, 0.0,  -1.0,  -2.0,  True),   # "gone down" (unextended)
    1:  (-np.inf, -5.0, -6.0,  -8.0,  True),   # down 5% or more
    2:  (-5.0, -4.0, -4.5, -4.5, False),
    3:  (-4.0, -3.0, -3.5, -3.5, False),
    4:  (-3.0, -2.0, -2.5, -2.5, False),
    5:  (-2.0, -1.0, -1.5, -1.5, False),
    6:  (-1.0,  0.0, -0.5, -0.5, False),
    7:  ( 0.0,  0.0,  0.0,  0.0, False),       # point mass "not changed"
    8:  ( 0.0,  1.0,  0.5,  0.5, False),
    9:  ( 1.0,  2.0,  1.5,  1.5, False),
    10: ( 2.0,  3.0,  2.5,  2.5, False),
    11: ( 3.0,  4.0,  3.5,  3.5, False),
    12: ( 4.0,  5.0,  4.5,  4.5, False),
    13: ( 5.0, np.inf, 6.0,  8.0, True),       # ">=5%"  OPEN (unextended -- q2 top bin)
    14: ( 5.0,  6.0,  5.5,  5.5, False),
    15: ( 6.0,  7.0,  6.5,  6.5, False),
    16: ( 7.0,  8.0,  7.5,  7.5, False),
    17: ( 8.0,  9.0,  8.5,  8.5, False),
    18: ( 9.0, 10.0,  9.5,  9.5, False),
    19: (10.0, np.inf, 11.0, 13.0, True),      # ">=10%" OPEN
    20: (10.0, 11.0, 10.5, 10.5, False),
    21: (11.0, 12.0, 11.5, 11.5, False),
    22: (12.0, 13.0, 12.5, 12.5, False),
    23: (13.0, 14.0, 13.5, 13.5, False),
    24: (14.0, 15.0, 14.5, 14.5, False),
    25: (15.0, np.inf, 16.0, 18.0, True),      # ">=15%" OPEN
}

# Bank of England pre-combined variable codes -> internal rank
MAP_AGG1 = {1:6, 2:5, 3:4, 4:3, 5:2, 6:1, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12,
            13:14, 14:15, 15:16, 16:17, 17:18, 18:19, 19:np.nan,
            20:20, 21:21, 22:22, 23:23, 24:24, 25:25}
MAP_AGG  = {1:0, 2:7, 3:8, 4:9, 5:10, 6:11, 7:12, 8:14, 9:15, 10:16, 11:17,
            12:18, 13:19, 14:np.nan, 15:20, 16:21, 17:22, 18:23, 19:24, 20:25}
MAP_Q2   = {1:0, 2:7, 3:8, 4:9, 5:10, 6:11, 7:12, 8:13, 9:np.nan}

# Ranks that collapse into the open ">=10%" bin (19) under harmonisation
HARMONISE_TO_19 = {20, 21, 22, 23, 24, 25}


def assign_rank_cens(row) -> float:
    """q2 only -- the original, censored bin (top bin = '>=5%')."""
    v = row["q2"]
    return MAP_Q2.get(int(v), np.nan) if pd.notna(v) else np.nan


def assign_rank_dec(row) -> float:
    """Best-available bin using the finest follow-up extension present."""
    if pd.notna(row["q2a_agg1"]):
        return MAP_AGG1.get(int(row["q2a_agg1"]), np.nan)
    if pd.notna(row["q2a_agg"]):
        return MAP_AGG.get(int(row["q2a_agg"]), np.nan)
    return assign_rank_cens(row)


def assign_rank_h10(rank_dec: float) -> float:
    """Harmonise: collapse any 10%+ subdivision back to the open '>=10%' bin."""
    if pd.notna(rank_dec) and int(rank_dec) in HARMONISE_TO_19:
        return 19.0
    return rank_dec


# 3. Wave-level statistics: quantiles (assumption-free) + moments (midpoint-based)

def wave_stats(ranks: pd.Series, weights: pd.Series, spec: str = "A") -> dict:
    """
    For one wave and one rank series (cens / dec / h10), compute:
      - quantiles q10, q25, median, q75, q90 (via linear CDF interpolation;
        NaN + flag if the quantile falls inside an open bin)
      - IQR, Bowley skewness
      - mean, SD under the chosen open-bin midpoint spec
      - top-bin diagnostics: share and edge of the open bin in this wave
    """
    mid_i = 2 if spec == "A" else 3
    df = pd.DataFrame({"r": ranks, "w": weights}).dropna()
    empty = dict(q10=np.nan, q25=np.nan, median=np.nan, q75=np.nan, q90=np.nan,
                 iqr=np.nan, bowley=np.nan, mean=np.nan, sd=np.nan,
                 flag_median=None, n=0, top_share=np.nan, top_edge=np.nan)
    if df.empty:
        return empty

    g = df.groupby("r")["w"].sum()
    g = g / g.sum()
    g = g.sort_index()
    ranks_sorted = g.index.to_numpy()
    p = g.values
    cum = np.cumsum(p)

    def interp_quantile(q):
        k = int(np.searchsorted(cum, q))
        k = min(k, len(p) - 1)
        lo, hi, _, _, is_open = BINS[int(ranks_sorted[k])]
        below = cum[k - 1] if k > 0 else 0.0
        if lo == hi:
            return 0.0, "point_mass"
        if is_open:
            return np.nan, "OPEN_BIN"
        return lo + (q - below) / p[k] * (hi - lo), None

    q10, _      = interp_quantile(0.10)
    q25, f25    = interp_quantile(0.25)
    q50, f50    = interp_quantile(0.50)
    q75, f75    = interp_quantile(0.75)
    q90, _      = interp_quantile(0.90)

    iqr = q75 - q25 if (np.isfinite(q25) and np.isfinite(q75)) else np.nan
    bowley = ((q75 + q25 - 2 * q50) / iqr
              if (np.isfinite(iqr) and iqr > 0 and np.isfinite(q50)) else np.nan)

    mids = np.array([BINS[int(r)][mid_i] for r in ranks_sorted])
    mean = float((p * mids).sum())
    sd = float(np.sqrt((p * (mids - mean) ** 2).sum()))

    open_up = [r for r in ranks_sorted if BINS[int(r)][4] and BINS[int(r)][0] >= 0]
    top_share = float(g.loc[open_up].sum()) if open_up else 0.0
    top_edge = float(BINS[int(max(open_up))][0]) if open_up else np.nan

    return dict(q10=q10, q25=q25, median=q50, q75=q75, q90=q90,
                iqr=iqr, bowley=bowley, mean=mean, sd=sd,
                flag_median=f50, n=len(df), top_share=top_share, top_edge=top_edge)


# 4. Build the full quarterly panel

def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["weight"].notna() & df["q2"].notna()].copy()
    df["rank_cens"] = df.apply(assign_rank_cens, axis=1)
    df["rank_dec"]  = df.apply(assign_rank_dec, axis=1)
    df["rank_h10"]  = df["rank_dec"].apply(assign_rank_h10)

    rows = []
    for yq, sub in df.groupby("yyyyqq"):
        rec = {"yyyyqq": int(yq)}
        for tag, col in [("cens", "rank_cens"), ("dec", "rank_dec"), ("h10", "rank_h10")]:
            for spec in ["A", "B"]:
                s = wave_stats(sub[col], sub["weight"], spec)
                sfx = f"_{tag}_spec{spec}"
                rec[f"mean{sfx}"] = s["mean"]
                rec[f"sd{sfx}"]   = s["sd"]
                if spec == "A":  # quantiles don't depend on spec
                    rec[f"q10_{tag}"]    = s["q10"]
                    rec[f"q25_{tag}"]    = s["q25"]
                    rec[f"median_{tag}"] = s["median"]
                    rec[f"q75_{tag}"]    = s["q75"]
                    rec[f"q90_{tag}"]    = s["q90"]
                    rec[f"iqr_{tag}"]    = s["iqr"]
                    rec[f"bowley_{tag}"] = s["bowley"]
                    rec[f"flagmed_{tag}"]  = s["flag_median"]
                    rec[f"topshare_{tag}"] = s["top_share"]
                    rec[f"topedge_{tag}"]  = s["top_edge"]
                    rec[f"n_{tag}"]         = s["n"]
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("yyyyqq").reset_index(drop=True)
    out["period"] = (out["yyyyqq"] // 100).astype(str) + "Q" + (out["yyyyqq"] % 100).astype(str)
    out["decensor_available"] = (out["yyyyqq"] >= 200803).astype(int)

    front = ["yyyyqq", "period", "decensor_available"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


# 5. Validation against the existing censored file (if present)

def validate_against_existing(panel: pd.DataFrame) -> None:
    if not VALIDATION_XLSX.exists():
        print(f"[validate] {VALIDATION_XLSX} not found -- skipping. "
              f"Run this once against the original file before trusting output.")
        return
    old = pd.read_excel(VALIDATION_XLSX, sheet_name=VALIDATION_SHEET)
    m = panel.merge(
        old[["yyyyqq", "q50_median", "mean_specA", "std_specA", "mean_specB", "std_specB"]],
        on="yyyyqq",
    )
    print("[validate] _cens reconstruction vs existing Constructed_quarterly_moments file:")
    for a, b in [("median_cens", "q50_median"), ("mean_cens_specA", "mean_specA"),
                 ("sd_cens_specA", "std_specA"), ("mean_cens_specB", "mean_specB"),
                 ("sd_cens_specB", "std_specB")]:
        diff = (m[a] - m[b]).abs().max()
        status = "OK" if diff < 1e-6 else "*** MISMATCH ***"
        print(f"  {a:18s} vs {b:12s}  maxdiff = {diff:.10f}   {status}")


# 6. Excel output

README_TEXT = [
    ("PURPOSE",
     "Single-pipeline reconstruction of UK household 12m-ahead inflation "
     "expectation moments from raw IAS microdata, at three censoring levels."),
    ("SOURCE", "BoE/Ipsos Inflation Attitudes Survey individual responses, sheet 'Dataset'."),
    ("QUESTION",
     "Q2A: 'How much would you expect prices in the shops generally to change "
     "over the next twelve months?' -- database variable q2 (NOT q2a, a follow-up)."),
    ("", ""),
    ("_cens series",
     "q2 only. Top bin = '5% or more', collapsed to a single assumed midpoint "
     "(Spec A: 6.0 / Spec B: 8.0). This is the ORIGINAL / naive series -- "
     "replicates Constructed_quarterly_moments_households.xlsx exactly."),
    ("_dec series",
     "De-censored using the survey's own follow-up questions: q2a (from "
     "2008Q3, splits >=5% into 5-6,...,9-10,'>=10%'), q2a4 (from 2022Q2, "
     "splits >=10% into 10-11,...,14-15,'>=15%'), q2a2 (from 2009Q1, splits "
     "the 'gone down' bin). Uses the FINEST split available in each wave. "
     "Only available from 2008Q3 (decensor_available=1)."),
    ("_h10 series",
     "De-censored AND harmonised: any respondent in the 10%+ range is "
     "re-pooled into a single open '>=10%' bin for the WHOLE 2008Q3-2025Q4 "
     "sample. This avoids a mechanical break in disagreement at 2022Q2, when "
     "the survey's own top bin narrows to 15%. THIS IS THE RECOMMENDED / "
     "REPORTED series for D_t. Note: the open-bin midpoint used for the "
     "harmonised '>=10%' group is 11.0 (Spec A) / 13.0 (Spec B) -- NOT 10. "
     "topedge_h10=10.0 records the CENSORING POINT (edge), not the imputed value."),
    ("", ""),
    ("MEDIAN INVARIANCE",
     "The median is IDENTICAL across _cens, _dec and _h10 in every wave: the "
     ">=5% share never exceeds ~0.4845 (2022Q3), so the median always falls "
     "in a closed bin and cannot be affected by any open-bin assumption. "
     "g_t_gap_median is therefore provably censoring-invariant."),
    ("MEAN / SD", "Depend on the open-bin spec (A conservative, B more extreme) and "
     "on which censoring level is used. Use _h10 for D_t (disagreement); "
     "the _cens SD is not credible post-2008 (flat through the 2022 episode "
     "because the open bin absorbed all tail heterogeneity)."),
    ("VALIDATION",
     "On each run, the _cens series is checked against "
     "Constructed_quarterly_moments_households.xlsx if present in the same "
     "folder; should match to machine precision. See console output."),
    ("REPLACES",
     "build_moments.py (produced _cens only) and build_decensored_moments.py "
     "(produced _cens + _dec; required a separately-extracted ias_extract.csv "
     "that this script now generates internally)."),
]

QUARTERLY_COLS = (
    ["yyyyqq", "period", "decensor_available"]
    + [f"{stat}_{tag}" for tag in ["cens", "dec", "h10"]
       for stat in ["q10", "q25", "median", "q75", "q90", "iqr", "bowley"]]
    + [f"{stat}_{tag}_spec{spec}" for tag in ["cens", "dec", "h10"]
       for stat in ["mean", "sd"] for spec in ["A", "B"]]
    + [f"{stat}_{tag}" for tag in ["cens", "dec", "h10"]
       for stat in ["topshare", "topedge", "n"]]
    + [f"flagmed_{tag}" for tag in ["cens", "dec", "h10"]]
)


def write_excel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readme = pd.DataFrame(README_TEXT, columns=["Item", "Detail"])
    main = panel[[c for c in QUARTERLY_COLS if c in panel.columns]].copy()

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        main.to_excel(xw, sheet_name="Quarterly_Moments", index=False)
        from openpyxl.utils import get_column_letter
        for sheet_name, frame in [("README", readme), ("Quarterly_Moments", main)]:
            ws = xw.book[sheet_name]
            for j, col in enumerate(frame.columns, start=1):
                q95 = frame[col].astype(str).str.len().quantile(0.95)
                width = max(12, min(60, max(len(str(col)), int(q95) if pd.notna(q95) else 12) + 2))
                ws.column_dimensions[get_column_letter(j)].width = width
    print(f"[write] {path}  (shape {main.shape})")


# Main

def main() -> int:
    raw = extract_raw(RAW_XLSX, RAW_SHEET, WANTED_COLS)
    panel = build_panel(raw)
    validate_against_existing(panel)
    write_excel(panel, OUTPUT_XLSX)

    print("\nLast 6 waves, cens vs dec vs h10 (Spec A mean/SD, median):")
    cols = ["period", "median_cens", "mean_cens_specA", "sd_cens_specA",
            "mean_dec_specA", "sd_dec_specA", "mean_h10_specA", "sd_h10_specA"]
    print(panel[cols].tail(6).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
