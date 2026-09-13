"""
Add the sterling effective exchange rate to the VAR-ready panel.

    python build_data_v3.py

The six-variable system has no domestic block, so every UK-specific inflation
disturbance falls into the unlabelled shocks. The exchange rate is the most
important UK inflation driver that is neither global commodity supply nor
global demand, and unlike Bank Rate it is not pinned at the lower bound for
most of the sample. It enters the seventh-variable robustness specifications
only; the baseline is unchanged at six variables.

    eri_t_yoy = 100 * (ERI_t / ERI_{t-4} - 1)

Year-on-year, matching the commodity and inflation series. A depreciation is
negative, which is the sign the sterling shock is restricted on.

The ONS download interleaves annual, quarterly and monthly rows in one column;
only rows matching "<YYYY> Q<n>" are read. Quarters from 2002Q1 are loaded
because the transformation needs four lags before the sample starts.

Inputs:  df_var_ready_decensored_EPcpi.xlsx, ONS series BK67 (dataset MRET)
Output:  df_var_ready_v3.xlsx
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_PANEL = Path("df_var_ready_decensored_EPcpi.xlsx")
SRC_ERI = Path("sterling_effective_exchange_rate_quarterly.csv")
OUT = Path("df_var_ready_v3.xlsx")

QPAT = re.compile(r"^\s*(\d{4})\s*Q([1-4])\s*$")


def read_ons_quarterly(path: Path, value_name: str) -> pd.DataFrame:
    """
    Parse an ONS time-series download. The file has a metadata header, then
    annual rows, then quarterly rows, then monthly rows, all in one column.
    Only quarterly rows are returned.
    """
    raw = pd.read_csv(path, header=None, names=["period", "value"],
                      dtype=str, skip_blank_lines=True)
    rows = []
    for period, value in zip(raw["period"], raw["value"]):
        if not isinstance(period, str):
            continue
        m = QPAT.match(period)
        if not m:
            continue
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        v = str(value).strip()
        if v == "":
            continue
        rows.append({"quarter": f"{m.group(1)}Q{m.group(2)}",
                     value_name: float(v)})
    out = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no quarterly rows parsed from {path}")
    return out


def main() -> int:
    if not SRC_PANEL.exists():
        print(f"ERROR: {SRC_PANEL} not found in the working directory.")
        return 1
    if not SRC_ERI.exists():
        print(f"ERROR: {SRC_ERI} not found in the working directory.")
        return 1

    df = pd.read_excel(SRC_PANEL)
    df["quarter"] = df["quarter"].astype(str)
    df = df.sort_values("quarter").reset_index(drop=True)
    print(f"[panel] {len(df)} quarters, {df.quarter.iloc[0]} to "
          f"{df.quarter.iloc[-1]}, {df.shape[1]} columns")

    eri = read_ons_quarterly(SRC_ERI, "eri_t_level")
    print(f"[eri]   {len(eri)} quarterly observations parsed, "
          f"{eri.quarter.iloc[0]} to {eri.quarter.iloc[-1]}")

    # year-on-year change computed on the FULL ERI history, before the merge,
    # so the four lags needed for 2003Q1 come from 2002Q1 rather than from
    # the estimation sample
    eri["eri_t_yoy"] = 100.0 * (eri["eri_t_level"] /
                                eri["eri_t_level"].shift(4) - 1.0)

    merged = df.merge(eri[["quarter", "eri_t_level", "eri_t_yoy"]],
                      on="quarter", how="left")

    miss_lvl = merged["eri_t_level"].isna().sum()
    miss_yoy = merged["eri_t_yoy"].isna().sum()
    print(f"[merge] missing eri_t_level {miss_lvl}, missing eri_t_yoy {miss_yoy}")
    if miss_yoy:
        bad = merged.loc[merged.eri_t_yoy.isna(), "quarter"].tolist()
        print(f"        quarters without eri_t_yoy: {bad}")
        print("        NOTE: any NaN inside the estimation window will drop "
              "that row from the VAR. Check before running R17/R18.")

    s = merged["eri_t_yoy"].dropna()
    print(f"[eri]   yoy mean {s.mean():+.3f}  sd {s.std():.3f}  "
          f"min {s.min():+.3f} ({merged.loc[s.idxmin(), 'quarter']})  "
          f"max {s.max():+.3f} ({merged.loc[s.idxmax(), 'quarter']})")

    # sanity: the two episodes an examiner will ask about
    for q in ("2008Q4", "2009Q1", "2016Q4", "2022Q3"):
        row = merged.loc[merged.quarter == q]
        if not row.empty:
            print(f"        {q}: level {row.eri_t_level.iat[0]:.2f}, "
                  f"yoy {row.eri_t_yoy.iat[0]:+.2f}%")

    # correlations that matter for the identification argument
    for c in ("pi_t_rpi", "pi_t_cpi", "q_t_commod_yoy", "a_t_kilian",
              "s_t_asinh100", "g_meandec_cpi"):
        if c in merged.columns:
            r = merged[["eri_t_yoy", c]].dropna().corr().iat[0, 1]
            print(f"[corr]  eri_t_yoy vs {c:18s} {r:+.4f}")

    merged.to_excel(OUT, index=False)
    print(f"\nwrote {OUT}  ({merged.shape[0]} rows, {merged.shape[1]} columns)")
    print("Point config_v3.DATA at this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
