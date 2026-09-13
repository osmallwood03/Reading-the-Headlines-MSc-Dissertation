"""
Merge the professional series with the IAS household moments and build the
object the SVAR explains, gap_t = E^H_t - E^P_t.

    python build_gap.py --panel <panel.csv> --workbook <hmt.xlsx>
                        --ias <moments.xlsx> [--salience <index.csv>]

Outputs gap_series.csv (RPI baseline, 92 quarters) and gap_series_cpi.csv (CPI
robustness, 88 quarters), and prints the 2008 and 2022 windows, sample moments,
and the correlation with the signed salience index at lags 0 and 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import load_panel
from validate_aggregation import apply_rule, load_workbook_average
from build_professional import (RULE, SAMPLE, assert_sample, build_series)

# The IAS moments file: 94 waves 2001Q1..2025Q4 (annual, Q1 only, in
# 2001-2002; quarterly from 2003Q1 -- which is what binds the sample).
IAS_SHEET = "Quarterly_Moments"
IAS_COLUMNS = {"q50_median": "E_H",   # household 12m-ahead median
               "std_specA": "D_H"}    # household disagreement

# First quarter whose E_P comes from the fitted route (splice boundary).
FITTED_FROM = pd.Period("2023Q2", freq="Q")

OUTPUT_COLUMNS = ["period", "E_H", "E_P", "gap", "D_H", "D_P", "n_P",
                  "source", "confidence", "pub_used"]


# IAS moments

def load_ias(xlsx_path: str | Path) -> pd.DataFrame:
    """
    Read the Quarterly_Moments sheet into (period, E_H, D_H).

    The period column is located by name (period/quarter/wave, case
    insensitive) and parsed to quarterly periods; anything unexpected fails
    loudly rather than being coerced.
    """
    df = pd.read_excel(xlsx_path, sheet_name=IAS_SHEET)

    period_col = next((c for c in df.columns
                       if str(c).strip().lower() in ("period", "quarter",
                                                     "wave")), None)
    if period_col is None:
        raise ValueError(
            f"no period column found in {IAS_SHEET}; columns are "
            f"{list(df.columns)}")
    missing = [c for c in IAS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{IAS_SHEET} lacks expected columns {missing}; columns are "
            f"{list(df.columns)}")

    out = df[[period_col, *IAS_COLUMNS]].rename(
        columns={period_col: "period", **IAS_COLUMNS}).copy()
    try:
        out["period"] = pd.PeriodIndex(
            [pd.Period(str(p).strip(), freq="Q") for p in out["period"]],
            freq="Q")
    except Exception as e:
        raise ValueError(
            f"cannot parse '{period_col}' as quarters, e.g. "
            f"{out['period'].head(3).tolist()}") from e
    return out


# Merge

def build_gap(prof: pd.DataFrame, ias: pd.DataFrame,
              measure: str) -> pd.DataFrame:
    """Merge E_P with the household moments; every quarter must match."""
    g = prof.merge(ias, on="period", how="left", validate="one_to_one")

    unmatched = g.loc[g["E_H"].isna(), "period"].astype(str).tolist()
    assert not unmatched, (
        f"{measure}: no IAS wave for {unmatched} -- the moments file is "
        "expected to cover every quarter from 2003Q1. No silent fill.")
    assert g["D_H"].notna().all(), (
        f"{measure}: std_specA missing at "
        f"{g.loc[g['D_H'].isna(), 'period'].astype(str).tolist()}")

    g["gap"] = g["E_H"] - g["E_P"]
    return g[OUTPUT_COLUMNS + ["D_P_status"]]


# Inspection

def _h(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def _window(g: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    per = pd.PeriodIndex(g["period"], freq="Q")
    return g[(per >= pd.Period(first, freq="Q"))
             & (per <= pd.Period(last, freq="Q"))]


def inspect(g: pd.DataFrame, measure: str, sal: pd.Series | None) -> None:
    _h(f"{measure}  --  gap_t = E_H - E_P")

    print("\n  2008Q2-2008Q4 (oil/GFC episode) and 2022Q1-2022Q4 (the peak):")
    print(f"    {'period':<8}{'E_H':>7}{'E_P':>8}{'gap':>8}   "
          f"{'source':<10}{'n_P':>4}")
    for _, r in pd.concat([_window(g, "2008Q2", "2008Q4"),
                           _window(g, "2022Q1", "2022Q4")]).iterrows():
        n_p = f"{int(r['n_P'])}" if pd.notna(r["n_P"]) else "--"
        print(f"    {str(r['period']):<8}{r['E_H']:>7.2f}{r['E_P']:>8.2f}"
              f"{r['gap']:>8.2f}   {r['source']:<10}{n_p:>4}")

    gap = g.set_index("period")["gap"]
    print(f"\n  full sample {g['period'].iloc[0]}..{g['period'].iloc[-1]} "
          f"({len(g)} quarters):")
    print(f"    min  gap : {gap.min():>7.3f}   at {gap.idxmin()}")
    print(f"    max  gap : {gap.max():>7.3f}   at {gap.idxmax()}")
    print(f"    mean gap : {gap.mean():>7.3f}")
    print(f"    max |gap|: {gap.abs().max():>7.3f}   at {gap.abs().idxmax()}")

    if sal is not None:
        both = pd.DataFrame({"gap": gap, "s_t": sal}).dropna()
        lag = pd.DataFrame({"gap": gap, "s_lag": sal.shift(1)}).dropna()
        print(f"\n  correlation with the signed salience index s_t "
              f"(n = {len(both)}):")
        print(f"    corr(gap_t, s_t)      : "
              f"{np.corrcoef(both['gap'], both['s_t'])[0, 1]:+.3f}")
        print(f"    corr(gap_t, s_t-1)    : "
              f"{np.corrcoef(lag['gap'], lag['s_lag'])[0, 1]:+.3f}   "
              "(salience lagged one quarter)")


# Plot

def plot(g: pd.DataFrame, measure: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = pd.PeriodIndex(g["period"], freq="Q")
    x = per.to_timestamp(how="start")
    boundary = FITTED_FROM.to_timestamp(how="start")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(x, g["E_H"], color="tab:red", lw=1.6,
             label=r"$E^H$ (IAS median, 12m ahead)")
    ax1.plot(x, g["E_P"], color="tab:blue", lw=1.6,
             label=rf"$E^P$ ({measure}, constant 12m horizon)")
    ax1.set_ylabel("per cent")
    ax1.legend(frameon=False, loc="upper left")

    ax2.axhline(0.0, color="0.6", lw=0.8)
    ax2.plot(x, g["gap"], color="tab:purple", lw=1.6,
             label=r"$g_t = E^H - E^P$")
    ax2.set_ylabel("pp")
    ax2.legend(frameon=False, loc="upper left")

    for ax in (ax1, ax2):
        ax.axvline(boundary, color="0.3", lw=1.0, ls="--")
        ax.axvspan(boundary, x[-1], color="0.85", alpha=0.5, zorder=0)
    ax1.annotate("published $\\leftarrow$ | $\\rightarrow$ fitted\n"
                 "(splice, 2023Q2)",
                 xy=(boundary, ax1.get_ylim()[1]),
                 xytext=(8, -12), textcoords="offset points",
                 fontsize=8, va="top")

    fig.suptitle(f"Household vs professional 12m inflation expectations "
                 f"({measure} baseline)" if measure == "RPI" else
                 f"Household vs professional 12m inflation expectations "
                 f"({measure} robustness)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\n  plot -> {path}")


# Entry point

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", default="panel_forecasters_june2026_in_.csv")
    p.add_argument("--workbook",
                   default="Database_of_Average_Forecasts_for_the_UK_Economy.xlsx")
    p.add_argument("--ias",
                   default="Constructed_quarterly_moments_households.xlsx")
    p.add_argument("--salience",
                   default=str(Path(__file__).resolve().parent.parent
                               / "salience" / "data" / "salience_index.csv"))
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    for f in (a.panel, a.workbook, a.ias):
        if not Path(f).exists():
            print(f"ERROR: cannot find {f}", file=sys.stderr)
            return 1

    # ---- professional side (step 3, imported, not reimplemented) ----------
    ind, _, _ = load_panel(a.panel)
    q4 = ind[ind["variable"].str.contains(r"\(Q4\)")].copy()
    assert int(q4["subm"].isna().sum()) == 0, "undated rows in the Q4 series"
    bench = load_workbook_average(a.workbook)
    bench_map = {(r.pub, r.variable, r.event_year): r.published
                 for r in bench.itertuples()}
    fit_map = {(r.pub, r.variable, r.event_year): r.fitted
               for r in apply_rule(q4, **RULE).itertuples()}

    # ---- household side ----------------------------------------------------
    ias = load_ias(a.ias)
    print(f"IAS moments: {len(ias)} waves, "
          f"{ias['period'].min()}..{ias['period'].max()}")

    # ---- salience (optional: the index lives in the salience folder) ------
    sal = None
    if Path(a.salience).exists():
        s = pd.read_csv(a.salience)
        sal = pd.Series(s["s_t"].values,
                        index=pd.PeriodIndex(s["quarter"], freq="Q"),
                        name="s_t")
    else:
        print(f"WARNING: salience index not found at {a.salience}; "
              "correlation block skipped")

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    filenames = {"RPI": "gap_series.csv", "CPI": "gap_series_cpi.csv"}

    for measure in ("RPI", "CPI"):
        prof = build_series(measure, bench_map, fit_map, q4)
        assert_sample(prof, measure)              # 92 / 88, contiguous, E_P
        g = build_gap(prof, ias, measure)
        assert g["gap"].notna().all()

        inspect(g, measure, sal)

        path = out / filenames[measure]
        g_out = g[OUTPUT_COLUMNS].assign(period=g["period"].astype(str))
        g_out.to_csv(path, index=False)
        print(f"\n  wrote {path}  ({len(g_out)} rows, "
              f"{'RPI baseline' if measure == 'RPI' else 'CPI robustness'})")

        plot(g, measure, out / f"gap_series_{measure.lower()}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
