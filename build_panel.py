"""
Load and clean the HM Treasury panel of independent forecasters.

    python build_panel.py --diagnose
    from build_panel import load_panel

Returns a tidy long frame and separates genuine forecaster rows from HMT's own
summary rows. Each cleaning rule is documented on the function that applies it;
--diagnose prints the audit report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration

DEFAULT_CSV = "panel_forecasters_june2026_in_.csv"
ENCODING = "cp1252"

# The six labels below appear in the `forecaster` column but are NOT
# forecasting institutions -- they are HMT's own summary rows, lifted
# verbatim from the published tables. Averaging over them would double-count
# the panel and contaminate any dispersion measure. They are split out rather
# than dropped because they serve as an external check on our own aggregation
# (see step 2 of the pipeline).
AGGREGATE_LABELS = [
    "Highest",                        # highest forecast in the current table
    "Lowest",                         # lowest forecast in the current table
    "Median",                         # median of the current table
    "City",                           # average of City forecasters
    "Independent",                    # average of non-City forecasters
    "Received this month (marked *)", # average of forecasts newly submitted
]

# Publication vintages whose `date_submission` column is not trustworthy.
#
# 2018-08: the ONLY month in all 186 for which the median forecast age is not
# zero. Median age is 9 months and just 9% of forecasts are <=2m old, against
# 75-96% in every neighbouring month, with 20 of 23 contributors all stamped
# Nov-2017. Real staleness does not arrive and depart in a single month and
# then vanish for 186 months: this is a corrupted submission column in one
# vintage of the source spreadsheet. The forecast VALUES look normal and are
# kept; only the dates are unusable, so any age-based rule must not be applied
# here. Flagged rather than dropped: step 3 falls back to the published
# workbook average for this month, which does not depend on submission dates.
UNRELIABLE_SUBMISSION_VINTAGES = {pd.Timestamp("2018-08-01")}

# Variables needed downstream. The (Q4) series are Q4-on-Q4 annual inflation
# rates -- fixed events at a point in time -- and are the basis of E^P.
# The (annual) series are calendar-year AVERAGES: a different concept, kept
# only for the independent validation of the interpolation in step 4.
# They must never be pooled with the (Q4) series.
TARGET_VARIABLES = [
    "RPI (Q4)",
    "CPI (Q4)",
    "RPI (annual)",
    "CPI (annual)",
]


# Cleaning

def _parse_dates(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """
    RULE 1 -- Parse the two month-stamps.

    `date_publication` and `date_submission` are 'Mon-YY' strings. Both are
    month resolution, so each is anchored to the first of the month; nothing
    downstream needs finer than monthly.

    `date_publication` is the vintage of the HMT release. `date_submission`
    is when that particular forecaster last sent HMT a forecast -- it is the
    single most important column in this file and the reason the panel is
    worth more than the published averages.
    """
    df = df.copy()
    df["pub"] = pd.to_datetime(df["date_publication"], format="%b-%y")
    df["subm"] = pd.to_datetime(df["date_submission"], format="%b-%y",
                                errors="coerce")
    log["n_subm_unparsed"] = int(df["subm"].isna().sum())
    return df


def _restrict_variables(df: pd.DataFrame, variables: list[str],
                        log: dict) -> pd.DataFrame:
    """
    RULE 2 -- Keep only the inflation variables we need.

    The file carries 46 variables (GDP, PSNB, sterling index, ...). Filtering
    early is not just for speed: `date_forecasted` holds financial years for
    some fiscal variables ('2025-26'), which would break the integer-year
    parse in rule 3. Restricting first means that parse can be strict.
    """
    log["n_rows_raw"] = len(df)
    out = df[df["variable"].isin(variables)].copy()
    log["n_rows_target_vars"] = len(out)
    return out


def _parse_event_year(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """
    RULE 3 -- `date_forecasted` -> integer event year.

    For the four target variables this is always a plain 4-digit year, so the
    parse is strict: anything that fails is a data problem we want to see, not
    something to coerce silently.
    """
    df = df.copy()
    yr = pd.to_numeric(df["date_forecasted"], errors="coerce")
    bad = yr.isna()
    log["n_event_year_unparsed"] = int(bad.sum())
    log["event_year_bad_values"] = sorted(
        df.loc[bad, "date_forecasted"].astype(str).unique()
    )[:10]
    df = df.loc[~bad].copy()
    df["event_year"] = yr.loc[~bad].astype(int)
    return df


def _split_aggregates(df: pd.DataFrame, log: dict
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RULE 4 -- Separate HMT's summary rows from real institutions.

    See AGGREGATE_LABELS. This classification is not taken on trust: the
    diagnostic report tests it by checking whether the 'Highest'/'Lowest'
    rows reproduce the max/min of the individual rows. They do, once the
    staleness window is applied -- which is what identifies them as summaries
    of this same panel rather than as forecasters in their own right.
    """
    is_agg = df["forecaster"].isin(AGGREGATE_LABELS)
    ind, agg = df.loc[~is_agg].copy(), df.loc[is_agg].copy()
    log["n_rows_individual"] = len(ind)
    log["n_rows_aggregate"] = len(agg)
    log["n_institutions"] = int(ind["forecaster"].nunique())
    return ind, agg


def _drop_impossible_vintages(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """
    RULE 5 -- Drop rows whose submission post-dates publication.

    A forecast cannot be submitted after the release that reports it. These
    are source errors (a handful, all with submission dates years ahead of
    publication). They are dropped rather than repaired because there is no
    way to know which of the two dates is wrong.
    """
    bad = df["subm"].notna() & (df["subm"] > df["pub"])
    log["n_dropped_impossible_vintage"] = int(bad.sum())
    log["dropped_impossible_examples"] = (
        df.loc[bad, ["pub", "subm", "forecaster", "variable"]]
          .head(6).to_dict("records")
    )
    return df.loc[~bad].copy()


def _add_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    RULE 6 -- Forecast age in months at the publication date.

    age = 0 means the forecast was submitted in the publication month (HMT
    marks these with an asterisk). age = 14 means HMT is still printing a
    forecast that is over a year old. This is the variable that decides
    whether E^P is measured or fabricated in 2022.
    """
    df = df.copy()
    df["age_m"] = ((df["pub"].dt.year - df["subm"].dt.year) * 12
                   + (df["pub"].dt.month - df["subm"].dt.month))
    # See UNRELIABLE_SUBMISSION_VINTAGES. age_m is retained for inspection but
    # must not be filtered on where this flag is False.
    df["subm_reliable"] = ~df["pub"].isin(UNRELIABLE_SUBMISSION_VINTAGES)
    return df


def _deduplicate(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """
    RULE 7 -- One forecast per (publication, variable, event, forecaster).

    A few publications carry two vintages for the same forecaster (e.g.
    Aug-18 lists ITEM Club with both an Oct-17 and a May-18 submission).
    HMT's own tables print only a contributor's latest forecast, so we keep
    the most recent submission and drop the superseded one. Rows with no
    submission date lose the tie-break and are dropped only if a dated
    duplicate exists.
    """
    key = ["pub", "variable", "event_year", "forecaster"]
    n_before = len(df)
    out = (df.sort_values(key + ["subm"], na_position="first")
             .drop_duplicates(subset=key, keep="last")
             .copy())
    log["n_dropped_superseded_duplicates"] = n_before - len(out)
    return out


def load_panel(csv_path: str | Path = DEFAULT_CSV,
               variables: list[str] | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load and clean the panel.

    Returns
    -------
    ind : DataFrame  -- one row per (pub, variable, event_year, institution)
    agg : DataFrame  -- HMT's own summary rows, same shape, kept for checking
    log : dict       -- audit trail of every row dropped and why
    """
    variables = variables or TARGET_VARIABLES
    log: dict = {"csv_path": str(csv_path)}

    # `table_number` mixes ints and strings ('1' and 1), which triggers a
    # pandas dtype warning. We never use it, so read it as string outright.
    df = pd.read_csv(csv_path, encoding=ENCODING,
                     dtype={"table_number": "string"})

    df = _parse_dates(df, log)
    df = _restrict_variables(df, variables, log)
    df = _parse_event_year(df, log)
    ind, agg = _split_aggregates(df, log)

    ind = _drop_impossible_vintages(ind, log)
    ind = _add_age(ind)
    ind = _deduplicate(ind, log)
    agg = _add_age(_drop_impossible_vintages(agg, {}))

    log["n_rows_clean"] = len(ind)
    cols = ["pub", "subm", "age_m", "subm_reliable", "variable", "event_year",
            "forecaster", "value"]
    return ind[cols].reset_index(drop=True), agg[cols].reset_index(drop=True), log


# Diagnostics

def _h(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def diagnose(ind: pd.DataFrame, agg: pd.DataFrame, log: dict) -> None:
    _h("1. CLEANING AUDIT  (every row dropped, and why)")
    print(f"  raw rows                                    {log['n_rows_raw']:>9,}")
    print(f"  ...of target variables                      {log['n_rows_target_vars']:>9,}")
    print(f"     HMT summary rows split out               {log['n_rows_aggregate']:>9,}")
    print(f"     individual-forecaster rows               {log['n_rows_individual']:>9,}")
    print(f"  RULE 5  dropped: submission after publication {log['n_dropped_impossible_vintage']:>7,}")
    print(f"  RULE 7  dropped: superseded duplicate vintage {log['n_dropped_superseded_duplicates']:>7,}")
    print(f"  RULE 3  dropped: event year unparseable       {log['n_event_year_unparsed']:>7,}")
    print(f"  ------------------------------------------------------")
    print(f"  clean individual rows                       {log['n_rows_clean']:>9,}")
    print(f"  distinct institutions                       {log['n_institutions']:>9,}")
    print(f"  rows with no submission date (age unknown)  "
          f"{int(ind['age_m'].isna().sum()):>9,}")
    if log["dropped_impossible_examples"]:
        print("\n  examples of impossible vintages dropped:")
        for r in log["dropped_impossible_examples"]:
            print(f"     {r['forecaster']:<32s} pub {r['pub']:%Y-%m}  "
                  f"submitted {r['subm']:%Y-%m}")

    _h("2. CLASSIFICATION CHECK  -- are the six labels really summary rows?")
    print("  If 'Highest'/'Lowest' reproduce the max/min of the individual rows,")
    print("  they summarise this panel and are correctly excluded from it.")
    print("  The window that reproduces them is also evidence on HMT's own recipe.\n")
    q4 = ind[ind["variable"].str.contains(r"\(Q4\)")]
    aq4 = agg[agg["variable"].str.contains(r"\(Q4\)")]
    key = ["pub", "variable", "event_year"]
    print(f"  {'window':<14s}{'Lowest == min':>16s}{'Highest == max':>16s}")
    for label, mask in [("submitted now", lambda d: d["age_m"] == 0),
                        ("age <= 1m", lambda d: d["age_m"] <= 1),
                        ("age <= 2m", lambda d: d["age_m"] <= 2),
                        ("age <= 3m", lambda d: d["age_m"] <= 3),
                        ("any age", lambda d: d["age_m"].notna() | True)]:
        sub = q4[mask(q4)]
        rng = sub.groupby(key)["value"].agg(["min", "max"]).reset_index()
        row = []
        for lab, col in [("Lowest", "min"), ("Highest", "max")]:
            a = (aq4[aq4["forecaster"] == lab][key + ["value"]]
                 .rename(columns={"value": lab}))
            m = rng.merge(a, on=key, how="inner")
            exact = ((m[lab] - m[col]).abs() < 1e-6).mean() if len(m) else np.nan
            row.append(f"{exact:>15.1%}")
        print(f"  {label:<14s}{row[0]}{row[1]}")

    _h("3. STALENESS  -- how old are the forecasts HMT prints? (Q4 series)")
    a = q4["age_m"].dropna()
    print(f"  share submitted in the publication month   {(a == 0).mean():>7.1%}")
    print(f"  share <= 2 months old                      {(a <= 2).mean():>7.1%}")
    print(f"  share > 6 months old                       {(a > 6).mean():>7.1%}")
    print(f"  share > 12 months old                      {(a > 12).mean():>7.1%}")
    print(f"  oldest forecast printed                    {int(a.max()):>5d} months")
    print("\n  age distribution (months):")
    vc = a.value_counts().sort_index()
    for k in list(vc.index)[:13]:
        bar = "#" * int(60 * vc[k] / vc.max())
        print(f"     {int(k):>3d}  {int(vc[k]):>5,}  {bar}")
    if len(vc) > 13:
        print(f"     {int(vc.index[13]):>3d}+ {int(vc.iloc[13:].sum()):>5,}  (tail)")

    _h("4. PANEL DEPTH  -- contributors per publication month")
    for v in ["RPI (Q4)", "CPI (Q4)"]:
        s = q4[q4["variable"] == v]
        n_all = s.groupby("pub")["forecaster"].nunique()
        n_fresh = (s[s["age_m"] <= 2].groupby("pub")["forecaster"].nunique()
                   .reindex(n_all.index).fillna(0))
        t = pd.DataFrame({"listed": n_all, "usable (<=2m)": n_fresh})
        t = t.groupby(t.index.year).agg(["min", "mean", "max"]).round(1)
        print(f"\n  {v}   [{s['pub'].min():%Y-%m} .. {s['pub'].max():%Y-%m}, "
              f"{s['pub'].nunique()} publication months]")
        print("     year   listed(min/mean/max)   usable<=2m(min/mean/max)")
        for y in t.index:
            r = t.loc[y]
            print(f"     {y}      {r[('listed','min')]:>4.0f} /{r[('listed','mean')]:>6.1f} /"
                  f"{r[('listed','max')]:>4.0f}        {r[('usable (<=2m)','min')]:>4.0f} /"
                  f"{r[('usable (<=2m)','mean')]:>6.1f} /{r[('usable (<=2m)','max')]:>4.0f}")

    _h("5. COVERAGE AT IAS FIELDWORK MONTHS  (Feb / May / Aug / Nov)")
    print("  A 12-month-ahead E^P needs BOTH the current-year and next-year")
    print("  Q4 anchors live in the same publication. Missing months are")
    print("  listed so the fallback rule is applied to a known, finite set.\n")
    for v in ["RPI (Q4)", "CPI (Q4)"]:
        s = q4[q4["variable"] == v]
        have = {(p.year, p.month, e) for p, e in
                zip(s["pub"], s["event_year"])}
        years = sorted({p.year for p in s["pub"]})
        miss = []
        for y in years:
            for m in (2, 5, 8, 11):
                if pd.Timestamp(y, m, 1) > s["pub"].max():
                    continue
                need = [(y, m, y), (y, m, y + 1)]
                gone = [f"{k[2]}Q4" for k in need if k not in have]
                if gone:
                    miss.append(f"{y}-{m:02d} (missing {', '.join(gone)})")
        print(f"  {v}: {len(miss)} of {4*len(years)} fieldwork months incomplete")
        for x in miss:
            print(f"       {x}")

    _h("6. THE FOUR SERIES, SIDE BY SIDE")
    t = (ind.groupby("variable")
            .agg(rows=("value", "size"),
                 institutions=("forecaster", "nunique"),
                 first_pub=("pub", "min"), last_pub=("pub", "max"),
                 n_pub_months=("pub", "nunique")))
    t["max_horizon_yrs"] = (ind.groupby("variable")
                            .apply(lambda d: (d["event_year"] - d["pub"].dt.year).max(),
                                   include_groups=False))
    print(t.to_string())
    print("\n  NOTE: (Q4) = Q4-on-Q4 annual rate, a fixed event at a point in")
    print("  time -> this is what E^P is built from.")
    print("  (annual) = calendar-year AVERAGE inflation, published only in")
    print("  Feb/May/Aug/Nov -> a different concept. Never pool the two.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="path to the panel CSV")
    p.add_argument("--diagnose", action="store_true", help="print audit report")
    p.add_argument("--out", default=None,
                   help="optional path to write the cleaned panel as CSV")
    a = p.parse_args()

    if not Path(a.csv).exists():
        print(f"ERROR: cannot find {a.csv}", file=sys.stderr)
        return 1

    ind, agg, log = load_panel(a.csv)
    if a.diagnose:
        diagnose(ind, agg, log)
    if a.out:
        ind.to_csv(a.out, index=False)
        print(f"\nwrote cleaned panel -> {a.out}  ({len(ind):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
