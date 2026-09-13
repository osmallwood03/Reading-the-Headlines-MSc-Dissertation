"""
Constant-horizon professional expectations E_P, disagreement D_P and
contributor count n_P, one row per quarter.

    RPI  2003Q1-2025Q4, 92 quarters, the baseline
    CPI  2004Q1-2025Q4, 88 quarters, robustness; CPI events start 2004Q4

Two routes, spliced. Published averages from the HMT workbook cover
2003Q1-2023Q1 and are interpolated to the constant horizon. From 2023Q2 the
recovered aggregation rule (mean over forecasts no more than two months old,
excluding the OBR, per validate_aggregation.py) is applied to the panel under
the same interpolation.

Fixed-event to constant-horizon interpolation follows Gerlach (2007) and
Dovern, Fritsche and Slacalek (2012). IAS fieldwork month m is in {2, 5, 8, 11}
and the event "Y Q4" is anchored at 15 November of year Y:

    d = m + 1               months from the anchor to fieldwork plus twelve
    w = 1 - d/12            weight on the near anchor F_Y
    E_P = w F_Y + (1 - w) F_{Y+1}

Q4 waves have w = 0, so no interpolation happens there. Nothing is ever
extrapolated.

D_P interpolates each surviving forecaster's own two anchors to the twelve-month
horizon first, then takes the dispersion across forecasters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import load_panel
from validate_aggregation import (OFFICIAL_FORECASTERS, apply_rule,
                                  load_workbook_average)

# Configuration

# Sample per measure: (first quarter, last quarter, expected length).
# RPI is bound at 2003Q1 by the IAS turning quarterly; CPI at 2004Q1 because
# CPI (Q4) events only exist from 2004Q4.
SAMPLE = {
    "RPI": ("2003Q1", "2025Q4", 92),
    "CPI": ("2004Q1", "2025Q4", 88),
}

# Splice point. The workbook (HMT's own published average) reaches the
# April-2023 publication, so 2023Q1 (Feb-2023 fieldwork) is the last quarter
# the published route can serve. Everything after is the fitted route.
LAST_PUBLISHED_QUARTER = pd.Period("2023Q1", freq="Q")

# First publication month in the forecaster panel. Before this, D_P and n_P
# cannot exist (E_P can: the workbook reaches back to 1986 for RPI).
PANEL_START = pd.Timestamp("2010-10-01")

# IAS fieldwork months by quarter. The month-m publication is used,
# contemporaneous with fieldwork (the strictly-predetermined m-1 alternative
# is not viable: January publications lack the h=+1 anchor).
FIELDWORK_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}

# The aggregation rule recovered in validate_aggregation.py: mean over
# forecasts <= 2 months old, excluding the OBR. Passed to apply_rule verbatim.
RULE = dict(max_age=2, stat="mean", drop_obr=True)

# The three exception cells, explicit and named.
#
# These are NOT found by searching backwards until something matches: each is
# a known, documented failure of the default publication month, resolved once,
# here, and counted in the output. Keyed by quarter; the same resolution
# applies to both measures because all three are publication-level facts.
#
# Confidence follows the splice map exactly: only 2010Q1 is a
# `timing_exception`, because Mar-2010 post-dates the Feb fieldwork --
# professionals get ~1 month more information than households. 2017Q2 and
# 2019Q4 instead use the nearest PRECEDING publication (no publication existed
# in the fieldwork month at all), which is predetermined at fieldwork and
# therefore keeps confidence `high`; pub_used records the substitution.

EXCEPTION_CELLS: dict[str, dict] = {
    "2010Q1": dict(pub=pd.Timestamp("2010-03-01"),
                   confidence="timing_exception",
                   reason="2011Q4 anchor first exists in the Mar-2010 "
                          "publication, after Feb fieldwork (RPI); the lone "
                          "Feb-2010 CPI 2011Q4 cell is distrusted, see "
                          "DISTRUSTED_WORKBOOK_CELLS"),
    "2017Q2": dict(pub=pd.Timestamp("2017-04-01"),
                   confidence="high",
                   reason="HMT did not publish in May 2017; "
                          "nearest preceding publication"),
    "2019Q4": dict(pub=pd.Timestamp("2019-10-01"),
                   confidence="high",
                   reason="HMT did not publish in Nov 2019; "
                          "nearest preceding publication"),
}


# The workbook is trusted except where diagnosed otherwise. Exactly one cell
# is distrusted: the CPI sheet carries a lone Feb-2010 value for the 2011Q4
# event (RPI does not). It is identical to the adjacent 2010Q4 cell (2.1) and
# sits 0.4 above the Mar/Apr/May path for the same event (1.7 / 1.8 / 1.8) --
# it fails the smoothness test that validates the Feb 2011-15 workbook cells
# Either a duplication error or a number lifted
# from the Feb medium-term table (a different, smaller panel). It is NOT used:
# CPI 2010Q1 takes the same Mar-2010 timing exception as RPI, which also keeps
# the two measures on the same publication month for the same quarter. The
# value is pinned so a revised workbook vintage forces a re-decision.
DISTRUSTED_WORKBOOK_CELLS: dict[tuple, float] = {
    ("CPI (Q4)", pd.Timestamp("2010-02-01"), 2011): 2.1,
}


def _assert_exceptions_still_needed(bench: pd.DataFrame, var: str) -> None:
    """
    Each exception exists because something is absent from (or distrusted in)
    the workbook. If a revised workbook vintage ever fills the gap or changes
    the distrusted cell, the exception must be re-decided explicitly -- not
    applied out of habit. So: assert the gaps, and pin the distrusted cell.
    """
    b = bench[bench["variable"] == var]
    feb10 = b[(b["pub"] == pd.Timestamp("2010-02-01"))
              & (b["event_year"] == 2011)]
    key = (var, pd.Timestamp("2010-02-01"), 2011)
    if key in DISTRUSTED_WORKBOOK_CELLS:
        assert (len(feb10) == 1
                and feb10["published"].iloc[0]
                == DISTRUSTED_WORKBOOK_CELLS[key]), (
            f"{var}: the distrusted Feb-2010 2011Q4 cell changed or vanished "
            "in this workbook vintage. Re-decide the 2010Q1 exception.")
    else:
        assert feb10.empty, (
            f"{var}: Feb-2010 now carries a 2011Q4 event in the workbook. "
            "The 2010Q1 timing exception may no longer be needed -- "
            "re-decide it.")
    for miss in ("2017-05-01", "2019-11-01"):
        assert b[b["pub"] == pd.Timestamp(miss)].empty, (
            f"{var}: the workbook now has a {miss[:7]} publication. The "
            "corresponding exception cell may no longer be needed -- "
            "re-decide it.")


# Interpolation

def _weight(quarter: int) -> float:
    """Weight on the near anchor F_Y. Q1: 0.75, Q2: 0.50, Q3: 0.25, Q4: 0."""
    d = FIELDWORK_MONTH[quarter] + 1
    return 1.0 - d / 12.0


def _interp_from(anchor_map: dict, pub: pd.Timestamp, var: str,
                 year: int, w: float) -> float | None:
    """
    E_P from a {(pub, variable, event_year): value} lookup, or None if an
    anchor that carries weight is missing. Never fills silently: the caller
    decides whether missing is fatal (the route in use) or expected (the
    cross-route diagnostic).
    """
    near = anchor_map.get((pub, var, year))
    far = anchor_map.get((pub, var, year + 1))
    if far is None or (w > 0 and near is None):
        return None
    return far if w == 0.0 else w * near + (1.0 - w) * far


# Disagreement: per-forecaster interpolation FIRST, then the SD

def _cross_section(q4: pd.DataFrame, pub: pd.Timestamp, var: str,
                   year: int, w: float) -> tuple[float, float, object, str]:
    """
    The surviving cross-section behind D_P and n_P: forecasts <= 2 months
    old, ex-OBR, submission dates reliable -- the same filter apply_rule
    applies (apply_rule itself cannot serve here because it aggregates over
    forecasters, and D_P needs each forecaster's own pair of anchors).

    Each forecaster's own F_Y and F_{Y+1} are interpolated to the 12-month
    horizon first; the SD is taken across those interpolated values. A
    forecaster missing either weighted anchor drops out -- horizons are never
    mixed. When w = 0 the horizon IS the far anchor, so only F_{Y+1} is
    required.

    Returns (mean, sd, n, status). n is the count behind both moments.
    """
    if pub < PANEL_START:
        return np.nan, np.nan, pd.NA, "pre-panel"

    sub = q4[(q4["pub"] == pub) & (q4["variable"] == var)
             & q4["subm_reliable"]
             & (q4["age_m"] <= RULE["max_age"])
             & ~q4["forecaster"].isin(OFFICIAL_FORECASTERS)]

    if sub.empty:
        raw = q4[(q4["pub"] == pub) & (q4["variable"] == var)]
        if raw.empty:
            return np.nan, np.nan, pd.NA, "no panel publication"
        if not raw["subm_reliable"].any():
            # Aug-2018: the submission column is corrupt for this vintage
            # (build_panel.UNRELIABLE_SUBMISSION_VINTAGES). The age filter
            # cannot be applied, so no cross-section survives.
            return np.nan, np.nan, pd.NA, "submission dates corrupt"
        return np.nan, np.nan, pd.NA, "no fresh submissions"

    if w == 0.0:
        vals = sub.loc[sub["event_year"] == year + 1, "value"]
        if vals.empty:
            return np.nan, np.nan, pd.NA, "panel hole: far anchor absent"
    else:
        # RULE 7 in build_panel guarantees one row per (pub, variable,
        # event_year, forecaster), so a strict pivot cannot collide.
        piv = sub.pivot(index="forecaster", columns="event_year",
                        values="value")
        if year + 1 not in piv.columns:
            # The Feb 2011-2015 hole: only h=-1 and h=0 events present.
            return np.nan, np.nan, pd.NA, "panel hole: far anchor absent"
        if year not in piv.columns:
            return np.nan, np.nan, pd.NA, "panel hole: near anchor absent"
        both = piv[[year, year + 1]].dropna()
        if both.empty:
            return np.nan, np.nan, pd.NA, "no forecaster holds both anchors"
        vals = w * both[year] + (1.0 - w) * both[year + 1]

    n = len(vals)
    if n == 1:
        return float(vals.iloc[0]), np.nan, 1, "single forecaster"
    return float(vals.mean()), float(vals.std(ddof=1)), n, "ok"


# The quarterly series

def build_series(measure: str, bench_map: dict, fit_map: dict,
                 q4: pd.DataFrame) -> pd.DataFrame:
    """
    One row per quarter for one measure. Underscore-prefixed columns are
    internal diagnostics (cross-route comparison) and are dropped on output.
    """
    var = f"{measure} (Q4)"
    start, end, _ = SAMPLE[measure]
    rows = []
    for p in pd.period_range(start, end, freq="Q"):
        y, q = p.year, p.quarter
        w = _weight(q)
        exc = EXCEPTION_CELLS.get(str(p))
        pub = exc["pub"] if exc else pd.Timestamp(y, FIELDWORK_MONTH[q], 1)
        published_route = p <= LAST_PUBLISHED_QUARTER

        route_map = bench_map if published_route else fit_map
        e_p = _interp_from(route_map, pub, var, y, w)
        if e_p is None:
            raise ValueError(
                f"{measure} {p}: an anchor is missing from the "
                f"{'published' if published_route else 'fitted'} source at "
                f"pub {pub:%Y-%m} (events {y}Q4 / {y + 1}Q4, w={w}). "
                "No silent fill -- decide explicitly.")

        cs_mean, d_p, n_p, status = _cross_section(q4, pub, var, y, w)

        rows.append(dict(
            period=p,
            E_P=round(e_p, 6),
            D_P=d_p,
            n_P=n_p,
            source="published" if published_route else "fitted",
            confidence=(exc["confidence"] if exc else "high")
                       if published_route else "fitted",
            pub_used=f"{pub:%Y-%m}",
            D_P_status=status,
            _w=w,
            _cs_mean=cs_mean,
            # the fitted-route E_P for EVERY quarter, where computable:
            # this is the overlap that quantifies the splice discontinuity.
            _ep_panel=(lambda v: np.nan if v is None else v)(
                _interp_from(fit_map, pub, var, y, w)),
        ))

    res = pd.DataFrame(rows)
    res["n_P"] = res["n_P"].astype("Int64")
    return res


def assert_sample(res: pd.DataFrame, measure: str) -> None:
    """The non-negotiables: exact length, contiguity, no NaN in E_P."""
    start, end, n_expected = SAMPLE[measure]
    per = pd.PeriodIndex(res["period"], freq="Q")
    assert len(res) == n_expected, (
        f"{measure}: {len(res)} quarters, expected {n_expected}")
    assert str(per[0]) == start and str(per[-1]) == end, (
        f"{measure}: sample runs {per[0]}..{per[-1]}, expected {start}..{end}")
    assert (np.diff(per.asi8) == 1).all(), f"{measure}: quarters not contiguous"
    assert res["E_P"].notna().all(), (
        f"{measure}: NaN in E_P at "
        f"{list(res.loc[res['E_P'].isna(), 'period'].astype(str))}")


# Report

def _h(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def report(res: pd.DataFrame, measure: str) -> None:
    start, end, n_expected = SAMPLE[measure]
    _h(f"{measure}  --  {start}..{end}")
    print(f"  quarters: {len(res)}  (expected {n_expected})   "
          f"contiguous: yes   E_P missing: {int(res['E_P'].isna().sum())}   "
          "ASSERTS PASS")

    print("\n  source x confidence:")
    tab = pd.crosstab(res["source"], res["confidence"])
    print("  " + tab.to_string().replace("\n", "\n  "))

    print("\n  exception cells applied "
          f"({sum(str(p) in EXCEPTION_CELLS for p in res['period'])} of "
          f"{len(EXCEPTION_CELLS)} defined):")
    for key, exc in EXCEPTION_CELLS.items():
        row = res[res["period"].astype(str) == key]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"    {key}  pub_used {r['pub_used']}  "
              f"confidence={r['confidence']:<17s}  E_P={r['E_P']:.3f}   "
              f"({exc['reason']})")

    print("\n  D_P availability (status of the panel cross-section):")
    for status, n in res["D_P_status"].value_counts().items():
        qs = res.loc[res["D_P_status"] == status, "period"].astype(str)
        detail = ""
        if status not in ("ok", "pre-panel"):
            detail = "   [" + ", ".join(qs) + "]"
        print(f"    {status:<32s}{n:>4}{detail}")

    ok = res[res["D_P"].notna()]
    if len(ok):
        print(f"    first quarter with D_P: {ok['period'].iloc[0]}")

    # Splice diagnostic: quantify the 2023Q2 discontinuity.
    ov = res[(res["source"] == "published") & res["_ep_panel"].notna()].copy()
    if len(ov):
        ov["err"] = ov["_ep_panel"] - ov["E_P"]
        _h(f"{measure}  --  SPLICE DIAGNOSTIC: fitted route vs published, "
           "on the overlap")
        print("  err = fitted-route E_P minus published-route E_P at IAS "
              "quarters.\n  This is the discontinuity the 2023Q2 splice could "
              "introduce.\n")
        print(f"  overlap quarters      : {len(ov)}  "
              f"({ov['period'].iloc[0]}..{ov['period'].iloc[-1]})")
        print(f"  mean |err|            : {ov['err'].abs().mean():.3f}")
        print(f"  median |err|          : {ov['err'].abs().median():.3f}")
        print(f"  max |err|             : {ov['err'].abs().max():.3f}   "
              f"(at {ov.loc[ov['err'].abs().idxmax(), 'period']})")
        recent = ov[ov["period"] >= pd.Period("2021Q1", freq="Q")]
        print(f"  mean err 2021Q1-2023Q1: {recent['err'].mean():+.3f}   "
              "(sign of the bias the fitted segment inherits)")
        last = ov.iloc[-1]
        print(f"  at the seam ({last['period']})    : published "
              f"{last['E_P']:.3f} vs fitted-route {last['_ep_panel']:.3f}  "
              f"(err {last['err']:+.3f})")

        print("\n  around the splice:")
        seam = res[(res["period"] >= pd.Period("2022Q3", freq="Q"))
                   & (res["period"] <= pd.Period("2024Q1", freq="Q"))]
        print(f"    {'period':<8}{'source':<11}{'E_P':>8}"
              f"{'fitted-route':>14}{'diff':>8}")
        for _, r in seam.iterrows():
            alt = (f"{r['_ep_panel']:>14.3f}" if pd.notna(r["_ep_panel"])
                   else f"{'--':>14}")
            diff = (f"{r['_ep_panel'] - r['E_P']:>+8.3f}"
                    if pd.notna(r["_ep_panel"]) and r["source"] == "published"
                    else f"{'':>8}")
            print(f"    {str(r['period']):<8}{r['source']:<11}"
                  f"{r['E_P']:>8.3f}{alt}{diff}")

    # coherence: on fitted quarters, interpolating the two rule means must
    # agree with the mean of per-forecaster interpolations up to composition
    # (forecasters holding only one anchor). E_P deliberately interpolates
    # the rule's per-event means -- the same estimator the published segment
    # uses (HMT's printed average is per event, over whoever is listed for
    # that event) -- so the splice does not change the construction. The
    # balanced per-forecaster set serves D_P and n_P.
    fit_ok = res[(res["source"] == "fitted") & res["_cs_mean"].notna()]
    if len(fit_ok):
        gap = (fit_ok["_cs_mean"] - fit_ok["E_P"]).abs()
        worst = fit_ok.loc[gap.idxmax()]
        print(f"\n  fitted-quarter coherence |mean-of-interp - "
              f"interp-of-means|: max {gap.max():.4f} at {worst['period']} "
              f"(composition: forecasters holding one anchor only)")

    # ---- thin quarters -----------------------------------------------------
    _h(f"{measure}  --  TEN QUARTERS WITH THE LOWEST n_P")
    low = (res[res["n_P"].notna()]
           .sort_values(["n_P", "period"])
           .head(10))
    print(f"  {'period':<8}{'n_P':>5}{'D_P':>8}{'E_P':>8}  "
          f"{'source':<11}{'pub_used':<9}")
    for _, r in low.iterrows():
        flag = "  <-- n_P < 10, flag in the chapter" if r["n_P"] < 10 else ""
        d_p = f"{r['D_P']:>8.3f}" if pd.notna(r["D_P"]) else f"{'--':>8}"
        print(f"  {str(r['period']):<8}{int(r['n_P']):>5}{d_p}"
              f"{r['E_P']:>8.3f}  {r['source']:<11}{r['pub_used']:<9}{flag}")
    if (low["n_P"] >= 10).all():
        print("  (no quarter falls below the n_P < 10 flag threshold)")


# Entry point

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", default="panel_forecasters_june2026_in_.csv")
    p.add_argument("--workbook",
                   default="Database_of_Average_Forecasts_for_the_UK_Economy.xlsx")
    p.add_argument("--outdir", default=None,
                   help="directory to write professional_<MEASURE>.csv; "
                        "omitted = print only, write nothing")
    a = p.parse_args()
    for f in (a.panel, a.workbook):
        if not Path(f).exists():
            print(f"ERROR: cannot find {f}", file=sys.stderr)
            return 1

    ind, _, _ = load_panel(a.panel)
    q4 = ind[ind["variable"].str.contains(r"\(Q4\)")].copy()

    # Same guard as validate_aggregation: undated rows must never meet an age
    # filter silently (NaN comparisons are False).
    n_undated = int(q4["subm"].isna().sum())
    assert n_undated == 0, (
        f"{n_undated} Q4 rows have no submission date -- decide explicitly.")

    bench = load_workbook_average(a.workbook)
    bench_map = {(r.pub, r.variable, r.event_year): r.published
                 for r in bench.itertuples()}

    fit = apply_rule(q4, **RULE)
    fit_map = {(r.pub, r.variable, r.event_year): r.fitted
               for r in fit.itertuples()}

    results = {}
    for measure in SAMPLE:
        _assert_exceptions_still_needed(bench, f"{measure} (Q4)")
        res = build_series(measure, bench_map, fit_map, q4)
        assert_sample(res, measure)
        report(res, measure)
        results[measure] = res

    if a.outdir:
        out = Path(a.outdir)
        out.mkdir(parents=True, exist_ok=True)
        for measure, res in results.items():
            path = out / f"professional_{measure}.csv"
            cols = ["period", "E_P", "D_P", "n_P", "source", "confidence",
                    "pub_used", "D_P_status"]
            res[cols].assign(period=res["period"].astype(str)).to_csv(
                path, index=False)
            print(f"\nwrote {path}  ({len(res)} rows)")
    else:
        print("\n(no files written -- pass --outdir to write the CSVs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
