"""
Four checks on the constant-horizon series from build_professional.py.

    python validate_professional.py --panel <panel.csv> --workbook <hmt.xlsx>

All four are reported. Nothing is tuned or repaired to make them pass.

1. Interpolation. E_P is rebuilt from the calendar-year series, same
   forecasters and same aggregation rule but a different fixed-event
   structure, and interpolated to the same horizon. The two are different
   concepts, so they will not agree exactly; the question is whether the
   interpolation is doing the work.
2. Splice discontinuity at 2023Q2. E_P for 2023Q1 computed both ways, plus a
   level-shift test of the fitted-route error over the published overlap.
3. Q4 exactness. In Q4 waves the weight on F_{Y+1} is one, so E_P must equal
   the next-year anchor. Asserted.
4. Regime dependence of the fitted segment. n_P and the fitted-versus-published
   error for every quarter of 2021-2023, before the rule is trusted blind from
   2023Q2.
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
from build_professional import (EXCEPTION_CELLS, FIELDWORK_MONTH,
                                LAST_PUBLISHED_QUARTER, RULE, SAMPLE,
                                build_series)

MEASURES = ("RPI", "CPI")

# Check 1: the (annual) route
#
# The (annual) events are calendar-year AVERAGES, so their natural anchor is
# the middle of the calendar year; the target is fieldwork + 12 months, i.e.
# the middle of month m in year Y+1. Linear interpolation between the two
# adjacent mid-year anchors gives, for target time  tau = (Y+1) + (m-0.5)/12
# and anchor times  a_A = A + 0.5 :
#
#   wave  m   target        anchors (A, A+1)   weight on A+1
#   Q1    2   mid-Feb Y+1   (Y,   Y+1)         0.625
#   Q2    5   mid-May Y+1   (Y,   Y+1)         0.875
#   Q3    8   mid-Aug Y+1   (Y+1, Y+2)         0.125
#   Q4    11  mid-Nov Y+1   (Y+1, Y+2)         0.375
#
# Q3/Q4 waves therefore need the two-year-ahead annual event, which the
# (annual) series carries (horizons up to 4 years). The conventions differ
# slightly from the Q4 route (mid-year anchors vs 15 November; mid-month
# target vs d = m+1) -- unavoidable, since the events measure different
# objects; the comparison is about co-movement, not identity.

ANNUAL_NEAR_OFFSET = {1: 0, 2: 0, 3: 1, 4: 1}   # near anchor year = Y + offset
ANNUAL_FAR_WEIGHT = {1: 0.625, 2: 0.875, 3: 0.125, 4: 0.375}


def _annual_ep(ann_map: dict, pub: pd.Timestamp, var_ann: str,
               y: int, q: int) -> float | None:
    a = y + ANNUAL_NEAR_OFFSET[q]
    wf = ANNUAL_FAR_WEIGHT[q]
    near = ann_map.get((pub, var_ann, a))
    far = ann_map.get((pub, var_ann, a + 1))
    if near is None or far is None:
        return None
    return (1.0 - wf) * near + wf * far


def _skip_reason(ann: pd.DataFrame, pub: pd.Timestamp, var_ann: str) -> str:
    """Why the (annual) route has no value here. Never a silent skip."""
    raw = ann[(ann["pub"] == pub) & (ann["variable"] == var_ann)]
    if raw.empty:
        return "publication absent"
    if not raw["subm_reliable"].any():
        return "submission dates corrupt (Aug-2018)"
    return "no fresh (<=2m) submissions in the medium-term table"


def check_1_interpolation(results: dict, ann: pd.DataFrame) -> None:
    _h("CHECK 1  --  INTERPOLATION: rebuild E_P from the (annual) series")
    print("  Same forecasters, same rule (mean, <=2m, ex-OBR), different")
    print("  fixed-event structure. Different concepts: a Q4-on-Q4 point rate")
    print("  vs a calendar-year average. Co-movement, not identity, is the")
    print("  test of whether the interpolation is doing the work.")

    # The undated rows: all 327 live in the
    # (annual) series, and the rule's age filter drops them SILENTLY (NaN
    # comparisons are False). Here that exclusion is made explicit, once.
    undated = ann[ann["subm"].isna()]
    n_obr = int(undated["forecaster"].isin(OFFICIAL_FORECASTERS).sum())
    print(f"\n  undated (annual) rows excluded by the rule: {len(undated)} "
          f"(of which OBR, excluded anyway: {n_obr}; "
          f"non-OBR lost to the age filter: {len(undated) - n_obr})")

    ann_fit = apply_rule(ann, **RULE)
    ann_map = {(r.pub, r.variable, r.event_year): r.fitted
               for r in ann_fit.itertuples()}

    for measure in MEASURES:
        res = results[measure]
        var_ann = f"{measure} (annual)"
        grid = pd.period_range("2010Q1", SAMPLE[measure][1], freq="Q")
        base = res.set_index("period")["E_P"]

        rows, skipped = [], []
        for p in grid:
            y, q = p.year, p.quarter
            pub = pd.Timestamp(y, FIELDWORK_MONTH[q], 1)
            ep_ann = _annual_ep(ann_map, pub, var_ann, y, q)
            if ep_ann is None:
                skipped.append(f"{p} ({_skip_reason(ann, pub, var_ann)})")
                rows.append((p, np.nan))
            else:
                rows.append((p, ep_ann))
        s_ann = pd.Series(dict(rows)).astype(float)
        both = pd.DataFrame({"base": base.reindex(s_ann.index),
                             "ann": s_ann}).dropna()

        d = pd.DataFrame({"base": base.reindex(s_ann.index),
                          "ann": s_ann}).diff().dropna()

        print(f"\n  {measure}: {len(both)} quarters compared "
              f"({both.index[0]}..{both.index[-1]})")
        print("    skipped, no (annual) route:")
        for s in skipped:
            print(f"      {s}")
        print("    note: 2010Q1 annual route uses Feb-2010 (the only "
              "publication the (annual)\n    series has there); the baseline "
              "uses the Mar-2010 timing exception.")
        print(f"    corr (levels)             : "
              f"{np.corrcoef(both['base'], both['ann'])[0, 1]:.3f}")
        print(f"    corr (quarterly changes)  : "
              f"{np.corrcoef(d['base'], d['ann'])[0, 1]:.3f}   "
              f"[n={len(d)}]")
        print(f"    mean |difference|         : "
              f"{(both['ann'] - both['base']).abs().mean():.3f}")
        print(f"    mean signed (ann - Q4)    : "
              f"{(both['ann'] - both['base']).mean():+.3f}   "
              "(concept gap: year-average vs Q4 point)")
        e = (both["ann"] - both["base"]).abs()
        print(f"    largest |difference|      : {e.max():.3f} at {e.idxmax()}")


# Check 2: the splice

def _t_of_mean(x: pd.Series) -> tuple[float, float, float, int]:
    """(mean, sd, t-stat of H0: mean = 0, n). No p-value: the errors are
    serially correlated, so a textbook p would overstate precision."""
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    return m, s, m / (s / np.sqrt(n)), n


def check_2_splice(results: dict) -> None:
    _h("CHECK 2  --  SPLICE DISCONTINUITY AT 2023Q2")
    print("  err = fitted-route E_P minus published E_P, on quarters where")
    print("  both exist. A level shift in err at the boundary would be")
    print("  inherited one-for-one by the gap g_t from 2023Q2 onward.")

    seam = pd.Period("2023Q1", freq="Q")
    for measure in MEASURES:
        res = results[measure].set_index("period")
        ov = res[(res["source"] == "published") & res["_ep_panel"].notna()]
        err = ov["_ep_panel"] - ov["E_P"]

        print(f"\n  {measure}:")
        e_pub = res.loc[seam, "E_P"]
        e_fit = res.loc[seam, "_ep_panel"]
        print(f"    2023Q1 published route    : {e_pub:.3f}")
        print(f"    2023Q1 fitted route       : {e_fit:.3f}")
        print(f"    difference at the seam    : {e_fit - e_pub:+.3f}")

        m, s, t, n = _t_of_mean(err)
        print(f"    level shift, full overlap : mean err {m:+.3f} "
              f"(sd {s:.3f}, t = {t:+.2f}, n = {n})")
        late = err[err.index >= pd.Period("2021Q1", freq="Q")]
        m, s, t, n = _t_of_mean(late)
        print(f"    level shift, 2021Q1-2023Q1: mean err {m:+.3f} "
              f"(sd {s:.3f}, t = {t:+.2f}, n = {n})   "
              "<- the regime that resembles the fitted segment")
        print("    (t-stats face serially correlated errors; read as "
              "descriptive, not as a clean 5% test)")

        # What the spliced series actually records across the boundary,
        # vs what a pure fitted-route series would have recorded.
        q2 = pd.Period("2023Q2", freq="Q")
        step_spliced = res.loc[q2, "E_P"] - res.loc[seam, "E_P"]
        step_fitted = res.loc[q2, "E_P"] - res.loc[seam, "_ep_panel"]
        print(f"    Delta E_P across boundary : {step_spliced:+.3f} spliced "
              f"vs {step_fitted:+.3f} within-fitted-route "
              f"(splice adds {step_spliced - step_fitted:+.3f})")


# Check 3: Q4 exactness

def check_3_q4_exact(results: dict, bench_map: dict, fit_map: dict) -> bool:
    _h("CHECK 3  --  Q4 EXACTNESS: E_P in Q4 waves equals the next-year "
       "anchor")
    print("  Q4 waves put weight 1.0 on F_{Y+1}: no interpolation happens,")
    print("  so E_P must equal the anchor exactly (to rounding, 1e-6).")

    ok = True
    for measure in MEASURES:
        res = results[measure]
        var = f"{measure} (Q4)"
        q4_rows = res[[pd.Period(p, freq="Q").quarter == 4
                       for p in res["period"].astype(str)]]
        worst, n_checked, failures = 0.0, 0, []
        for _, r in q4_rows.iterrows():
            p = r["period"]
            pub = pd.Timestamp(f"{r['pub_used']}-01")
            amap = bench_map if r["source"] == "published" else fit_map
            anchor = amap.get((pub, var, p.year + 1))
            assert anchor is not None, (
                f"{measure} {p}: next-year anchor missing from the "
                f"{r['source']} source -- cannot check exactness")
            dev = abs(r["E_P"] - anchor)
            worst = max(worst, dev)
            n_checked += 1
            if dev > 1e-6:
                failures.append((p, r["E_P"], anchor, dev))
        status = "PASS" if not failures else "FAIL"
        print(f"\n  {measure}: {n_checked} Q4 waves checked, "
              f"max |E_P - anchor| = {worst:.2e}   {status}")
        for p, e, anc, dev in failures:
            print(f"    FAIL {p}: E_P {e} vs anchor {anc} (dev {dev:.6f})")
            ok = False
    return ok


# Check 4: the fitted rule in a fast-moving regime

def check_4_regime(results: dict) -> None:
    _h("CHECK 4  --  REGIME-DEPENDENCE: the rule through 2021-2023, "
       "quarter by quarter")
    print("  err = fitted-route E_P minus published E_P. From 2023Q2 there is")
    print("  no published benchmark -- that is precisely where the rule runs")
    print("  blind, so read its 2021-2023Q1 behaviour as the evidence base.")

    window = pd.period_range("2021Q1", "2023Q4", freq="Q")
    for measure in MEASURES:
        res = results[measure].set_index("period")
        print(f"\n  {measure}:")
        print(f"    {'period':<8}{'source':<11}{'n_P':>4}{'E_P':>8}"
              f"{'fitted-route':>14}{'published':>11}{'err':>8}")
        for p in window:
            r = res.loc[p]
            n_p = f"{int(r['n_P'])}" if pd.notna(r["n_P"]) else "--"
            fitted = (f"{r['_ep_panel']:.3f}" if pd.notna(r["_ep_panel"])
                      else "--")
            if r["source"] == "published":
                pub_v = f"{r['E_P']:.3f}"
                err = (f"{r['_ep_panel'] - r['E_P']:+.3f}"
                       if pd.notna(r["_ep_panel"]) else "--")
            else:
                pub_v, err = "--", "--"
            print(f"    {str(p):<8}{r['source']:<11}{n_p:>4}"
                  f"{r['E_P']:>8.3f}{fitted:>14}{pub_v:>11}{err:>8}")
        ov = res[(res.index.isin(window)) & (res["source"] == "published")
                 & res["_ep_panel"].notna()]
        e = ov["_ep_panel"] - ov["E_P"]
        print(f"    2021Q1-2023Q1 summary: mean err {e.mean():+.3f}, "
              f"mean |err| {e.abs().mean():.3f}, worst {e.abs().max():.3f} "
              f"at {e.abs().idxmax()}")
    print("\n  Signing the risk: the rule understates E_P when inflation")
    print("  rises (errs negative above) and by symmetry overstates it in a")
    print("  fast disinflation -- so 2023Q2-2025Q4 E_P is, if anything, too")
    print("  high and the gap g_t too small. Conservative, still wrong;")
    print("  state it in the chapter; this is a known weakness.")


# Entry point

def _h(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", default="panel_forecasters_june2026_in_.csv")
    p.add_argument("--workbook",
                   default="Database_of_Average_Forecasts_for_the_UK_Economy.xlsx")
    a = p.parse_args()
    for f in (a.panel, a.workbook):
        if not Path(f).exists():
            print(f"ERROR: cannot find {f}", file=sys.stderr)
            return 1

    ind, _, _ = load_panel(a.panel)
    q4 = ind[ind["variable"].str.contains(r"\(Q4\)")].copy()
    ann = ind[ind["variable"].str.contains(r"\(annual\)")].copy()
    assert int(q4["subm"].isna().sum()) == 0, "undated rows in the Q4 series"

    bench = load_workbook_average(a.workbook)
    bench_map = {(r.pub, r.variable, r.event_year): r.published
                 for r in bench.itertuples()}
    fit_map = {(r.pub, r.variable, r.event_year): r.fitted
               for r in apply_rule(q4, **RULE).itertuples()}

    # The baseline under test: exactly what build_professional builds.
    results = {m: build_series(m, bench_map, fit_map, q4) for m in MEASURES}

    check_1_interpolation(results, ann)
    check_2_splice(results)
    q4_ok = check_3_q4_exact(results, bench_map, fit_map)
    check_4_regime(results)

    _h("VERDICT")
    print("  1. interpolation      : reported above (co-movement, no "
          "pass/fail line)")
    print("  2. splice at 2023Q2   : reported above (seam difference and "
          "level-shift t)")
    print(f"  3. Q4 exactness       : {'PASS' if q4_ok else 'FAIL'}")
    print("  4. regime-dependence  : reported above (2021-2023 table)")
    return 0 if q4_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
