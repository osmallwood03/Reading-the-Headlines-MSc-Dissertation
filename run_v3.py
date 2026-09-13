"""
Command entry points for the empirical chapter.

    python run_v3.py diagnostics    lag order under both reduced forms, roots
    python run_v3.py soe            the ladder under both reduced forms
    python run_v3.py ladder         four identification schemes
    python run_v3.py baseline       IRF, FEVD and historical decomposition
    python run_v3.py sensitivity    FEVD threshold and persistence
    python run_v3.py episodes       2008 / 2011 / 2015 / 2022
    python run_v3.py grid           every specification, resumable
    python run_v3.py grid R17_eri   one specification
    python run_v3.py r16            split system, per-draw difference
    python run_v3.py eri            R17, R18 and R19
    python run_v3.py bootstrap 500  wild bootstrap, checkpoints every 25
    python run_v3.py combine        everything into results_v3.xlsx

Run diagnostics and soe first. Block exogeneity changes the reduced form, so it
has to be settled before the scheme ladder means anything, and the choice of
baseline scheme then depends on the ladder table.

The bootstrap is a Rademacher wild bootstrap on the reduced-form residuals. The
design matrix is held fixed, so the sample, lag structure and survey dummy are
common across replications and only the residual signs are resampled. Each
replication is re-identified from scratch, narrative restriction included, so
the pooled bands mix identification and sampling uncertainty. The refit uses
restricted_ols: an unrestricted refit would drop block exogeneity in every
replication and produce bands for a model that was never estimated.
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from svar_lib_v3 import (fit_var, select_order, bands, VARFit, shock_index,
                         restricted_ols)
import config_v3 as C
import identify_v3 as I

OUT = C.OUT
H = C.H


def setup(spec_name="baseline", strict_narrative=True, with_narrative=True):
    """Build the fit, the restrictions and the narrative checker for a spec."""
    spec = C.SPECS[spec_name]
    df = C.frame(spec)
    endog = spec["endog"]
    fit = fit_var(df, endog, spec.get("p", 1),
                  spec.get("exog", C.BASE_EXOG),
                  n_glo=C.n_glo_for(spec))
    quarters = np.asarray(df["quarter"].iloc[-fit.T:])
    specs = C.specs_for(spec)
    shock_names = [sp.name for sp in specs]
    narr = None
    if with_narrative:
        narr = I.narrative_checker(fit, fit.names, quarters, endog[3],
                                   shock_names, strict=strict_narrative)
    return spec, df, fit, quarters, specs, narr


def _target_of(spec):
    """The variable the summary tables report. Gap if present, else E^H/E^P."""
    g = [v for v in spec["endog"] if v.startswith("g_")]
    return g[0] if g else [v for v in spec["endog"] if v.startswith("E_")][0]


def _write(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"    wrote {name}.csv")


def diagnostics():
    """Section 5.1 tables, now including the ERI."""
    from statsmodels.tsa.stattools import adfuller
    from scipy import stats

    df = C.load()
    V = C.BASE_ENDOG
    V7 = V + [C.ERI]

    print("\n=== LAG ORDER SELECTION (six-variable baseline) ===")
    print("-- unrestricted reduced form --")
    lo = select_order(df, V, maxlags=4, exog=C.BASE_EXOG)
    print(lo.round(3)); print("selected:", lo.attrs["selected"])
    lo.to_csv(OUT / "tab_lagorder_v3.csv")

    # Block exogeneity drops n_dom * n_glo * p coefficients from the global
    # equations, so the penalty term applies to a different parameter count
    # and the restricted system can select a different p. p=1 must be
    # re-earned here, not carried over.
    print("\n-- SOE reduced form (q_t, a_t equations restricted) --")
    lo_soe = select_order(df, V, maxlags=4, exog=C.BASE_EXOG, n_glo=C.N_GLO)
    print(lo_soe.round(3)); print("selected:", lo_soe.attrs["selected"])
    lo_soe.to_csv(OUT / "tab_lagorder_soe_v3.csv")
    if lo_soe.attrs["selected"]["BIC"] != lo.attrs["selected"]["BIC"]:
        print("  *** BIC CHOICE CHANGES UNDER SOE. The draft says p=1 was "
              "chosen by BIC/HQ; that sentence needs revisiting. ***")

    print("\n=== UNIT ROOT (ADF, constant, AIC lag) ===")
    rows = []
    for c in V7:
        s = adfuller(df[c].dropna().values, regression="c", autolag="AIC")
        rows.append({"variable": c, "adf": s[0], "pvalue": s[1],
                     "lags": s[2], "reject_5pc": s[1] < 0.05})
    adf = pd.DataFrame(rows); print(adf.round(4).to_string(index=False))
    _write(adf, "tab_adf_v3")

    print("\n=== COLLINEARITY, six-variable and seven-variable ===")
    allrows = []
    for tag, VV in (("six", V), ("seven_with_eri", V7)):
        M = df[VV].dropna()
        for c in VV:
            o = [v for v in VV if v != c]
            X = np.column_stack([np.ones(len(M))] + [M[v] for v in o])
            b = np.linalg.lstsq(X, M[c], rcond=None)[0]
            r2 = 1 - ((M[c] - X @ b) ** 2).sum() / \
                     ((M[c] - M[c].mean()) ** 2).sum()
            allrows.append({"system": tag, "variable": c,
                            "R2_on_others": r2, "VIF": 1 / (1 - r2)})
        Z = (M - M.mean()) / M.std()
        print(f"  [{tag}] condition number (standardised) = "
              f"{np.linalg.cond(Z.values):.2f}")
    vif = pd.DataFrame(allrows)
    print(vif.round(3).to_string(index=False))
    _write(vif, "tab_vif_v3")

    M = df[V7].dropna()
    M.corr().to_csv(OUT / "tab_corr_v3.csv")
    print(f"\ncorr(s,pi)  = {M.corr().loc['s_t_asinh100','pi_t_rpi']:+.4f}")
    print(f"corr(s,g)   = {M.corr().loc['s_t_asinh100','g_meandec_cpi']:+.4f}")
    print(f"corr(eri,pi)= {M.corr().loc[C.ERI,'pi_t_rpi']:+.4f}")
    print(f"corr(eri,s) = {M.corr().loc[C.ERI,'s_t_asinh100']:+.4f}")

    print("\n=== SALIENCE MOMENTS (motivates the wild bootstrap) ===")
    for c in ["s_t_asinh100", "s_t_raw"]:
        x = df[c].dropna()
        print(f"  {c:16s} skew {stats.skew(x, bias=False):6.3f}  "
              f"kurtosis {stats.kurtosis(x, bias=False)+3:6.3f}")

    ng = C.n_glo_for(C.SPECS["baseline"])
    fit = fit_var(df, V, C.BASE_P, C.BASE_EXOG, n_glo=ng)
    ev = np.abs(np.linalg.eigvals(fit.companion()))
    print(f"\nbaseline fit: T={fit.T}, params/equation={fit.X.shape[1]}, "
          f"df={fit.T - fit.X.shape[1]}, max|eig|={ev.max():.4f}")
    if fit.is_soe:
        n_ex, n_v, pp = fit.n_exog, fit.n, fit.p
        k_glo = n_ex + ng * pp
        print(f"  SOE ON (n_glo={ng}): global equations use {k_glo} "
              f"regressors, domestic {fit.X.shape[1]}; "
              f"{ng * (n_v - ng) * pp} coefficients dropped in total")
        print(f"  max |Phi_12| relative = {fit.phi12_norm():.3e}")
        fit_un = fit_var(df, V, C.BASE_P, C.BASE_EXOG, n_glo=None)
        ev_un = np.abs(np.linalg.eigvals(fit_un.companion()))
        print(f"  max|eig| unrestricted = {ev_un.max():.4f} vs "
              f"SOE {ev.max():.4f}")
    mult, change, sd = I.episode_scale(df)
    print(f"episode scaling: {I.EPISODE_FROM}->{I.EPISODE_TO} moves s_t by "
          f"{change:.4f} asinh units = {mult:.2f} sd; gap sd = "
          f"{df.g_meandec_cpi.std():.4f}pp")
    json.dump({"T": int(fit.T), "df": int(fit.T - fit.X.shape[1]),
               "max_eig": float(ev.max()), "episode_sd_multiple": float(mult),
               "gap_sd": float(df.g_meandec_cpi.std())},
              open(OUT / "diagnostics_v3.json", "w"), indent=2)


def ladder(n_keep=600):
    """
    THE DECISION RUN. Four nested schemes on the baseline system.

    Read the output before choosing a baseline. The question the table answers
    is whether the narrative restriction is still needed once the exact zeros
    are imposed.

    DECISION RULE, fixed in advance so the choice is not made on the gap
    response:
      - if L1 self-attribution (FEVD of s_t at h=0) is at least 0.35 AND
        within about 0.10 of L2, DROP the narrative restriction. L1 becomes
        the baseline, the 2022-conditioning objection disappears, and the
        narrative scheme is reported as robustness.
      - otherwise KEEP it. L2 becomes the baseline and the text must say the
        restriction was necessary, with L1 reported to show why.
    Do not choose on which scheme gives the more congenial gap response.
    """
    spec, df, fit, quarters, zero_specs, _ = setup()
    ineq_specs = C.make_specs(spec["endog"], zeros=False)

    print("=== SCHEME LADDER ===")
    print("L0 signs, weak inequalities   (v1 scheme)")
    print("L1 signs, exact zeros         (the change on its own)")
    print("L2 exact zeros + narrative    (proposed v3 baseline)")
    print("L3 weak inequalities + narrative (v2 baseline)\n")

    lad = I.scheme_ladder(fit, spec["endog"], quarters, zero_specs,
                          ineq_specs, H=H, n_keep=n_keep)

    rows = []
    for label, res in lad.items():
        if res["kept"] == 0:
            print(f"{label}: NO ADMISSIBLE DRAWS"); continue
        I.check_zeros_ok(res, label)
        I.check_soe_ok(res, label)
        rows.append(I.summary_row(res, "g_meandec_cpi", label))
        r = rows[-1]
        print(f"{label:20s} acc={r['acceptance_rate']:7.3%}  "
              f"FEVD(s,h0)={r['fevd_s_h0']:.3f}  "
              f"gap h0={r['gap_h0']:+.4f} [{r['gap_h0_p16']:+.4f},"
              f"{r['gap_h0_p84']:+.4f}]  "
              f"h4={r['gap_h4']:+.4f} [{r['gap_h4_p16']:+.4f},"
              f"{r['gap_h4_p84']:+.4f}]"
              f"{'  SIG_h4' if r['sig_h4'] else ''}")

    out = pd.DataFrame(rows)
    _write(out, "ladder_v3")

    if {"L1_signs_zeros", "L2_zeros_narrative"} <= set(out.label):
        f1 = out.loc[out.label == "L1_signs_zeros", "fevd_s_h0"].iat[0]
        f2 = out.loc[out.label == "L2_zeros_narrative", "fevd_s_h0"].iat[0]
        print(f"\nDECISION INPUT: self-attribution L1={f1:.3f}, L2={f2:.3f}, "
              f"difference {f2 - f1:+.3f}")
        if f1 >= 0.35 and (f2 - f1) <= 0.10:
            print("  -> rule says DROP the narrative restriction; "
                  "set BASELINE_SCHEME='L1' below.")
        else:
            print("  -> rule says KEEP the narrative restriction; "
                  "L2 is the baseline.")
    print("\nStop here and read the table before running anything else.")


BASELINE_SCHEME = "L2"      # set to "L1" if the ladder says to drop narrative


def baseline(n_keep=2000):
    with_narr = BASELINE_SCHEME == "L2"
    spec, df, fit, quarters, specs, narr = setup(with_narrative=with_narr)
    mult, change, sd = I.episode_scale(df)
    print(f"=== BASELINE (scheme {BASELINE_SCHEME}, "
          f"narrative {'ON' if with_narr else 'OFF'}) ===")
    print(f"episode scaling: {change:.4f} asinh units = {mult:.2f} sd")

    res = I.identify(fit, specs, narr, H=H, n_keep=n_keep, want_hd=True)
    print(f"kept {res['kept']}/{res['draws']} draws, "
          f"acceptance {res['acceptance']:.4%}")
    if not I.check_zeros_ok(res, "baseline"):
        print("ABORT: the zeros are not binding. Do not report this run.")
        return
    if not I.check_soe_ok(res, "baseline"):
        print("ABORT: block exogeneity is not binding. Do not report this run.")
        return

    irf = I.response_table(res)
    _write(irf, "irf_baseline_v3")
    _write(I.response_table(res, scale=mult), "irf_baseline_v3_episode_scaled")
    _write(I.fevd_table(res), "fevd_baseline_v3")

    names = res["names"]
    gi, si = names.index("g_meandec_cpi"), names.index("s_t_asinh100")
    k = shock_index(res["shock_names"], "media_salience")

    g = irf[irf.variable == "g_meandec_cpi"]
    print("\nGap response, per 1sd shock (pp):")
    print(g[g.h <= 8][["h", "median", "p16", "p84", "excludes_zero"]]
          .round(4).to_string(index=False))

    fmed, _, _ = bands(res["fevd"][:, :, :, k])
    print(f"\nFEVD(s_t) h=0: {fmed[0, si]:.3f}   "
          f"FEVD(gap) h=8: {fmed[8, gi]:.3f}")

    # impact response of every variable, the table that caught the v1 problem
    med, lo, hi = bands(res["Theta"][:, :, :, k])
    print("\nImpact (h=0) response of every variable:")
    for i, v in enumerate(names):
        print(f"  {v:22s} {med[0,i]:+10.4f} [{lo[0,i]:+.4f}, {hi[0,i]:+.4f}]")

    if res["hd"] is not None:
        hm, hl, hh = bands(res["hd"][:, :, gi, k])
        hd = pd.DataFrame({"quarter": quarters, "salience_contrib": hm,
                           "p16": hl, "p84": hh})
        for j, sh in enumerate(res["shock_names"]):
            hd[f"contrib_{sh}"] = np.percentile(res["hd"][:, :, gi, j],
                                                50, axis=0)
        _write(hd, "histdecomp_gap_v3")
        print("\nHistorical decomposition, 2021-2023:")
        print(hd[hd.quarter.str[:4].isin(["2021", "2022", "2023"])]
              [["quarter", "salience_contrib", "p16", "p84"]]
              .round(3).to_string(index=False))

    np.save(OUT / "theta_baseline_v3.npy", res["Theta"])


def sensitivity(n_keep=400):
    """The two restrictions considered and rejected, re-tested under zeros."""
    spec, df, fit, quarters, specs, narr = setup()
    shock_names = [sp.name for sp in specs]
    gi = fit.names.index("g_meandec_cpi")
    si = fit.names.index("s_t_asinh100")
    rows = []

    for thr in (0.0, 0.2, 0.4, 0.5, 0.6):
        fev = I.fevd_share_checker(fit.names, "s_t_asinh100", shock_names,
                                   threshold=thr)
        extra = I.both(narr, fev) if thr > 0 else narr
        res = I.identify(fit, specs, extra, H=H, n_keep=n_keep, want_hd=False)
        if res["kept"] == 0:
            print(f"  fevd>={thr}: no draws"); continue
        r = I.summary_row(res, "g_meandec_cpi", f"fevd_{thr}")
        r.update({"restriction": "fevd_threshold", "value": thr})
        rows.append(r)
        print(f"  fevd>={thr:.1f}  acc={r['acceptance_rate']:7.3%}  "
              f"FEVD(s,h0)={r['fevd_s_h0']:.3f}  h4={r['gap_h4']:+.4f}")

    for through in (1, 2, 3, 4):
        sp = I.persistence_specs(specs, spec["endog"], through=through)
        res = I.identify(fit, sp, narr, H=H, n_keep=n_keep, want_hd=False)
        if res["kept"] == 0:
            print(f"  persist h<={through}: no draws"); continue
        r = I.summary_row(res, "g_meandec_cpi", f"persist_{through}")
        r.update({"restriction": "persistence_through_h", "value": through})
        rows.append(r)
        print(f"  persist<=h{through} acc={r['acceptance_rate']:7.3%}  "
              f"h4={r['gap_h4']:+.4f}")

    _write(pd.DataFrame(rows), "sensitivity_v3")


EPISODES = {"2008_financial_crisis": ("2008Q3", "2008Q4"),
            "2011_rpi_peak":         ("2011Q3", "2011Q4"),
            "2015_quiet_control":    ("2015Q2", "2015Q3"),
            "2022_cost_of_living":   ("2022Q2", "2022Q3")}


def episodes(n_keep=300):
    """
    Point the narrative restriction at each episode in turn.

    This is a comparison across episodes, NOT a placebo test. 2011Q3-Q4 carried
    RPI inflation above five per cent, so reproducing the result there is
    corroboration. 2015 is the only genuine quiet control.
    """
    spec, df, fit, quarters, specs, _ = setup(with_narrative=False)
    shock_names = [sp.name for sp in specs]
    rows = []
    for name, targ in EPISODES.items():
        narr = I.narrative_checker(fit, fit.names, quarters, spec["endog"][3],
                                   shock_names, targets=targ, strict=True)
        res = I.identify(fit, specs, narr, H=H, n_keep=n_keep, want_hd=False,
                         max_draws=400_000)
        if res["kept"] == 0:
            print(f"  {name}: no admissible draws"); continue
        I.check_zeros_ok(res, name)
        r = I.summary_row(res, "g_meandec_cpi", name,
                          {"episode": name, "targets": "/".join(targ)})
        rows.append(r)
        print(f"  {name:24s} acc={r['acceptance_rate']:7.3%} "
              f"h0={r['gap_h0']:+.4f}{'*' if r['sig_h0'] else ' '} "
              f"h4={r['gap_h4']:+.4f}{'*' if r['sig_h4'] else ' '}")
    _write(pd.DataFrame(rows), "episodes_v3")


GRIDFILE = OUT / "grid_v3.csv"


def grid(n_keep=300, max_draws=800_000, only=None, resume=True):
    """Every specification, one row each. Resumable: rerun to fill gaps."""
    done = set()
    if resume and GRIDFILE.exists():
        done = set(pd.read_csv(GRIDFILE).spec.unique())
        print(f"resuming; {len(done)} specs already recorded")

    todo = [only] if only else [k for k in C.SPECS if k != "baseline"]
    for name in todo:
        if name in done and not only:
            continue
        spec = C.SPECS[name]
        try:
            df = C.frame(spec)
            fit = fit_var(df, spec["endog"], spec.get("p", 1),
                          spec.get("exog", C.BASE_EXOG),
                          n_glo=C.n_glo_for(spec))
            quarters = np.asarray(df["quarter"].iloc[-fit.T:])
            specs = C.specs_for(spec)
            shock_names = [sp.name for sp in specs]

            if BASELINE_SCHEME == "L2":
                narr = I.narrative_checker(fit, fit.names, quarters,
                                           spec["endog"][3], shock_names,
                                           strict=False)
                if narr.missing:
                    scheme = (f"NO NARRATIVE: {'/'.join(narr.missing)} "
                              f"outside sample")
                    extra = None
                else:
                    scheme = "zeros + narrative"
                    extra = narr
            else:
                scheme, extra = "zeros only", None

            res = I.identify(fit, specs, extra, H=H, n_keep=n_keep,
                             max_draws=max_draws, want_hd=False)
            if res["kept"] == 0:
                _append_grid([{"spec": name, "status": "no draws",
                               "draws": res["draws"], "note": spec["note"]}])
                print(f"  [{name}] NO ADMISSIBLE DRAWS after {res['draws']}")
                continue

            rows = []
            targets = [v for v in spec["endog"]
                       if v.startswith("g_") or v.startswith("E_")]
            for tv in targets:
                r = I.summary_row(res, tv, name,
                                  {"spec": name, "status": "ok",
                                   "scheme": scheme, "T": fit.T,
                                   "note": spec["note"],
                                   "kept_target_met": res["kept"] >= n_keep,
                                   "hit_max_draws": res["draws"] >= max_draws})
                rows.append(r)
                print(f"  [{name:22s}/{tv:18s}] T={fit.T:3d} "
                      f"acc={res['acceptance']:7.3%} "
                      f"FEVD(s,h0)={r['fevd_s_h0']:.3f} "
                      f"h0={r['gap_h0']:+.4f} h4={r['gap_h4']:+.4f}"
                      f"{'  PARTIAL' if res['kept'] < n_keep else ''}")
            if "NO NARRATIVE" in scheme:
                print(f"      note: {scheme}")
            _append_grid(rows)
        except Exception as e:
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            _append_grid([{"spec": name,
                           "status": f"FAILED {type(e).__name__}",
                           "note": spec.get("note", "")}])


def _append_grid(rows):
    d = pd.DataFrame(rows)
    if GRIDFILE.exists():
        old = pd.read_csv(GRIDFILE)
        old = old[~old.spec.isin(d.spec.unique())]
        d = pd.concat([old, d], ignore_index=True)
    d.to_csv(GRIDFILE, index=False)


def r16(n_keep=2000):
    """The split system. E^H and E^P separately, with the per-draw difference."""
    spec, df, fit, quarters, specs, narr = setup("R16_split_EH_EP")
    mult, _, _ = I.episode_scale(df)
    res = I.identify(fit, specs, narr, H=H, n_keep=n_keep, want_hd=True)
    print(f"kept {res['kept']}/{res['draws']}, acc {res['acceptance']:.4%}")
    if res["kept"] == 0:
        return
    I.check_zeros_ok(res, "R16")

    _write(I.response_table(res), "irf_r16_v3")
    d = I.perdraw_difference(res, "E_H_mean_dec_h10", "E_P_cpi")
    _write(d, "r16_perdraw_gap_v3")
    _write(I.perdraw_difference(res, "E_H_mean_dec_h10", "E_P_cpi",
                                scale=mult), "r16_perdraw_gap_v3_episode")
    print("\nPer-draw E^H - E^P:")
    print(d[d.h <= 8].round(4).to_string(index=False))

    k = shock_index(res["shock_names"], "media_salience")
    ehi = res["names"].index("E_H_mean_dec_h10")
    epi = res["names"].index("E_P_cpi")
    hm_h, _, _ = bands(res["hd"][:, :, ehi, k])
    hm_p, _, _ = bands(res["hd"][:, :, epi, k])
    hd = pd.DataFrame({"quarter": quarters, "salience_to_EH": hm_h,
                       "salience_to_EP": hm_p})
    _write(hd, "histdecomp_r16_v3")
    print(f"\ncorrelation of the two HD paths: "
          f"{np.corrcoef(hm_h, hm_p)[0,1]:.4f}")


def eri(n_keep=400):
    """
    The domestic block. Answers: is UK inflation required to be globally
    determined?

    R17 puts the sterling ERI in the system with the same three labelled
    shocks. R18 additionally labels a sterling depreciation shock. R19 adds
    the ERI to the split system.

    R17 is the primary answer. If the gap response survives it, the reply to
    the referee is that the model does not require UK inflation to be globally
    determined, and that adding an explicit domestic cost-push channel leaves
    the result where it was.
    """
    rows = []
    for name in ("R17_eri", "R18_eri_sterling_shock", "R19_eri_split"):
        spec = C.SPECS[name]
        df = C.frame(spec)
        fit = fit_var(df, spec["endog"], spec.get("p", 1),
                      spec.get("exog", C.BASE_EXOG),
                      n_glo=C.n_glo_for(spec))
        quarters = np.asarray(df["quarter"].iloc[-fit.T:])
        specs = C.specs_for(spec)
        shock_names = [sp.name for sp in specs]
        narr = None
        if BASELINE_SCHEME == "L2":
            narr = I.narrative_checker(fit, fit.names, quarters,
                                       spec["endog"][3], shock_names,
                                       strict=True)
        print(f"\n=== {name} ({fit.n} variables, "
              f"{len(specs)} labelled shocks) ===")
        res = I.identify(fit, specs, narr, H=H, n_keep=n_keep,
                         max_draws=2_000_000, want_hd=True)
        print(f"kept {res['kept']}/{res['draws']}, acc {res['acceptance']:.4%}")
        if res["kept"] == 0:
            print("  no admissible draws; raise max_draws or drop a "
                  "restriction, and say so in the text")
            continue
        I.check_zeros_ok(res, name)

        _write(I.response_table(res), f"irf_{name}_v3")
        for tv in [v for v in spec["endog"]
                   if v.startswith("g_") or v.startswith("E_")]:
            rows.append(I.summary_row(res, tv, name, {"spec": name}))

        # the ERI response to a salience shock, and vice versa where labelled
        irf = I.response_table(res)
        e = irf[irf.variable == C.ERI]
        print("ERI response to a salience shock:")
        print(e[e.h <= 4][["h", "median", "p16", "p84", "excludes_zero"]]
              .round(4).to_string(index=False))

        if "sterling_depreciation" in shock_names:
            st = I.response_table(res, shock="sterling_depreciation")
            _write(st, f"irf_{name}_sterling_shock_v3")
            g = st[st.variable == "g_meandec_cpi"]
            print("Gap response to a STERLING shock:")
            print(g[g.h <= 8][["h", "median", "p16", "p84", "excludes_zero"]]
                  .round(4).to_string(index=False))

        if name == "R19_eri_split" and res["kept"]:
            _write(I.perdraw_difference(res, "E_H_mean_dec_h10", "E_P_cpi"),
                   "r19_perdraw_gap_v3")

    if rows:
        _write(pd.DataFrame(rows), "eri_summary_v3")


def soe(n_keep=600):
    """
    THE SOE DECISION RUN. Read this before rewriting any prose.

    Estimates the baseline system under the four-scheme ladder TWICE, once
    with the unrestricted reduced form and once with block exogeneity, and
    reports the three things that decide whether the restriction was hygiene
    or a real change to the story:

      1. ACCEPTANCE RATE. If it collapses by an order of magnitude relative to
         the unrestricted fit, the identified set is being carved out by a
         handful of episodes and 2,000 retained draws stops being reassuring.
         That needs saying in the text whatever the IRFs look like.

      2. L0 vs L1 UNDER SOE. With domestic feedback removed, the difference
         between weak inequalities and exact zeros is now the marginal
         contribution of the exact-zero assumption alone, uncontaminated by
         the lag dynamics. This comparison is cleaner than it was before.

      3. THE GAP RESPONSE. Unchanged means the earlier commodity and activity
         declines were a property of the unrestricted lag dynamics and nothing
         was resting on them. Materially changed means the headline was
         leaning on the global block, which is worth knowing now.

    The global-block IRF column is the complementarity check: zero at every
    horizon under L1/L2, non-zero and decaying under L0/L3. If L0 shows zeros
    too, something is wrong with the reduced form, not with the scheme.
    """
    spec = C.SPECS["baseline"]
    df = C.frame(spec)
    endog = spec["endog"]
    ineq_specs = C.make_specs(endog, zeros=False)
    zero_specs = C.make_specs(endog, zeros=True)

    print("=== SOE COMPARISON: unrestricted vs block-exogenous reduced form ===")
    rows = []
    for tag, ng in (("unrestricted", None), ("soe", C.N_GLO)):
        fit = fit_var(df, endog, spec.get("p", 1),
                      spec.get("exog", C.BASE_EXOG), n_glo=ng)
        quarters = np.asarray(df["quarter"].iloc[-fit.T:])
        ev = np.abs(np.linalg.eigvals(fit.companion())).max()
        print(f"\n--- {tag}  (T={fit.T}, max|eig|={ev:.4f}) ---")

        lad = I.scheme_ladder(fit, endog, quarters, zero_specs,
                              ineq_specs, H=H, n_keep=n_keep)
        for label, res in lad.items():
            if res["kept"] == 0:
                print(f"  {label}: NO ADMISSIBLE DRAWS")
                rows.append({"reduced_form": tag, "label": label,
                             "status": "no draws", "draws": res["draws"]})
                continue
            r = I.summary_row(res, "g_meandec_cpi", label)
            r.update({"reduced_form": tag, "status": "ok", "max_eig": ev,
                      "T": fit.T})
            # global-block response, the complementarity check
            k = shock_index(res["shock_names"], "media_salience")
            gm, _, _ = bands(res["Theta"][:, :, :, k])
            r["q_h4"] = gm[4, 0]
            r["a_h4"] = gm[4, 1]
            r["glob_max_abs"] = float(np.abs(gm[:, :C.N_GLO]).max())
            rows.append(r)
            print(f"  {label:20s} acc={r['acceptance_rate']:7.3%}  "
                  f"FEVD(s,h0)={r['fevd_s_h0']:.3f}  "
                  f"gap h0={r['gap_h0']:+.4f} h4={r['gap_h4']:+.4f}  "
                  f"q(h4)={r['q_h4']:+.3f} a(h4)={r['a_h4']:+.3f}  "
                  f"|glob|max={r['glob_max_abs']:.2e}")

    out = pd.DataFrame(rows)
    _write(out, "soe_comparison_v3")

    ok = out[out.status == "ok"] if "status" in out else out
    if len(ok):
        print("\n--- READ THESE TWO NUMBERS ---")
        for label in ("L1_signs_zeros", "L2_zeros_narrative"):
            sub = ok[ok.label == label]
            if len(sub) == 2:
                a = sub[sub.reduced_form == "unrestricted"].iloc[0]
                b = sub[sub.reduced_form == "soe"].iloc[0]
                ratio = (b.acceptance_rate / a.acceptance_rate
                         if a.acceptance_rate else float("nan"))
                print(f"{label}: acceptance {a.acceptance_rate:.3%} -> "
                      f"{b.acceptance_rate:.3%}  (x{ratio:.2f})"
                      f"{'   *** ORDER-OF-MAGNITUDE DROP, SAY SO IN THE TEXT ***' if ratio < 0.1 else ''}")
                print(f"{label}: gap h4 {a.gap_h4:+.4f} -> {b.gap_h4:+.4f}  "
                      f"(change {b.gap_h4 - a.gap_h4:+.4f})")
        s = ok[ok.reduced_form == "soe"]
        l0 = s[s.label == "L0_signs_weak"]
        l1 = s[s.label == "L1_signs_zeros"]
        if len(l0) and len(l1):
            print(f"\nL0 vs L1 under SOE (marginal value of the exact zero):")
            print(f"  FEVD(s,h0) {l0.fevd_s_h0.iat[0]:.3f} -> "
                  f"{l1.fevd_s_h0.iat[0]:.3f}")
            print(f"  gap h4     {l0.gap_h4.iat[0]:+.4f} -> "
                  f"{l1.gap_h4.iat[0]:+.4f}")
            print(f"  |global IRF|max {l0.glob_max_abs.iat[0]:.2e} (expected "
                  f"non-zero) -> {l1.glob_max_abs.iat[0]:.2e} "
                  f"(expected machine precision)")
    print("\nStop here and read the table before rewriting anything.")


BOOT_STORE = OUT / "boot_theta_v3.npy"
BOOT_COUNT = OUT / "boot_count_v3.txt"


def bootstrap(n_reps=300, keep_per_boot=4, checkpoint_every=25):
    """
    Wild bootstrap (Rademacher), resumable. Pooled draws are appended to disk
    after every checkpoint, so an interrupted run loses at most one chunk.

    Rerun the same command to add more replications; the seed offset advances
    so the additional replications are new draws rather than repeats.
    """
    with_narr = BASELINE_SCHEME == "L2"
    spec, df, fit, quarters, specs, narr = setup(with_narrative=with_narr)
    n, k = fit.n, fit.X.shape[1]
    offset = int(open(BOOT_COUNT).read()) if BOOT_COUNT.exists() else 0
    rng = np.random.default_rng(20260816 + offset)
    print(f"=== WILD BOOTSTRAP ({n_reps} reps, offset {offset}) ===")

    buf, fails = [], 0
    for b in range(n_reps):
        e = rng.choice([-1.0, 1.0], size=(fit.T, 1))
        Yb = fit.X @ fit.B + fit.U * e
        # restricted_ols, NOT lstsq. Under SOE an unrestricted refit of each
        # replication would silently drop block exogeneity and return bands
        # for a model that was never estimated. With n_glo=None it reduces to
        # exactly the previous unrestricted lstsq call.
        Bb = restricted_ols(fit.X, Yb, n, fit.p, fit.n_exog, fit.n_glo)
        Ub = Yb - fit.X @ Bb
        fitb = VARFit(Y=Yb, X=fit.X, B=Bb, U=Ub,
                      Sigma=(Ub.T @ Ub) / (fit.T - k), p=fit.p, n=n,
                      n_exog=fit.n_exog, names=fit.names, n_glo=fit.n_glo)
        nb = None
        if with_narr:
            nb = I.narrative_checker(fitb, fitb.names, quarters,
                                     spec["endog"][3],
                                     [sp.name for sp in specs], strict=True)
        try:
            r = I.identify(fitb, specs, nb, H=H, n_keep=keep_per_boot,
                           max_draws=800_000, seed=int(rng.integers(1e9)),
                           want_hd=False)
        except np.linalg.LinAlgError:
            fails += 1; continue
        if r["kept"] == 0:
            fails += 1; continue
        buf.append(r["Theta"])

        if (b + 1) % checkpoint_every == 0:
            _flush_boot(buf); buf = []
            print(f"  checkpoint at rep {b+1}/{n_reps}, failures {fails}")
    _flush_boot(buf)
    open(BOOT_COUNT, "w").write(str(offset + n_reps))

    Th = np.load(BOOT_STORE)
    print(f"pooled draws now {Th.shape[0]}, failures this batch {fails}")
    kk = shock_index([sp.name for sp in specs], "media_salience")
    gi = fit.names.index("g_meandec_cpi")
    med, lo, hi = bands(Th[:, :, gi, kk])
    rows = [{"h": h, "median": med[h], "p16": lo[h], "p84": hi[h],
             "excludes_zero": bool((lo[h] > 0) or (hi[h] < 0))}
            for h in range(len(med))]
    _write(pd.DataFrame(rows), "irf_bootstrap_v3")
    print("\nPooled gap response (identification + sampling):")
    for h in (0, 1, 2, 4, 8, 12):
        s = "SIG" if (lo[h] > 0 or hi[h] < 0) else ""
        print(f"  h={h:2d}  {med[h]:+.4f}  [{lo[h]:+.4f}, {hi[h]:+.4f}]  {s}")


def _flush_boot(buf):
    if not buf:
        return
    new = np.concatenate(buf, axis=0)
    if BOOT_STORE.exists():
        new = np.concatenate([np.load(BOOT_STORE), new], axis=0)
    np.save(BOOT_STORE, new)


def combine():
    """Every CSV in results_v3 into one workbook, with a contents sheet."""
    files = sorted(OUT.glob("*.csv"))
    if not files:
        print("no CSVs found"); return
    path = OUT / "results_v3.xlsx"
    contents = []
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame({"sheet": ["Contents"]}).to_excel(
            xl, sheet_name="Contents", index=False)
        for f in files:
            d = pd.read_csv(f)
            sheet = f.stem[:31]
            d.to_excel(xl, sheet_name=sheet, index=False)
            contents.append({"sheet": sheet, "source": f.name,
                             "rows": len(d), "cols": d.shape[1]})
        pd.DataFrame(contents).to_excel(xl, sheet_name="Contents", index=False)
    print(f"wrote {path} with {len(files)} sheets")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ladder"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "diagnostics":  diagnostics()
    elif cmd == "soe":        soe()
    elif cmd == "ladder":     ladder()
    elif cmd == "baseline":   baseline()
    elif cmd == "sensitivity": sensitivity()
    elif cmd == "episodes":   episodes()
    elif cmd == "grid":       grid(only=arg)
    elif cmd == "r16":        r16()
    elif cmd == "eri":        eri()
    elif cmd == "bootstrap":  bootstrap(int(arg) if arg else 300)
    elif cmd == "combine":    combine()
    else:
        print(__doc__)
