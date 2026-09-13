"""
Rotation sampling, the narrative restriction and the scheme ladder.

Dispatch is on the restrictions themselves: if any ShockSpec carries zeros the
ARW construction is used, so a flag cannot disagree with the scheme it labels.
Every run reports the largest zero violation; anything above 1e-10 relative
means the construction is not binding and the run should not be reported.

Nothing indexes the salience shock positionally, since R18 has four labelled
shocks. Callers pass shock_names and look up "media_salience".

scheme_ladder() estimates the same system under four nested schemes, so the
narrative restriction is judged on the acceptance rate and self-attribution
rather than retained by habit.

Block exogeneity lives in Phi, not in B_0, so the rotation loop, the matcher
and the ARW construction are untouched by it; it arrives here only as
diagnostics carried into the summary rows.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from svar_lib_v3 import (ma_coefficients, irf_from_B0, fevd_from_theta,
                         draw_rotation, build_rotation_with_zeros,
                         match_shocks, historical_decomposition, bands,
                         zero_violation, soe_violation, shock_index)

EPISODE_FROM, EPISODE_TO = "2021Q4", "2022Q3"
NARRATIVE_TARGETS = ("2022Q2", "2022Q3")
ZERO_ALARM = 1e-10          # relative; above this the run is not reportable


# Extra restrictions

def persistence_specs(base_specs, endog, through=3):
    """Sensitivity only. Extend salience positivity to horizon `through`."""
    s = copy.deepcopy(base_specs)
    i = [k for k, sp in enumerate(s) if sp.name == "media_salience"][0]
    for h in range(2, through + 1):
        s[i].restrictions[h] = {endog[3]: "+"}
    return s


def narrative_checker(fit, names, quarters, svar_name, shock_names,
                      targets=NARRATIVE_TARGETS, strict=True):
    """
    Antolin-Diaz and Rubio-Ramirez (2018): the media salience shock must be
    the largest single contributor to s_t in the named quarters.

    strict=True makes an out-of-sample target a hard error rather than a
    silently vacuous restriction. In v1 that bug let R13_pre2020 run under a
    weaker scheme than its label claimed.
    """
    si = names.index(svar_name)
    sk = shock_index(shock_names, "media_salience")
    qs = [str(q) for q in quarters]
    idx = [i for i, q in enumerate(qs) if q in targets]
    missing = [t for t in targets if t not in qs]
    if missing and strict:
        raise ValueError(
            f"narrative targets {missing} are outside this sample "
            f"({qs[0]}..{qs[-1]}); pass strict=False to run without them")

    def check(fit_, B0, Theta, cols):
        if not idx:
            return True
        # Only the target quarters and only the s_t row are needed, so the
        # full (T, n, n) historical decomposition is not computed. Pure
        # optimisation: the accept/reject decision is identical.
        eps = fit_.U @ np.linalg.inv(B0).T
        H_ = Theta.shape[0] - 1
        for t in idx:
            contrib = np.zeros(Theta.shape[2])
            for j in range(0, min(t, H_) + 1):
                contrib += Theta[j, si, :] * eps[t - j, :]
            if np.argmax(np.abs(contrib)) != cols[sk]:
                return False
        return True

    check.n_targets = len(idx)
    check.missing = missing
    check.targets = tuple(targets)
    return check


def fevd_share_checker(names, svar_name, shock_names,
                       threshold=0.5, horizon=0):
    """Sensitivity only. Selecting on a reported outcome; not in the baseline."""
    si = names.index(svar_name)
    sk = shock_index(shock_names, "media_salience")

    def check(fit_, B0, Theta, cols):
        return fevd_from_theta(Theta)[horizon, si, cols[sk]] >= threshold
    return check


def both(*checkers):
    def check(*a):
        return all(c(*a) for c in checkers)
    return check


# Rotation loop

def identify(fit, specs, extra=None, H=20, n_keep=1000,
             max_draws=1_500_000, seed=20260816, want_hd=True):
    """
    Returns a dict with Theta, fevd, hd, acceptance, draws, shock_names and
    the worst zero violation observed.

    Whether the ARW construction is used is decided by the specs themselves,
    never by a caller-supplied flag.
    """
    use_zeros = any(sp.n_zeros > 0 for sp in specs)
    rng = np.random.default_rng(seed)
    Psi = ma_coefficients(fit, H)
    P = np.linalg.cholesky(fit.Sigma)
    names = fit.names
    shock_names = [sp.name for sp in specs]

    T_, F_, D_ = [], [], []
    draws = 0
    worst_zero = 0.0
    worst_soe = 0.0
    while len(T_) < n_keep and draws < max_draws:
        draws += 1
        if use_zeros:
            Q = build_rotation_with_zeros(P, Psi, specs, names, rng)
            if Q is None:
                continue
        else:
            Q = draw_rotation(fit.n, rng)
        B0 = P @ Q
        Th = irf_from_B0(Psi, B0)
        m = match_shocks(Th, specs, names)
        if m is None:
            continue
        cols, signs = m
        B0s = B0.copy()
        for c, s in zip(cols, signs):
            B0s[:, c] = s * B0[:, c]
        Th2 = irf_from_B0(Psi, B0s)
        if extra is not None and not extra(fit, B0s, Th2, cols):
            continue
        worst_zero = max(worst_zero, zero_violation(Th2, specs, names, cols))
        if fit.is_soe:
            ks = shock_index(shock_names, "media_salience")
            worst_soe = max(worst_soe,
                            soe_violation(Th2, fit.n_glo, cols[ks]))
        T_.append(Th2[:, :, cols])
        F_.append(fevd_from_theta(Th2)[:, :, cols])
        if want_hd:
            D_.append(historical_decomposition(fit, B0s, Th2)[:, :, cols])

    return {
        "Theta": np.array(T_),
        "fevd": np.array(F_),
        "hd": np.array(D_) if want_hd and D_ else None,
        "acceptance": len(T_) / draws if draws else 0.0,
        "draws": draws,
        "kept": len(T_),
        "names": names,
        "shock_names": shock_names,
        "use_zeros": use_zeros,
        "worst_zero": worst_zero,
        "is_soe": bool(fit.is_soe),
        "n_glo": fit.n_glo,
        "phi12_norm": float(fit.phi12_norm()) if fit.is_soe else float("nan"),
        "soe_worst": worst_soe if fit.is_soe else float("nan"),
    }


def check_zeros_ok(res, label="") -> bool:
    """Print and test the zero-violation guard."""
    if not res["use_zeros"]:
        return True
    w = res["worst_zero"]
    ok = w <= ZERO_ALARM
    flag = "OK" if ok else "*** ZEROS NOT BINDING ***"
    print(f"    zero check{(' ' + label) if label else ''}: "
          f"max relative violation {w:.3e}  {flag}")
    return ok


def check_soe_ok(res, label="") -> bool:
    """
    Print the block-exogeneity diagnostics and test the two that are testable.

    phi12_norm is ALWAYS expected at machine precision when SOE is on: it is a
    property of the estimated reduced form and has nothing to do with the
    scheme. Failure means restricted_ols was bypassed somewhere, which is the
    exact class of bug this project has hit before.

    soe_worst is expected at machine precision only under the exact-zero
    schemes. Under weak inequalities it is reported, not judged, because
    b_glo is free to be non-zero there by design.
    """
    if not res.get("is_soe"):
        return True
    p12, sw = res["phi12_norm"], res["soe_worst"]
    ok_p12 = p12 <= ZERO_ALARM
    lab = (" " + label) if label else ""
    print(f"    SOE check{lab}: max |Phi_12| (rel) {p12:.3e}  "
          f"{'OK' if ok_p12 else '*** BLOCK EXOGENEITY NOT IMPOSED ***'}")
    if res["use_zeros"]:
        ok_sw = sw <= ZERO_ALARM
        print(f"    SOE check{lab}: max global IRF (rel) {sw:.3e}  "
              f"{'OK, zero at all horizons' if ok_sw else '*** GLOBAL BLOCK MOVES ***'}")
        return ok_p12 and ok_sw
    print(f"    SOE check{lab}: max global IRF (rel) {sw:.3e}  "
          f"(non-zero EXPECTED under weak inequalities; b_glo is free)")
    return ok_p12


# The scheme ladder

def scheme_ladder(fit, endog, quarters, base_zero_specs, base_ineq_specs,
                  H=20, n_keep=600, max_draws=1_200_000, seed=20260816):
    """
    Estimate the same system under four nested schemes.

      L0  signs only, weak inequalities            (the v1 scheme)
      L1  signs only, exact zeros                  (the change on its own)
      L2  exact zeros + narrative restriction      (proposed v3 baseline)
      L3  weak inequalities + narrative            (the v2 baseline)

    The point of L1 is to find out whether the narrative restriction is still
    doing work once the zeros are imposed. If L1 already delivers acceptable
    self-attribution to s_t, the narrative restriction can be demoted to
    robustness and the 2022-conditioning objection disappears with it.
    """
    out = {}
    for label, specs, use_narr in (
            ("L0_signs_weak",       base_ineq_specs, False),
            ("L1_signs_zeros",      base_zero_specs, False),
            ("L2_zeros_narrative",  base_zero_specs, True),
            ("L3_weak_narrative",   base_ineq_specs, True)):
        shock_names = [sp.name for sp in specs]
        extra = None
        if use_narr:
            extra = narrative_checker(fit, fit.names, quarters, endog[3],
                                      shock_names, strict=True)
        res = identify(fit, specs, extra, H=H, n_keep=n_keep,
                       max_draws=max_draws, seed=seed, want_hd=True)
        res["label"] = label
        out[label] = res
    return out


# Scaling and reporting helpers

def episode_scale(df, svar_name="s_t_asinh100",
                  frm=EPISODE_FROM, to=EPISODE_TO):
    """
    How many one-standard-deviation shocks the 2022 episode corresponds to.
    Returns (multiplier, change_in_asinh_units, sd). NaNs when an endpoint
    is outside the sample, so episode scaling is simply unavailable there.
    """
    s = df.set_index("quarter")[svar_name]
    if frm not in s.index or to not in s.index:
        return np.nan, np.nan, float(df[svar_name].std())
    change = float(s.loc[to] - s.loc[frm])
    sd = float(df[svar_name].std())
    return change / sd, change, sd


def response_table(res, horizons=None, scale=1.0, shock="media_salience"):
    """Median and 68% bands for every variable, optionally rescaled."""
    k = shock_index(res["shock_names"], shock)
    Theta = res["Theta"]
    horizons = horizons if horizons is not None else range(Theta.shape[1])
    med, lo, hi = bands(Theta[:, :, :, k])
    rows = []
    for h in horizons:
        for i, v in enumerate(res["names"]):
            rows.append({"h": h, "variable": v,
                         "median": scale * med[h, i],
                         "p16": scale * lo[h, i], "p84": scale * hi[h, i],
                         "excludes_zero": (lo[h, i] > 0) or (hi[h, i] < 0)})
    return pd.DataFrame(rows)


def fevd_table(res, shock="media_salience"):
    k = shock_index(res["shock_names"], shock)
    fmed, flo, fhi = bands(res["fevd"][:, :, :, k])
    return pd.DataFrame([{"h": h, "variable": v, "median_share": fmed[h, i],
                          "p16": flo[h, i], "p84": fhi[h, i]}
                         for h in range(fmed.shape[0])
                         for i, v in enumerate(res["names"])])


def perdraw_difference(res, a, b, scale=1.0, shock="media_salience"):
    """
    Per-draw difference between two variables' responses, e.g. E^H - E^P.
    This is the correct object; differencing two medians is not.
    """
    k = shock_index(res["shock_names"], shock)
    ia, ib = res["names"].index(a), res["names"].index(b)
    d = res["Theta"][:, :, ia, k] - res["Theta"][:, :, ib, k]
    med, lo, hi = bands(d)
    return pd.DataFrame({
        "h": np.arange(len(med)), "median": scale * med,
        "p16": scale * lo, "p84": scale * hi,
        "excludes_zero": (lo > 0) | (hi < 0)})


def summary_row(res, target, label, extra_fields=None):
    """One row of the comparison tables: acceptance, self-attribution, gap."""
    k = shock_index(res["shock_names"], "media_salience")
    names = res["names"]
    si = names.index([v for v in names if v.startswith("s_t")][0])
    gi = names.index(target)
    med, lo, hi = bands(res["Theta"][:, :, gi, k])
    fmed, _, _ = bands(res["fevd"][:, :, :, k])
    row = {
        "label": label, "target": target,
        "kept": res["kept"], "draws": res["draws"],
        "acceptance_rate": res["acceptance"],
        "uses_zeros": res["use_zeros"], "worst_zero": res["worst_zero"],
        "is_soe": res.get("is_soe", False),
        "phi12_norm": res.get("phi12_norm", float("nan")),
        "soe_worst": res.get("soe_worst", float("nan")),
        "fevd_s_h0": fmed[0, si], "fevd_gap_h8": fmed[8, gi],
        "gap_h0": med[0], "gap_h0_p16": lo[0], "gap_h0_p84": hi[0],
        "gap_h4": med[4], "gap_h4_p16": lo[4], "gap_h4_p84": hi[4],
        "gap_h8": med[8], "gap_h8_p16": lo[8], "gap_h8_p84": hi[8],
        "sig_h0": bool((lo[0] > 0) or (hi[0] < 0)),
        "sig_h4": bool((lo[4] > 0) or (hi[4] < 0)),
    }
    if extra_fields:
        row.update(extra_fields)
    return row
