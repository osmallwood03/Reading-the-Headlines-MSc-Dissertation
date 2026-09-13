"""
Data loading, bridging, restrictions and the specification grid.

The baseline imposes exact zeros on the impact response of world commodity
prices and global real activity to a salience shock. A weak inequality is not
the conservative version of that: the extra region it admits consists of
rotations in which a UK cost-of-living news shock lowers world prices and
global activity within the quarter, and with corr(s_t, pi_t) = 0.88 that is
the leakage the design exists to shut. The weak-inequality scheme survives as
R00 and is reported as robustness.

Block exogeneity restricts the commodity and activity equations to global lags
only (Cushman and Zha 1997; Zha 1999). It is complementary to the zeros rather
than redundant: the zeros shut the contemporaneous channel, block exogeneity
shuts the feedback one quarter later. R20 is the unrestricted reduced form,
retained as robustness.

The seventh-variable specifications add the sterling effective exchange rate.
R17 keeps the same three labelled shocks, R18 additionally labels a sterling
depreciation shock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from svar_lib_v3 import ShockSpec

DATA = Path("df_var_ready_v3.xlsx")     # produced by build_data_v3.py
OUT = Path("results_v3"); OUT.mkdir(exist_ok=True)
FIGS = Path("figures_v3"); FIGS.mkdir(exist_ok=True)


# Data loading and bridging

def load(bridge: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Load the VAR-ready file and bridge the E_P_cpi gaps.

    E_P_cpi is missing in six quarters:
      2003Q1-Q4        the HMT workbook carries no CPI forecast before 2004
      2017Q2, 2019Q4   documented exception cells

    Treatment:
      2003Q1-Q4       E_P_RPI minus the mean RPI-CPI forecast wedge over
                      2004Q1-2007Q4, which is 0.7594pp. This is a cross-index
                      imputation, not an interpolation.
      2017Q2, 2019Q4  linear interpolation of E_P_cpi

    Set bridge=False for the shorter complete-case sample, spec R11.
    """
    df = pd.read_excel(DATA)
    df["quarter"] = df["quarter"].astype(str)
    df = df.sort_values("quarter").reset_index(drop=True)
    df["period"] = pd.PeriodIndex(df["quarter"], freq="Q")

    if bridge:
        m = (df.quarter >= "2004Q1") & (df.quarter <= "2007Q4")
        wedge = (df.loc[m, "E_P_RPI"] - df.loc[m, "E_P_cpi"]).mean()
        pre = df.quarter <= "2003Q4"
        df.loc[pre & df.E_P_cpi.isna(), "E_P_cpi"] = \
            df.loc[pre & df.E_P_cpi.isna(), "E_P_RPI"] - wedge
        df["E_P_cpi"] = df["E_P_cpi"].interpolate(limit_direction="both")
        df["E_P_rpi_rebuilt"] = df["E_P_rpi_rebuilt"].interpolate(
            limit_direction="both")
        if verbose:
            print(f"[load] bridged E_P_cpi using wedge = {wedge:.4f}pp")

    df["g_meandec_cpi"]    = df["E_H_mean_dec_h10"]  - df["E_P_cpi"]
    df["g_median_cpi"]     = df["E_H_median_dec"]    - df["E_P_cpi"]
    df["g_meandec_rpi_rb"] = df["E_H_mean_dec_h10"]  - df["E_P_rpi_rebuilt"]
    df["g_median_rpi_rb"]  = df["E_H_median_dec"]    - df["E_P_rpi_rebuilt"]
    df["g_meandec_specB"]  = df["E_H_mean_dec_h10_specB"] - df["E_P_cpi"]
    return df


def diff_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        d[c] = d[c].diff()
    return d


# Baseline system

BASE_ENDOG = ["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi",
              "s_t_asinh100", "D_t_std_dec_h10", "g_meandec_cpi"]
BASE_EXOG = ["d_may2020"]
BASE_P = 1
H = 20
N_KEEP = 2000

# --- small-open-economy block exogeneity -----------------------------------
# N_GLO counts the leading variables of endog that form the exogenous global
# block. It is 2 (q_t, a_t) in every system in this grid, including the seven-
# and eight-variable ones, because the ERI is a UK price and belongs to the
# domestic block.
N_GLO = 2
BASE_SOE = True          # set False to revert the reduced form to unrestricted


def n_glo_for(spec: dict) -> int | None:
    """The global-block size a grid entry runs under, or None for no SOE."""
    return N_GLO if spec.get("soe", BASE_SOE) else None

ERI = "eri_t_yoy"

PRETTY = {
    "q_t_commod_yoy":     r"Commodity prices $q_t$",
    "a_t_kilian":         r"Global activity $a_t$",
    "pi_t_rpi":           r"RPI inflation $\pi_t$",
    "pi_t_cpi":           r"CPI inflation $\pi_t$",
    "s_t_asinh100":       r"Signed salience $s_t$",
    "s_t_exboe_asinh100": r"Signed salience, ex-BoE $s_t$",
    "D_t_std_dec_h10":    r"Disagreement $D_t$",
    "D_t_std_dec_h10_specB": r"Disagreement $D_t$ (spec B)",
    "D_t_std_dec_full":   r"Disagreement $D_t$ (unharmonised)",
    "g_meandec_cpi":      r"Expectations gap $g_t$",
    "g_median_cpi":       r"Expectations gap $g_t$ (median)",
    "g_meandec_rpi_rb":   r"Expectations gap $g_t$ (RPI)",
    "E_H_mean_dec_h10":   r"Household $E^H_t$",
    "E_P_cpi":            r"Professional $E^P_t$",
    "eri_t_yoy":          r"Sterling ERI $e_t$",
}


# Restrictions

def make_specs(endog: list, zeros: bool = True,
               sterling: bool = False) -> list:
    """
    Build the labelled shocks for a given variable ordering. Positions are
    looked up by name, so alternative variable choices work unchanged.

    zeros=True   exact zeros on q_t and a_t at h=0 under the salience shock
                 (BASELINE)
    zeros=False  the old weak inequalities, spec R00_weak_ineq

    sterling=True adds a fourth labelled shock, a sterling depreciation.
                 Requires the ERI to be in endog.

    THE STERLING SHOCK AND WHAT IT CAN AND CANNOT SEPARATE. It is restricted
    to depreciate the exchange rate on impact and after one quarter, and to
    raise domestic inflation after one quarter rather than on impact, since
    pass-through is not instantaneous. It carries the same zeros on q_t and
    a_t as the salience shock, for the same reason: the IMF index is in
    dollars, so a sterling move does not shift it. Nothing forces a rotation
    to satisfy the salience restrictions and fail the sterling ones, so where
    a single column could serve either the matcher's assignment is not unique.
    That shows up as wider bands, which is the honest outcome. R17, which
    puts the ERI in the system WITHOUT labelling a fourth shock, is the
    cleaner test of whether UK inflation is required to be globally
    determined, and is reported first.
    """
    q, a, pi = endog[0], endog[1], endog[2]
    s, D = endog[3], endog[4]

    comm = ShockSpec("commodity_supply", {0: {q: "+", a: "-", pi: "+"}})
    dem  = ShockSpec("global_demand",    {0: {q: "+", a: "+", pi: "+"}})

    if zeros:
        sal = ShockSpec("media_salience",
                        restrictions={0: {s: "+", D: "+"},
                                      1: {s: "+", D: "+"}},
                        zeros={0: [q, a]})
    else:
        sal = ShockSpec("media_salience",
                        restrictions={0: {q: "<=0", a: "<=0", s: "+", D: "+"},
                                      1: {s: "+", D: "+"}})

    specs = [comm, dem, sal]

    if sterling:
        if ERI not in endog:
            raise ValueError(f"sterling shock requested but {ERI} not in endog")
        stg = ShockSpec("sterling_depreciation",
                        restrictions={0: {ERI: "-"},
                                      1: {ERI: "-", pi: "+"}},
                        zeros={0: [q, a]})
        specs.append(stg)

    return specs


# The specification grid.
# Each entry notes what it varies and why.
# Every spec inherits zeros=True unless it says otherwise.

SPECS = {
    "baseline": dict(
        note="Baseline. p=1, de-censored mean vs CPI, RPI fundamentals, "
             "exact zeros on q and a at h=0.",
        endog=BASE_ENDOG, p=1),

    # --- the identification change itself, reported as robustness --------
    "R00_weak_ineq": dict(
        note="Old scheme: weak inequalities q<=0, a<=0 instead of exact "
             "zeros. The earlier baseline, demoted to robustness.",
        endog=BASE_ENDOG, p=1, zeros=False),

    # --- lag order (section 4.1) -----------------------------------------
    "R01_lag2": dict(note="Lag order p=2.", endog=BASE_ENDOG, p=2),
    "R02_lag3": dict(note="Lag order p=3 (AIC/FPE choice).",
                     endog=BASE_ENDOG, p=3),

    # --- gap definition (section 3.3) ------------------------------------
    "R03_gap_median": dict(
        note="Gap from the household median rather than the de-censored mean.",
        endog=[*BASE_ENDOG[:5], "g_median_cpi"], p=1),

    # --- professional comparator (section 3.3) ---------------------------
    "R04_gap_rpi": dict(
        note="Gap against the RPI-based professional consensus.",
        endog=[*BASE_ENDOG[:5], "g_meandec_rpi_rb"], p=1),

    # --- fundamentals inflation (section 3.4) ----------------------------
    "R05_pi_cpi": dict(
        note="CPI rather than RPI in the fundamentals slot.",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_cpi",
               "s_t_asinh100", "D_t_std_dec_h10", "g_meandec_cpi"], p=1),

    # --- salience variants (sections 3.5 and 4.5) ------------------------
    "R06_exboe": dict(
        note="Ex-Bank-of-England salience index (section 4.5).",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi",
               "s_t_exboe_asinh100", "D_t_std_dec_h10", "g_meandec_cpi"], p=1),

    # --- disagreement measure (section 3.2) ------------------------------
    "R07_D_specB": dict(
        note="Disagreement on specification B.",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi",
               "s_t_asinh100", "D_t_std_dec_h10_specB", "g_meandec_cpi"], p=1),

    "R08_unharmonised": dict(
        note="Unharmonised de-censoring (section 3.2).",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi",
               "s_t_asinh100", "D_t_std_dec_full", "g_meandec_cpi"], p=1),

    # --- first differences (section 4.1) ---------------------------------
    "R10_differences": dict(
        note="System in first differences. The h=4 band collapses to "
             "numerical noise; do not report it as a precise effect.",
        endog=BASE_ENDOG, p=1, difference=True),

    # --- complete-case sample --------------------------------------------
    "R11_nobridge": dict(
        note="No bridging of E_P_cpi; complete cases only.",
        endog=BASE_ENDOG, p=1, bridge=False),

    # --- sample windows ---------------------------------------------------
    "R12_from2008": dict(
        note="Post-2008Q3 sample, where the de-censoring uses the survey's "
             "own follow-up questions throughout.",
        endog=BASE_ENDOG, p=1, start="2008Q3"),

    "R13_pre2020": dict(
        note="Sample ends 2019Q4. The narrative targets are outside this "
             "window, so it CANNOT run the baseline scheme. Flag wherever "
             "it appears.",
        endog=BASE_ENDOG, p=1, end="2019Q4"),

    "R14_no_dummy": dict(
        note="May 2020 dummy dropped.",
        endog=BASE_ENDOG, p=1, exog=[]),

    "R15_h0only": dict(
        note="Sign restrictions imposed at h=0 only; the zeros are h=0 "
             "restrictions already and are unaffected.",
        endog=BASE_ENDOG, p=1, h0_only=True),

    # --- auxiliary split system (section 4.4) -----------------------------
    "R16_split_EH_EP": dict(
        note="Auxiliary 7-variable system replacing g_t with E^H and E^P "
             "entered separately (section 4.4).",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi", "s_t_asinh100",
               "D_t_std_dec_h10", "E_H_mean_dec_h10", "E_P_cpi"], p=1),

    # --- NEW: the domestic block -----------------------------------------
    "R17_eri": dict(
        note="Sterling ERI added as a seventh variable, same three labelled "
             "shocks. Tests whether UK inflation is required to be "
             "globally determined.",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi", "s_t_asinh100",
               "D_t_std_dec_h10", "g_meandec_cpi", ERI], p=1),

    "R18_eri_sterling_shock": dict(
        note="Sterling ERI added AND a sterling depreciation shock labelled, "
             "so a domestic cost-push channel is identified explicitly "
             "rather than left among the unlabelled shocks.",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi", "s_t_asinh100",
               "D_t_std_dec_h10", "g_meandec_cpi", ERI], p=1,
        sterling=True),

    "R19_eri_split": dict(
        note="ERI plus the split system: eight variables, E^H and E^P "
             "separate. Tests whether the section 5.3 finding survives a "
             "domestic block. Expect low acceptance.",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi", "s_t_asinh100",
               "D_t_std_dec_h10", "E_H_mean_dec_h10", "E_P_cpi", ERI], p=1),

    # --- NEW: small-open-economy block exogeneity -------------------------
    # NUMBERING NOTE. R04 is already the RPI professional comparator, so the
    # SOE specs take R20 onwards. Anywhere a draft or handoff note says
    # "R04 block-exogenous" the label is wrong and must be changed.
    "R20_no_soe": dict(
        note="Unrestricted reduced form: the PRE-SOE baseline, demoted to "
             "robustness. This is the spec that produced the commodity and "
             "activity declines at h>0 under a salience shock. Report it "
             "alongside the baseline so the change is visible.",
        endog=BASE_ENDOG, p=1, soe=False),

    "R21_soe_weak_ineq": dict(
        note="SOE reduced form WITH weak inequalities instead of exact "
             "zeros. Isolates the marginal contribution of the exact zero: "
             "under SOE the domestic block can no longer re-inject a global "
             "response, so whatever global movement survives here is "
             "Phi_11^h b_glo and nothing else. The global IRF is NOT "
             "expected to be zero in this spec.",
        endog=BASE_ENDOG, p=1, zeros=False),

    "R22_soe_split": dict(
        note="SOE on the split system. THE ONE THAT MATTERS. The section 5.3 "
             "finding (professionals respond ~70pc more than households) is "
             "the claim most damaged if the salience shock carries a global "
             "component, so it is the claim that has to survive the "
             "restriction that removes one.",
        endog=["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi", "s_t_asinh100",
               "D_t_std_dec_h10", "E_H_mean_dec_h10", "E_P_cpi"], p=1),
}


def specs_for(spec: dict) -> list:
    """Build the ShockSpec list implied by a grid entry."""
    s = make_specs(spec["endog"],
                   zeros=spec.get("zeros", True),
                   sterling=spec.get("sterling", False))
    if spec.get("h0_only"):
        for sh in s:
            sh.restrictions = {0: sh.restrictions.get(0, {})}
    return s


def frame(spec: dict) -> pd.DataFrame:
    """Apply the sample and transformation options of a grid entry."""
    df = load(bridge=spec.get("bridge", True), verbose=False)
    if spec.get("difference"):
        df = diff_frame(df, spec["endog"])
    if spec.get("start"):
        df = df[df.quarter >= spec["start"]]
    if spec.get("end"):
        df = df[df.quarter <= spec["end"]]
    return df.reset_index(drop=True)
