"""
Every figure in the dissertation that is not produced by the estimation run
itself.

    python figures.py

Reads the saved CSVs in results_v3/ rather than re-running identification, so
the figures are fast and match the tabulated numbers by construction. Run the
pipeline first. The two descriptive figures additionally need the estimation
panel, df_var_ready_v3.xlsx.

House style throughout: serif type, muted navy, red and amber, legends below
the axes, shaded episodes.

    fig1_2_gap              the household-professional gap
    fig2b_salience_gap      the gap against signed salience
    fig07_irf_baseline_v3   six-panel IRF with identification bands
    fig08_gap_bands_v3      identification against bootstrap
    fig09_fevd_v3           FEVD of salience, disagreement and the gap
    fig10_histdecomp_v3     historical decomposition of the gap
    fig11_grid_v3           robustness forest plot at h=0 and h=4
    fig12_sensitivity_v3    FEVD threshold and persistence
    fig13_episodes_v3       episode comparison
    fig14_r16_v3            E^H against E^P and the per-draw difference
    fig15_ladder_v3         the four identification schemes
    fig16_eri_v3            the domestic block

fig14 is drawn twice. fig14_r16() writes the identification-only version and
fig14_r16_bootstrap() overwrites it with both bands, which is the version the
draft uses; the order in __main__ is therefore load-bearing. Keep a copy of the
first if you want to compare:

    cp figures_v3/fig14_r16_v3.pdf figures_v3/fig14_r16_v3_idonly.pdf

The two descriptive figures reproduce config_v3.load(bridge=True) rather than
importing it, so they can be regenerated without the rest of the pipeline on
the path. The gap must be built after bridging E_P_cpi: the raw column is 86
observations and does not match the descriptive table. Both plot the same gap
array on the same y-limits so they are comparable by eye, which the self-check
at the end of descriptive_figures() asserts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path("results_v3")
FIG = Path("figures_v3"); FIG.mkdir(exist_ok=True)
DATA = Path("df_var_ready_v3.xlsx")

NAVY, RED, AMBER, GREY = "#1f3a5f", "#a4243b", "#d8973c", "#8a8a8a"
HMAX = 20
YLIM = (-2.1, 3.4)          # shared by both descriptive figures
EPISODES = [("2008Q1", "2008Q4"), ("2022Q1", "2022Q4")]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


# --- from figures_v3.py ---------------------------------------------


PRETTY = {
    "q_t_commod_yoy":   r"Commodity prices $q_t$",
    "a_t_kilian":       r"Global activity $a_t$",
    "pi_t_rpi":         r"RPI inflation $\pi_t$",
    "s_t_asinh100":     r"Signed salience $s_t$",
    "D_t_std_dec_h10":  r"Disagreement $D_t$",
    "g_meandec_cpi":    r"Expectations gap $g_t$",
    "E_H_mean_dec_h10": r"Household $E^H_t$",
    "E_P_cpi":          r"Professional $E^P_t$",
    "eri_t_yoy":        r"Sterling ERI $e_t$",
}


LADDER_PRETTY = {
    "L0_signs_weak":      "signs only,\nweak inequalities",
    "L1_signs_zeros":     "signs only,\nexact zeros",
    "L2_zeros_narrative": "exact zeros\n+ narrative",
    "L3_weak_narrative":  "weak inequalities\n+ narrative",
}


EP1, EP2 = ("2008Q1", "2009Q4"), ("2021Q4", "2023Q4")


def _read(name):
    p = OUT / f"{name}.csv"
    if not p.exists():
        print(f"  [skip] {p} not found")
        return None
    return pd.read_csv(p)


def _legend_below(fig, handles, ncol=2, y=-0.04):
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, y))


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name); plt.close(fig)
    print(f"  wrote {name}")


def fig07_irf():
    d = _read("irf_baseline_v3")
    if d is None: return
    vars_ = [v for v in PRETTY if v in set(d.variable)]
    ncol = 3
    nrow = int(np.ceil(len(vars_) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.2, 2.5 * nrow),
                             sharex=True)
    for i, ax in enumerate(np.ravel(axes)):
        if i >= len(vars_):
            ax.axis("off"); continue
        s = d[d.variable == vars_[i]].sort_values("h")
        ax.fill_between(s.h, s.p16, s.p84, color=NAVY, alpha=0.20, lw=0)
        ax.plot(s.h, s["median"], color=NAVY, lw=1.6)
        ax.axhline(0, color=RED, lw=0.8, ls="--")
        ax.set_title(PRETTY[vars_[i]], fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in np.ravel(axes)[-ncol:]:
        ax.set_xlabel("quarters after the shock", fontsize=8)
    _legend_below(fig, [
        plt.Line2D([], [], color=NAVY, lw=1.6, label="median response"),
        Patch(facecolor=NAVY, alpha=0.20, label="68% identification band")])
    _save(fig, "fig07_irf_baseline_v3.pdf")


def fig08_gap_bands():
    a, b = _read("irf_baseline_v3"), _read("irf_bootstrap_v3")
    if a is None or b is None: return
    g = a[a.variable == "g_meandec_cpi"].sort_values("h")
    b = b.sort_values("h")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, s, ttl, col in [
            (axes[0], g, "Identification uncertainty only", NAVY),
            (axes[1], b, "Identification + sampling (wild bootstrap)", RED)]:
        ax.fill_between(s.h, s.p16, s.p84, color=col, alpha=0.20, lw=0)
        ax.plot(s.h, s["median"], color=col, lw=1.7)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("quarters after the shock", fontsize=8)
    axes[0].set_ylabel("response of $g_t$ (pp)", fontsize=8)
    _legend_below(fig, [
        plt.Line2D([], [], color=NAVY, lw=1.7, label="identification only"),
        plt.Line2D([], [], color=RED, lw=1.7, label="wild bootstrap"),
        Patch(facecolor=GREY, alpha=0.25, label="68% band")], ncol=3, y=-0.08)
    _save(fig, "fig08_gap_bands_v3.pdf")


def fig09_fevd():
    d = _read("fevd_baseline_v3")
    if d is None: return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for v, col in [("s_t_asinh100", NAVY), ("D_t_std_dec_h10", AMBER),
                   ("g_meandec_cpi", RED)]:
        s = d[d.variable == v].sort_values("h")
        if s.empty: continue
        ax.plot(s.h, 100 * s.median_share, color=col, lw=1.6,
                label=PRETTY.get(v, v))
        ax.fill_between(s.h, 100 * s.p16, 100 * s.p84, color=col,
                        alpha=0.14, lw=0)
    ax.set_xlabel("forecast horizon, quarters"); ax.set_ylabel("per cent")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=8)
    _save(fig, "fig09_fevd_v3.pdf")


def fig10_histdecomp():
    d = _read("histdecomp_gap_v3")
    if d is None: return
    q = pd.Index(d.quarter.astype(str)); x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for a, bq in (EP1, EP2):
        if a in q and bq in q:
            ax.axvspan(q.get_loc(a), q.get_loc(bq), color=GREY,
                       alpha=0.13, lw=0)
    ax.fill_between(x, d.p16, d.p84, color=NAVY, alpha=0.18, lw=0)
    ax.bar(x, d.salience_contrib, color=NAVY, width=0.85)
    ax.axhline(0, color="k", lw=0.7)
    step = max(1, len(q) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(q[::step], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("contribution to $g_t$ (pp)")
    _legend_below(fig, [
        Patch(facecolor=NAVY, label="media salience shock"),
        Patch(facecolor=NAVY, alpha=0.18, label="68% band"),
        Patch(facecolor=GREY, alpha=0.13, label="2008 and 2022 episodes")],
        ncol=3, y=-0.10)
    _save(fig, "fig10_histdecomp_v3.pdf")


def fig11_grid():
    d = _read("grid_v3")
    if d is None: return
    d = d[d.status == "ok"].copy()
    if d.empty: return
    d["label2"] = np.where(d.target.astype(str).str.startswith("g_"),
                           d.spec, d.spec + " (" + d.target + ")")
    d["flag"] = d.scheme.astype(str).str.contains("NO NARRATIVE")
    d = d.sort_values("spec").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.34 * len(d) + 1.6),
                             sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, (m, lo, hi, ttl) in zip(axes, [
            ("gap_h0", "gap_h0_p16", "gap_h0_p84", "Impact, $h=0$"),
            ("gap_h4", "gap_h4_p16", "gap_h4_p84", "One year, $h=4$")]):
        cols = [AMBER if f else NAVY for f in d.flag]
        ax.hlines(y, d[lo], d[hi], color=cols, lw=2.2, alpha=0.55)
        ax.scatter(d[m], y, color=cols, s=22, zorder=3)
        ax.axvline(0, color=RED, lw=0.9, ls="--")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("response of $g_t$ (pp)", fontsize=8)
    axes[0].set_yticks(y); axes[0].set_yticklabels(d.label2, fontsize=7.5)
    _legend_below(fig, [
        plt.Line2D([], [], color=NAVY, marker="o", lw=2.2,
                   label="baseline scheme"),
        plt.Line2D([], [], color=AMBER, marker="o", lw=2.2,
                   label="narrative restriction not applicable")],
        ncol=2, y=-0.02)
    _save(fig, "fig11_grid_v3.pdf")


def fig12_sensitivity():
    d = _read("sensitivity_v3")
    if d is None: return
    fev = d[d.restriction == "fevd_threshold"].sort_values("value")
    per = d[d.restriction == "persistence_through_h"].sort_values("value")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, s, col, xl, ttl in [
            (axes[0], fev, NAVY, r"required FEVD share of $s_t$ at $h=0$",
             "FEVD threshold attenuates the effect"),
            (axes[1], per, AMBER, "persistence restriction imposed through h",
             "Persistence restriction is redundant")]:
        if s.empty: continue
        ax.fill_between(s.value, s.gap_h4_p16, s.gap_h4_p84, color=col,
                        alpha=0.20, lw=0)
        ax.plot(s.value, s.gap_h4, color=col, lw=1.7, marker="o", ms=4)
        ax.axhline(0, color=RED, lw=0.9, ls="--")
        ax.set_xlabel(xl, fontsize=8); ax.set_title(ttl, fontsize=9)
    axes[0].set_ylabel("response of $g_t$ at $h=4$ (pp)", fontsize=8)
    _save(fig, "fig12_sensitivity_v3.pdf")


def fig13_episodes():
    d = _read("episodes_v3")
    if d is None: return
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(d) + 1.8))
    y = np.arange(len(d))[::-1]; off = 0.14
    ax.hlines(y + off, d.gap_h0_p16, d.gap_h0_p84, color=NAVY, lw=2.2,
              alpha=0.55)
    ax.scatter(d.gap_h0, y + off, color=NAVY, s=24, zorder=3)
    ax.hlines(y - off, d.gap_h4_p16, d.gap_h4_p84, color=AMBER, lw=2.2,
              alpha=0.65)
    ax.scatter(d.gap_h4, y - off, color=AMBER, s=24, zorder=3)
    ax.axvline(0, color=RED, lw=0.9, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(d.episode, fontsize=8.5)
    ax.set_xlabel("response of $g_t$ (pp)", fontsize=8)
    _legend_below(fig, [
        plt.Line2D([], [], color=NAVY, marker="o", lw=2.2, label="$h=0$"),
        plt.Line2D([], [], color=AMBER, marker="o", lw=2.2, label="$h=4$")],
        ncol=2, y=-0.02)
    _save(fig, "fig13_episodes_v3.pdf")


def fig14_r16():
    a, d = _read("irf_r16_v3"), _read("r16_perdraw_gap_v3")
    if a is None or d is None: return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    for v, col in [("E_H_mean_dec_h10", NAVY), ("E_P_cpi", AMBER)]:
        s = a[a.variable == v].sort_values("h")
        if s.empty: continue
        ax.fill_between(s.h, s.p16, s.p84, color=col, alpha=0.18, lw=0)
        ax.plot(s.h, s["median"], color=col, lw=1.7, label=PRETTY.get(v, v))
    ax.axhline(0, color=RED, lw=0.8, ls="--")
    ax.set_title("Household and professional responses", fontsize=9)
    ax.set_xlabel("quarters after the shock", fontsize=8)
    ax.set_ylabel("response (pp)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]; d = d.sort_values("h")
    ax.fill_between(d.h, d.p16, d.p84, color=RED, alpha=0.18, lw=0)
    ax.plot(d.h, d["median"], color=RED, lw=1.7)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(r"Per-draw difference $E^H_t - E^P_t$", fontsize=9)
    ax.set_xlabel("quarters after the shock", fontsize=8)
    _save(fig, "fig14_r16_v3.pdf")


def fig15_ladder():
    """NEW. What the zeros buy, and what the narrative restriction adds."""
    d = _read("ladder_v3")
    if d is None: return
    d = d.set_index("label").reindex(
        [k for k in LADDER_PRETTY if k in set(d.label)]).reset_index()
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    ax.bar(x, d.fevd_s_h0, color=NAVY, width=0.6)
    ax.axhline(0.35, color=RED, lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([LADDER_PRETTY[l] for l in d.label], fontsize=7.5)
    ax.set_ylabel(r"FEVD share of $s_t$ at $h=0$", fontsize=8)
    ax.set_title("Does the labelled shock explain $s_t$?", fontsize=9)
    for i, v in enumerate(d.fevd_s_h0):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)

    ax = axes[1]
    for j, (m, lo, hi, col, lab) in enumerate([
            ("gap_h0", "gap_h0_p16", "gap_h0_p84", NAVY, "$h=0$"),
            ("gap_h4", "gap_h4_p16", "gap_h4_p84", AMBER, "$h=4$")]):
        off = 0.15 * (1 if j else -1)
        ax.vlines(x + off, d[lo], d[hi], color=col, lw=2.2, alpha=0.6)
        ax.scatter(x + off, d[m], color=col, s=26, zorder=3, label=lab)
    ax.axhline(0, color=RED, lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([LADDER_PRETTY[l] for l in d.label], fontsize=7.5)
    ax.set_ylabel("response of $g_t$ (pp)", fontsize=8)
    ax.set_title("Gap response under each scheme", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig15_ladder_v3.pdf")


def fig16_eri():
    """NEW. The domestic block: does the result need a seventh variable?"""
    base = _read("irf_baseline_v3")
    rows, labels = [], []
    for name, lab in [("irf_R17_eri_v3", "R17  ERI added"),
                      ("irf_R18_eri_sterling_shock_v3",
                       "R18  ERI + sterling shock"),
                      ("irf_R19_eri_split_v3", "R19  ERI + split system")]:
        d = _read(name)
        if d is not None:
            rows.append(d); labels.append(lab)
    if base is None and not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    series = []
    if base is not None:
        series.append((base[base.variable == "g_meandec_cpi"],
                       "baseline, six variables", NAVY))
    for d, lab, col in zip(rows, labels, [AMBER, RED, GREY]):
        g = d[d.variable == "g_meandec_cpi"]
        if not g.empty:
            series.append((g, lab, col))
    for s, lab, col in series:
        s = s.sort_values("h")
        ax.plot(s.h, s["median"], color=col, lw=1.7, label=lab)
        ax.fill_between(s.h, s.p16, s.p84, color=col, alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("Gap response to a salience shock", fontsize=9)
    ax.set_xlabel("quarters after the shock", fontsize=8)
    ax.set_ylabel("response of $g_t$ (pp)", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    st = _read("irf_R18_eri_sterling_shock_sterling_shock_v3")
    if st is not None:
        g = st[st.variable == "g_meandec_cpi"].sort_values("h")
        ax.fill_between(g.h, g.p16, g.p84, color=RED, alpha=0.18, lw=0)
        ax.plot(g.h, g["median"], color=RED, lw=1.7)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title("Gap response to a sterling depreciation shock",
                     fontsize=9)
        ax.set_xlabel("quarters after the shock", fontsize=8)
    else:
        ax.axis("off")
    _save(fig, "fig16_eri_v3.pdf")


# --- from figures_r16_boot.py ---------------------------------------


EH, EP = "E_H_mean_dec_h10", "E_P_cpi"


def _panel(ax, idd, boot, title, ylab=None):
    """One panel: identification band under, bootstrap band over."""
    if idd is not None:
        s = idd.sort_values("h")
        s = s[s.h <= HMAX]
        ax.fill_between(s.h, s.p16, s.p84, color=NAVY, alpha=0.20, lw=0,
                        zorder=1)
        ax.plot(s.h, s["median"], color=NAVY, lw=1.7, zorder=3)
    if boot is not None:
        b = boot.sort_values("h")
        b = b[b.h <= HMAX]
        ax.fill_between(b.h, b.p16, b.p84, color=RED, alpha=0.10, lw=0,
                        zorder=2)
        ax.plot(b.h, b.p16, color=RED, lw=0.9, ls=":", zorder=4)
        ax.plot(b.h, b.p84, color=RED, lw=0.9, ls=":", zorder=4)
        ax.plot(b.h, b["median"], color=RED, lw=1.5, ls="--", zorder=5)
    ax.axhline(0, color="k", lw=0.8, ls="--", zorder=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("quarters after the shock", fontsize=8)
    ax.tick_params(labelsize=8)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8)


def fig14_r16_bootstrap():
    a = _read("irf_r16_v3")
    d = _read("r16_perdraw_gap_v3")
    beh = _read("irf_bootstrap_r16_EH_v3")
    bep = _read("irf_bootstrap_r16_EP_v3")
    bdf = _read("r16_perdraw_gap_bootstrap_v3")

    if a is None and beh is None:
        print("nothing to plot"); return

    ideh = a[a.variable == EH] if a is not None else None
    idep = a[a.variable == EP] if a is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    _panel(axes[0], ideh, beh, r"Household $E^H_t$",
           ylab="response (pp)")
    _panel(axes[1], idep, bep, r"Professional $E^P_t$")
    _panel(axes[2], d,    bdf, r"Per-draw difference $E^H_t - E^P_t$")

    # E^H and E^P share a scale: the claim is that the professional response
    # is the larger of the two, which independent y-axes would conceal. The
    # difference panel is a different object and keeps its own scale.
    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(lo, hi); axes[1].set_ylim(lo, hi)
    axes[1].tick_params(labelleft=False)

    _ = [ax.margins(x=0.02) for ax in axes]
    fig.legend(handles=[
        plt.Line2D([], [], color=NAVY, lw=1.7, label="median, identification"),
        Patch(facecolor=NAVY, alpha=0.20, label="68% identification band"),
        plt.Line2D([], [], color=RED, lw=1.5, ls="--",
                   label="median, wild bootstrap"),
        Patch(facecolor=RED, alpha=0.10, label="68% bootstrap band"),
    ], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(FIG / "fig14_r16_v3.pdf")
    plt.close(fig)
    print("  wrote fig14_r16_v3.pdf")


def table_for_text():
    """Print the numbers the split-system section quotes, so none is
    transcribed by eye."""
    a = _read("irf_r16_v3")
    d = _read("r16_perdraw_gap_v3")
    beh = _read("irf_bootstrap_r16_EH_v3")
    bep = _read("irf_bootstrap_r16_EP_v3")
    bdf = _read("r16_perdraw_gap_bootstrap_v3")
    if beh is None:
        return

    def peak(t, sign=+1):
        t = t.sort_values("h")
        i = (sign * t["median"]).idxmax()
        return t.loc[i, "h"], t.loc[i, "median"]

    def sigrun(t):
        t = t.sort_values("h")
        hs = t[t.excludes_zero].h.tolist()
        return hs

    print("\n=== NUMBERS FOR SECTION 5.3 ===")
    for nm, idt, bt, sgn in (
            ("E^H", a[a.variable == EH] if a is not None else None, beh, +1),
            ("E^P", a[a.variable == EP] if a is not None else None, bep, +1),
            ("E^H - E^P", d, bdf, -1)):
        print(f"\n{nm}")
        if idt is not None and len(idt):
            h, v = peak(idt, sgn)
            print(f"  identification : peak {v:+.4f} at h={int(h)}, "
                  f"excludes zero at h={sigrun(idt)}")
        h, v = peak(bt, sgn)
        print(f"  wild bootstrap : peak {v:+.4f} at h={int(h)}, "
              f"excludes zero at h={sigrun(bt)}")

    # the cross-system agreement claim in the draft
    g = _read("irf_bootstrap_v3")
    if g is not None and bdf is not None:
        print("\ncross-system check (draft claims agreement at h=2):")
        for h in (0, 1, 2, 3, 4):
            print(f"  h={h}  six-var gap boot {g.iloc[h]['median']:+.4f}   "
                  f"R16 per-draw boot {bdf.iloc[h]['median']:+.4f}")


# --- from makefigs.py -----------------------------------------------


def load():
    """config_v3.load(bridge=True), reproduced."""
    df = pd.read_excel(DATA)
    df["quarter"] = df["quarter"].astype(str)
    df = df.sort_values("quarter").reset_index(drop=True)
    df["period"] = pd.PeriodIndex(df["quarter"], freq="Q")

    m = (df.quarter >= "2004Q1") & (df.quarter <= "2007Q4")
    wedge = (df.loc[m, "E_P_RPI"] - df.loc[m, "E_P_cpi"]).mean()
    pre = df.quarter <= "2003Q4"
    df.loc[pre & df.E_P_cpi.isna(), "E_P_cpi"] = \
        df.loc[pre & df.E_P_cpi.isna(), "E_P_RPI"] - wedge
    df["E_P_cpi"] = df["E_P_cpi"].interpolate(limit_direction="both")
    df["E_P_rpi_rebuilt"] = df["E_P_rpi_rebuilt"].interpolate(
        limit_direction="both")
    print(f"[load] wedge = {wedge:.4f}pp")

    df["g_meandec_cpi"] = df["E_H_mean_dec_h10"] - df["E_P_cpi"]
    df["g_meandec_rpi_rb"] = df["E_H_mean_dec_h10"] - df["E_P_rpi_rebuilt"]
    return df


def _shade(ax, df):
    x = df["period"].dt.to_timestamp()
    for a, b in EPISODES:
        lo = pd.Period(a, "Q").to_timestamp()
        hi = pd.Period(b, "Q").to_timestamp(how="end")
        ax.axvspan(lo, hi, color=GREY, alpha=0.18, lw=0, zorder=0)
    return x


def fig_gap(df):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = _shade(ax, df)
    ax.axhline(0, color="k", lw=0.9, zorder=1)
    ax.plot(x, df["g_meandec_rpi_rb"], color=RED, lw=1.6,
            label=r"$g_t$ vs RPI-based $E^P$")
    ax.plot(x, df["g_meandec_cpi"], color=AMBER, lw=1.6, ls="--",
            label=r"$g_t$ vs CPI-based $E^P$")
    ax.set_ylim(*YLIM)
    ax.set_ylabel("Percentage points")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_2_gap.{ext}")
    plt.close(fig)
    print("  wrote fig1_2_gap.pdf/.png")


def fig_salience_gap(df, r_full, r_post):
    """Stacked, not side by side, and one shared axis in the top panel.

    A twin axis rescales the gap against the salience range and makes the same
    numbers look like a different series from Figure 2. One axis, same limits.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 7.2))

    x = _shade(ax1, df)
    ax1.axhline(0, color="k", lw=0.9, zorder=1)
    ax1.plot(x, df["s_t_asinh100"], color=AMBER, lw=1.3,
             label=r"Signed salience $s_t$")
    ax1.plot(x, df["g_meandec_cpi"], color=NAVY, lw=1.5,
             label=r"Expectations gap $g_t$")
    ax1.set_ylim(*YLIM)
    ax1.set_ylabel(r"$g_t$ (pp) and $s_t$ (index)")
    ax1.set_title("Coverage and the wedge, 2003--2025", fontsize=9)

    post = df["quarter"] >= "2008Q1"
    ax2.axhline(0, color="k", lw=0.9, zorder=1)
    ax2.scatter(df.loc[~post, "s_t_asinh100"], df.loc[~post, "g_meandec_cpi"],
                facecolors="none", edgecolors=NAVY, s=22, lw=0.9,
                label="pre-2008 quarters")
    ax2.scatter(df.loc[post, "s_t_asinh100"], df.loc[post, "g_meandec_cpi"],
                color=NAVY, s=22)
    b = np.polyfit(df.loc[post, "s_t_asinh100"], df.loc[post, "g_meandec_cpi"], 1)
    xs = np.linspace(df["s_t_asinh100"].min(), df["s_t_asinh100"].max(), 50)
    ax2.plot(xs, np.polyval(b, xs), color=RED, lw=1.2, ls="--")
    ax2.set_ylim(*YLIM)
    ax2.set_xlabel(r"Signed salience $s_t$")
    ax2.set_ylabel(r"Expectations gap $g_t$ (pp)")
    ax2.set_title("The same relation, cross-section", fontsize=9)
    ax2.text(0.98, 0.96, f"$r = {r_full:.2f}$ full\n$r = {r_post:.2f}$ post--2008",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig2b_salience_gap.{ext}")
    plt.close(fig)
    print("  wrote fig2b_salience_gap.pdf/.png")


def descriptive_figures():
    """Figures 2 and 3, and the checks the draft quotes."""
    df = load()
    g = df["g_meandec_cpi"]
    post = df["quarter"] >= "2008Q1"
    r_full = np.corrcoef(df["s_t_asinh100"], g)[0, 1]
    r_post = np.corrcoef(df.loc[post, "s_t_asinh100"], g[post])[0, 1]

    print(f"[check] T = {len(df)}, gap mean {g.mean():.2f}, sd {g.std():.2f}, "
          f"min {g.min():.2f}, max {g.max():.2f}")
    print(f"[check] corr(s_t, g_t) full {r_full:+.3f}, post-2008 {r_post:+.3f}")

    fig_gap(df)
    fig_salience_gap(df, r_full, r_post)

    # Both figures must plot the same numbers.
    assert g.equals(df["g_meandec_cpi"]), "the two figures use different gap arrays"
    print("[check] both figures plot an identical gap array")


if __name__ == "__main__":
    print("building figures from results_v3/ ...")
    for fn in (fig07_irf, fig08_gap_bands, fig09_fevd, fig10_histdecomp,
               fig11_grid, fig12_sensitivity, fig13_episodes, fig14_r16,
               fig15_ladder, fig16_eri):
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")

    # Must follow fig14_r16: it overwrites the same file with both bands.
    for fn in (fig14_r16_bootstrap, table_for_text, descriptive_figures):
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("done.")
