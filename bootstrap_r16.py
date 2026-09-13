"""
Wild bootstrap for the split system, with E^H and E^P entered separately in
place of g_t, refit under block exogeneity.

    python bootstrap_r16.py probe            5 reps, times the run, no store
    python bootstrap_r16.py run 500          resumable
    python bootstrap_r16.py run 500 --reset  start the store from scratch
    python bootstrap_r16.py report           re-read the store, rewrite CSVs

Two places need the restriction, not one: the point estimate the bootstrap
resamples around, and the refit inside every replication. Restricting only the
first would put the domestic-lag coefficients back into the global equations on
every replication.

This is a separate file because run_v3.bootstrap() is hardwired to the
six-variable baseline and reports g_meandec_cpi, which the split system does
not contain. It writes its own store and never touches boot_theta_v3.npy. The
store filename carries the scheme, so a rerun cannot pool draws estimated under
different reduced forms.

The replication counter advances with the store at every checkpoint. Written
only at the end, an interrupted run would leave draws on disk with no counter,
and the rerun would reseed at the same offset and append duplicates that
double-weight the bands.

Three responses are written out: E^H, E^P and the per-draw difference. The
difference is taken within each retained rotation and only then pooled;
differencing two pooled medians is a different quantity. "E^P excludes zero and
E^H does not" is not a test that the two differ, which is what the per-draw
difference is for.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from svar_lib_v3 import fit_var, bands, VARFit, shock_index, restricted_ols
import config_v3 as C
import identify_v3 as I

OUT = C.OUT
H = C.H

SPEC_NAME = "R16_split_EH_EP"
EH, EP = "E_H_mean_dec_h10", "E_P_cpi"

BOOT_STORE = OUT / "boot_theta_r16_soe_v3.npy"
BOOT_COUNT = OUT / "boot_count_r16_soe_v3.txt"
N_GLO = 2   # q_t, a_t are endog[0], endog[1] in every spec this file uses

BASELINE_SCHEME = "L2"          # must match run_v3.BASELINE_SCHEME
SEED_BASE = 20260816


def _setup():
    """Fit, restrictions and narrative checker for R16."""
    spec = C.SPECS[SPEC_NAME]
    df = C.frame(spec)
    endog = spec["endog"]
    fit = fit_var(df, endog, spec.get("p", 1), spec.get("exog", C.BASE_EXOG),
                  n_glo=N_GLO)
    quarters = np.asarray(df["quarter"].iloc[-fit.T:])
    specs = C.specs_for(spec)
    return spec, df, fit, quarters, specs


def _write(d, name):
    d.to_csv(OUT / f"{name}.csv", index=False)
    print(f"    wrote {name}.csv")


def _flush(buf):
    if not buf:
        return
    new = np.concatenate(buf, axis=0)
    if BOOT_STORE.exists():
        new = np.concatenate([np.load(BOOT_STORE), new], axis=0)
    np.save(BOOT_STORE, new)


def _one_rep(fit, specs, quarters, endog, rng, with_narr,
             keep_per_boot, max_draws):
    """One wild-bootstrap replication. Returns Theta or None."""
    n, k = fit.n, fit.X.shape[1]
    e = rng.choice([-1.0, 1.0], size=(fit.T, 1))
    Yb = fit.X @ fit.B + fit.U * e
    Bb = restricted_ols(fit.X, Yb, n, fit.p, fit.n_exog, fit.n_glo)
    Ub = Yb - fit.X @ Bb
    fitb = VARFit(Y=Yb, X=fit.X, B=Bb, U=Ub,
                  Sigma=(Ub.T @ Ub) / (fit.T - k), p=fit.p, n=n,
                  n_exog=fit.n_exog, names=fit.names, n_glo=fit.n_glo)
    nb = None
    if with_narr:
        nb = I.narrative_checker(fitb, fitb.names, quarters, endog[3],
                                 [sp.name for sp in specs], strict=True)
    try:
        r = I.identify(fitb, specs, nb, H=H, n_keep=keep_per_boot,
                       max_draws=max_draws, seed=int(rng.integers(1e9)),
                       want_hd=False)
    except np.linalg.LinAlgError:
        return None
    if r["kept"] == 0:
        return None
    return r["Theta"], r["acceptance"], r["worst_zero"]


def probe(n_reps=5, keep_per_boot=4, max_draws=800_000):
    """
    Time a handful of replications before committing to a long run.

    R16 is a seven-variable system carrying the same two exact zeros and the
    narrative restriction as the baseline, so its acceptance rate is lower and
    its per-replication cost higher. Find out by how much before queueing 500.
    """
    spec, df, fit, quarters, specs = _setup()
    with_narr = BASELINE_SCHEME == "L2"
    print(f"=== PROBE: {SPEC_NAME}, {fit.n} variables, "
          f"{len(specs)} labelled shocks, narrative={with_narr} ===")
    print(f"    T={fit.T}, params/equation={fit.X.shape[1]}, "
          f"df={fit.T - fit.X.shape[1]}")

    rng = np.random.default_rng(SEED_BASE)
    t0 = time.time()
    ok, accs = 0, []
    for b in range(n_reps):
        out = _one_rep(fit, specs, quarters, spec["endog"], rng,
                       with_narr, keep_per_boot, max_draws)
        if out is None:
            print(f"    rep {b+1}: FAILED (no admissible draws)")
            continue
        _, acc, wz = out
        ok += 1
        accs.append(acc)
        print(f"    rep {b+1}: acceptance {acc:.4%}, worst zero {wz:.2e}")
    el = time.time() - t0
    per = el / max(n_reps, 1)
    print(f"\n    {ok}/{n_reps} replications succeeded")
    if accs:
        print(f"    median acceptance {np.median(accs):.4%}")
    print(f"    {per:.1f}s per replication")
    for target in (250, 500, 1000):
        print(f"    projected {target} reps: {per*target/60:.1f} min")
    print("\n    If this is too slow, lower keep_per_boot (pooled draws are "
          "reps x keep) or raise max_draws only if failures are common.")


def run(n_reps=500, keep_per_boot=4, checkpoint_every=25,
        max_draws=800_000, reset=False):
    """Resumable wild bootstrap. Rerun to add replications."""
    if reset:
        for p in (BOOT_STORE, BOOT_COUNT):
            if p.exists():
                p.unlink()
                print(f"    removed {p.name}")

    spec, df, fit, quarters, specs = _setup()
    with_narr = BASELINE_SCHEME == "L2"
    offset = int(open(BOOT_COUNT).read()) if BOOT_COUNT.exists() else 0
    rng = np.random.default_rng(SEED_BASE + offset)

    print(f"=== WILD BOOTSTRAP, {SPEC_NAME} "
          f"({n_reps} reps, offset {offset}, narrative={with_narr}) ===")
    t0 = time.time()
    buf, fails, accs, worst = [], 0, [], 0.0

    for b in range(n_reps):
        out = _one_rep(fit, specs, quarters, spec["endog"], rng,
                       with_narr, keep_per_boot, max_draws)
        if out is None:
            fails += 1
            continue
        Th, acc, wz = out
        buf.append(Th)
        accs.append(acc)
        worst = max(worst, wz)

        if (b + 1) % checkpoint_every == 0:
            _flush(buf); buf = []
            # The counter advances WITH the store, not after the loop. If it
            # were written only at the end, an interrupted run would leave
            # draws on disk and no counter, and the rerun would reseed at the
            # same offset and append byte-identical duplicates that silently
            # double-weight the percentile bands.
            open(BOOT_COUNT, "w").write(str(offset + b + 1))
            el = time.time() - t0
            rate = el / (b + 1)
            print(f"    checkpoint {b+1}/{n_reps}  failures {fails}  "
                  f"{el/60:.1f}m elapsed, ~{rate*(n_reps-b-1)/60:.1f}m left",
                  flush=True)
    _flush(buf)
    open(BOOT_COUNT, "w").write(str(offset + n_reps))

    print(f"\n    failures this batch {fails}/{n_reps}")
    if accs:
        print(f"    median acceptance {np.median(accs):.4%}")
    print(f"    worst zero violation {worst:.3e} "
          f"{'OK' if worst <= I.ZERO_ALARM else '*** NOT BINDING ***'}")
    report()


def report():
    """Read the pooled store and write the three response tables."""
    if not BOOT_STORE.exists():
        print("no bootstrap store found; run first")
        return
    spec, df, fit, quarters, specs = _setup()
    Th = np.load(BOOT_STORE)
    shock_names = [sp.name for sp in specs]
    k = shock_index(shock_names, "media_salience")
    ih, ip = fit.names.index(EH), fit.names.index(EP)
    mult, _, _ = I.episode_scale(df)

    print(f"\n=== POOLED: {Th.shape[0]} draws "
          f"({Th.shape[0]//4} reps x 4 kept, nominal) ===")

    def tab(arr, scale=1.0):
        med, lo, hi = bands(arr)
        return pd.DataFrame({
            "h": np.arange(len(med)),
            "median": scale * med, "p16": scale * lo, "p84": scale * hi,
            "excludes_zero": (lo > 0) | (hi < 0)})

    eh = tab(Th[:, :, ih, k])
    ep = tab(Th[:, :, ip, k])
    diff = tab(Th[:, :, ih, k] - Th[:, :, ip, k])

    _write(eh,   "irf_bootstrap_r16_EH_v3")
    _write(ep,   "irf_bootstrap_r16_EP_v3")
    _write(diff, "r16_perdraw_gap_bootstrap_v3")
    if np.isfinite(mult):
        _write(tab(Th[:, :, ih, k] - Th[:, :, ip, k], scale=mult),
               "r16_perdraw_gap_bootstrap_v3_episode")

    # one tidy long table for the write-up
    longt = pd.concat([eh.assign(object="E_H"), ep.assign(object="E_P"),
                       diff.assign(object="E_H_minus_E_P")],
                      ignore_index=True)
    _write(longt, "r16_bootstrap_summary_v3")

    for nm, t in (("E^H", eh), ("E^P", ep), ("E^H - E^P", diff)):
        print(f"\n{nm} (identification + sampling):")
        for h in (0, 1, 2, 3, 4, 8, 12):
            r = t.iloc[h]
            s = "SIG" if r.excludes_zero else ""
            print(f"    h={h:2d}  {r['median']:+.4f}  "
                  f"[{r['p16']:+.4f}, {r['p84']:+.4f}]  {s}")

    # side-by-side with the identification-only run, if present
    a = OUT / "irf_r16_v3.csv"
    if a.exists():
        idr = pd.read_csv(a)
        print("\n=== identification-only vs pooled, h=0..4 ===")
        print(f"{'h':>3} {'obj':<14} {'id median':>10} {'id sig':>7} "
              f"{'boot median':>12} {'boot sig':>9}")
        for nm, var, t in (("E^H", EH, eh), ("E^P", EP, ep)):
            s = idr[idr.variable == var].sort_values("h")
            for h in range(5):
                print(f"{h:>3} {nm:<14} {s.iloc[h]['median']:>10.4f} "
                      f"{str(bool(s.iloc[h]['excludes_zero'])):>7} "
                      f"{t.iloc[h]['median']:>12.4f} "
                      f"{str(bool(t.iloc[h]['excludes_zero'])):>9}")
    b = OUT / "r16_perdraw_gap_v3.csv"
    if b.exists():
        pdr = pd.read_csv(b).sort_values("h")
        print("\nper-draw difference, identification vs pooled, h=0..4:")
        for h in range(5):
            print(f"  h={h}  id {pdr.iloc[h]['median']:+.4f} "
                  f"({bool(pdr.iloc[h]['excludes_zero'])})   "
                  f"boot {diff.iloc[h]['median']:+.4f} "
                  f"({bool(diff.iloc[h]['excludes_zero'])})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    arg = None
    for a in sys.argv[2:]:
        if not a.startswith("--"):
            arg = a
    reset = "--reset" in sys.argv
    if cmd == "probe":
        probe(int(arg) if arg else 5)
    elif cmd == "run":
        run(int(arg) if arg else 500, reset=reset)
    elif cmd == "report":
        report()
    else:
        print(__doc__)
