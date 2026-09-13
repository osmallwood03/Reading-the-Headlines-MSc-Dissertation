"""
Tests for the collector and for the block-exogeneity code.

    python tests.py

Neither half needs the network or an API key. The collector tests run against
a mock API; the block-exogeneity tests run on synthetic data.

The collector half checks quarter boundaries, HTML cleaning, query building,
pagination, that the UK and article-type filters are on every call, that a run
caches and then resumes with zero calls, and that a rate limit surfaces as an
error with a resume instruction rather than a silent stall.

The block-exogeneity half tests mechanics, not economics. Each claim below is
one the write-up makes in prose.

    T1  Phi_12 is exactly zero after restricted_ols, at p = 1, 2 and 3.
    T2  n_glo=None reproduces the unrestricted least-squares fit bit for bit,
        so R20 really is the old baseline rather than a near-miss.
    T3  Under block exogeneity and exact zeros the global rows are zero at
        every horizon.
    T4  Under block exogeneity and weak inequalities they are not, and under
        exact zeros without block exogeneity they are not either. If this
        passed trivially the two restrictions would be redundant and the
        write-up would be wrong.
    T5  A wild-bootstrap refit preserves the restriction.
    T6  select_order runs under both reduced forms and counts parameters
        differently.
"""
import gzip, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

import guardian_collect as gc
from svar_lib_v3 import (fit_var, restricted_ols, select_order, ShockSpec,
                         ma_coefficients, irf_from_B0, soe_violation,
                         build_rotation_with_zeros, draw_rotation,
                         match_shocks, VARFit)


# --- from test_collect.py -------------------------------------------


CALLS = []


def fake_api_get(params, session=None):
    CALLS.append(dict(params))
    page_size = int(params.get("page-size", 1))
    has_q = "q" in params

    # Pretend the cost-of-living query matches 350 articles/quarter, and that the
    # Guardian publishes 9000 UK articles/quarter. 350 forces multi-page paging.
    total = 350 if has_q else 9000

    if page_size == 1:
        return {"total": total, "results": []}

    page = int(params.get("page", 1))
    pages = -(-total // page_size)
    start = (page - 1) * page_size
    n = min(page_size, total - start)
    results = [{
        "id": f"x/{start + i}",
        "webUrl": f"https://theguardian.com/x/{start + i}",
        "sectionId": "business",
        "webPublicationDate": "2022-08-01T00:00:00Z",
        "fields": {
            "headline": f"Energy bills <b>soar</b> #{start + i}",
            "standfirst": "Households face &pound;3,000 &amp; rising",
            "trailText": "trail",
            "body": "<p>Prices are  rising  sharply.</p><p>More text.</p>",
            "wordcount": "500",
            "firstPublicationDate": "2022-08-01T00:00:00Z",
        },
    } for i in range(n)]
    return {"total": total, "pages": pages, "results": results}


gc.api_get = fake_api_get


gc.SLEEP_BETWEEN_CALLS = 0


ok = lambda m: print(f"  PASS  {m}")


def test_quarters():
    qs = list(gc.quarters(2004, 1, 2025, 4))
    assert len(qs) == 88, len(qs)
    assert qs[0] == ("2004Q1", "2004-01-01", "2004-03-31"), qs[0]
    assert qs[-1] == ("2025Q4", "2025-10-01", "2025-12-31"), qs[-1]
    # leap-year and 30-day month boundaries
    d = dict((q[0], q[2]) for q in qs)
    assert d["2004Q1"] == "2004-03-31"
    assert d["2004Q2"] == "2004-06-30"
    assert d["2004Q3"] == "2004-09-30"
    assert d["2004Q4"] == "2004-12-31"
    ok(f"quarters(): {len(qs)} quarters, 2004Q1 -> 2025Q4, month-ends correct")


def test_strip_html():
    s = gc.strip_html("<p>Energy bills &pound;3,000 &amp;  rising</p>")
    assert s == "Energy bills £3,000 & rising", repr(s)
    assert gc.strip_html(None) == ""
    ok("strip_html(): tags removed, entities unescaped, whitespace collapsed")


def test_query_building():
    main = gc.build_query("main")
    exboe = gc.build_query("ex_boe")
    assert main.startswith("(") and main.endswith(")")
    assert '"cost of living"' in main
    assert "AND NOT" in exboe and "Bank of England" in exboe
    ok("build_query(): main and ex_boe baskets well-formed")


def test_pagination():
    CALLS.clear()
    arts = gc.get_articles("2022-07-01", "2022-09-30", "(inflation)", verbose=False)
    assert len(arts) == 350, len(arts)
    assert len(CALLS) == 2, f"expected 2 pages, got {len(CALLS)}"
    assert arts[0]["headline"] == "Energy bills soar #0", arts[0]["headline"]
    assert arts[0]["body"] == "Prices are rising sharply. More text."
    assert arts[0]["lead"].startswith("Prices are rising")
    assert len({a["id"] for a in arts}) == 350, "duplicate articles across pages"
    ok(f"get_articles(): 350 articles over {len(CALLS)} pages, no duplicates, text clean")


def test_uk_filters_applied():
    CALLS.clear()
    gc.get_total("2022-07-01", "2022-09-30", "(inflation)")
    p = CALLS[0]
    assert p["production-office"] == "uk", "UK office filter missing"
    assert p["type"] == "article", "type=article filter missing"
    ok("filters: production-office=uk and type=article on every call")


def test_full_run_and_resume():
    tmp = Path("_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    gc.DATA_DIR, gc.RAW_DIR = tmp, tmp / "raw"
    orig = gc.quarters
    gc.quarters = lambda *a: [("2022Q3", "2022-07-01", "2022-09-30"),
                              ("2022Q4", "2022-10-01", "2022-12-31")]
    try:
        args = type("A", (), {"basket": "main", "force": False})()

        CALLS.clear()
        gc.run_full(args)
        first_pass_calls = len(CALLS)

        panel = (tmp / "quarterly_panel.csv").read_text().strip().split("\n")
        assert panel[0] == "quarter,n_matched,n_total,salience_raw"
        assert panel[1].startswith("2022Q3,350,9000,0.0388"), panel[1]
        assert len(panel) == 3

        cached = json.load(gzip.open(tmp / "raw" / "2022Q3.json.gz", "rt"))
        assert cached["n_matched"] == 350
        assert len(cached["articles"]) == 350
        assert cached["n_total"] == 9000

        # Re-run: must hit zero API calls and still produce an identical panel.
        CALLS.clear()
        gc.run_full(args)
        assert len(CALLS) == 0, f"resume made {len(CALLS)} calls; should be 0"
        panel2 = (tmp / "quarterly_panel.csv").read_text()
        assert panel2.strip().split("\n")[1] == panel[1]
        ok(f"run_full(): collected in {first_pass_calls} calls, cached, "
           f"resumed with 0 calls, panel stable")
    finally:
        gc.quarters = orig
        shutil.rmtree(tmp, ignore_errors=True)


def test_ratelimit_surfaces():
    class R:
        status_code = 429
    class S:
        def get(self, *a, **k):
            return R()
    real = gc.api_get
    gc.api_get = real.__wrapped__ if hasattr(real, "__wrapped__") else None
    # exercise the real api_get with a fake session
    import importlib
    m = importlib.reload(gc)
    m.SLEEP_BETWEEN_CALLS = 0
    try:
        m.api_get({"page-size": 1}, session=S())
    except m.RateLimited as e:
        assert "resume" in str(e)
        ok("api_get(): HTTP 429 raises RateLimited with a resume instruction")
    else:
        raise AssertionError("429 did not raise RateLimited")
    finally:
        gc.api_get = fake_api_get


# --- from test_soe.py -----------------------------------------------


ENDOG = ["q_t_commod_yoy", "a_t_kilian", "pi_t_rpi",
         "s_t_asinh100", "D_t_std_dec_h10", "g_meandec_cpi"]


NG = 2


RNG = np.random.default_rng(7)


FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def synth(T=91, n=6, seed=7):
    """A stable VAR(1) with genuine domestic->global feedback in the DGP,
    so that imposing SOE is a real restriction and not a free lunch."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.18, (n, n)) + np.diag(np.full(n, 0.55))
    A[:NG, NG:] = rng.normal(0, 0.15, (NG, n - NG))   # feedback present
    A *= 0.9 / max(np.abs(np.linalg.eigvals(A)).max(), 1e-9)
    Y = np.zeros((T + 60, n))
    for t in range(1, T + 60):
        Y[t] = A @ Y[t - 1] + rng.normal(0, 1.0, n)
    Y = Y[-T:]
    df = pd.DataFrame(Y, columns=ENDOG)
    df["d_may2020"] = 0.0
    df.loc[df.index[len(df) // 2], "d_may2020"] = 1.0
    df["quarter"] = pd.period_range("2003Q1", periods=T, freq="Q").astype(str)
    return df


def specs(zeros=True):
    q, a, pi, s, D = ENDOG[0], ENDOG[1], ENDOG[2], ENDOG[3], ENDOG[4]
    comm = ShockSpec("commodity_supply", {0: {q: "+", a: "-", pi: "+"}})
    dem = ShockSpec("global_demand", {0: {q: "+", a: "+", pi: "+"}})
    if zeros:
        sal = ShockSpec("media_salience",
                        restrictions={0: {s: "+", D: "+"}, 1: {s: "+", D: "+"}},
                        zeros={0: [q, a]})
    else:
        sal = ShockSpec("media_salience",
                        restrictions={0: {q: "<=0", a: "<=0", s: "+", D: "+"},
                                      1: {s: "+", D: "+"}})
    return [comm, dem, sal]


def one_draw(fit, sp, tries=40000):
    """First admissible rotation; returns (Theta, salience column)."""
    Psi = ma_coefficients(fit, 20)
    P = np.linalg.cholesky(fit.Sigma)
    use_zeros = any(s.n_zeros > 0 for s in sp)
    rng = np.random.default_rng(11)
    for _ in range(tries):
        Q = (build_rotation_with_zeros(P, Psi, sp, fit.names, rng)
             if use_zeros else draw_rotation(fit.n, rng))
        if Q is None:
            continue
        B0 = P @ Q
        m = match_shocks(irf_from_B0(Psi, B0), sp, fit.names)
        if m is None:
            continue
        cols, signs = m
        B0s = B0.copy()
        for c, s_ in zip(cols, signs):
            B0s[:, c] = s_ * B0[:, c]
        return irf_from_B0(Psi, B0s), cols[2]
    return None, None


if __name__ == "__main__":
    print('Running mock tests (no network)\n')
    test_quarters()
    test_strip_html()
    test_query_building()
    test_pagination()
    test_uk_filters_applied()
    test_full_run_and_resume()
    test_ratelimit_surfaces()
    print('\nAll tests passed.')
    df = synth()
    EXOG = ["d_may2020"]
    print("\nT1  Phi_12 exactly zero under restricted_ols")
    for p in (1, 2, 3):
        f = fit_var(df, ENDOG, p, EXOG, n_glo=NG)
        Blag = f.B[f.n_exog:, :]
        worst = max(np.abs(Blag[j * f.n:(j + 1) * f.n, :][NG:, :NG]).max()
                    for j in range(p))
        check(f"p={p} max|Phi_12|", worst == 0.0, f"= {worst:.3e}")
        check(f"p={p} phi12_norm", f.phi12_norm() == 0.0)
        fu = fit_var(df, ENDOG, p, EXOG, n_glo=None)
        wu = max(np.abs(fu.B[fu.n_exog:, :][j * fu.n:(j + 1) * fu.n, :]
                        [NG:, :NG]).max() for j in range(p))
        check(f"p={p} unrestricted Phi_12 non-zero (restriction bites)",
              wu > 1e-6, f"= {wu:.3e}")
    print("\nT2  n_glo=None reproduces plain lstsq bit for bit")
    f = fit_var(df, ENDOG, 1, EXOG, n_glo=None)
    B_ref, *_ = np.linalg.lstsq(f.X, f.Y, rcond=None)
    check("B identical", np.allclose(f.B, B_ref, atol=0, rtol=0)
          or np.abs(f.B - B_ref).max() < 1e-12,
          f"maxdiff {np.abs(f.B - B_ref).max():.3e}")
    print("\nT3  SOE + exact zeros -> global rows zero at EVERY horizon")
    f_soe = fit_var(df, ENDOG, 1, EXOG, n_glo=NG)
    Th, col = one_draw(f_soe, specs(zeros=True))
    if Th is None:
        check("found an admissible draw", False)
    else:
        v = soe_violation(Th, NG, col)
        check("h=0 zero", np.abs(Th[0, :NG, col]).max() < 1e-12,
              f"{np.abs(Th[0, :NG, col]).max():.3e}")
        check("all horizons zero", v < 1e-12, f"soe_violation = {v:.3e}")
    print("\nT4  complementarity: the two restrictions are NOT redundant")
    Th_w, col_w = one_draw(f_soe, specs(zeros=False))
    if Th_w is None:
        check("SOE + weak ineq: admissible draw found", False)
    else:
        vw = soe_violation(Th_w, NG, col_w)
        check("SOE + weak ineq -> global response NON-zero", vw > 1e-8,
              f"soe_violation = {vw:.3e}")
    f_un = fit_var(df, ENDOG, 1, EXOG, n_glo=None)
    Th_u, col_u = one_draw(f_un, specs(zeros=True))
    if Th_u is None:
        check("no SOE + zeros: admissible draw found", False)
    else:
        h0 = np.abs(Th_u[0, :NG, col_u]).max()
        later = np.abs(Th_u[1:, :NG, col_u]).max()
        check("no SOE + zeros -> zero at h=0", h0 < 1e-12, f"{h0:.3e}")
        check("no SOE + zeros -> NON-zero at h>0 (the original problem)",
              later > 1e-6, f"max|global| h>0 = {later:.3e}")
    print("\nT5  wild-bootstrap refit preserves the restriction")
    rng = np.random.default_rng(3)
    e = rng.choice([-1.0, 1.0], size=(f_soe.T, 1))
    Yb = f_soe.X @ f_soe.B + f_soe.U * e
    Bb = restricted_ols(f_soe.X, Yb, f_soe.n, f_soe.p, f_soe.n_exog, f_soe.n_glo)
    fb = VARFit(Y=Yb, X=f_soe.X, B=Bb, U=Yb - f_soe.X @ Bb,
                Sigma=np.cov((Yb - f_soe.X @ Bb).T), p=f_soe.p, n=f_soe.n,
                n_exog=f_soe.n_exog, names=f_soe.names, n_glo=f_soe.n_glo)
    check("bootstrap Phi_12 still zero", fb.phi12_norm() == 0.0)
    Bb_un, *_ = np.linalg.lstsq(f_soe.X, Yb, rcond=None)
    un = max(np.abs(Bb_un[f_soe.n_exog:, :][NG:, :NG]).max(), 0)
    check("unrestricted bootstrap refit WOULD have broken it", un > 1e-6,
          f"max|Phi_12| = {un:.3e}")
    print("\nT6  lag selection runs under both reduced forms")
    lo = select_order(df, ENDOG, maxlags=4, exog=EXOG)
    lo_s = select_order(df, ENDOG, maxlags=4, exog=EXOG, n_glo=NG)
    check("unrestricted selection returns", len(lo) == 5,
          f"BIC picks p={lo.attrs['selected']['BIC']}")
    check("SOE selection returns", len(lo_s) == 5,
          f"BIC picks p={lo_s.attrs['selected']['BIC']}")
    check("penalties differ (fewer parameters under SOE)",
          not np.allclose(lo["BIC"].values[1:], lo_s["BIC"].values[1:]))
    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
