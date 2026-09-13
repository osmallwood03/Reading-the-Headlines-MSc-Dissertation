"""
Reduced form, rotation sampling and impulse responses for a sign- and
zero-restricted SVAR.

Zero restrictions are part of the baseline rather than a variant, so rotations
are drawn following Arias, Rubio-Ramirez and Waggoner (2018) rather than
Haar-uniform: a randomly drawn column satisfies an exact zero with probability
zero. ARW require columns to be built in descending order of the number of
zeros they carry, because constructing a two-zero column after two
orthogonality constraints have already been imposed draws from a different
conditional distribution. build_rotation_with_zeros sorts by zero count and
completes the remainder with a Haar fill.

Violations are measured rather than assumed: zero_violation() reports the
largest response that was meant to be zero, scaled by the norm of its column,
and every run prints it.

restricted_ols() estimates the reduced form equation by equation with domestic
lags excluded from the global equations. With n_glo=None it reduces to the
plain least-squares fit bit for bit. Estimation is OLS, not GLS: with different
regressors across equations the system is a SUR, so this is consistent but not
efficient, and the wild bootstrap carries the sampling uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

ZTOL_REL = 1e-9          # relative tolerance for "this response is zero"


# 1. REDUCED-FORM VAR

@dataclass
class VARFit:
    Y: np.ndarray          # (T, n) endogenous, aligned to estimation sample
    X: np.ndarray          # (T, k) regressors incl. deterministics and lags
    B: np.ndarray          # (k, n) OLS coefficients
    U: np.ndarray          # (T, n) residuals
    Sigma: np.ndarray      # (n, n) residual covariance
    p: int
    n: int
    n_exog: int            # number of deterministic/exogenous columns
    names: list
    n_glo: int | None = None   # size of the exogenous global block, or None

    @property
    def T(self):
        return self.Y.shape[0]

    @property
    def is_soe(self) -> bool:
        return self.n_glo is not None and self.n_glo > 0

    def companion(self) -> np.ndarray:
        n, p = self.n, self.p
        Blag = self.B[self.n_exog:, :]
        A = np.zeros((n * p, n * p))
        A[:n, :] = Blag.T
        if p > 1:
            A[n:, :-n] = np.eye(n * (p - 1))
        return A

    def phi12_norm(self) -> float:
        """
        Largest |coefficient| on a domestic lag in a global equation, relative
        to the largest coefficient anywhere in the lag block. Zero by
        construction when the SOE restriction is imposed; reported on every
        run for the same reason zero_violation() is, namely that this project
        has previously shipped a restriction flag the estimator ignored.
        """
        if not self.is_soe:
            return float("nan")
        ng, n, p = self.n_glo, self.n, self.p
        Blag = self.B[self.n_exog:, :]                 # (n*p, n)
        scale = np.abs(Blag).max() or 1.0
        worst = 0.0
        for j in range(p):
            block = Blag[j * n:(j + 1) * n, :]         # (n, n) lag-j
            worst = max(worst, np.abs(block[ng:, :ng]).max())
        return float(worst / scale)


def build_matrices(df: pd.DataFrame, endog: list, p: int,
                   exog: list | None = None, const: bool = True):
    """Construct Y and X for a VAR(p). Rows with any NaN are dropped."""
    exog = exog or []
    data = df[endog + exog].to_numpy(dtype=float)
    n = len(endog)
    rows_Y, rows_X = [], []
    for t in range(p, data.shape[0]):
        block = data[t - p:t + 1]
        if np.isnan(block).any():
            continue
        x = []
        if const:
            x.append(1.0)
        if exog:
            x.extend(data[t, n:])
        for j in range(1, p + 1):
            x.extend(data[t - j, :n])
        rows_Y.append(data[t, :n])
        rows_X.append(x)
    return np.asarray(rows_Y), np.asarray(rows_X)


def soe_regressor_mask(n: int, p: int, n_exog: int, n_glo: int,
                       eq: int) -> np.ndarray:
    """
    Boolean mask over the columns of X giving the regressors equation `eq` is
    allowed to use under block exogeneity.

    Deterministics are always allowed. Global equations (eq < n_glo) may use
    global lags only; domestic equations may use everything. Variables must be
    ORDERED GLOBAL FIRST in endog for this to mean what it says — fit_var
    cannot check that for you, so the ordering is a caller obligation.
    """
    mask = np.ones(n_exog + n * p, dtype=bool)
    if eq < n_glo:
        for j in range(p):
            start = n_exog + j * n
            mask[start + n_glo:start + n] = False
    return mask


def restricted_ols(X: np.ndarray, Y: np.ndarray, n: int, p: int,
                   n_exog: int, n_glo: int | None) -> np.ndarray:
    """
    Equation-by-equation OLS honouring the SOE restriction. Returns B of the
    same (k, n) shape as unrestricted OLS, with exact zeros in the restricted
    positions, so companion() and everything downstream are unchanged.

    Shared by fit_var and the wild bootstrap. The bootstrap MUST use this: an
    unrestricted refit of each replication would silently discard the
    restriction and produce bands for a model that was never estimated.
    """
    if n_glo is None or n_glo <= 0:
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return B
    k = X.shape[1]
    B = np.zeros((k, n))
    for eq in range(n):
        m = soe_regressor_mask(n, p, n_exog, n_glo, eq)
        b, *_ = np.linalg.lstsq(X[:, m], Y[:, eq], rcond=None)
        B[m, eq] = b
    return B


def fit_var(df: pd.DataFrame, endog: list, p: int,
            exog: list | None = None, const: bool = True,
            n_glo: int | None = None) -> VARFit:
    """
    n_glo=None  unrestricted VAR (the pre-SOE reduced form).
    n_glo=k     first k variables of `endog` form an exogenous global block:
                their equations exclude all domestic lags.
    """
    Y, X = build_matrices(df, endog, p, exog, const)
    if Y.size == 0:
        raise ValueError("no usable rows; check for NaNs in the endogenous set")
    T, n = Y.shape
    n_exog = (1 if const else 0) + len(exog or [])
    if n_glo is not None and not (0 < n_glo < n):
        raise ValueError(f"n_glo={n_glo} must satisfy 0 < n_glo < n={n}")
    B = restricted_ols(X, Y, n, p, n_exog, n_glo)
    U = Y - X @ B
    # Common divisor across equations. Under SOE the global equations use
    # fewer regressors, so per-equation dof differ; T - k_max is the
    # conservative common choice and keeps Sigma comparable with the
    # unrestricted fit.
    k_max = X.shape[1]
    Sigma = (U.T @ U) / (T - k_max)
    return VARFit(Y=Y, X=X, B=B, U=U, Sigma=Sigma, p=p, n=n,
                  n_exog=n_exog, names=list(endog), n_glo=n_glo)


def select_order(df: pd.DataFrame, endog: list, maxlags: int = 4,
                 exog: list | None = None,
                 n_glo: int | None = None) -> pd.DataFrame:
    """
    Information criteria on a COMMON sample (all fits use maxlags rows).

    n_glo re-runs the selection under block exogeneity. This is NOT cosmetic.
    Dropping n_dom * n_glo * p coefficients from the global equations changes
    the parameter count the penalty term is applied to, so the restricted
    system can select a different lag order from the unrestricted one. Do not
    carry p=1 over from the unrestricted fit without re-checking it here.
    """
    rows = []
    n = len(endog)
    n_ex = 1 + len(exog or [])
    _, Xmax = build_matrices(df, endog, maxlags, exog)
    T_common = Xmax.shape[0]
    for p in range(0, maxlags + 1):
        Y, X = build_matrices(df, endog, max(p, 1), exog)
        Y, X = Y[-T_common:], X[-T_common:]
        X = X[:, :(n_ex + n * p)] if p else X[:, :n_ex]
        B = restricted_ols(X, Y, n, p, n_ex, n_glo if p else None)
        U = Y - X @ B
        S = (U.T @ U) / T_common
        _, ld = np.linalg.slogdet(S)
        if n_glo and p:
            npar = (n_glo * (n_ex + n_glo * p)
                    + (n - n_glo) * (n_ex + n * p))
        else:
            npar = X.shape[1] * n
        kbar = npar / n            # average regressors per equation, for FPE
        rows.append({
            "p": p,
            "AIC":  ld + 2 * npar / T_common,
            "BIC":  ld + np.log(T_common) * npar / T_common,
            "HQIC": ld + 2 * np.log(np.log(T_common)) * npar / T_common,
            "FPE":  np.exp(ld) * ((T_common + kbar) /
                                  (T_common - kbar)) ** n,
        })
    out = pd.DataFrame(rows).set_index("p")
    out.attrs["selected"] = {c: int(out[c].idxmin()) for c in out.columns}
    out.attrs["n_glo"] = n_glo
    return out


# 2. IMPULSE RESPONSES / FEVD / HISTORICAL DECOMPOSITION

def ma_coefficients(fit: VARFit, H: int) -> np.ndarray:
    n, p = fit.n, fit.p
    A = fit.companion()
    J = np.zeros((n, n * p)); J[:, :n] = np.eye(n)
    Psi = np.zeros((H + 1, n, n))
    Ah = np.eye(n * p)
    for h in range(H + 1):
        Psi[h] = J @ Ah @ J.T
        Ah = Ah @ A
    return Psi


def irf_from_B0(Psi: np.ndarray, B0: np.ndarray) -> np.ndarray:
    return np.einsum("hij,jk->hik", Psi, B0)


def fevd_from_theta(Theta: np.ndarray) -> np.ndarray:
    C = np.cumsum(Theta ** 2, axis=0)
    tot = C.sum(axis=2, keepdims=True)
    return C / np.where(tot == 0, np.nan, tot)


def historical_decomposition(fit: VARFit, B0: np.ndarray,
                             Theta: np.ndarray) -> np.ndarray:
    """Contribution of each shock to each variable, shape (T, n_var, n_shock)."""
    eps = fit.U @ np.linalg.inv(B0).T
    T, n = eps.shape
    H = Theta.shape[0] - 1
    hd = np.zeros((T, n, n))
    for t in range(T):
        for j in range(0, min(t, H) + 1):
            hd[t] += Theta[j] * eps[t - j][None, :]
    return hd


# 3. RESTRICTIONS

@dataclass
class ShockSpec:
    """
    One labelled structural shock.

    restrictions: {horizon: {variable_name: '+' | '-' | '<=0' | '>=0'}}
    zeros:        {horizon: [variable_name, ...]}   exact zero restrictions
    """
    name: str
    restrictions: dict = field(default_factory=dict)
    zeros: dict = field(default_factory=dict)

    @property
    def n_zeros(self) -> int:
        return sum(len(v) for v in self.zeros.values())


def _check_column(Theta: np.ndarray, col: int, spec: ShockSpec,
                  names: list, sign: int, ztol_rel: float = ZTOL_REL) -> bool:
    for h, reqs in spec.restrictions.items():
        if h >= Theta.shape[0]:
            return False
        for var, rule in reqs.items():
            v = sign * Theta[h, names.index(var), col]
            if   rule == "+"   and not v > 0:   return False
            elif rule == "-"   and not v < 0:   return False
            elif rule == "<=0" and not v <= 0:  return False
            elif rule == ">=0" and not v >= 0:  return False
    for h, vars_ in spec.zeros.items():
        if h >= Theta.shape[0]:
            return False
        scale = np.linalg.norm(Theta[h, :, col]) or 1.0
        for var in vars_:
            if abs(Theta[h, names.index(var), col]) > ztol_rel * scale:
                return False
    return True


def match_shocks(Theta: np.ndarray, specs: list, names: list):
    """
    Assign each labelled shock to a distinct column, with a sign flip if
    needed. Returns (cols, signs) or None.

    Specs carrying zero restrictions are matched FIRST, because far fewer
    columns can satisfy them, which prunes the search early.
    """
    n = Theta.shape[1]
    order = sorted(range(len(specs)), key=lambda i: -specs[i].n_zeros)
    feasible = []
    for i in order:
        opts = [(c, s) for c in range(n) for s in (1, -1)
                if _check_column(Theta, c, specs[i], names, s)]
        if not opts:
            return None
        feasible.append(opts)
    for combo in _assignments(feasible):
        cols = [c for c, _ in combo]
        if len(set(cols)) == len(cols):
            out_c = [0] * len(specs)
            out_s = [1] * len(specs)
            for slot, (c, s) in zip(order, combo):
                out_c[slot], out_s[slot] = c, s
            return out_c, out_s
    return None


def _assignments(feasible, prefix=None, used=None):
    prefix = prefix or []; used = used or set()
    i = len(prefix)
    if i == len(feasible):
        yield list(prefix); return
    for c, s in feasible[i]:
        if c in used:
            continue
        yield from _assignments(feasible, prefix + [(c, s)], used | {c})


def zero_violation(Theta: np.ndarray, specs: list, names: list,
                   cols: list) -> float:
    """
    Largest relative violation of any zero restriction, across all specs.
    Should be at machine precision when the zeros are imposed by
    construction. Reported on every run as a guard against the zeros flag
    being silently ignored, which has happened before in this project.
    """
    worst = 0.0
    for spec, col in zip(specs, cols):
        for h, vars_ in spec.zeros.items():
            scale = np.linalg.norm(Theta[h, :, col]) or 1.0
            for var in vars_:
                worst = max(worst,
                            abs(Theta[h, names.index(var), col]) / scale)
    return worst


def soe_violation(Theta: np.ndarray, n_glo: int, col: int) -> float:
    """
    Largest relative response of the GLOBAL block to shock `col`, across all
    horizons. Scaled pointwise by the norm of the response vector at that
    horizon, so it is comparable with zero_violation().

    INTERPRETATION DEPENDS ON THE SCHEME, and this is the thing to get right
    in the write-up:

      exact-zero schemes (L1/L2)  b_glo = 0, so this should be at machine
                                  precision at EVERY horizon. Anything larger
                                  means either the restriction is not binding
                                  in the reduced form or the zeros are not
                                  binding in the rotation.

      weak-inequality schemes     b_glo <= 0 and generally non-zero, so this
      (L0/L3)                     is NOT expected to be zero. It measures the
                                  decay of Phi_11^h b_glo. A large number is
                                  informative, not an error.
    """
    H = Theta.shape[0]
    worst = 0.0
    for h in range(H):
        scale = np.linalg.norm(Theta[h, :, col]) or 1.0
        worst = max(worst, np.abs(Theta[h, :n_glo, col]).max() / scale)
    return float(worst)


# 4. ROTATIONS

def draw_rotation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform orthogonal matrix via QR of a Gaussian matrix."""
    Z = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    return Q * np.sign(np.diag(R))[None, :]


def build_rotation_with_zeros(P, Psi, specs, names, rng):
    """
    Arias, Rubio-Ramirez and Waggoner (2018) construction.

    Columns carrying zero restrictions are built FIRST, in descending order
    of the number of zeros they carry, each drawn uniformly from the null
    space of its own zero restrictions stacked on the columns already fixed.
    The remainder of the orthogonal matrix is then completed with a Haar
    fill. Returns None if the constraint stack leaves no null space.

    The descending order is not cosmetic. ARW's parameterisation requires
    z_1 >= z_2 >= ... >= z_n; building a two-zero column after two
    orthogonality constraints have already been imposed draws from a
    different conditional distribution than the one the theorem covers.
    """
    n = P.shape[0]
    with_zeros = [s for s in specs if s.n_zeros > 0]
    with_zeros.sort(key=lambda s: -s.n_zeros)

    Qcols = []
    for spec in with_zeros:
        Rrows = []
        for h, vars_ in spec.zeros.items():
            M_h = Psi[h] @ P
            for var in vars_:
                Rrows.append(M_h[names.index(var), :])
        stack = Rrows + [q for q in Qcols]
        M = np.vstack(stack) if stack else np.zeros((0, n))
        if M.shape[0] == 0:
            q = rng.standard_normal(n)
        else:
            _, sv, Vt = np.linalg.svd(M)
            rank = int((sv > 1e-12 * (sv[0] if sv.size else 1.0)).sum())
            N = Vt[rank:].T
            if N.shape[1] == 0:
                return None
            q = N @ rng.standard_normal(N.shape[1])
        nrm = np.linalg.norm(q)
        if nrm < 1e-12:
            return None
        Qcols.append(q / nrm)

    if not Qcols:
        return draw_rotation(n, rng)

    Q = np.column_stack(Qcols)
    Zfill = rng.standard_normal((n, n - Q.shape[1]))
    for _ in range(2):                      # twice for numerical stability
        Zfill -= Q @ (Q.T @ Zfill)
    Qf, _ = np.linalg.qr(Zfill)
    return np.hstack([Q, Qf])


# 5. SUMMARY HELPERS

def bands(arr: np.ndarray, lo=16, hi=84, axis=0):
    """Pointwise median and percentile bands across draws."""
    return (np.percentile(arr, 50, axis=axis),
            np.percentile(arr, lo, axis=axis),
            np.percentile(arr, hi, axis=axis))


def shock_index(shock_names: list, name: str = "media_salience") -> int:
    """Position of a labelled shock. Never index it positionally."""
    return list(shock_names).index(name)
