"""
uaf.analytics — 1D analysis: fixed points, watershed, δ*, Kramers
==================================================================
Verified analytical tools from v5.0/v5.1. Single source of truth.
"""

import numpy as np
from scipy.optimize import brentq
from uaf.core import rhs_1d, rhs_hmf, DEFAULTS


# ── Fixed points and watershed ───────────────────────────────────────────────
def find_fixed_points(rhs_fn, n_scan=5000):
    """
    Find all fixed points of a 1D RHS by sign-change scan + brentq.
    Returns list of (A*, lambda) where lambda = rhs'(A*).
    """
    A_grid = np.linspace(0.001, 0.999, n_scan)
    vals   = [rhs_fn(a) for a in A_grid]
    fps = []
    for i in range(len(A_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            try:
                a   = brentq(rhs_fn, A_grid[i], A_grid[i + 1], xtol=1e-10)
                lam = (rhs_fn(a + 1e-6) - rhs_fn(a - 1e-6)) / 2e-6
                fps.append((a, lam))
            except Exception:
                pass
    return fps


def get_watershed(rhs_fn):
    """Return the unstable fixed point (watershed A*_uns), or None."""
    for a, lam in find_fixed_points(rhs_fn):
        if lam > 0:
            return a
    return None


def watershed_1d(delta=None, f=None, **kw):
    """Watershed for 1D system at given parameters."""
    return get_watershed(lambda A: rhs_1d(A, delta=delta, f=f, **kw))


def watershed_hmf(chi, delta=None, f=None, **kw):
    """Watershed for HMF system with susceptibility χ."""
    return get_watershed(lambda A: rhs_hmf(A, chi, delta=delta, f=f, **kw))


# ── Bifurcation point δ* ──────────────────────────────────────────────────────
def delta_star_1d(f=None, lo=0.005, hi=0.040, n=3000):
    """Find saddle-node δ* for 1D system: where watershed disappears."""
    for d in np.linspace(lo, hi, n):
        if watershed_1d(delta=d, f=f) is None:
            return d
    return None


def delta_star_hmf(chi, f=None, lo=0.005, hi=0.080, n=3000):
    """Find δ* for HMF system at susceptibility χ."""
    for d in np.linspace(lo, hi, n):
        if watershed_hmf(chi, delta=d, f=f) is None:
            return d
    return None


# ── LST slope (dA*/df) ───────────────────────────────────────────────────────
def lst_slope_1d(delta=None, f_vals=(0.0, 0.001, 0.002, 0.003, 0.004, 0.005)):
    """Linear-shift-theorem slope dA*_uns/df for 1D system."""
    rows = [(f, watershed_1d(delta=delta, f=f)) for f in f_vals]
    rows = [(f, a) for f, a in rows if a is not None]
    if len(rows) < 3:
        return None
    return float(np.polyfit([r[0] for r in rows], [r[1] for r in rows], 1)[0])


def lst_slope_hmf(chi, delta=None,
                  f_vals=(0.0, 0.001, 0.002, 0.003, 0.004, 0.005)):
    """LST slope for HMF system at susceptibility χ."""
    rows = [(f, watershed_hmf(chi, delta=delta, f=f)) for f in f_vals]
    rows = [(f, a) for f, a in rows if a is not None]
    if len(rows) < 3:
        return None
    return float(np.polyfit([r[0] for r in rows], [r[1] for r in rows], 1)[0])


# ── Quasipotential and Kramers escape ────────────────────────────────────────
def quasipotential(rhs_fn, A_grid=None):
    """V(A) = −∫ rhs dA. Returns (A_grid, V)."""
    if A_grid is None:
        A_grid = np.linspace(0.005, 0.995, 4000)
    V = np.zeros_like(A_grid)
    for i in range(1, len(A_grid)):
        mid = 0.5 * (A_grid[i] + A_grid[i - 1])
        V[i] = V[i - 1] - rhs_fn(mid) * (A_grid[i] - A_grid[i - 1])
    return A_grid, V


def barrier_height(rhs_fn):
    """ΔV between life attractor and watershed. Returns barrier or None."""
    fps = find_fixed_points(rhs_fn)
    stable   = [a for a, l in fps if l < 0 and a > 0.3]
    unstable = [a for a, l in fps if l > 0]
    if not stable or not unstable:
        return None
    A_grid, V = quasipotential(rhs_fn)
    V_life = float(np.interp(max(stable), A_grid, V))
    V_uns  = float(np.interp(unstable[0], A_grid, V))
    return V_uns - V_life


def kramers_rate(rhs_fn, sigma):
    """Approximate Kramers escape rate ~ exp(−2ΔV/σ²)."""
    dV = barrier_height(rhs_fn)
    if dV is None or dV <= 0:
        return None
    return float(np.exp(-2 * dV / sigma**2))


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("uaf.analytics self-test")
    print(f"  watershed_1d(δ=0.012)      = {watershed_1d(delta=0.012):.5f}")
    print(f"  watershed_hmf(χ=4, δ=0.012)= {watershed_hmf(4.0, delta=0.012):.5f}")
    print(f"  δ*_1d(f=0.002)             = {delta_star_1d(f=0.002):.5f}")
    print(f"  lst_slope_1d(δ=0.012)      = {lst_slope_1d(delta=0.012):.2f}")
    chi = 4.0
    a = watershed_hmf(chi, delta=0.012)
    print(f"  χ·A*_uns (χ=4)             = {chi*a:.5f}  (expect ≈ 0.200)")
