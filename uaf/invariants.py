"""
uaf.invariants — Verified reference formulas from v5.1
=======================================================
These are the formulas v5.2 is testing. They are stored here as the
"prediction" against which ABM and real-data measurements are compared.

ALL verified INSIDE the HMF model (CV < 0.003). v5.2 asks: do they hold
OUTSIDE it?
"""

import numpy as np
from uaf.core import DEFAULTS


# ── Verified constants (v5.1) ────────────────────────────────────────────────
K_INF      = DEFAULTS['delta'] / DEFAULTS['alpha_s']   # δ/α_s = 0.200
K0         = 0.593     # correction factor (δ/δ* ∈ [0.70, 0.88])
SLOPE_1D   = -28.0     # dA*_uns/df in 1D
CHI_SLOPE  = -16.61    # χ·slope invariant
C_PI       = 0.30      # precision-capital coefficient


# ── Predicted watershed (the central invariant) ──────────────────────────────
def predicted_watershed(chi, delta=None, f=0.0, alpha_s=None, alpha_l=None):
    """
    v5.1 prediction: A*_uns = δ / (α_s·χ + α_l) − (16.61/χ)·f

    This is what ABM simulations (Q20) must reproduce if the invariant
    holds outside mean-field.
    """
    p = DEFAULTS
    delta   = p['delta']   if delta   is None else delta
    alpha_s = p['alpha_s'] if alpha_s is None else alpha_s
    alpha_l = p['alpha_l'] if alpha_l is None else alpha_l
    base  = delta / (alpha_s * chi + alpha_l)
    floor_term = (CHI_SLOPE / chi) * f if chi > 0 else 0.0
    return base + floor_term


def predicted_K(chi, delta=None, alpha_s=None, alpha_l=None):
    """
    Predicted invariant K = χ·A*_uns(f=0) = δ·χ/(α_s·χ + α_l).
    → δ/α_s as χ → ∞.
    """
    p = DEFAULTS
    delta   = p['delta']   if delta   is None else delta
    alpha_s = p['alpha_s'] if alpha_s is None else alpha_s
    alpha_l = p['alpha_l'] if alpha_l is None else alpha_l
    return delta * chi / (alpha_s * chi + alpha_l)


# ── Optimal learning rate (Q8) ───────────────────────────────────────────────
def eta_optimal(eta_min, eta_max):
    """η* = √(η_min · η_max) — geometric mean (Q8)."""
    return float(np.sqrt(eta_min * eta_max))


# ── Floor design rule (Q15) ──────────────────────────────────────────────────
def floor_for_target(A_uns_zero, target, mk):
    """
    Required floor f* to reach watershed `target` from baseline A_uns_zero.
    f* = (A_uns_zero − target)·<k> / 16.61
    """
    return max(0.0, (A_uns_zero - target) * mk / abs(CHI_SLOPE))


# ── Verdict helper ───────────────────────────────────────────────────────────
def verdict(measured, predicted, tol_hold=0.05, tol_partial=0.20):
    """
    Compare measured vs predicted. Returns ('ДЕРЖИТСЯ'|'ЧАСТИЧНО'|'НЕ ДЕРЖИТСЯ',
    relative_deviation).
    """
    if predicted == 0:
        return ("N/A", np.nan)
    dev = abs(measured - predicted) / abs(predicted)
    if dev < tol_hold:
        label = "ДЕРЖИТСЯ"
    elif dev < tol_partial:
        label = "ЧАСТИЧНО"
    else:
        label = "НЕ ДЕРЖИТСЯ"
    return (label, dev)


# ── Reference table ──────────────────────────────────────────────────────────
INVARIANT_TABLE = {
    'K_inf':     ('δ/α_s',            K_INF),
    'chi_A':     ('χ·A*_uns',         K_INF),
    'chi_slope': ('χ·slope',          CHI_SLOPE),
    'slope_1d':  ('dA*/df',           SLOPE_1D),
    'k0':        ('correction factor', K0),
    'c_pi':      ('precision coeff',   C_PI),
}


if __name__ == "__main__":
    print("uaf.invariants self-test")
    print(f"  K_inf = δ/α_s = {K_INF:.4f}")
    for chi in [2, 4, 8, 20]:
        a = predicted_watershed(chi)
        print(f"  χ={chi:>3}: A*_pred={a:.5f}  χ·A*={chi*a:.5f}  "
              f"K_pred={predicted_K(chi):.5f}")
    lab, dev = verdict(0.198, K_INF)
    print(f"  verdict(0.198 vs {K_INF}): {lab} (dev={dev:.3f})")
