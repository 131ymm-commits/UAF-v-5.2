"""
uaf.core — Master equation, RHS, integrators
=============================================
Single source of truth for UAF dynamics. Experiments IMPORT from here,
never copy. Verified parameters from v5.0/v5.1.

Master equation:
    dA_i/dτ = α_s·C_ij·A_j·(1−A_i) + α_l·Π_i·PE_i·(1−A_i)
            + f·(1−A_i) − δ·(1−0.3·A_i)
"""

import numpy as np


# ── Verified default parameters (v5.0/v5.1) ──────────────────────────────────
DEFAULTS = dict(
    alpha_s = 0.06,   # social/TSV coupling
    alpha_l = 0.01,   # learning/FEP coupling
    f       = 0.002,  # basal floor
    delta   = 0.012,  # decay
    A_c     = 1.0,    # closure ceiling
    Pi      = 1.0,    # default precision
    decay_floor_coeff = 0.3,  # the 0.3 in (1 − 0.3·A)
)

DELTA_STAR = 0.0148   # saddle-node bifurcation point at f=0.002


# ── 1D mean-field RHS (single agent / uniform field) ─────────────────────────
def rhs_1d(A, delta=None, f=None, alpha_s=None, alpha_l=None, Pi=None):
    """
    1D reduction: single agent or uniform mean-field with self-coupling.
    TSV term uses A² (agent couples to field = itself).
    """
    p = DEFAULTS
    delta   = p['delta']   if delta   is None else delta
    f       = p['f']       if f       is None else f
    alpha_s = p['alpha_s'] if alpha_s is None else alpha_s
    alpha_l = p['alpha_l'] if alpha_l is None else alpha_l
    Pi      = p['Pi']      if Pi      is None else Pi

    A = float(np.clip(A, 1e-9, 1 - 1e-9))
    tsv = alpha_s * A**2 * (1 - A)
    fep = alpha_l * Pi * A * (1 - A)
    bas = f * (1 - A)
    dec = delta * (1 - p['decay_floor_coeff'] * A)
    return tsv + fep + bas - dec


def rhs_hmf(A, chi, delta=None, f=None, alpha_s=None, alpha_l=None, Pi=None):
    """
    Heterogeneous mean-field RHS with susceptibility χ = <k²>/<k>.
    χ replaces the effective coupling degree in the TSV term.
    This is the form used to derive χ·A*_uns = δ/α_s in v5.1.
    """
    p = DEFAULTS
    delta   = p['delta']   if delta   is None else delta
    f       = p['f']       if f       is None else f
    alpha_s = p['alpha_s'] if alpha_s is None else alpha_s
    alpha_l = p['alpha_l'] if alpha_l is None else alpha_l
    Pi      = p['Pi']      if Pi      is None else Pi

    A = float(np.clip(A, 1e-9, 1 - 1e-9))
    tsv = alpha_s * chi * A * (1 - A)
    fep = alpha_l * Pi * A * (1 - A)
    bas = f * (1 - A)
    dec = delta * (1 - p['decay_floor_coeff'] * A)
    return tsv + fep + bas - dec


# ── Full N-agent network RHS (ABM — no mean-field) ───────────────────────────
def rhs_network(A_vec, adjacency, delta=None, f=None,
                alpha_s=None, alpha_l=None, Pi_vec=None):
    """
    Full agent-based RHS on an explicit network.
    A_vec    : (N,) closure values
    adjacency: (N,N) coupling matrix C_ij (catalytic weights)
    Returns dA_vec/dτ — no mean-field approximation.

    THIS is what Phase-1 (Q20) uses to crash-test the HMF invariant.
    Each agent sees the actual sum over its real neighbours.
    """
    p = DEFAULTS
    delta   = p['delta']   if delta   is None else delta
    f       = p['f']       if f       is None else f
    alpha_s = p['alpha_s'] if alpha_s is None else alpha_s
    alpha_l = p['alpha_l'] if alpha_l is None else alpha_l

    A = np.clip(A_vec, 1e-9, 1 - 1e-9)
    N = len(A)
    Pi = np.ones(N) if Pi_vec is None else Pi_vec

    # TSV: each agent i receives sum_j C_ij · A_j, modulated by (1−A_i)
    field = adjacency @ A                       # (N,) neighbour field
    tsv   = alpha_s * field * (1 - A)
    fep   = alpha_l * Pi * A * (1 - A)
    bas   = f * (1 - A)
    dec   = delta * (1 - p['decay_floor_coeff'] * A)
    return tsv + fep + bas - dec


# ── Integrators ───────────────────────────────────────────────────────────────
def integrate_1d(A0, T=3000, dt=0.4, sigma=0.0, seed=None, **rhs_kwargs):
    """Euler-Maruyama integration of 1D system. Returns trajectory array."""
    if seed is not None:
        np.random.seed(seed)
    A = float(A0)
    traj = [A]
    for _ in range(int(T / dt)):
        dW = np.random.normal(0, dt**0.5) if sigma > 0 else 0.0
        A = float(np.clip(A + rhs_1d(A, **rhs_kwargs) * dt + sigma * dW,
                          1e-9, 1 - 1e-9))
        traj.append(A)
    return np.array(traj)


def integrate_network(A0_vec, adjacency, T=3000, dt=0.4,
                      sigma=0.0, seed=None, **rhs_kwargs):
    """
    Euler-Maruyama integration of full network. Returns (steps, N) array.
    Used by Phase-1 ABM experiments.
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.array(A0_vec, dtype=float)
    N = len(A)
    traj = [A.copy()]
    for _ in range(int(T / dt)):
        dW = np.random.normal(0, dt**0.5, N) if sigma > 0 else 0.0
        dA = rhs_network(A, adjacency, **rhs_kwargs)
        A = np.clip(A + dA * dt + sigma * dW, 1e-9, 1 - 1e-9)
        traj.append(A.copy())
    return np.array(traj)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("uaf.core self-test")
    print(f"  rhs_1d(0.5)        = {rhs_1d(0.5):.6f}")
    print(f"  rhs_hmf(0.5, χ=4)  = {rhs_hmf(0.5, 4.0):.6f}")
    # tiny network
    adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
    A0  = np.array([0.3, 0.5, 0.7])
    print(f"  rhs_network        = {rhs_network(A0, adj)}")
    traj = integrate_1d(0.7, T=500, dt=0.5)
    print(f"  integrate_1d → A_final = {traj[-1]:.5f}")
