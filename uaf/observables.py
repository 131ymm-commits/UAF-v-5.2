"""
uaf.observables — Early-warning signals from A(t)
==================================================
The v5.1 observational set, extracted from a single time series:
    AR1     → proximity to tipping (δ/δ*)        [EXP 060]
    γ₁      → type of tipping (bifurcation/noise) [EXP 061]
    Var(A)  → sign of co-evolution β             [EXP 074]

These are the quantities Phase-3 (real data) will test.
"""

import numpy as np


def ar1(series):
    """Lag-1 autocorrelation. → 1 near tipping (critical slowing down)."""
    x = np.asarray(series, dtype=float)
    if len(x) < 3:
        return np.nan
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:]  - x[1:].mean()
    denom = np.std(x[:-1]) * np.std(x[1:])
    return float(np.mean(x0 * x1) / (denom + 1e-12))


def variance(series):
    """Variance of the series. Rises near tipping; Var(A)→sign(β) in v5.1."""
    return float(np.var(np.asarray(series, dtype=float)))


def skewness(series):
    """Skewness γ₁. << 0: bifurcation tipping; ≈ 0: noise tipping."""
    x = np.asarray(series, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    return float(np.mean((x - mu)**3) / sd**3)


def lambda_from_ar1(ar1_val, dt=1.0):
    """Estimate eigenvalue λ from AR1: AR1 = exp(λ·dt) → λ = log(AR1)/dt."""
    a = float(np.clip(ar1_val, 1e-6, 1 - 1e-6))
    return np.log(a) / dt


def rolling_ews(series, window=200, step=None):
    """
    Compute AR1, Var, Skew in rolling windows.
    Returns dict of lists. Used to detect TRENDS (rising AR1 = approaching).
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if step is None:
        step = max(1, window // 2)
    ar1_list, var_list, skew_list, idx = [], [], [], []
    for i in range(window, n + 1, step):
        seg = x[i - window:i]
        ar1_list.append(ar1(seg))
        var_list.append(variance(seg))
        skew_list.append(skewness(seg))
        idx.append(i)
    return dict(idx=idx, ar1=ar1_list, var=var_list, skew=skew_list)


def ews_trend(rolling_result, key='ar1'):
    """Linear trend (slope) of a rolling EWS. Rising AR1 trend = warning."""
    vals = rolling_result[key]
    if len(vals) < 3:
        return 0.0
    return float(np.polyfit(range(len(vals)), vals, 1)[0])


def full_ews_report(series, dt=1.0):
    """Complete EWS snapshot from a single series."""
    a   = ar1(series)
    return dict(
        ar1       = a,
        variance  = variance(series),
        skewness  = skewness(series),
        lambda_est = lambda_from_ar1(a, dt),
    )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("uaf.observables self-test")
    # AR(1) process with known autocorrelation
    np.random.seed(0)
    rho = 0.9
    x = [0.0]
    for _ in range(2000):
        x.append(rho * x[-1] + np.random.normal(0, 0.1))
    x = np.array(x) + 0.5
    print(f"  AR1 (true ρ=0.9): {ar1(x):.4f}")
    print(f"  Var:              {variance(x):.6f}")
    print(f"  Skew:             {skewness(x):.4f}")
    print(f"  λ_est:            {lambda_from_ar1(ar1(x)):.4f}")
    roll = rolling_ews(x, window=300)
    print(f"  AR1 trend:        {ews_trend(roll,'ar1'):+.6f}")
