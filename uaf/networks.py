"""
uaf.networks — Topologies, susceptibility χ, network statistics
================================================================
Generates explicit networks (for ABM) AND computes χ = <k²>/<k>
(for HMF). Phase-1 experiments compare the two.

KEY QUANTITY:
    χ = <k²>/<k>          (undirected susceptibility)
    χ_dir = <k_in·k_out>/<k>  (directed)

The v5.1 invariant: χ·A*_uns = δ/α_s.
"""

import numpy as np


# ── Susceptibility from a degree sequence ────────────────────────────────────
def chi_from_degrees(degrees):
    """χ = <k²>/<k> from an explicit degree sequence."""
    k = np.asarray(degrees, dtype=float)
    mk  = k.mean()
    mk2 = (k**2).mean()
    return mk2 / mk if mk > 0 else 0.0


def chi_dir_from_degrees(k_in, k_out):
    """χ_dir = <k_in·k_out>/<k> from in/out degree sequences."""
    k_in  = np.asarray(k_in, dtype=float)
    k_out = np.asarray(k_out, dtype=float)
    mk = 0.5 * (k_in.mean() + k_out.mean())
    return float(np.mean(k_in * k_out) / mk) if mk > 0 else 0.0


def chi_from_adjacency(adjacency):
    """χ = <k²>/<k> directly from an adjacency matrix (row sums = degree)."""
    deg = np.asarray(adjacency).sum(axis=1)
    return chi_from_degrees(deg)


# ── Network generators (explicit edges for ABM) ──────────────────────────────
def make_ba(N, m, seed=None):
    """
    Barabási-Albert network via preferential attachment.
    Returns symmetric adjacency matrix (N,N) with 0/1 entries.
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((N, N))
    # start with m fully-connected seed nodes
    for i in range(m):
        for j in range(i + 1, m):
            adj[i, j] = adj[j, i] = 1
    targets = list(range(m))
    repeated = list(range(m)) * m   # for preferential attachment
    for new in range(m, N):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rng.choice(repeated))
        for t in chosen:
            adj[new, t] = adj[t, new] = 1
        repeated.extend(chosen)
        repeated.extend([new] * m)
    return adj


def make_er(N, mean_degree, seed=None):
    """Erdős-Rényi G(N, p) with p = <k>/(N−1). Returns adjacency."""
    rng = np.random.default_rng(seed)
    p = mean_degree / (N - 1)
    adj = (rng.random((N, N)) < p).astype(float)
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    return adj


def make_watts_strogatz(N, k, beta, seed=None):
    """
    Watts-Strogatz small-world. k = each node connected to k nearest
    neighbours (k even), beta = rewiring probability.
    High clustering for low beta — used in Q21 (clustering test).
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((N, N))
    half = k // 2
    for i in range(N):
        for j in range(1, half + 1):
            adj[i, (i + j) % N] = adj[(i + j) % N, i] = 1
    # rewire
    for i in range(N):
        for j in range(1, half + 1):
            if rng.random() < beta:
                nb = (i + j) % N
                adj[i, nb] = adj[nb, i] = 0
                new = rng.integers(N)
                while new == i or adj[i, new] > 0:
                    new = rng.integers(N)
                adj[i, new] = adj[new, i] = 1
    return adj


def make_regular(N, k):
    """k-regular ring lattice. χ = k exactly."""
    adj = np.zeros((N, N))
    half = k // 2
    for i in range(N):
        for j in range(1, half + 1):
            adj[i, (i + j) % N] = adj[(i + j) % N, i] = 1
    return adj


def make_star(N):
    """Star: node 0 is hub connected to all others."""
    adj = np.zeros((N, N))
    adj[0, 1:] = 1
    adj[1:, 0] = 1
    return adj


# ── Clustering coefficient ───────────────────────────────────────────────────
def clustering_coefficient(adjacency):
    """Global clustering coefficient (transitivity)."""
    A = (np.asarray(adjacency) > 0).astype(float)
    deg = A.sum(axis=1)
    triangles = np.trace(A @ A @ A) / 6.0
    triples = np.sum(deg * (deg - 1)) / 2.0
    return float(3 * triangles / triples) if triples > 0 else 0.0


# ── BA analytical degree distribution (for HMF) ──────────────────────────────
def ba_stats(m, N=200):
    """Analytical BA degree distribution P(k)~2m²/k³. Returns (<k>, χ)."""
    k_max = max(m + 1, int(np.sqrt(N)))
    k = np.arange(m, k_max + 1, dtype=float)
    P = 2 * m**2 / k**3
    P /= P.sum()
    mk  = float(np.dot(k, P))
    mk2 = float(np.dot(k**2, P))
    return mk, mk2 / mk


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("uaf.networks self-test")
    adj = make_ba(200, 3, seed=1)
    deg = adj.sum(axis=1)
    print(f"  BA(200,3): <k>={deg.mean():.3f}  χ={chi_from_adjacency(adj):.3f}")
    print(f"  BA analytical χ = {ba_stats(3,200)[1]:.3f}")
    adj_ws = make_watts_strogatz(200, 6, 0.1, seed=1)
    print(f"  WS(200,6,0.1): χ={chi_from_adjacency(adj_ws):.3f}  "
          f"C={clustering_coefficient(adj_ws):.4f}")
    adj_er = make_er(200, 6, seed=1)
    print(f"  ER(200,<k>=6): χ={chi_from_adjacency(adj_er):.3f}  "
          f"C={clustering_coefficient(adj_er):.4f}")
