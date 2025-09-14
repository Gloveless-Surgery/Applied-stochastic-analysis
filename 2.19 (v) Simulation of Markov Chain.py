import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# part (v)

P = np.array([
    [0, 0, 0, 0, 1/2, 0, 1/2, 0, 0],  
    [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0],  
    [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0],  
    [1/2, 0, 0, 0, 1/2, 0, 0, 0, 0],  
    [1/2, 0, 0, 0, 0,   0, 1/2, 0, 0],
    [0, 0, 0, 1/2, 0,   0, 1/2, 0, 0],
    [0, 0, 0, 1/2, 0,   0, 0,   0, 1/2],
    [0, 0, 0, 0, 0,   0, 0,   0, 1],
    [0, 0, 0, 0, 0,   0, 0,   0, 1],
], dtype=float)




def _inverse_draw(p_row, rng):
    """
    Generate Y_i from row p_row via inversion:
    choose j with sum_{k=0}^{j-1} p_row[k] < U <= sum_{k=0}^j p_row[k].
    """
    u = rng.random()                # U ~ Uniform(0,1)
    cdf = np.cumsum(p_row)
    j = np.searchsorted(cdf, u, side="left")
    return int(j)

def simulate_mc_inversion(P, i0, N, seed=None):
    """
    Algorithm:
      1. Choose X_0 = i0; set n = 1
      2. Generate Y_{i0} and set X_1 = Y_{i0}
      3. If n < N then set i = X_n, generate Y_i, set n := n+1 and X_n := Y_i; else stop.
    States are 0..S-1.
    """
    S = P.shape[0]
    
    rng = np.random.default_rng(seed)

    X = np.empty(N + 1, dtype=int)
    X[0] = i0
    n = 1
    # step 2
    X[1] = _inverse_draw(P[i0], rng)

    # steps 3–4
    while n < N:
        i = X[n]
        n += 1
        X[n] = _inverse_draw(P[i], rng)
    return X

# MFPT via simulation 

def first_passage_time(P, start, target, max_steps=100, seed=None):
    """
    T = min{n >= 1 : X_n = target} starting at X_0 = start.
    Returns np.inf if not hit within max_steps.
    """
    rng = np.random.default_rng(seed)
    x = start
    for n in range(1, max_steps + 1):
        x = _inverse_draw(P[x], rng)
        if x == target:
            return n
    return np.inf

def estimate_mfpt(P, start, target, n_runs, max_steps=100_000, seed=None):
    """
    Monte-Carlo estimate of E_start[T_target] using the inversion simulator.
    Returns (mean, stderr, hits, misses).
    """
    rng = np.random.default_rng(seed)
    times = []
    misses = 0
    for _ in range(n_runs):
        # re-seed per run by advancing RNG to keep independence
        t = first_passage_time(P, start, target, max_steps=max_steps, seed=rng.integers(2**63))
        if np.isfinite(t):
            times.append(t)
        else:
            misses += 1
    if not times:
        return np.inf, np.nan, 0, misses
    arr = np.asarray(times, float)
    mean = arr.mean()
    stderr = arr.std(ddof=1)/np.sqrt(arr.size) if arr.size > 1 else np.nan
    return mean, stderr, arr.size, misses


traj = simulate_mc_inversion(P, i0=0, N=20)
mfpt_mean, mfpt_se, hits, misses = estimate_mfpt(P, start=0, target=8, n_runs=1000)
print(traj)
print(mfpt_mean, mfpt_se, hits, misses)