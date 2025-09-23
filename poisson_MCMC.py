import math
import random
from typing import Tuple, Iterable
import numpy as np

def log_pois_pmf(i: int, lam: float) -> float: # Poisson distribution
    if i < 0:
        return float("-inf")
    return -lam + i * math.log(lam) - math.lgamma(i + 1)


def H_entry(i: int, j: int, M: int) -> float: # Transition matrix
    d = j - i
    if d == 0 or abs(d) > M:
        return 0.0
    if d > 0:
        return (2.0 / 3.0) * (1.0 / M)  # step to the right
    else:
        return (1.0 / 3.0) * (1.0 / M)  # step to the left


def accept_prob(i: int, j: int, lam: float, M: int) -> float: # acceptance probability
    Hij = H_entry(i, j, M)
    if Hij == 0.0:
        return 0.0
    Hji = H_entry(j, i, M)
    log_num = log_pois_pmf(j, lam) + (math.log(Hji) if Hji > 0 else float("-inf"))
    log_den = log_pois_pmf(i, lam) + math.log(Hij)
    lr = log_num - log_den
    return 1.0 if lr >= 0 else math.exp(lr)

def MCMC(lam: float, M: int, N: int, x0: int, seed: int = None) -> Tuple[np.ndarray, float]:
    """
    Metropolis-Hastings MCMC to sample from Poisson(lam) using proposal H with max step size M.
    Returns (samples, acceptance_rate).
    """
    rng = np.random.default_rng(seed)

    samples = np.empty(N, dtype=int)
    samples[0] = x0
    accepts = 0

    for t in range(1, N):
        # Propose new state
        if rng.random() < 2.0/ 3.0:
            y = samples[t - 1] + rng.integers(1, M + 1)  # step to the right
        else:
            y = samples[t - 1] - rng.integers(1, M + 1)  # step to the left

        a = accept_prob(samples[t - 1], y, lam, M)

        if rng.random() < a:
            samples[t] = y # accepted new step is added
            accepts += 1
        else:
            samples[t] = samples[t - 1] # rejected, stay at current state
    
    acceptance_rate = accepts / (N - 1)

    return samples, acceptance_rate

# Part (iii)

def running_average(xs: Iterable[float]) -> Tuple[np.ndarray, float]:
    """
    Compute running averages of the input sequence.
    """
    xs = np.asarray(xs)
    n = len(xs)
    if n == 0:
        return np.array([])

    run_avg = np.empty(n, dtype=float)
    run_sum = 0.0
    for i in range(n):
        run_sum += xs[i]
        run_avg[i] = run_sum / (i + 1)
    return run_avg, run_avg[-1]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lam, M, n = 5.0, 5, 500
    xs, acc = MCMC(lam=lam, M=M, N=n, x0=0, seed=None)

    print(f"acceptance ≈ {acc:.3f}, mean ≈ {xs.mean():.3f}, var ≈ {xs.var(ddof=1):.3f}")

    # check against the true pmf
    vals, counts = np.unique(xs, return_counts=True)
    ks = np.arange(vals.min(), vals.max() + 1)
    pmf = np.exp([log_pois_pmf(int(k), lam) for k in ks])

    plt.bar(vals, counts / counts.sum(), width=0.9, label="empirical with npts = " + str(n) + " , M = " + str(M))
    plt.plot(ks, pmf, "o--", label="Poisson pmf")
    plt.legend(); plt.tight_layout(); plt.show()

    run_avg = running_average(xs)[0]
    plt.xlabel("number of steps n")
    plt.ylabel("running average")
    plt.plot(run_avg, label="running average")
    plt.show()


# Part (iv)

def empirical_covariance_centered(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    C(t) for t=0..max_lag using a single chain x (post burn-in).
    Uses centered, unbiased time average:  (1/(n-t)) sum_{s=1}^{n-t} (x_s-x̄)(x_{s+t}-x̄).
    """
    x = x.astype(float)
    x = x - x.mean()                # centering
    n = len(x)
    max_lag = min(max_lag, n - 1)
    covs = np.empty(max_lag + 1)
    # vectorized dot-products per lag
    for t in range(max_lag + 1):
        covs[t] = np.dot(x[: n - t], x[t:]) / (n - t)
    return covs

# Parameters
lam = 5.0
M_list = [1, 2, 5, 10]     # proposal ranges to compare
N_long = 8000               # chain length
burn_frac = 0.1            # burn-in as fraction of N_long
max_lag = 50
seed0 = 2025

# Run chains, compute C(t), and plot
plt.figure(figsize=(7, 4))
for idx, M in enumerate(M_list):
    xs, acc = MCMC(lam=lam, M=M, N=N_long, x0=int(lam), seed=None) # Run MCMC for each M
    post = xs[int(burn_frac * N_long):]    # burn-in
    covs = empirical_covariance_centered(post, max_lag=max_lag)
    plt.plot(range(max_lag + 1), covs, marker="o", label=f"M={M} (acc~{acc:.3f})")

plt.xlabel("lag t")
plt.ylabel("Empirical covariance C(t)")
plt.title("Metropolis–Hastings Poisson — covariance decay vs. M")
plt.legend()
plt.tight_layout()
plt.show()