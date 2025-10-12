# --- Stationary Gaussian process simulation by spectral method (Algorithm 5.13) ---
# Uses numpy's FFT to simulate zero-mean stationary GPs on [-L/2, L/2).

import numpy as np
import matplotlib.pyplot as plt

# Covariance functions 
def cov_exponential(t):
    # C(t) = exp(-|t|)
    return np.exp(-np.abs(t))

def cov_gaussian(t):
    # C(t) = (1/sqrt(2π)) * exp(-t^2/2)
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * t**2)

def cov_matern32(t):
    # Matérn ν=3/2 with unit length-scale and unit variance:
    # C(t) = (1 + |t|) * exp(-|t|)
    a = np.abs(t)
    return (1 + a) * np.exp(-a)

# Spectral simulator (Algorithm 5.13) 
def simulate_gp_from_cov(C, L=16.0, N=2048, n_paths=5, rng=None):
    """
    Simulate 'n_paths' independent real-valued trajectories on [-L/2, L/2).
    Returns:
        t   : time grid (length N)
        X   : array shape (n_paths, N) with real-valued samples
        Chat: discrete spectral weights used internally
    """
    if rng is None:
        rng = np.random.default_rng()

    # Time grid centered at 0 (periodic)
    dt = L / N
    t = (np.arange(N) - N//2) * dt  # centered grid

    # Sampled covariance on this grid
    C_t = C(t)

    # Discrete spectral coefficients (nonnegative)
    C_t_shift = np.fft.ifftshift(C_t) # need to shift for FFT
    Chat = np.real(np.fft.fft(C_t_shift)) * dt  # ≈ ∫ C(t) e^{-iωt} dt via Riemann sum
    Chat = np.maximum(Chat, 0.0)               # clip tiny negatives from roundoff 

    # Draw complex-Gaussian ξ_k with variance 0.5 * Chat[k]
    A = rng.standard_normal((n_paths, N))
    B = rng.standard_normal((n_paths, N))
    xi = (np.sqrt(0.5 * Chat)[None, :] * (A + 1j*B)) # shape (n_paths, N)

    # Inverse FFT to obtain complex process; scale by sqrt(N) to match variance convention
    X_complex = np.fft.ifft(xi, axis=1) * np.sqrt(N)

    # Two independent real processes: Re and Im. Use them alternately for independence.
    X_real = np.real(X_complex)
    X_imag = np.imag(X_complex)
    X = np.empty((n_paths, N))
    for i in range(n_paths):
        X[i] = X_real[i] if i % 2 == 0 else X_imag[i]

    # Shift time back to [-L/2, L/2)
    t = np.fft.fftshift(t)

    return t, X, Chat

# Empirical covariance checker 
def empirical_covariance(X):
    """
    computes sample (circular) covariance as a function of lag (periodic on N).
    Returns cov(lag) aligned so that index N//2 corresponds to zero lag.
    """
    n_paths, N = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    F = np.fft.fft(Xc, axis=1)
    S = (F * np.conj(F)).real / N        # periodogram
    cov_circ = np.real(np.fft.ifft(S, axis=1))
    cov_mean = cov_circ.mean(axis=0)
    return np.fft.fftshift(cov_mean)

# simulate and plot
def run_demo(L=16.0, N=2048, n_paths=5, n_for_cov=300):
    rng = np.random.default_rng(12345)

    covs = {
        "Exponential  C(t)=exp(-|t|)": cov_exponential,
        "Gaussian     C(t)=(1/sqrt(2π)) exp(-t^2/2)": cov_gaussian,
        "Matérn-3/2   C(t)=(1+|t|) exp(-|t|)": cov_matern32,
    }

    # Trajectories (part ii)
    for name, C in covs.items():
        t, X, _ = simulate_gp_from_cov(C, L=L, N=N, n_paths=n_paths, rng=rng)
        plt.figure(figsize=(8, 4))
        for i in range(n_paths):
            plt.plot(t, X[i])
        plt.xlabel("t"); plt.ylabel("X(t)")
        plt.title(f"{name} — {n_paths} trajectories on [-L/2,L/2) with L={L}, N={N}")
        plt.tight_layout(); plt.show()

    # Covariance verification (part iii)
    for name, C in covs.items():
        t, X, _ = simulate_gp_from_cov(C, L=L, N=N, n_paths=n_for_cov, rng=rng)
        emp_cov = empirical_covariance(X)

        dt = L / N
        t_grid = (np.arange(N) - N//2) * dt
        C_th = C(np.abs(t_grid))  # stationary & even

        # scale to match lag-0
        scale = emp_cov[N//2] / C_th[N//2] if C_th[N//2] != 0 else 1.0
        C_th_scaled = C_th * scale

        plt.figure(figsize=(8, 4))
        plt.plot(t_grid, emp_cov, label="Empirical (circular) covariance")
        plt.plot(t_grid, C_th_scaled, linestyle="--",
                 label="Target covariance (scaled to match lag 0)")
        plt.xlabel("lag t"); plt.ylabel("covariance")
        plt.title(f"Covariance check — {name} (avg over {n_for_cov} paths)")
        plt.legend(); plt.tight_layout(); plt.show()

# Run demo
if __name__ == "__main__":
    run_demo(L=16.0, N=2048, n_paths=5, n_for_cov=300)
