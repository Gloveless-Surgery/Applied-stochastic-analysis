import numpy as np 
from typing import Tuple, Iterable

# birth–death rates
def birth_rate(i):     # i >= 0
    return 1.5         # arrivals
def death_rate(i):
    if i <= 0:  return 0.0
    if i <= 4:  return 1.0
    return 2.0


def Q_row(i, S):
    """
    Returns row vector q[i, :] of length S.
    Only i-1 and i+1 non zero.
    If i+1 == S, we raise to signal the cap was hit.
    """
    if i+1 >= S:
        raise RuntimeError("Hit S cap; increase S.")
    row = np.zeros(S, dtype=float)
    lam = birth_rate(i)
    mu  = death_rate(i)
    if i > 0:
        row[i-1] = mu
    row[i+1] = lam
    row[i]   = -(lam + mu)
    return row


def kinetic_MC(i0=10, S=200, T=np.inf, seed=None,
                  stop_when_hit= int):
    """
    Kinetic MC, mirroring your loop and API:
      - builds rates = Q[i, :] (on the fly)
      - zeroes self-transition
      - total_rate = -Q[i,i]
      - dt ~ Exp(1/total_rate)
      - j = rng.choice(S, p=probs)

    stop_when_hit:
      - None: run until hitting state 0 (cleared)
      - int k: stop when you reach state k (e.g. 20) or 0, whichever first.

    Returns times[], states[] (including initial time/state at 0).
    """
    assert S > 1 and 0 <= i0 < S
    rng = np.random.default_rng(seed)

    times  = [0.0]
    states = [i0]
    t = 0.0
    i = i0

    while t < T:
        # stopping conditions
        if i == 0: break
        if stop_when_hit is not None and i == stop_when_hit: break

        rates = Q_row(i, S)         # <- "Q[i, :]"
        rates[i] = 0.0                    # no self-transition
        total_rate = -Q_row(i, S)[i]  # i.e. rates.sum()

        if total_rate <= 0:
            break  # absorbing 

        # Sample time to next event
        dt = rng.exponential(1.0 / total_rate)
        t += dt
        if t >= T: break

        # Sample next state
        probs = rates / total_rate
        j = rng.choice(S, p=probs)

        times.append(t)
        states.append(j)
        i = j

    return np.array(times), np.array(states)


# (ii)(a) Average time to clear the inbox from 10
def estimate_mean_clear_time(N=2000, i0=10, S=200, seed=None):
    ts = np.empty(N, dtype=float)
    for n in range(N):
        t, x = kinetic_MC(i0=i0, S=S, seed=None, stop_when_hit=None)
        ts[n] = t[-1] # last time in the path is the hitting time of 0
    mean = ts.mean()
    se   = ts.std(ddof=1) / np.sqrt(N)
    return mean, se


# (ii)(b) Probability you clear before ever reaching 20


def estimate_prob_clear_before_20(N=2000, i0=10, S=200, seed=None):
    wins = 0
    for n in range(N):
        t, x = kinetic_MC(i0=i0, S=S, seed=None,
                             stop_when_hit=20)
        if x[-1] == 0:
            wins += 1
    p_hat = wins / N
    se    = np.sqrt(p_hat * (1 - p_hat) / N)   # binomial SE
    return p_hat, se

m, m_se = estimate_mean_clear_time(N=10000, i0=10, S=200, seed=None)
p, p_se = estimate_prob_clear_before_20(N=10000, i0=10, S=200, seed=None)
print("E[time to clear | X0=10] ≈", m, "±", 1.96*m_se)
print("P(clear before hitting 20 | X0=10] ≈", p, "±", 1.96*p_se)