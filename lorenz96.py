import numpy as np
import pandas as pd
from scipy.integrate import odeint


def lorenz(x_, t, p):
    """
    Partial derivatives for Lorenz-96 ODE.
    """
    f_ = 10
    dx_dt = np.zeros(p)
    for i in np.arange(p):
        dx_dt[i] = (x_[(i + 1) % p] - x_[(i - 2) % p]) * x_[(i - 1) % p] - x_[i] + f_
    return dx_dt


def simulate_lorenz_96(n_neurons, length, random_state=None):
    delta_t = 0.1
    sd = 0.1
    burn_in = 1000

    # Solve ODE
    rng = np.random.RandomState(seed=random_state)
    x0 = rng.normal(scale=0.01, size=n_neurons)

    t = np.linspace(0, (length + burn_in) * delta_t, length + burn_in)
    x_ = odeint(lorenz, x0, t, args=(n_neurons,))
    x_ += rng.normal(scale=sd, size=(length + burn_in, n_neurons))

    # Granger causality ground truth
    gc_ = np.zeros((n_neurons, n_neurons), dtype=int)
    for i in np.arange(n_neurons):  # row: reason series, col: affected series
        gc_[i, i] = 1
        gc_[(i + 1) % n_neurons, i] = 1
        gc_[(i - 1) % n_neurons, i] = 1
        gc_[(i - 2) % n_neurons, i] = 1
    x_ = pd.DataFrame(data=x_[burn_in:], columns=[f'G{i}' for i in np.arange(1, n_neurons + 1)])
    return x_, gc_
