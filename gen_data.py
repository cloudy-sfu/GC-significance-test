"""
doi:10.1016/j.physleta.2004.02.032
"""
import numpy as np
import matplotlib.pyplot as plt


def eq15(t, s, scale, c=0.5, random_state=None):
    """
    Generate 3 time series with linear causality
    :param t: The length of the time series
    :param s: The beginning of the time series which isn't chaotic
    :param scale: The scale of the noise, which fits Gaussian distribution N(0, scale^2)
    :param c:
    :param random_state:
    :return: time series x, y, z
    """
    rng = np.random.RandomState(seed=random_state)
    x = np.zeros(shape=t)
    y = np.zeros(shape=t)
    z = np.zeros(shape=t)
    x[0] = 0.01
    y[0] = 0.01
    z[0] = 0.01
    r1 = rng.normal(scale=scale, size=t)
    r2 = rng.normal(scale=scale, size=t)
    r3 = rng.normal(scale=scale, size=t)
    for i in np.arange(1, t):
        x[i] = 0.2 * x[i - 1] + r1[i]
        y[i] = 0.5 * y[i - 1] + 0.5 * x[i - 1] + r2[i]
        z[i] = 0.4 * z[i - 1] + 0.3 * y[i - 1] + c * x[i - 1] + r3[i]
    return x[s:], y[s:], z[s:]


def eq16(t, s, c=0.5):
    """
    Generate 3 time series with nonlinear causality
    :param t: The length of the time series
    :param s: The beginning of the time series which isn't chaotic
    :param c:
    :return: time series x, y, z
    """
    x = np.zeros(shape=t)
    y = np.zeros(shape=t)
    z = np.zeros(shape=t)
    x[0] = 0.01
    y[0] = 0.01
    z[0] = 0.01
    for i in np.arange(1, t):
        x[i] = 3.4 * x[i - 1] * (1 - x[i - 1] ** 2) * np.exp(-x[i - 1] ** 2)
        y[i] = 3.4 * y[i - 1] * (1 - y[i - 1] ** 2) * np.exp(-y[i - 1] ** 2) + 0.5 * x[i - 1]
        z[i] = 3.4 * z[i - 1] * (1 - z[i - 1] ** 2) * np.exp(-z[i - 1] ** 2) + 0.3 * y[i - 1] + c * x[i - 1]
    return x[s:], y[s:], z[s:]


def draw_3_nodes(x, y, z):
    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))
    ax[0, 0].scatter(x[1:], x[:-1], s=9, color='k')
    ax[0, 0].set_xlabel('x(n)')
    ax[0, 0].set_ylabel('x(n-1)')
    ax[0, 1].scatter(y[1:], y[:-1], s=9, color='k')
    ax[0, 1].set_xlabel('y(n)')
    ax[0, 1].set_ylabel('y(n-1)')
    ax[0, 2].scatter(z[1:], z[:-1], s=9, color='k')
    ax[0, 2].set_xlabel('z(n)')
    ax[0, 2].set_ylabel('z(n-1)')
    ax[1, 0].scatter(y[1:], x[:-1], s=9, color='k')
    ax[1, 0].set_xlabel('y(n)')
    ax[1, 0].set_ylabel('x(n-1)')
    ax[1, 1].scatter(z[1:], x[:-1], s=9, color='k')
    ax[1, 1].set_xlabel('z(n)')
    ax[1, 1].set_ylabel('x(n-1)')
    ax[1, 2].scatter(z[1:], y[:-1], s=9, color='k')
    ax[1, 2].set_xlabel('z(n)')
    ax[1, 2].set_ylabel('y(n-1)')
    return f, ax


if __name__ == '__main__':
    x1, y1, z1 = eq16(t=1200, s=200)
    f1, ax1 = draw_3_nodes(x1, y1, z1)
    f1.savefig('results/gen_data_eq16.svg')
    plt.close(f1)

    x2, y2, z2 = eq15(t=1200, s=200, scale=0.1, random_state=306)
    f2, ax2 = draw_3_nodes(x2, y2, z2)
    f2.savefig('results/gen_data_eq15.svg')
    plt.close(f2)
