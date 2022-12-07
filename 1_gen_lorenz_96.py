"""
Reference: https://github.com/iancovert/Neural-GC/blob/master/synthetic.py
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lorenz96 import simulate_lorenz_96
from statsmodels.tsa.stattools import adfuller, pacf


def moving_window(ts, k_):
    """
    Make moving window samples from time series.
    :param ts: Time series.
    :param k_: Length of the window.
    :return: x_, y_: fraction used as input, fraction used as output.
    """
    length = ts.shape[0]
    y_ = ts[k_:]
    indices = np.tile(np.arange(k_), [length - k_, 1]) + np.arange(length - k_)[:, np.newaxis]
    x_ = ts[indices]
    return x_, y_


# %% Generate dataset
x, gc = simulate_lorenz_96(n_neurons=10, length=2000, random_state=353479)

# %% Split training and testing set.
train_size = round(0.8 * x.shape[0])
x_train, x_test = x[:train_size], x[train_size:]

# %% ADF test (stationary)
# H_0: has unit root = non-stationary
adf = x_train.apply(adfuller, axis=0).T
adf.columns = ['adf', 'pvalue', 'usedlag', 'nobs', 'critical values', 'icbest']
adf.to_excel('results/lorenz96_adf_test.xlsx')

# %% PACF determining the order (AR -> p -> PACF)
pacf_ = x_train.apply(lambda z: pacf(z, nlags=15), axis=0)
pacf_['N_significance'] = np.bitwise_or(pacf_ < -0.05, pacf_ > 0.05).sum(axis=1)
pacf_.to_excel('results/lorenz96_pacf_order.xlsx')

# %% Moving window
x_train, y_train = moving_window(x_train.values, k_=6)
x_test, y_test = moving_window(x_test.values, k_=6)
with open('raw/lorenz96.pkl', 'wb') as f:
    pickle.dump([x_train, y_train, x_test, y_test], f)

# %% Draw ground truth
fig, ax = plt.subplots(figsize=(5, 4))
mask = np.zeros_like(gc, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(1 - gc, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.0f')
ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('results/lorenz96_ground_truth.eps')
plt.close(fig)

with open('raw/lorenz96_truth.pkl', 'wb') as f:
    pickle.dump(gc, f)
