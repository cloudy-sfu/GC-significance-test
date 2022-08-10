"""
Granger causality

CC ground truth:    linear eq15 c=0.5 noise=0.01
GC distribution:    F
GC method:          pairwise
Prevent sparsity:   N
Prediction model:   Linear regression
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from gen_data import eq15
from scipy.stats import f
from itertools import product

q = 1  # N of restricted variables
k = 2  # N of variables in UR model
train_size = 800  # round(length(time series) * 0.8)
n = 199  # N of obs

nodes = eq15(t=1200, s=200, scale=0.01, random_state=306)
n_nodes = 3
f_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)

for i, j in product(range(n_nodes), range(n_nodes)):
    if i == j:
        continue
    ni, nj = nodes[i], nodes[j]

    # make dataset
    x_ur = np.stack([ni[:-1], nj[:-1]], axis=0).T  # UR: y_{t} ~ y_{t-1} + x_{t-1}
    y_ur = nj[1:, np.newaxis]
    x_r = nj[:-1, np.newaxis]  # R: y_{t} ~ y_{t-1}
    y_r = nj[1:, np.newaxis]
    x_ur_train, x_ur_test = x_ur[:train_size], x_ur[train_size:]
    y_ur_train, y_ur_test = y_ur[:train_size], y_ur[train_size:]
    x_r_train, x_r_test = x_r[:train_size], x_r[train_size:]
    y_r_train, y_r_test = y_r[:train_size], y_r[train_size:]

    # train model
    ur = LinearRegression().fit(x_ur_train, y_ur_train)
    r = LinearRegression().fit(x_r_train, y_r_train)

    # evaluate
    y_ur_test_hat = ur.predict(x_ur_test)
    y_r_test_hat = r.predict(x_r_test)
    ssr_ur = np.sum((y_ur_test_hat - y_ur_test) ** 2)
    ssr_r = np.sum((y_r_test_hat - y_r_test) ** 2)

    # hypothesis testing
    f_val[i, j] = (ssr_r - ssr_ur) * (n - k) / ssr_ur / q  # [(ssr_r - ssr_ur) / q] / [ssr_ur / (n-k)]
    p_val[i, j] = 1 - f(dfn=q, dfd=n - k).cdf(f_val[i, j])

print(f_val)
