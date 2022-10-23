import pickle

import numpy as np
from scipy.stats import f, wilcoxon
from sklearn.linear_model import LinearRegression
from statsmodels.tools.tools import add_constant

# %% Read dataset
with open('raw/lorenz96.pkl', 'rb') as f_:
    x_train, y_train, x_test, y_test = pickle.load(f_)

# %% Constants
n_nodes = x_train.shape[2]
k = 2 * x_train.shape[1] + 1
n = x_test.shape[0]
q = x_train.shape[1]
f_val = np.full((n_nodes, n_nodes), np.nan)
fp_val = np.full((n_nodes, n_nodes), np.nan)
z_val = np.full((n_nodes, n_nodes), np.nan)
zp_val = np.full((n_nodes, n_nodes), np.nan)

# %% i -> j
rng = np.random.RandomState(seed=23846)

for i in range(n_nodes):
    x_ur_train = np.swapaxes(x_train, 1, 2).reshape(x_train.shape[0], -1)
    x_ur_train = add_constant(x_ur_train, prepend=False)
    x_ur_test = np.swapaxes(x_test, 1, 2).reshape(x_test.shape[0], -1)
    x_ur_test = add_constant(x_ur_test, prepend=False)

    x_r_train = np.delete(x_train, i, axis=2)
    x_r_train = np.swapaxes(x_r_train, 1, 2).reshape(x_train.shape[0], -1)
    x_r_train = add_constant(x_r_train, prepend=False)
    x_r_test = np.delete(x_test, i, axis=2)
    x_r_test = np.swapaxes(x_r_test, 1, 2).reshape(x_test.shape[0], -1)
    x_r_test = add_constant(x_r_test, prepend=False)

    ur = LinearRegression()
    ur.fit(x_ur_train, y_train)
    r = LinearRegression()
    r.fit(x_r_train, y_train)
    y_ur_test_hat = ur.predict(x_ur_test)
    y_r_test_hat = r.predict(x_r_test)

    # https://doi.org/10.1016/j.cmpb.2022.106669 Therefore, we decided to test if the error obtained by the model,
    # which uses both time series for prediction is significantly smaller than the error obtained by the model using
    # only one time series as an input by using Wilcoxon signed-rank test [52].
    err_ur = (y_ur_test_hat - y_test) ** 2
    err_r = (y_r_test_hat - y_test) ** 2
    wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx')
    w, p, z = wilcoxon_results.statistic, wilcoxon_results.pvalue, wilcoxon_results.zstatistic
    zp_val[i, :] = p
    z_val[i, :] = z

    ssr_ur = np.sum(err_ur, axis=0)
    ssr_r = np.sum(err_ur, axis=0)
    f_val[i, :] = (ssr_r - ssr_ur) * (n - k) / ssr_ur / q  # [(ssr_r - ssr_ur) / q] / [ssr_ur / (n-k)]
    fp_val[i, :] = 1 - f(dfn=q, dfd=n - k).cdf(f_val[i, :])

# %% Export results
with open('raw/lorenz96_conditional_no_bootstrap_stats.pkl', 'wb') as f_:
    pickle.dump({'f_val': f_val, 'f_p': fp_val, 'z_val': z_val, 'z_p': zp_val}, f_)
