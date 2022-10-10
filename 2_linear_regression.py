import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from statsmodels.tools.tools import add_constant

# %% Read dataset
with open('raw/indexes_std.pkl', 'rb') as f:
# with open('raw/lorenz96.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

# %% Constants
n_nodes = x_train.shape[2]
z_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)

# %% i -> j
rng = np.random.RandomState(seed=23846)

for i in range(n_nodes):
    x_ur_train = np.swapaxes(x_train, 1, 2).reshape(x_train.shape[0], -1)
    x_ur_train = add_constant(x_ur_train, prepend=False)
    x_ur_test = np.swapaxes(x_test, 1, 2).reshape(x_test.shape[0], -1)
    x_ur_test = add_constant(x_ur_test, prepend=False)

    x_r_train = x_train.copy()
    rng.shuffle(x_r_train[:, :, i])
    x_r_train = np.swapaxes(x_r_train, 1, 2).reshape(x_train.shape[0], -1)
    x_r_train = add_constant(x_r_train, prepend=False)
    x_r_test = x_test.copy()
    rng.shuffle(x_r_test[:, :, i])
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

    p_val[i, :] = p
    z_val[i, :] = z

# %% Export results
with open('raw/indexes_conditional_LR_wilcoxon_stats.pkl', 'wb') as f:
# with open('raw/lorenz96_conditional_LR_wilcoxon_stats.pkl', 'wb') as f:
    pickle.dump({'z_val': z_val, 'p_val': p_val}, f)
with open('raw/indexes_names.pkl', 'rb') as f:
    col_names = pickle.load(f)

# %% Causality heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(p_val, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(p_val, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.2f')
# add the column names as labels
ax.set_yticklabels(col_names, rotation=0)
ax.set_xticklabels(col_names, rotation=90)

ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('results/indexes_conditional_LR_wilcoxon.svg')
# heatmap.get_figure().savefig('results/lorenz96_conditional_LR_wilcoxon.svg')
plt.close(fig)
