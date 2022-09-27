from sklearn.linear_model import LinearRegression
from scipy.stats import f
from itertools import product
from statsmodels.tools.tools import add_constant
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('raw/lorenz96.pkl', 'rb') as file:
    x_train, y_train, x_test, y_test = pickle.load(file)

n_nodes = x_train.shape[2]
q = x_train.shape[1]
k = 2 * x_train.shape[1] + 1
n = x_test.shape[0]
f_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)

seq_along_nodes = np.arange(n_nodes)
for i, j in product(seq_along_nodes, seq_along_nodes):
    if i == j:
        continue
    x_ur_train = np.swapaxes(x_train[:, :, [i, j]], 1, 2).reshape(x_train.shape[0], -1)
    x_ur_train = add_constant(x_ur_train, prepend=False)
    y_ur_train = y_train[:, j]
    x_ur_test = np.swapaxes(x_test[:, :, [i, j]], 1, 2).reshape(x_test.shape[0], -1)
    x_ur_test = add_constant(x_ur_test, prepend=False)
    y_ur_test = y_test[:, j]

    x_r_train = x_ur_train[:, q:]
    y_r_train = y_ur_train.copy()
    x_r_test = x_ur_test[:, q:]
    y_r_test = y_ur_test.copy()

    ur = LinearRegression()
    ur.fit(x_ur_train, y_ur_train)
    r = LinearRegression()
    r.fit(x_r_train, y_r_train)

    y_ur_test_hat = ur.predict(x_ur_test)
    y_r_test_hat = r.predict(x_r_test)
    ssr_ur = np.sum((y_ur_test_hat - y_ur_test) ** 2)
    ssr_r = np.sum((y_r_test_hat - y_r_test) ** 2)

    f_val[i, j] = (ssr_r - ssr_ur) * (n - k) / ssr_ur / q  # [(ssr_r - ssr_ur) / q] / [ssr_ur / (n-k)]
    p_val[i, j] = 1 - f(dfn=q, dfd=n - k).cdf(f_val[i, j])

with open('raw/lorenz96_pairwise_LR_F_stats.pkl', 'wb') as file:
    pickle.dump({'f_val': f_val, 'p_val': p_val}, file)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(p_val, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(p_val, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.2f')
ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('results/lorenz96_pairwise_LR_F.svg')
plt.close(fig)
