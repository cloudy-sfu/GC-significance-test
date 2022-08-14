import pickle
from itertools import product

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from statsmodels.tools.tools import add_constant

with open('raw/indexes_std.pkl', 'rb') as file:
    x_train, x_test, y_train, y_test = pickle.load(file)

n_nodes = x_train.shape[2]
q = x_train.shape[1]  # N of restricted variables
k = 2 * x_train.shape[1] + 1  # N of variables in UR model
n = x_test.shape[0]  # N of obs
simu = 100  # N of bootstrap simulation
n_cpu = joblib.cpu_count()
p_val = np.full((n_nodes, n_nodes), np.nan)
ssr_ur = np.zeros((n_nodes, n_nodes))
ssr_r = np.zeros((n_nodes, n_nodes, simu))

seq_along_nodes = np.arange(n_nodes)
for i, j in product(seq_along_nodes, seq_along_nodes):
    if i == j:
        continue
    print('i=', i, 'j=', j)

    x_ur_train = np.swapaxes(x_train[:, :, [i, j]], 1, 2).reshape(x_train.shape[0], -1)
    x_ur_train = add_constant(x_ur_train, prepend=False)
    y_ur_train = y_train[:, j]
    x_ur_test = np.swapaxes(x_test[:, :, [i, j]], 1, 2).reshape(x_test.shape[0], -1)
    x_ur_test = add_constant(x_ur_test, prepend=False)
    y_ur_test = y_test[:, j]

    x_r_train_basic = x_ur_train.copy()
    y_r_train = y_ur_train.copy()
    x_r_test_basic = x_ur_test.copy()
    y_r_test = y_ur_test.copy()
    x_r_train_boot, x_r_test_boot = [], []
    rng = np.random.RandomState(seed=306)
    for _ in np.arange(simu):
        rng.shuffle(x_r_train_basic[:, :q])
        x_r_train_boot.append(x_r_train_basic.copy())
        rng.shuffle(x_r_test_basic[:, :q])
        x_r_test_boot.append(x_r_test_basic.copy())
    del x_r_train_basic, x_r_test_basic

    ur = GridSearchCV(
        SVR(max_iter=5000),
        {'kernel': ['rbf', 'poly'], 'C': [0.01, 1, 10]},
        scoring='neg_root_mean_squared_error'
    )
    ur.fit(x_ur_train, y_ur_train)
    y_ur_test_hat = ur.predict(x_ur_test)
    ssr_ur[i, j] = np.sum((y_ur_test_hat - y_ur_test) ** 2)

    def simulation(x_r_train, x_r_test):
        r = GridSearchCV(
            SVR(max_iter=5000),
            {'kernel': ['rbf', 'poly'], 'C': [0.01, 1, 10]},
            scoring='neg_root_mean_squared_error'
        )
        r.fit(x_r_train, y_r_train)
        y_r_test_hat = r.predict(x_r_test)
        return np.sum((y_r_test_hat - y_r_test) ** 2)

    ssr_r[i, j] = joblib.Parallel(n_jobs=n_cpu)(
        joblib.delayed(simulation)(x_r_train_boot[m], x_r_test_boot[m])
        for m in np.arange(simu)
    )
    p_val[i, j] = np.mean(ssr_ur[i, j] > ssr_r[i, j])

with open('raw/indexes_svm_gc.pkl', 'wb') as file:
    pickle.dump({'p_val': p_val, 'ssr_r': ssr_r, 'ssr_ur': ssr_ur}, file)
with open('raw/indexes_names.pkl', 'rb') as file:
    col_names = pickle.load(file)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(p_val, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(p_val, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.2f')
# add the column names as labels
ax.set_yticklabels(col_names, rotation=0)
ax.set_xticklabels(col_names, rotation=90)
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('results/indexes_svm_heatmap.svg')
plt.close(fig)
