import pickle
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# with open('raw/indexes_std.pkl', 'rb') as f:
with open('raw/lorenz96.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

n_nodes = x_train.shape[2]
z_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)
rng = np.random.RandomState(seed=23846)
n_cpu = joblib.cpu_count()

for i in tqdm(range(n_nodes)):
    x_ur_train = np.swapaxes(x_train, 1, 2).reshape(x_train.shape[0], -1)
    x_ur_test = np.swapaxes(x_test, 1, 2).reshape(x_test.shape[0], -1)

    x_r_train = x_train.copy()
    rng.shuffle(x_r_train[:, :, i])
    x_r_train = np.swapaxes(x_r_train, 1, 2).reshape(x_train.shape[0], -1)
    x_r_test = x_test.copy()
    rng.shuffle(x_r_test[:, :, i])
    x_r_test = np.swapaxes(x_r_test, 1, 2).reshape(x_test.shape[0], -1)

    @ignore_warnings(category=ConvergenceWarning)
    def simulation(j):
        if i == j:
            return np.nan, np.nan
        ur = GridSearchCV(
            SVR(max_iter=5000),
            {'kernel': ['rbf', 'poly'], 'C': [0.01, 0.1, 1, 10]},
            scoring='neg_root_mean_squared_error'
        )
        ur.fit(x_ur_train, y_train[:, j])
        r = GridSearchCV(
            SVR(max_iter=5000),
            {'kernel': ['rbf', 'poly'], 'C': [0.01, 0.1, 1, 10]},
            scoring='neg_root_mean_squared_error'
        )
        r.fit(x_r_train, y_train[:, j])
        y_ur_test_hat = ur.predict(x_ur_test)
        y_r_test_hat = r.predict(x_r_test)
        err_ur = (y_ur_test_hat - y_test[:, j]) ** 2
        err_r = (y_r_test_hat - y_test[:, j]) ** 2
        wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx')
        w, p, z = wilcoxon_results.statistic, wilcoxon_results.pvalue, wilcoxon_results.zstatistic
        return p, z
    
    parallel_results = joblib.Parallel(n_jobs=n_cpu)(
        joblib.delayed(simulation)(j) for j in range(n_nodes)
    )
    parallel_results = np.array(parallel_results)
    p_val[i, :] = parallel_results[:, 0]
    z_val[i, :] = parallel_results[:, 1]

# %% Export results
# with open('raw/indexes_conditional_SVM_wilcoxon_stats.pkl', 'wb') as f:
with open('raw/lorenz96_conditional_SVM_wilcoxon_stats.pkl', 'wb') as f:
    pickle.dump({'z_val': z_val, 'p_val': p_val}, f)
# with open('raw/indexes_names.pkl', 'rb') as f:
#     col_names = pickle.load(f)

# %% Causality heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(p_val, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(p_val, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.2f')
# add the column names as labels
# ax.set_yticklabels(col_names, rotation=0)
# ax.set_xticklabels(col_names, rotation=90)

ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
# heatmap.get_figure().savefig('results/indexes_conditional_SVM_wilcoxon.eps')
heatmap.get_figure().savefig('results/lorenz96_conditional_SVM_wilcoxon.eps')
plt.close(fig)
