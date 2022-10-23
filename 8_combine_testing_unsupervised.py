import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2
from view_graph import draw


def mesh_grid(x, y):
    xx = np.tile(x, (y.shape[0], 1))
    yy = np.tile(y[:, np.newaxis], (1, x.shape[0]))
    return xx, yy


# %% Applying indexes dataset
with open('raw/indexes_conditional_SVM_wilcoxon_stats.pkl', 'rb') as f:
    SVM = pickle.load(f)
with open('raw/indexes_conditional_LSTM_L2_0.01_wilcoxon_stats.pkl', 'rb') as f:
    LSTM_L2 = pickle.load(f)
with open('raw/indexes_conditional_MLP_L2_0.1_wilcoxon_stats.pkl', 'rb') as f:
    MLP_L2 = pickle.load(f)
with open('raw/indexes_names.pkl', 'rb') as f:
    col_names = pickle.load(f)
p = np.stack([SVM['p_val'], LSTM_L2['p_val'], MLP_L2['p_val']], axis=2)
k = p.shape[2]


# %% Determine the strictness of testing
@np.vectorize
def model_corr_strict(c, alpha):
    chi_ = -2 * np.sum(np.log(p), axis=2) * c
    pred = chi2.cdf(chi_, df=2 * k) > 1 - alpha
    return np.mean(pred)


alpha_ = np.array([0.01, 0.05, 0.1])
c_ = np.linspace(0.5, 2, 51)
alpha_grid, c_grid = mesh_grid(alpha_, c_)
p_ = model_corr_strict(c=c_grid, alpha=alpha_grid)

fig, ax = plt.subplots(figsize=(10, 8))
for j in range(p_.shape[1]):
    ax.plot(c_, p_[:, j], label=f'alpha = {alpha_[j]}')
ax.legend()
ax.set_xlabel('χ^2 multiplier')
ax.set_ylabel('Ratio of causality pairs')
fig.savefig('results/indexes_chisq_multiplier.eps')
plt.close(fig)

# %% Select the best chi2 multiplier
c0 = 1.5
chi_0 = -2 * np.sum(np.log(p), axis=2) * c0
p_0 = 1 - chi2.cdf(chi_0, df=2 * k)

with open('raw/indexes_combined_chisq_p.pkl', 'wb') as f:
    pickle.dump(p_0, f)

# %% Causality heatmap (indexes)
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(p_0, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(p_0, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.2f')
ax.set_yticklabels(col_names, rotation=0)
ax.set_xticklabels(col_names, rotation=90)
ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
ax.set_title(f'c0 = {c0.__round__(3)}')
fig.subplots_adjust(bottom=0.27, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('results/indexes_combined_chisq.eps')
plt.close(fig)

# %% Output as items
h = draw(
    adjacency=p_0 < 0.01,
    labels=col_names,
    output_path='results/indexes_combined_conclusion.html'
)
