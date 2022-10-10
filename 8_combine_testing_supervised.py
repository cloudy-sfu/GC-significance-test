import pickle
from scipy.stats import chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def remove_diag(A):
    return A[~np.eye(A.shape[0], dtype=bool)].flatten()


# %% Load dataset
with open('raw/lorenz96_conditional_SVM_wilcoxon_stats.pkl', 'rb') as f:
    SVM = pickle.load(f)
with open('raw/lorenz96_conditional_LSTM_L2_0.01_wilcoxon_stats.pkl', 'rb') as f:
    LSTM_L2 = pickle.load(f)
with open('raw/lorenz96_conditional_MLP_L2_0.1_wilcoxon_stats.pkl', 'rb') as f:
    MLP_L2 = pickle.load(f)
with open('raw/lorenz96_truth.pkl', 'rb') as f:
    ground_truth = pickle.load(f)
p = np.stack([SVM['p_val'], LSTM_L2['p_val'], MLP_L2['p_val']], axis=2)
pred_truth = remove_diag(ground_truth)

# %% Calc model correlation coefficient
k = p.shape[2]


@np.vectorize
def model_corr_optimize(c):
    chi_ = -2 * np.sum(np.log(p), axis=2) * c
    pred = chi2.cdf(chi_, df=2 * k) > 0.99  # confidence level: 95%
    return - np.mean(remove_diag(pred) == pred_truth)


result = minimize(model_corr_optimize, x0=np.array(1), method='Nelder-Mead',
                  options={'initial_simplex': np.array([[0.1], [1]])})
c0 = result.x[0]

# %% Apply this multiplier to hypothesis testing
chi_0 = -2 * np.sum(np.log(p), axis=2) * c0
p_0 = 1 - chi2.cdf(chi_0, df=2 * k)

with open('raw/lorenz96_combined_chisq_p.pkl', 'wb') as f:
    pickle.dump(p_0, f)

# %% Causality heatmap (lorenz96)
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(p_0, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(p_0, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.2f')
ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
ax.set_title(f'c0 = {c0.__round__(3)}')
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('results/lorenz96_combined_chisq.svg')
plt.close(fig)
