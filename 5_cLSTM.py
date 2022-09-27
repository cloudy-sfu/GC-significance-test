import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lorenz96 import simulate_lorenz_96
from method_alex2018.clstm import cLSTM, train_model_ista

# %% Load data
# x, gc = simulate_lorenz_96(n_neurons=10, length=2000, random_state=353479)
with open('raw/indexes_alex2018.pkl', 'rb') as f:
    x = pickle.load(f)

# %% Define
device = torch.device('cuda')
X = torch.tensor(x.values[np.newaxis, :], dtype=torch.float32, device=device)
clstm = cLSTM(X.shape[-1], hidden=100).cuda(device=device)

# %% Train
train_loss_list = train_model_ista(
    clstm, X, context=10, lam=10.0, lam_ridge=0.1, lr=1e-3, max_iter=20000
)
# torch.onnx.export(clstm, X, "raw/lorenz96_clstm.onnx", verbose=True)
torch.onnx.export(clstm, X, "raw/indexes_clstm_L0.1.onnx", verbose=True)

# %% Granger causality
GC_est = clstm.GC(threshold=False).cpu().data.numpy().T

# %% Save results
# with open('raw/lorenz96_conditional_cLSTM_stats.pkl', 'wb') as f:
#     pickle.dump(GC_est, f)
with open('raw/indexes_conditional_cLSTM_L0.1_stats.pkl', 'wb') as f:
    pickle.dump(GC_est, f)

# %% Load results
# with open('raw/lorenz96_conditional_cLSTM_stats.pkl', 'rb') as f:
#     GC_est = pickle.load(f)

# %%
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(GC_est, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(GC_est > 0, mask=mask, square=True, linewidths=.5, cmap='coolwarm_r',
                      vmin=0, vmax=1, annot=GC_est, fmt='.4f')

# add the column names as labels
col_names = x.columns
ax.set_yticklabels(col_names, rotation=0)
ax.set_xticklabels(col_names, rotation=90)

ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
# heatmap.get_figure().savefig('results/lorenz96_conditional_cLSTM.svg')
heatmap.get_figure().savefig('results/indexes_conditional_cLSTM_L0.1.svg')
plt.close(fig)
