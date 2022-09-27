"""
Accuracy in Lorenz96 dataset
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


def remove_diag(A):
    return A[~np.eye(A.shape[0], dtype=bool)].flatten()


# %% Load results
with open('raw/lorenz96_conditional_LR_wilcoxon_stats.pkl', 'rb') as f:
    LR = pickle.load(f)
LR = (LR['p_val'] <= 0.01).astype('int')
with open('raw/lorenz96_conditional_SVM_wilcoxon_stats.pkl', 'rb') as f:
    SVM = pickle.load(f)
SVM = (SVM['p_val'] <= 0.01).astype('int')
with open('raw/lorenz96_conditional_LSTM_noreg_wilcoxon_stats.pkl', 'rb') as f:
    LSTM_noreg = pickle.load(f)
LSTM_noreg = (LSTM_noreg['p_val'] <= 0.01).astype('int')
with open('raw/lorenz96_conditional_LSTM_L2_0.01_wilcoxon_stats.pkl', 'rb') as f:
    LSTM_L2 = pickle.load(f)
LSTM_L2 = (LSTM_L2['p_val'] <= 0.01).astype('int')
with open('raw/lorenz96_conditional_cLSTM_stats.pkl', 'rb') as f:
    cLSTM = pickle.load(f)
cLSTM = (cLSTM > 0).astype('int')
# with open('raw/lorenz96_conditional_BHT_ARIMA_wilcoxon_stats.pkl', 'rb') as f:
#     bht = pickle.load(f)
# bht = (bht['p_val'] <= 0.01).astype('int')
with open('raw/lorenz96_conditional_MLP_L2_0.1_wilcoxon_stats.pkl', 'rb') as f:
    mlp_l2 = pickle.load(f)
mlp_l2 = (mlp_l2['p_val'] <= 0.01).astype('int')
with open('raw/lorenz96_conditional_MLP_wilcoxon_stats.pkl', 'rb') as f:
    mlp = pickle.load(f)
mlp = (mlp['p_val'] <= 0.01).astype('int')
with open('raw/lorenz96_truth.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

# %%
prediction = {
    'Linear regression': LR,
    'SVM': SVM,
    'LSTM': LSTM_noreg,
    'LSTM-L2': LSTM_L2,
    'cLSTM': cLSTM,
    # 'BHT-ARIMA': bht,
    'MLP': mlp,
    'MLP-L2': mlp_l2,
    'MLP-L2 & LSTM-L2 & SVM': np.bitwise_and(np.bitwise_and(mlp_l2, LSTM_L2), SVM)
}

# %%
fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
plt.subplots_adjust(left=0.05, right=0.95)

accuracy = {}
pred_truth = remove_diag(ground_truth)
for name, pred in prediction.items():
    accuracy[name] = np.sum(remove_diag(pred) == pred_truth)
print(accuracy)
