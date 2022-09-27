import pickle
import numpy as np

# %%
with open('raw/indexes_conditional_SVM_wilcoxon_stats.pkl', 'rb') as f:
    SVM = pickle.load(f)
with open('raw/indexes_conditional_LSTM_L2_0.01_wilcoxon_stats.pkl', 'rb') as f:
    LSTM_L2 = pickle.load(f)
with open('raw/indexes_conditional_MLP_L2_0.1_wilcoxon_stats.pkl', 'rb') as f:
    MLP_L2 = pickle.load(f)
with open('raw/indexes_names.pkl', 'rb') as f:
    col_names = pickle.load(f)

# %%
confidence_level = 0.05 ** (1/3)
pSVM = (SVM['p_val'] <= confidence_level).astype('int')
pLSTM_L2 = (LSTM_L2['p_val'] <= confidence_level).astype('int')
pMLP_L2 = (MLP_L2['p_val'] <= confidence_level).astype('int')

# %%
combined = np.bitwise_and(np.bitwise_and(pSVM, pLSTM_L2), pMLP_L2)
# combined = np.sum([pSVM, pLSTM_L2, pMLP_L2], axis=0) >= 2
for i, j in zip(*np.where(combined)):
    print(col_names[i], '->', col_names[j])
