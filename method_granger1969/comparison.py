"""
Accuracy in Lorenz96 dataset
"""
import pickle

import numpy as np
import pandas as pd


def remove_diag(A):
    return A[~np.eye(A.shape[0], dtype=bool)].flatten()


with open('raw/lorenz96_conditional_no_bootstrap_stats.pkl', 'rb') as f:
    nb = pickle.load(f)
nb_f = (nb['f_p'] <= 0.01).astype('int')
nb_z = (nb['z_p'] <= 0.01).astype('int')
with open('raw/lorenz96_conditional_bootstrap_stats.pkl', 'rb') as f:
    b = pickle.load(f)
b_f = (nb['f_p'] <= 0.01).astype('int')
b_z = (nb['z_p'] <= 0.01).astype('int')
with open('raw/lorenz96_truth.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

prediction = {
    'F test': nb_f,
    'Wilcoxon test': nb_z,
    'bootstrap and F test': b_f,
    'bootstrap and Wilcoxon test': b_z
}

accuracy = []
pred_truth = remove_diag(ground_truth)
for name, pred in prediction.items():
    accuracy.append({
        'Name': name,
        'Accuracy': np.sum(remove_diag(pred) == pred_truth)
    })
accuracy = pd.DataFrame(accuracy)
print(accuracy)
