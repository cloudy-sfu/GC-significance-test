"""
Granger causality

CC ground truth:    nonlinear eq16 c=0.5
GC distribution:    bootstrap
GC method:          pairwise
Prevent sparsity:   Y
Prediction model:   SVM regression C=0.1, kernel='rbf' (carefully tuned)
"""
import numpy as np
from sklearn.svm import SVR
from gen_data import eq16
from tqdm import tqdm
from itertools import product

q = 1  # N of restricted variables
k = 2  # N of variables in UR model
train_size = 800  # round(length(time series) * 0.8)
n = 199  # N of obs
simu = 100  # N of bootstrap simulation

nodes = eq16(t=1200, s=200)
n_nodes = 3
p_val = np.full((n_nodes, n_nodes), np.nan)

for i, j in product(range(n_nodes), range(n_nodes)):
    if i == j:
        continue
    print("i=", i, "j=", j)
    ni, nj = nodes[i], nodes[j]

    # make dataset
    x_ur = np.stack([ni[:-1], nj[:-1]], axis=0).T  # UR: y_{t} ~ y_{t-1} + x_{t-1}
    y_ur = nj[1:]
    x_r = nj[:-1, np.newaxis]  # R: y_{t} ~ y_{t-1}
    y_r = nj[1:]
    x_ur_train, x_ur_test = x_ur[:train_size], x_ur[train_size:]
    y_ur_train, y_ur_test = y_ur[:train_size], y_ur[train_size:]
    x_r_train, x_r_test, y_r_train, y_r_test = x_ur_train.copy(), x_ur_test.copy(), y_ur_train.copy(), y_ur_test.copy()
    rng = np.random.RandomState(seed=306)

    # train model
    ur = SVR(kernel='rbf', C=0.1).fit(x_ur_train, y_ur_train)

    # evaluate
    y_ur_test_hat = ur.predict(x_ur_test)
    ssr_ur = np.sum((y_ur_test_hat - y_ur_test) ** 2)
    ssr_r = np.zeros(simu)

    for k in tqdm(np.arange(simu)):
        rng.shuffle(x_r_train[:, 0])  # shuffle x_{t-1} -> R: y_{t} ~ y_{t-1}
        rng.shuffle(x_r_test[:, 0])

        # train model
        r = SVR(kernel='rbf', C=0.1).fit(x_r_train, y_r_train)

        # evaluate
        y_r_test_hat = r.predict(x_r_test)
        ssr_r[k] = np.sum((y_r_test_hat - y_r_test) ** 2)

    # hypothesis testing
    p_val[i, j] = np.mean(ssr_ur > ssr_r)

print(p_val)
print(p_val < 0.05)

"""
Ground truth:
[[False  True  True]
 [False False  True]
 [False False False]]
"""
