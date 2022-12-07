import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon
import tensorflow as tf
from keras.layers import *

# %% Load data
with open('raw/lorenz96.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

# %% Constants
n_nodes = x_train.shape[2]
z_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)
rng = np.random.RandomState(seed=23846)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed
# Although random seed is fixed, we still observe uncertainty in training neural network.
tf.keras.utils.set_random_seed(748697)

# %% Train
for i in range(n_nodes):
    x_r_train = x_train.copy()
    rng.shuffle(x_r_train[:, :, i])
    x_r_test = x_test.copy()
    rng.shuffle(x_r_test[:, :, i])

    for j in range(n_nodes):
        if i == j:
            continue
        l0 = Input(shape=(x_train.shape[1], x_train.shape[2]))
        l1 = LSTM(8)(l0)
        l2 = Dense(8, activation='relu')(l1)
        l3 = Dense(1, activation='linear')(l2)
        ur = tf.keras.Model(l0, l3)
        r = tf.keras.models.clone_model(ur)

        ur.compile(optimizer='adam', loss='mse')
        ur.fit(
            x_train, y_train[:, j], validation_data=(x_test, y_test[:, j]),
            epochs=5000, batch_size=2000, callbacks=[stop_early], verbose=0
        )
        r.compile(optimizer='adam', loss='mse')
        r.fit(
            x_r_train, y_train[:, j], validation_data=(x_r_test, y_test[:, j]),
            epochs=5000, batch_size=2000, callbacks=[stop_early], verbose=0
        )

        y_ur_test_hat = ur.predict(x_test, batch_size=2000)
        y_r_test_hat = r.predict(x_r_test, batch_size=2000)
        err_ur = (y_ur_test_hat.flatten() - y_test[:, j]) ** 2
        err_r = (y_r_test_hat.flatten() - y_test[:, j]) ** 2
        wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx')
        w, p, z = wilcoxon_results.statistic, wilcoxon_results.pvalue, wilcoxon_results.zstatistic

        p_val[i, j] = p
        z_val[i, j] = z

        print('i =', i, 'j =', j, 'p =', p)

# %% Export results
with open('raw/lorenz96_conditional_LSTM_noreg_wilcoxon_stats.pkl', 'wb') as f:
    pickle.dump({'z_val': z_val, 'p_val': p_val}, f)

# %% Causality heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(p_val, dtype=bool)
mask[np.diag_indices_from(mask)] = True
heatmap = sns.heatmap(p_val, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=True, fmt='.2f')
ax.set_ylabel('Cause')
ax.set_xlabel('Effect')
fig.subplots_adjust(bottom=0.15, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('results/lorenz96_conditional_LSTM_noreg_wilcoxon.eps')
plt.close(fig)
