import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon
import tensorflow as tf
from keras.layers import *

with open('raw/lorenz96.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

n_nodes = x_train.shape[2]
z_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)
rng = np.random.RandomState(seed=23846)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)
tf.random.set_seed(seed=748697)

for i in range(n_nodes):
    x_ur_train = np.swapaxes(x_train, 1, 2).reshape(x_train.shape[0], -1)
    x_ur_test = np.swapaxes(x_test, 1, 2).reshape(x_test.shape[0], -1)

    x_r_train = x_train.copy()
    rng.shuffle(x_r_train[:, :, i])
    x_r_train = np.swapaxes(x_r_train, 1, 2).reshape(x_train.shape[0], -1)
    x_r_test = x_test.copy()
    rng.shuffle(x_r_test[:, :, i])
    x_r_test = np.swapaxes(x_r_test, 1, 2).reshape(x_test.shape[0], -1)

    for j in range(n_nodes):
        if i == j:
            continue
        l0 = Input(shape=x_ur_train.shape[1])
        l1 = Dense(256, activation='relu')(l0)
        l2 = Dense(256, activation='relu')(l1)
        l3 = Dropout(0.1)(l2)
        l4 = Dense(64, activation='relu')(l3)
        l5 = Dense(1)(l4)
        r = tf.keras.Model(l0, l5)
        ur = tf.keras.models.clone_model(r)

        r.compile(optimizer='adam', loss='mse')
        r.fit(
            x_r_train, y_train[:, j], validation_data=(x_r_test, y_test[:, j]),
            epochs=5000, batch_size=1000, callbacks=[stop_early], verbose=0
        )  # restricted model
        y_r_test_hat = r.predict(x_r_test, batch_size=2000)

        ur.compile(optimizer='adam', loss='mse')
        ur.fit(
            x_ur_train, y_train[:, j], validation_data=(x_ur_test, y_test[:, j]),
            epochs=5000, batch_size=1000, callbacks=[stop_early], verbose=0
        )  # unrestricted model
        y_ur_test_hat = ur.predict(x_ur_test, batch_size=2000)

        err_ur = (y_ur_test_hat.flatten() - y_test[:, j]) ** 2
        err_r = (y_r_test_hat.flatten() - y_test[:, j]) ** 2
        wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx')
        w, p, z = wilcoxon_results.statistic, wilcoxon_results.pvalue, wilcoxon_results.zstatistic

        p_val[i, j] = p
        z_val[i, j] = z

        print('i =', i, 'j =', j, 'p =', p)

# %% Export results
with open('raw/lorenz96_conditional_MLP_wilcoxon_stats.pkl', 'wb') as f:
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
heatmap.get_figure().savefig('results/lorenz96_conditional_MLP_wilcoxon.svg')
plt.close(fig)
