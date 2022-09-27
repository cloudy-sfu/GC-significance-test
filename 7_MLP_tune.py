import pickle

import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.regularizers import L2
from scipy.stats import wilcoxon

with open('raw/lorenz96.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

n_nodes = x_train.shape[2]
z_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)
rng = np.random.RandomState(seed=23846)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)

i = 2
j = 5

x_ur_train = np.swapaxes(x_train, 1, 2).reshape(x_train.shape[0], -1)
x_ur_test = np.swapaxes(x_test, 1, 2).reshape(x_test.shape[0], -1)

x_r_train = x_train.copy()
rng.shuffle(x_r_train[:, :, i])
x_r_train = np.swapaxes(x_r_train, 1, 2).reshape(x_train.shape[0], -1)
x_r_test = x_test.copy()
rng.shuffle(x_r_test[:, :, i])
x_r_test = np.swapaxes(x_r_test, 1, 2).reshape(x_test.shape[0], -1)

l0 = Input(shape=x_ur_train.shape[1])
l1 = Dense(256, activation='relu', kernel_regularizer=L2(0.1))(l0)
l2 = Dense(256, activation='relu', kernel_regularizer=L2(0.1))(l1)
l3 = Dropout(0.1)(l2)
l4 = Dense(64, activation='relu')(l3)
l5 = Dense(1)(l4)
r = tf.keras.Model(l0, l5)
ur = tf.keras.models.clone_model(r)

r.compile(optimizer='adam', loss='mse')
r.fit(
    x_r_train, y_train[:, j], validation_data=(x_r_test, y_test[:, j]),
    epochs=5000, batch_size=2000, callbacks=[stop_early]
)  # restricted model
y_r_test_hat = r.predict(x_r_test, batch_size=2000)

# ur.compile(optimizer='adam', loss='mse')
# ur.fit(
#     x_ur_train, y_train[:, j], validation_data=(x_ur_test, y_test[:, j]),
#     epochs=5000, batch_size=2000, callbacks=[stop_early]
# )  # unrestricted model
# y_ur_test_hat = ur.predict(x_ur_test, batch_size=2000)

# err_ur = (y_ur_test_hat.flatten() - y_test[:, j]) ** 2
# err_r = (y_r_test_hat.flatten() - y_test[:, j]) ** 2
# wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx')
# w, p, z = wilcoxon_results.statistic, wilcoxon_results.pvalue, wilcoxon_results.zstatistic
#
# print(p)
