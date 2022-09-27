import pickle

import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.regularizers import L2
from scipy.stats import wilcoxon

tf.config.list_physical_devices('GPU')


with open('raw/lorenz96.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

n_nodes = x_train.shape[2]
z_val = np.full((n_nodes, n_nodes), np.nan)
p_val = np.full((n_nodes, n_nodes), np.nan)
rng = np.random.RandomState(seed=23846)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)

i = 1
j = 4

x_r_train = x_train.copy()
rng.shuffle(x_r_train[:, :, i])
x_r_test = x_test.copy()
rng.shuffle(x_r_test[:, :, i])

l0 = Input(shape=(x_train.shape[1], x_train.shape[2]))
l1 = LSTM(8, kernel_regularizer=L2(0.01))(l0)
l2 = Dense(8, activation='relu', kernel_regularizer=L2(0.01))(l1)
l3 = Dense(1, activation='linear')(l2)
r = tf.keras.Model(l0, l3)
ur = tf.keras.models.clone_model(r)

r.compile(optimizer='adam', loss='mse')
r.fit(
    x_r_train, y_train[:, j], validation_data=(x_r_test, y_test[:, j]),
    epochs=5000, batch_size=2000, callbacks=[stop_early]
)  # restricted model
y_r_test_hat = r.predict(x_r_test, batch_size=2000)

ur.compile(optimizer='adam', loss='mse')
ur.fit(
    x_train, y_train[:, j], validation_data=(x_test, y_test[:, j]),
    epochs=5000, batch_size=2000, callbacks=[stop_early]
)  # unrestricted model
y_ur_test_hat = ur.predict(x_test, batch_size=2000)

err_ur = (y_ur_test_hat.flatten() - y_test[:, j]) ** 2
err_r = (y_r_test_hat.flatten() - y_test[:, j]) ** 2
wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx')
w, p, z = wilcoxon_results.statistic, wilcoxon_results.pvalue, wilcoxon_results.zstatistic

print(p)
