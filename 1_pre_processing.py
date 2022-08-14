import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def moving_window(ts, k_):
    """
    Make moving window samples from time series.
    :param ts: Time series.
    :param k_: Length of the window.
    :return: x_, y_: fraction used as input, fraction used as output.
    """
    length = ts.shape[0]
    y_ = ts[k_:]
    indices = np.tile(np.arange(k_), [length - k_, 1]) + np.arange(length - k_)[:, np.newaxis]
    x_ = ts[indices]
    return x_, y_


# %% Read the dataset.
with open('data/stock_indexes.pkl', 'rb') as f:
    indexes = pickle.load(f)

# %% Save column names.
with open('raw/indexes_names.pkl', 'wb') as f:
    pickle.dump(indexes.columns.tolist(), f)

# %% Split training and testing set.
train_size = round(0.8 * indexes.shape[0])
indexes_train, indexes_test = indexes.values[:train_size], indexes.values[train_size:]

# %% Standardization.
std_scaler = StandardScaler()
indexes_train_std = std_scaler.fit_transform(indexes_train)
indexes_test_std = std_scaler.transform(indexes_test)
with open('raw/indexes_std_scaler.pkl', 'wb') as f:
    pickle.dump(std_scaler, f)

# %% Moving window
x_train, y_train = moving_window(indexes_train_std, k_=10)
x_test, y_test = moving_window(indexes_test_std, k_=10)
with open('raw/indexes_std.pkl', 'wb') as f:
    pickle.dump([x_train, x_test, y_train, y_test], f)
