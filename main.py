import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import time

df = pd.read_csv("data/prices-split-adjusted.csv", index_col=0)

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10
test_set_size_percentage = 10


# feature engineering
def add_features(df):
    df['range_hl'] = df['high'] - df['low']
    df['range_co'] = df['close'] - df['open']
    df['cross_hl'] = df['high'] * df['low']
    df['cross_co'] = df['close'] * df['open']
    df['cross_hc'] = df['close'] * df['high']
    df['cross_ho'] = df['high'] * df['open']
    df['cross_lc'] = df['low'] * df['close']
    df['cross_lo'] = df['low'] * df['open']
    return df


# Use Lasso Regression to select feature
def feature_selection(X_train, Y_train):
    from sklearn.linear_model import LassoCV
    regr = LassoCV(cv=5, random_state=101)
    regr.fit(X_train, Y_train)

    model_coef = pd.Series(regr.coef_, index = list(X_train.columns[:]))

    top_coef = model_coef.sort_values()
    return list(top_coef[top_coef != 0].index)


def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['volume'] = min_max_scaler.fit_transform(df['volume'].values.reshape(-1, 1))
    df['range_hl'] = min_max_scaler.fit_transform(df['range_hl'].values.reshape(-1, 1))
    df['range_co'] = min_max_scaler.fit_transform(df['range_co'].values.reshape(-1, 1))
    df['cross_hl'] = min_max_scaler.fit_transform(df['cross_hl'].values.reshape(-1, 1))
    df['cross_co'] = min_max_scaler.fit_transform(df['cross_co'].values.reshape(-1, 1))
    df['cross_hc'] = min_max_scaler.fit_transform(df['cross_hc'].values.reshape(-1, 1))
    df['cross_ho'] = min_max_scaler.fit_transform(df['cross_ho'].values.reshape(-1, 1))
    df['cross_lc'] = min_max_scaler.fit_transform(df['cross_lc'].values.reshape(-1, 1))
    df['cross_lo'] = min_max_scaler.fit_transform(df['cross_lo'].values.reshape(-1, 1))
    # target
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1, 1))
    return df


# function to create train, validation, test data given preprocessed stock data
def load_reduced_features(stock):
    feature_cols = ['open', 'high', 'range_co']
    target_col = 'close'

    features = stock[feature_cols].values
    target = stock[target_col].values

    valid_set_size = int(np.round(valid_set_size_percentage / 100 * features.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * features.shape[0]))
    train_set_size = features.shape[0] - (valid_set_size + test_set_size)

    x_train = features[:train_set_size, :]
    y_train = target[:train_set_size]

    x_valid = features[train_set_size:train_set_size + valid_set_size, :]
    y_valid = target[train_set_size:train_set_size + valid_set_size]

    x_test = features[train_set_size + valid_set_size:, :]
    y_test = target[train_set_size + valid_set_size:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_data(stock):
    feature_cols = ['open', 'high', 'low', 'volume', 'range_hl', 'range_co',
                    'cross_hl', 'cross_co', 'cross_hc', 'cross_ho', 'cross_lc', 'cross_lo']

    target_col = 'close'

    features = stock[feature_cols].values
    target = stock[target_col].values

    valid_set_size = int(np.round(valid_set_size_percentage / 100 * features.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * features.shape[0]))
    train_set_size = features.shape[0] - (valid_set_size + test_set_size)

    x_train = features[:train_set_size, :]
    y_train = target[:train_set_size]

    x_valid = features[train_set_size:train_set_size + valid_set_size, :]
    y_valid = target[train_set_size:train_set_size + valid_set_size]

    x_test = features[train_set_size + valid_set_size:, :]
    y_test = target[train_set_size + valid_set_size:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# choose one stock
df_stock = df[df.symbol == 'EQIX'].copy()
df_stock.drop(['symbol'], 1, inplace=True)
df_stock = add_features(df_stock)

cols = list(df_stock.columns.values)

# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# create train, test data
x_train, y_train, x_valid, y_valid, x_test, y_test = load_reduced_features(df_stock_norm)


'''--------------------------------------Model Training--------------------------------------'''

from mgbdt import MGBDT
from multi_xgb_model import MultiXGBModel
from linear_model import LinearModel

# Create a multi-layerd GBDTs
net = MGBDT(loss="MSELoss", target_lr=0.01, epsilon=0.01)

# add several target-propogation layers
# F, G represent the forward mapping and inverse mapping
# (in the paper, Ji Feng et.al used gradient boosting decision tree)
net.add_layer("tp_layer",
              F=MultiXGBModel(input_size=3, output_size=5, learning_rate=0.01, max_depth=7, num_boost_round=5),
              G=None)
net.add_layer("tp_layer",
              F=MultiXGBModel(input_size=5, output_size=3, learning_rate=0.01, max_depth=7, num_boost_round=5),
              G=MultiXGBModel(input_size=3, output_size=5, learning_rate=0.01, max_depth=7, num_boost_round=5))
net.add_layer("tp_layer",
              F=MultiXGBModel(input_size=3, output_size=1, learning_rate=0.01, max_depth=7, num_boost_round=5),
              G=MultiXGBModel(input_size=1, output_size=3, learning_rate=0.01, max_depth=7, num_boost_round=5))

# init the forward mapping
net.init(x_train, n_rounds=5)

start_time = time.time()
# fit the dataset
net.fit(x_train, y_train, n_epochs=10, eval_sets=[(x_test, y_test)], eval_metric="mean_squared_error")
print('Time:{}'.format(time.time() - start_time))

# prediction
y_train_pred = net.forward(x_train)
y_valid_pred = net.forward(x_valid)
y_test_pred = net.forward(x_test)

'''
# get the hidden outputs
# hiddens[0] represent the input data
# hiddens[1] represent the output of the first layer
# hiddens[2] represent the output of the second layer
# hiddens[3] represent the output of the final layer (same as y_pred)
# hiddens = net.get_hiddens(x_test)
'''


'''--------------------------------------Show Predictions--------------------------------------'''

plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)

plt.plot(np.arange(y_train.shape[0]), y_train[:], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

plt.subplot(1,2,2)

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

