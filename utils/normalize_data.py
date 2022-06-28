import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1)) # MinMaxScaler sẽ đưa các biến về miền giá trị [0, 1]


def normalize_and_split_train_data(df, target_column):
    # extract target column from dataframe
    data = df.filter([target_column])
    dataset = data.values

    train_data_len = len(df)

    # Scale data
    scaled_data = scaler.fit_transform(dataset)  # Tranform dữ liệu về khoảng [0-1]

    train_data = scaled_data[0:train_data_len, :]

    # Chia dữ liệu thằng tập x và y trong đó x là 60 dòng trước và y là dòng hiện tại
    x_train, y_train = [], []
    for i in range(60, train_data_len):
        x_train.append(train_data[i - 60:i, 0])  # x_train la du lieu 60 dong dau [i-60, i]
        y_train.append(train_data[i, 0])  # y_train la du lieu dong tiep theo [i]

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train


def normalize_and_split_valid_data(df, target_column):
    # extract target column from dataframe
    df = df.filter([target_column])
    data = df.tail(60)
    dataset = data.values

    # Scale data
    scaled_data = scaler.fit_transform(dataset)  # Tranform dữ liệu về khoảng [0-1]

    x_val = [scaled_data[0:60, 0]]
    x_val = np.array(x_val)

    return x_val


def inverse_transform(predict_result):
    return scaler.inverse_transform(predict_result)
