import os
import pandas as pd
import numpy as np
import xgboost as xgb

from . import preprocessing_data
from . import normalize_data
from . import build_and_train_lstm_model
from . import build_and_train_xgboost_model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def get_exist_data(interval):
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Data' + interval + '.csv')
    file_exists = os.path.exists(relative_path)

    if file_exists:
        print("Relative path" + relative_path)
        df = pd.read_csv(relative_path, sep='\t', encoding='utf-8')
    else:
        df = get_new_data(interval)

    df = df.tail(1000)
    df.index = range(0, (1000, len(df))[len(df) < 1000], 1)

    return df


def get_new_data(interval):
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Data' + interval + '.csv')

    df = preprocessing_data.read_data(interval)
    df = preprocessing_data.add_roc_column(df)
    df = preprocessing_data.add_ma_and_bollingband_column(df)
    df = preprocessing_data.add_rsi_column(df)
    df = df.tail(1000)
    df.index = range(0, (1000, len(df))[len(df) < 1000], 1)
    df.to_csv(relative_path, sep='\t', encoding='utf-8')

    return df


def predict_with_lstm_model(df, interval, target_column):
    x_val = normalize_data.normalize_and_split_valid_data(df, target_column)
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    lstm_model = build_n_train_lstm_model(df, interval, target_column)

    predicted_closing_price = lstm_model.predict(x_val)
    predicted_closing_price = normalize_data.inverse_transform(predicted_closing_price)

    return predicted_closing_price


def build_n_train_lstm_model(df, interval, target_column):
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'models',
                                 "lstm_model_" + interval + target_column + ".h5")
    file_exists = os.path.exists(relative_path)

    if file_exists:
        lstm_model = load_model(relative_path)
    else:
        lstm_model = build_n_train_new_lstm_model(df, interval, target_column)

    return lstm_model


def build_n_train_new_lstm_model(df, interval, target_column):
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'models',
                                 "lstm_model_" + interval + target_column + ".h5")

    x_train, y_train = normalize_data.normalize_and_split_train_data(df, target_column)

    lstm_model = build_and_train_lstm_model.build_and_train_lstm_model(x_train, y_train)
    lstm_model.save(relative_path)

    return lstm_model


def predict_with_xgboost_model(df, interval, target_column):
    x_val = normalize_data.normalize_and_split_valid_data(df, target_column)
    # print(x_val)
    val = xgb.DMatrix(x_val)
    # print(val)
    xgboost_model = build_n_train_xgboost_model(df, interval, target_column)

    predicted_closing_price = xgboost_model.predict(val)
    predicted_closing_price = np.reshape(predicted_closing_price, (predicted_closing_price.shape[0], 1))
    predicted_closing_price = normalize_data.inverse_transform(predicted_closing_price)

    return predicted_closing_price


def build_n_train_xgboost_model(df, interval, target_column):
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'models',
                                 "xgboost_model_" + interval + target_column + ".txt")
    file_exists = os.path.exists(relative_path)

    if not file_exists:
        build_n_train_new_xgboost_model(df, interval, target_column)

    xgboost_model = xgb.Booster()
    xgboost_model.load_model(relative_path)

    return xgboost_model


def build_n_train_new_xgboost_model(df, interval, target_column):
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'models',
                                 "xgboost_model_" + interval + target_column + ".txt")

    x_train, y_train = normalize_data.normalize_and_split_train_data(df, target_column)
    xgboost_model = build_and_train_xgboost_model.build_and_train_xgboost_model(x_train, y_train)
    xgboost_model.save_model(relative_path)
