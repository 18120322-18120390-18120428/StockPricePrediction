import math
import os
import datetime
import pandas as pd
import numpy as np
import csv
import pandas as pd
import btalib
from pathlib import Path

from datetime import datetime
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

api_key = "dbIXwZgzdtTXEyGLh0WqXWusgmNQrAmIgs6iERES4ZLJxl6aXv4ql60JH19ToL9b"
api_secret = "dCeqx1LTfkWxkxxtVXMzVrwn53MQucQetZ7AFddYlhDUnrwpJjEebiA9f5HGK7Bh"

client = Client(api_key, api_secret)


# client.API_URL = 'https://testnet.binance.vision/api'


def read_data(interval):
    symbol = 'BNBBTC'
    # symbol = "BTCUSDT"
    timestamp = client._get_earliest_valid_timestamp(symbol, interval)

    historical_data = client.get_historical_klines(symbol, interval, timestamp, limit=1000)

    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
               'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
    hist_df = pd.DataFrame(historical_data, columns=columns)
    hist_df = hist_df.iloc[:, 0:6]

    for i in range(0, len(hist_df)):
        hist_df["Date"][i] = datetime.fromtimestamp(hist_df["Date"][i] / 1000)
    # hist_df["Date"] = hist_df["Date"]/1000
    # hist_df["Date"] = pd.to_datetime(hist_df.Date, format="%Y-%m-%d")

    # hist_df.index = hist_df['Date']

    hist_df['Date'] = hist_df['Date'].astype(str)
    for column in hist_df.columns:
        if column != 'Date':
            hist_df[column] = hist_df[column].astype(float)

    # df.drop('Date', inplace=True, axis=1)
    # Sort cột index (lúc này tương ứng với cột Date) theo thứ tự ngày tăng dần
    # hist_df = hist_df.sort_index(ascending=True, axis=0)

    result_df = hist_df.tail(1000)
    result_df.index = range(0, (1000, len(result_df))[len(result_df) < 1000], 1)

    return result_df


def add_roc_column(df):
    num_row = len(df)

    df["ROC"] = np.NaN
    for i in range(1, num_row):
        df["ROC"][i] = (df["Close"][i] / df["Close"][i - 1] - 1) * 100
    return df


def add_rsi_column(df):
    NUM_SESSION = 14
    num_row = len(df)
    closing_price_up = 0
    closing_price_down = 0
    number_up = 0
    number_down = 0

    df["RSI"] = np.NaN

    for i in range(NUM_SESSION + 1, num_row):
        if i == NUM_SESSION + 1:
            for j in range(i - NUM_SESSION, i):
                if df["ROC"][j] > 0:
                    closing_price_up += df["ROC"][j]
                    number_up += 1
                elif df["ROC"][j] < 0:
                    closing_price_down -= df["ROC"][j]
                    number_down += 1
        else:
            if df["ROC"][i - 1] > 0:
                closing_price_up += df["ROC"][i - 1]
                number_up += 1
            elif df["ROC"][i - 1] < 0:
                closing_price_down -= df["ROC"][i - 1]
                number_down += 1
            if df["ROC"][i - NUM_SESSION - 1] > 0:
                closing_price_up -= df["ROC"][i - NUM_SESSION - 1]
                number_up -= 1
            elif df["ROC"][i - NUM_SESSION - 1] < 0:
                closing_price_down += df["ROC"][i - NUM_SESSION - 1]
                number_down -= 1
        RS = (closing_price_up / number_up) / (closing_price_down / number_down)
        df["RSI"][i] = 100 - 100 / (1 + RS)
    return df


def add_ma_and_bollingband_column(df):
    PERIOD = 20
    M = 2
    num_row = len(df)
    closing_price = 0

    df["MA"] = df["Close"].rolling(PERIOD).mean()
    df["UpperBand"] = np.NaN
    df["LowerBand"] = np.NaN

    standard_deviation = 0

    for i in range(PERIOD, num_row):
        standard_deviation = 0
        for j in range(i - PERIOD, i):
            standard_deviation += pow(df["MA"][i] - df["Close"][j], 2)
        standard_deviation = math.sqrt(standard_deviation / (PERIOD - 1))

        df["UpperBand"][i] = df["MA"][i] + M * standard_deviation
        df["LowerBand"][i] = df["MA"][i] - M * standard_deviation

    return df

# def add_rsi_column(df):
#     NUM_SESSION = 14
#     num_row = len(df)
#     closing_price_up = 0
#     closing_price_down = 0
#     number_up = 0
#     number_down = 0
#
#     df["RSI"] = np.NaN
#
#     for i in range(NUM_SESSION + 1, num_row):
#         if i == NUM_SESSION + 1:
#             for j in range(i - NUM_SESSION, i):
#                 if df["ROC"][j] > 0:
#                     closing_price_up += df["ROC"][j]
#                     number_up += 1
#                 elif df["ROC"][j] < 0:
#                     closing_price_down -= df["ROC"][j]
#                     number_down += 1
#         else:
#             if df["ROC"][i - 1] > 0:
#                 closing_price_up += df["ROC"][i - 1]
#                 number_up += 1
#             elif df["ROC"][i - 1] < 0:
#                 closing_price_down -= df["ROC"][i - 1]
#                 number_down += 1
#             if df["ROC"][i - NUM_SESSION - 1] > 0:
#                 closing_price_up -= df["ROC"][i - NUM_SESSION - 1]
#                 number_up -= 1
#             elif df["ROC"][i - NUM_SESSION - 1] < 0:
#                 closing_price_down += df["ROC"][i - NUM_SESSION - 1]
#                 number_down -= 1
#         RS = (closing_price_up / number_up) / (closing_price_down / number_down)
#         df["RSI"][i] = 100 - 100 / (1 + RS)
#     return df
#
#
# def add_ma_and_bollingband_column(df):
#     NUM_SESSION = 20
#     M = 2
#     num_row = len(df)
#     closing_price = 0
#
#     df["MA"] = np.NaN
#     df["UpperBand"] = np.NaN
#     df["LowerBand"] = np.NaN
#
#     standard_deviation = 0
#
#     for i in range(NUM_SESSION, num_row):
#         if i == NUM_SESSION:
#             for j in range(i - NUM_SESSION, i):
#                 closing_price += df["Close"][j]
#         else:
#             closing_price += df["Close"][i - 1]
#             closing_price -= df["Close"][i - NUM_SESSION - 1]
#         df["MA"][i] = closing_price / NUM_SESSION
#
#         standard_deviation = 0
#         for j in range(i - NUM_SESSION, i):
#             standard_deviation += pow(df["MA"][i] - df["Close"][j], 2)
#         standard_deviation = math.sqrt(standard_deviation / (NUM_SESSION - 1))
#
#         df["UpperBand"][i] = df["MA"][i] + M * standard_deviation
#         df["LowerBand"][i] = df["MA"][i] - M * standard_deviation
#
#     return df
