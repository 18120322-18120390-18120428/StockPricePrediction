import json
import numpy as np
from django.shortcuts import render
from random import randrange

from utils.main import get_data, predict_with_lstm_model, predict_with_xgboost_model
from .constants import intervals, features


def index(request):
    selected_interval = "1d"
    selected_feature = "Close"

    if request.method == 'GET':
        print(request.GET)
        if request.GET.__contains__("interval"):
            selected_interval = request.GET["interval"]
        if request.GET.__contains__("feature"):
            selected_feature = request.GET["feature"]

    df = get_data(selected_interval)

    main_chart_columns = ["Date", "Low", "Open", "Close", "High", selected_feature,
               "LSTM " + selected_feature + " Prediction", "Xgboost " + selected_feature + " Prediction"]
    rsi_chart_columns = ["Date", "RSI", "LSTM RSI Predict", "Xgboost RSI Predict"]

    df["LSTM " + selected_feature + " Prediction"] = np.NaN
    df["Xgboost " + selected_feature + " Prediction"] = np.NaN
    df["LSTM RSI Predict"] = np.NaN
    df["Xgboost RSI Predict"] = np.NaN

    candle_stick_data = [main_chart_columns]
    rsi_data = [rsi_chart_columns]
    for i in range(0, len(df)):
        rsi_data.append([json.dumps(df["Date"][i]), df["RSI"][i], None, None])

    candle_stick_data.extend(df.loc[:, main_chart_columns].values.tolist())
    rsi_data.extend(df.loc[:, rsi_chart_columns].values.tolist())

    # Predict for selected_feature
    lstm_feature_predict = predict_with_lstm_model(df, selected_interval, selected_feature)
    xgboost_feature_predict = predict_with_xgboost_model(df, selected_interval, selected_feature)

    candle_stick_data.append(
        ["Predict", None, None, None, None, None, float(lstm_feature_predict[0][0]), float(xgboost_feature_predict[0][0])])

    # Predict for rsi feature
    lstm_rsi_predict = predict_with_lstm_model(df, selected_interval, "RSI")
    xgboost_rsi_predict = predict_with_xgboost_model(df, selected_interval, "RSI")

    rsi_data.append(["Predict", None, float(lstm_rsi_predict[0][0]), float(xgboost_rsi_predict[0][0])])

    modified_candle_stick_data = json.dumps(candle_stick_data)
    modified_rsi_data = json.dumps(rsi_data)

    return render(request, 'index.html', {
        "candle_stick_data": modified_candle_stick_data,
        "rsi_data": modified_rsi_data,
        "intervals": intervals,
        "features": features,
        "selectedInterval": selected_interval,
        "selectedFeature": selected_feature,
    })
