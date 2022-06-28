import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense


def build_and_train_lstm_model(x_train, y_train):

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    return lstm_model

