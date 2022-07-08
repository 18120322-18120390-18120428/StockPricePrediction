from django.apps import AppConfig
from utils.main import get_new_data, build_n_train_new_lstm_model, build_n_train_new_xgboost_model
from StockPrice.constants import intervals, features


class StockpriceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'StockPrice'

    def ready(self):
        for interval in intervals:
            df = get_new_data(interval.value)
            for feature in features:
                build_n_train_new_lstm_model(df, interval.value, feature.value)
                build_n_train_new_xgboost_model(df, interval.value, feature.value)

