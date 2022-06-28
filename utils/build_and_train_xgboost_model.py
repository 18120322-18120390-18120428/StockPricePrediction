import os
import xgboost as xgb


def build_and_train_xgboost_model(x_train, y_train):
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_model')
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.01)

    xgb_model.fit(x_train, y_train)
    return xgb_model
