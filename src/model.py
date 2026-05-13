import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, BayesianRidge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler


# =========================
# MACHINE LEARNING MODELS
# =========================
def run_models(features, train, test, target='Solar_Power'):

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    models = {
        "Linear Regression": LinearRegression(),
        "Bayesian Regression": BayesianRidge(),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        trained_models[name] = model

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(results), trained_models


# =========================
# TIME SERIES MODELS
# =========================
def run_time_series(df):

    df_ts = df.copy()
    df_ts = df_ts.set_index('Date').sort_index()

    train = df_ts.loc[:'2024-12-31']
    test  = df_ts.loc['2025-01-01':]

    y_train = train['Solar_Power']
    y_test  = test['Solar_Power']

    # ARIMA
    arima = ARIMA(y_train, order=(1, 0, 0)).fit()
    pred_arima = arima.forecast(len(y_test))

    # SARIMA
    sarima = SARIMAX(
        y_train,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 365)
    ).fit()

    pred_sarima = sarima.forecast(len(y_test))

    # SARIMAX
    exog = ['Solar_Irradiance', 'Temperature', 'Specific_Humidity', 'Wind_Speed']

    sarimax = SARIMAX(
        y_train,
        exog=train[exog],
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 365)
    ).fit()

    pred_sarimax = sarimax.forecast(len(y_test), exog=test[exog])

    results = {
        "ARIMA_RMSE": np.sqrt(mean_squared_error(y_test, pred_arima)),
        "SARIMA_RMSE": np.sqrt(mean_squared_error(y_test, pred_sarima)),
        "SARIMAX_RMSE": np.sqrt(mean_squared_error(y_test, pred_sarimax))
    }

    return results


# =========================
# RNN MODEL
# =========================
def run_rnn(train, test, features, target='Solar_Power'):

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[features])
    X_test  = scaler_X.transform(test[features])

    y_train = scaler_y.fit_transform(train[target].values.reshape(-1, 1))
    y_test  = scaler_y.transform(test[target].values.reshape(-1, 1))

    def create_sequences(X, y, window=7):
        X_seq, y_seq = [], []

        for i in range(window, len(X)):
            X_seq.append(X[i - window:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_test_seq, y_test_seq   = create_sequences(X_test, y_test)

    model = Sequential([
        SimpleRNN(16, activation='tanh', input_shape=(7, len(features))),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=10,
        batch_size=16,
        verbose=0
    )

    predictions = model.predict(X_test_seq)

    return predictions