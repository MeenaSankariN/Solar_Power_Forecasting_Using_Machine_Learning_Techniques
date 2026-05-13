import pandas as pd
from model import run_models, run_time_series, run_rnn
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "solar_data.csv")


def main():

    df = pd.read_csv(data_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    train = df[df['Date'] < '2025-01-01']
    test  = df[df['Date'] >= '2025-01-01']

    features = ['Solar_Irradiance', 'Temperature', 'Specific_Humidity', 'Wind_Speed']

    # ML models
    ml_results, models = run_models(features, train, test)
    print("\nML Results:\n", ml_results)

    # Time series
    ts_results = run_time_series(df)
    print("\nTime Series Results:\n", ts_results)

    # RNN
    rnn_pred = run_rnn(train, test, features)
    print("\nRNN completed")


if __name__ == "__main__":
    main()