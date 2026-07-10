import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def prepare_series(series):

    series = series.dropna()

    return series


def train_test_split_ts(
    series,
    test_size=0.2
):

    cutoff = int(
        len(series)
        * (1 - test_size)
    )

    train = series.iloc[:cutoff]
    test = series.iloc[cutoff:]

    return train, test


def evaluate_forecast(
    actual,
    predicted
):

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": mean_squared_error(
            actual,
            predicted,
            squared=False
        )
    }


def export_results(
    dataframe,
    filename
):

    dataframe.to_excel(
        filename,
        index=False
    )

    print(
        f"Exported to {filename}"
    )
