import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


def compare_ar_vs_leading_indicator(
    df,
    target,
    leading_factor,
    lag=1
):

    data = df[
        [target, leading_factor]
    ].copy()

    data[leading_factor] = (
        data[leading_factor].shift(lag)
    )

    data = data.dropna()

    y = data[target]

    x = data[[leading_factor]]

    ar_model = SARIMAX(
        y,
        order=(1, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    arx_model = SARIMAX(
        y,
        exog=x,
        order=(1, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    return pd.DataFrame({
        "Model": [
            "AR(1)",
            f"ARX(1)+{leading_factor}"
        ],
        "AIC": [
            ar_model.aic,
            arx_model.aic
        ],
        "BIC": [
            ar_model.bic,
            arx_model.bic
        ]
    })


def predictive_gain(
    df,
    target,
    leading_factor,
    lag=1
):

    result = compare_ar_vs_leading_indicator(
        df,
        target,
        leading_factor,
        lag
    )

    gain = (
        result.iloc[0]["AIC"]
        - result.iloc[1]["AIC"]
    )

    return {
        "Target": target,
        "Leading Factor": leading_factor,
        "AIC Gain": gain
    }
