import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


def validate_arma_model(
    series,
    p,
    q,
    alpha=0.05,
    nb_lags_test=12
):

    series = series.dropna()

    model = ARIMA(
        series,
        order=(p,0,q)
    ).fit()

    residuals = model.resid

    ljung_box = acorr_ljungbox(
        residuals,
        lags=[nb_lags_test],
        return_df=True
    )

    pvalue = ljung_box["lb_pvalue"].iloc[0]

    if pvalue > alpha:

        decision = (
            "Non rejet H0 : "
            "ARMA adéquat"
        )

    else:

        decision = (
            "Rejet H0 : "
            "ARMA insuffisant"
        )

    return pd.DataFrame({

        "Model":[
            f"ARMA({p},{q})"
        ],

        "AIC":[
            model.aic
        ],

        "BIC":[
            model.bic
        ],

        "Ljung-Box p-value":[
            pvalue
        ],

        "Decision":[
            decision
        ]
    })
