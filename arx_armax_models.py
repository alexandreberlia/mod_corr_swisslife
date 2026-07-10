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

from statsmodels.tsa.statespace.sarimax import SARIMAX


def compare_arma_vs_leading_indicator(
    data, p, q,
    target,
    leading_factor,
    lag=1
):
    """
    Compare :

    ARMA(p,q)

    contre

    ARMAX(p,q)

    où le leading_factor est ajouté
    comme variable explicative.
    """

    df = data[[target, leading_factor]].copy()

    df[leading_factor] = (
        df[leading_factor]
        .shift(lag)
    )

    df = df.dropna()

    y = df[target]

    x = df[[leading_factor]]

    arma_model = SARIMAX(
        y,
        order=(p, 0, q),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    armax_model = SARIMAX(
        y,
        exog=x,
        order=(p, 0, q),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    return pd.DataFrame({

        "Model":[
            f"ARMA({p},{q})",
            f"ARMAX({p},{q})"
        ],

        "AIC":[
            arma_model.aic,
            armax_model.aic
        ],

        "BIC":[
            arma_model.bic,
            armax_model.bic
        ]
    })


from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def best_model_with_leading_indicator(
    data,
    target,
    leading_factor,
    max_ar=5,
    max_ma=5,
    lag=1
):

    df = data[[target, leading_factor]].copy()

    df[leading_factor] = df[leading_factor].shift(lag)

    df = df.dropna()

    y = df[target]

    x = df[[leading_factor]]

    results = []

    # --------------------
    # AR
    # --------------------

    for p in range(1, max_ar + 1):

        try:

            model = AutoReg(
                y,
                lags=p,
                old_names=False
            ).fit()

            results.append({
                "Model Type": "AR",
                "Model": f"AR({p})",
                "AIC": model.aic,
                "BIC": model.bic
            })

        except:
            pass

    # --------------------
    # ARMA
    # --------------------

    for p in range(max_ar + 1):

        for q in range(max_ma + 1):

            if p == 0 and q == 0:
                continue

            try:

                model = ARIMA(
                    y,
                    order=(p, 0, q)
                ).fit()

                results.append({
                    "Model Type": "ARMA",
                    "Model": f"ARMA({p},{q})",
                    "AIC": model.aic,
                    "BIC": model.bic
                })

            except:
                pass

    # --------------------
    # ARX
    # --------------------

    for p in range(1, max_ar + 1):

        try:

            model = SARIMAX(
                y,
                exog=x,
                order=(p,0,0),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            coef = model.params[leading_factor]
            pval = model.pvalues[leading_factor]

            results.append({
                "Model Type": "ARX",
                "Model": f"ARX({p})",
                "AIC": model.aic,
                "BIC": model.bic,
                "Leading Coef": coef,
                "Leading p-value": pval
            })

        except:
            pass

    # --------------------
    # ARMAX
    # --------------------

    for p in range(max_ar + 1):

        for q in range(max_ma + 1):

            if p == 0 and q == 0:
                continue

            try:

                model = SARIMAX(
                    y,
                    exog=x,
                    order=(p,0,q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)

                coef = model.params[leading_factor]
                pval = model.pvalues[leading_factor]

                results.append({
                    "Model Type": "ARMAX",
                    "Model": f"ARMAX({p},{q})",
                    "AIC": model.aic,
                    "BIC": model.bic,
                    "Leading Coef": coef,
                    "Leading p-value": pval
                })

            except:
                pass

    results = pd.DataFrame(results)

    results = results.sort_values("AIC")

    return results
