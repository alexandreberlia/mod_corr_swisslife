import pandas as pd

from statsmodels.tsa.arima.model import ARIMA


def test_arma(
    series,
    max_ar=5,
    max_ma=5
):

    results = []

    series = series.dropna()

    for p in range(max_ar + 1):

        for q in range(max_ma + 1):

            if p == 0 and q == 0:
                continue

            try:

                model = ARIMA(
                    series,
                    order=(p, 0, q)
                ).fit()

                results.append({
                    "Model": f"ARMA({p},{q})",
                    "AR": p,
                    "MA": q,
                    "AIC": model.aic,
                    "BIC": model.bic
                })

            except Exception:
                pass

    return pd.DataFrame(results).sort_values("AIC")


def best_arma_model(
    series,
    max_ar=5,
    max_ma=5
):

    return test_arma(
        series,
        max_ar=max_ar,
        max_ma=max_ma
    ).iloc[0]


def all_variables_arma(
    df,
    max_ar=5,
    max_ma=5
):

    results = []

    for col in df.columns:

        try:

            best = best_arma_model(
                df[col],
                max_ar=max_ar,
                max_ma=max_ma
            )

            results.append({
                "Variable": col,
                "Best Model": best["Model"],
                "AIC": best["AIC"],
                "BIC": best["BIC"]
            })

        except Exception:
            pass

    return pd.DataFrame(results).sort_values("AIC")
