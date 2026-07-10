import pandas as pd

from statsmodels.tsa.ar_model import AutoReg


def test_ar(series, max_lag=12):

    results = []

    series = series.dropna()

    for lag in range(1, max_lag + 1):

        try:

            model = AutoReg(
                series,
                lags=lag,
                old_names=False
            ).fit()

            results.append({
                "Model": f"AR({lag})",
                "Lag": lag,
                "AIC": model.aic,
                "BIC": model.bic,
                "R2": model.rsquared if hasattr(model, "rsquared") else None
            })

        except Exception:
            pass

    return pd.DataFrame(results).sort_values("AIC")


def best_ar_model(series, max_lag=12):

    results = test_ar(series, max_lag)

    return results.iloc[0]


def all_variables_ar(df, max_lag=12):

    output = []

    for col in df.columns:

        try:

            best = best_ar_model(df[col], max_lag)

            output.append({
                "Variable": col,
                "Best Model": best["Model"],
                "AIC": best["AIC"],
                "BIC": best["BIC"]
            })

        except Exception:
            pass

    return pd.DataFrame(output).sort_values("AIC")
