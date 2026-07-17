from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import (
    adfuller,
    kpss
)


def integration_order(series):

    series = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if len(series) < 30:
        return "Insufficient observations"

    try:
        if adfuller(
            series,
            autolag="AIC"
        )[1] < 0.05:

            return "I(0)"

    except Exception:
        pass

    try:
        if adfuller(
            series.diff().dropna(),
            autolag="AIC"
        )[1] < 0.05:

            return "I(1)"

    except Exception:
        pass

    try:
        if adfuller(
            series.diff().diff().dropna(),
            autolag="AIC"
        )[1] < 0.05:

            return "I(2)"

    except Exception:
        pass

    return "> I(2)"


def stationarity_report():

    results = []

    for df_name, dataframe in dict_of_df.items():

        for col in dataframe.columns:

            series = pd.to_numeric(
                dataframe[col],
                errors="coerce"
            ).dropna()

            if len(series) < 30:
                continue

            try:

                adf_stat, adf_p, *_ = adfuller(
                    series,
                    autolag="AIC"
                )

            except Exception:

                adf_stat = np.nan
                adf_p = np.nan

            try:

                kpss_stat, kpss_p, *_ = kpss(
                    series,
                    regression="c",
                    nlags="auto"
                )

            except Exception:

                kpss_stat = np.nan
                kpss_p = np.nan

            results.append({

                "DataFrame": df_name,

                "Variable": col,

                "ADF Statistic": adf_stat,

                "ADF p-value": adf_p,

                "ADF Stationary":
                    adf_p < 0.05
                    if pd.notna(adf_p)
                    else np.nan,

                "KPSS Statistic": kpss_stat,

                "KPSS p-value": kpss_p,

                "KPSS Stationary":
                    kpss_p > 0.05
                    if pd.notna(kpss_p)
                    else np.nan,

                "Integration Order":
                    integration_order(series)

            })

    return pd.DataFrame(results)


def stationarity_block(
        block,
        block_name="Unnamed Block"):

    if not isinstance(
            block,
            pd.DataFrame):

        raise ValueError(
            "block must be a DataFrame"
        )

    results = []

    for col in block.columns:

        series = (
            pd.to_numeric(
                block[col],
                errors="coerce"
            )
            .dropna()
        )

        if len(series) < 30:
            continue

        try:

            adf_stat, adf_p, *_ = adfuller(
                series,
                autolag="AIC"
            )

        except Exception:

            adf_stat = np.nan
            adf_p = np.nan

        try:

            kpss_stat, kpss_p, *_ = kpss(
                series,
                regression="c",
                nlags="auto"
            )

        except Exception:

            kpss_stat = np.nan
            kpss_p = np.nan

        results.append({

            "Block": block_name,

            "Variable": col,

            "ADF Statistic": adf_stat,

            "ADF p-value": adf_p,

            "ADF Stationary":
                adf_p < 0.05
                if pd.notna(adf_p)
                else np.nan,

            "KPSS Statistic": kpss_stat,

            "KPSS p-value": kpss_p,

            "KPSS Stationary":
                kpss_p > 0.05
                if pd.notna(kpss_p)
                else np.nan,

            "Integration Order":
                integration_order(
                    series
                )
        })

    return pd.DataFrame(results)
