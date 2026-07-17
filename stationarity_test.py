from statsmodels.tsa.stattools import adfuller, kpss


def integration_order(
        series):

    series = pd.to_numeric(
        series,
        errors='coerce'
    ).dropna()

    if len(series) < 30:
        return "Insufficient observations"

    try:

        adf_level = adfuller(
            series,
            autolag='AIC'
        )[1]

        if adf_level < 0.05:
            return "I(0)"

    except Exception:
        pass

    try:

        adf_diff1 = adfuller(
            series.diff().dropna(),
            autolag='AIC'
        )[1]

        if adf_diff1 < 0.05:
            return "I(1)"

    except Exception:
        pass

    try:

        adf_diff2 = adfuller(
            series.diff().diff().dropna(),
            autolag='AIC'
        )[1]

        if adf_diff2 < 0.05:
            return "I(2)"

    except Exception:
        pass

    return "> I(2)"


def stationarity_report(
        target="all",
        save_excel='no',
        excel_name='Stationarity_Report',
        cell_width=75):
    
    results = []

    # Cas 1 : tous les DataFrames du dictionnaire
    if target == "all":

        datasets = list(
            dict_of_df.items()
        )

    # Cas 2 : un DataFrame unique
    elif isinstance(target, pd.DataFrame):

        datasets = [
            ("CUSTOM_BLOCK", target)
        ]

    # Cas 3 : dictionnaire de blocs
    elif isinstance(target, dict):

        datasets = list(
            target.items()
        )

    else:

        raise ValueError(
            "target must be 'all', a DataFrame, or a dict of DataFrames."
        )

    for df_name, dataframe in datasets:

        for col in dataframe.columns:

            series = pd.to_numeric(
                dataframe[col],
                errors='coerce'
            ).dropna()

            if len(series) < 30:
                continue

            try:

                adf_stat, adf_p, _, _, _, _ = adfuller(
                    series,
                    autolag='AIC'
                )

            except Exception:

                adf_stat = np.nan
                adf_p = np.nan

            try:

                kpss_stat, kpss_p, _, _ = kpss(
                    series,
                    regression='c',
                    nlags='auto'
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

    results_df = pd.DataFrame(
        results
    )

    if save_excel == "yes":

        results_df.to_excel(
            f"{excel_name}.xlsx",
            index=False
        )

        wb = load_workbook(
            f"{excel_name}.xlsx"
        )

        adjust_dimensions(
            wb,
            max_column_width=cell_width
        )

        wb.save(
            f"{excel_name}.xlsx"
        )

    return results_df
