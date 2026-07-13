from statsmodels.tsa.stattools import adfuller, kpss
convert_to_weeks()
import pandas as pd

excel_file = pd.ExcelFile("cleaned_macro_data.xlsx")

dict_of_df = {}

for sheet in excel_file.sheet_names:

    temp_df = pd.read_excel(
        excel_file,
        sheet_name=sheet,
        index_col=0
    )

    temp_df.index = pd.to_datetime(
        temp_df.index
    )

    dict_of_df[sheet] = temp_df

for i, (name, dataframe) in enumerate(
    dict_of_df.items(),
    start=1
):
    globals()[f"df{i}"] = dataframe
def integration_order(series):

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
    except:
        pass

    try:
        adf_diff1 = adfuller(
            series.diff().dropna(),
            autolag='AIC'
        )[1]

        if adf_diff1 < 0.05:
            return "I(1)"
    except:
        pass

    try:
        adf_diff2 = adfuller(
            series.diff().diff().dropna(),
            autolag='AIC'
        )[1]

        if adf_diff2 < 0.05:
            return "I(2)"
    except:
        pass

    return "> I(2)"


def stationarity_report(
        save_excel='no',
        excel_name='Stationarity_Report',
        cell_width=75):

    results = []

    for df_name, dataframe in dict_of_df.items():

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
            except:
                adf_stat = np.nan
                adf_p = np.nan

            try:
                kpss_stat, kpss_p, _, _ = kpss(
                    series,
                    regression='c',
                    nlags='auto'
                )
            except:
                kpss_stat = np.nan
                kpss_p = np.nan

            if pd.notna(adf_p):
                adf_stationary = adf_p < 0.05
            else:
                adf_stationary = np.nan

            if pd.notna(kpss_p):
                kpss_stationary = kpss_p > 0.05
            else:
                kpss_stationary = np.nan

            results.append({
                "DataFrame": df_name,
                "Variable": col,
                "ADF Statistic": adf_stat,
                "ADF p-value": adf_p,
                "ADF Stationary": adf_stationary,
                "KPSS Statistic": kpss_stat,
                "KPSS p-value": kpss_p,
                "KPSS Stationary": kpss_stationary,
                "Integration Order": integration_order(series)
            })

    results_df = pd.DataFrame(results)

    if save_excel == 'yes':

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


def stationarity_summary():

    df = stationarity_report()

    summary = df[
        [
            "Variable",
            "ADF p-value",
            "KPSS p-value",
            "Integration Order"
        ]
    ]

    summary = summary.sort_values(
        by="Integration Order"
    )

    return summary


def print_stationarity_summary():

    summary = stationarity_summary()

    print("\n")
    print("=" * 100)
    print("STATIONARITY SUMMARY")
    print("=" * 100)

    print(summary.to_string(index=False))

    return summary
