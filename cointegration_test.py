from statsmodels.tsa.stattools import coint

def cointegration_report(
        significance_level=0.05,
        save_excel='no',
        excel_name='Cointegration_Report',
        cell_width=75):

    results = []

    for df_name, dataframe in dict_of_df.items():

        columns = dataframe.columns.tolist()

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                var1 = columns[i]
                var2 = columns[j]

                merged = pd.concat(
                    [
                        dataframe[var1],
                        dataframe[var2]
                    ],
                    axis=1
                ).dropna()

                if len(merged) < 50:
                    continue

                try:

                    score, pvalue, critical_values = coint(
                        merged[var1],
                        merged[var2]
                    )

                    results.append({

                        "DataFrame": df_name,

                        "Variable 1": var1,

                        "Variable 2": var2,

                        "Cointegration Statistic": score,

                        "p-value": pvalue,

                        "5% Critical Value": critical_values[1],

                        "Cointegrated": pvalue < significance_level

                    })

                except:

                    continue

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="p-value"
    )

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
            f"{excel_name}.xlsx")

def significant_cointegrated_pairs(
        significance_level=0.05):

    df = cointegration_report(
        significance_level=significance_level
    )

    return df[
        df["Cointegrated"] == True
    ]

def cointegration_I1_only(
        significance_level=0.05):

    stationarity_df = stationarity_report()

    order_dict = dict(
        zip(
            stationarity_df["Variable"],
            stationarity_df["Integration Order"]
        )
    )

    results = []

    for df_name, dataframe in dict_of_df.items():

        columns = dataframe.columns.tolist()

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                var1 = columns[i]
                var2 = columns[j]

                if order_dict.get(var1) != "I(1)":
                    continue

                if order_dict.get(var2) != "I(1)":
                    continue

                merged = pd.concat(
                    [
                        dataframe[var1],
                        dataframe[var2]
                    ],
                    axis=1
                ).dropna()

                if len(merged) < 50:
                    continue

                try:

                    score, pvalue, critical_values = coint(
                        merged[var1],
                        merged[var2]
                    )

                    results.append({

                        "Variable 1": var1,

                        "Variable 2": var2,

                        "p-value": pvalue,

                        "Cointegrated":
                            pvalue < significance_level

                    })

                except:

                    continue

    return pd.DataFrame(results).sort_values(
        by="p-value"
    )

    return results_df
