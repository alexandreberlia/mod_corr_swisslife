from statsmodels.tsa.stattools import coint

def cointegration_report(dict_of_df,
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

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import coint


def cointegration_block(
        block,
        block_name="Unnamed Block",
        significance_level=0.05,
        min_observations=50,
        save_excel=False,
        excel_name=None):
    """
    Teste la cointégration d'Engle-Granger entre toutes les paires
    de variables contenues dans un bloc.

    Parameters
    ----------
    block : pd.DataFrame
        Bloc de séries temporelles à tester.

    block_name : str, default="Unnamed Block"
        Nom du bloc économique.

    significance_level : float, default=0.05
        Seuil utilisé pour rejeter l'hypothèse nulle
        d'absence de cointégration.

    min_observations : int, default=50
        Nombre minimal d'observations communes requis.

    save_excel : bool, default=False
        Si True, exporte les résultats vers Excel.

    excel_name : str, optional
        Nom du fichier Excel sans extension.

    Returns
    -------
    pd.DataFrame
        Résultats des tests de cointégration par paire.
    """

    if not isinstance(block, pd.DataFrame):
        raise TypeError(
            "block must be a pandas DataFrame."
        )

    if block.shape[1] < 2:
        raise ValueError(
            "The block must contain at least two variables."
        )

    if not 0 < significance_level < 1:
        raise ValueError(
            "significance_level must be between 0 and 1."
        )

    if min_observations < 1:
        raise ValueError(
            "min_observations must be strictly positive."
        )

    results = []

    columns = block.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            var1 = columns[i]
            var2 = columns[j]

            merged = pd.concat(
                [
                    pd.to_numeric(
                        block[var1],
                        errors="coerce"
                    ).rename(var1),
                    pd.to_numeric(
                        block[var2],
                        errors="coerce"
                    ).rename(var2)
                ],
                axis=1
            ).dropna()

            n_observations = len(merged)

            if n_observations < min_observations:
                results.append({
                    "Block": block_name,
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "Observations": n_observations,
                    "Cointegration Statistic": np.nan,
                    "p-value": np.nan,
                    "Critical Value 1%": np.nan,
                    "Critical Value 5%": np.nan,
                    "Critical Value 10%": np.nan,
                    "Cointegrated": np.nan,
                    "Status": "Insufficient observations"
                })
                continue

            try:
                score, pvalue, critical_values = coint(
                    merged[var1],
                    merged[var2],
                    trend="c",
                    autolag="aic"
                )

                cointegrated = (
                    pvalue < significance_level
                )

                results.append({
                    "Block": block_name,
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "Observations": n_observations,
                    "Cointegration Statistic": score,
                    "p-value": pvalue,
                    "Critical Value 1%": critical_values[0],
                    "Critical Value 5%": critical_values[1],
                    "Critical Value 10%": critical_values[2],
                    "Cointegrated": cointegrated,
                    "Status": (
                        "Cointegrated"
                        if cointegrated
                        else "Not cointegrated"
                    )
                })

            except Exception as error:
                results.append({
                    "Block": block_name,
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "Observations": n_observations,
                    "Cointegration Statistic": np.nan,
                    "p-value": np.nan,
                    "Critical Value 1%": np.nan,
                    "Critical Value 5%": np.nan,
                    "Critical Value 10%": np.nan,
                    "Cointegrated": np.nan,
                    "Status": f"Test failed: {error}"
                })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by="p-value",
            na_position="last"
        ).reset_index(drop=True)

    if save_excel:

        if excel_name is None:
            safe_block_name = (
                block_name
                .replace(" ", "_")
                .replace("/", "_")
            )

            excel_name = (
                f"{safe_block_name}_Cointegration_Report"
            )

        results_df.to_excel(
            f"{excel_name}.xlsx",
            index=False,
            engine="openpyxl"
        )

    return results_df

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
