import pandas as pd
from scipy.stats import chi2


def granger_vecm(
        vecm_results,
        significance_level=0.05):

    variables = vecm_results.names

    gamma = vecm_results.gamma

    n = len(variables)

    results = []

    for causing_idx, causing in enumerate(variables):

        for caused_idx, caused in enumerate(variables):

            if causing == caused:
                continue

            statistic = 0

            for lag in range(vecm_results.k_ar - 1):

                column_index = causing_idx + lag * n

                coefficient = gamma[
                    caused_idx,
                    column_index
                ]

                statistic += coefficient ** 2

            pvalue = 1 - chi2.cdf(
                statistic,
                df=vecm_results.k_ar - 1
            )

            results.append({

                "Cause": causing,

                "Effect": caused,

                "Statistic": statistic,

                "P-value": pvalue,

                "Granger Causality":
                    pvalue < significance_level

            })

    return pd.DataFrame(results)


def display_vecm_granger(
        vecm_results,
        significance_level=0.05):

    results = granger_vecm(
        vecm_results,
        significance_level
    )

    significant = results[
        results["Granger Causality"]
    ]

    print()
    print("=" * 100)
    print("VECM GRANGER CAUSALITY")
    print("=" * 100)

    display(significant)

    return significant


def export_vecm_granger(
        vecm_results,
        excel_name="VECM_Granger"):

    results = granger_vecm(
        vecm_results
    )

    results.to_excel(
        f"{excel_name}.xlsx",
        index=False
    )

    return results


def vecm_granger_matrix(
        vecm_results):

    results = granger_vecm(
        vecm_results
    )

    matrix = results.pivot(
        index="Cause",
        columns="Effect",
        values="P-value"
    )

    return matrix
