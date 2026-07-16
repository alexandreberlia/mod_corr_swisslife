import pandas as pd
from statmodels.tsa.vector_ar.vecm import(VECM,select_order,select_coint_rank)


def granger_vecm(
        vecm_results,
        significance_level=0.05):
 
    results = []

    for caused in consumer_vecm.names:

        for causing in consumer_vecm.names:

            if caused == causing:
                    continue

            test = consumer_vecm.test_granger_causality(
                caused=caused,
                causing=causing
            )

            results.append({

                "Cause": causing,

                "Effect": caused,

                "Statistic": test.test_statistic,

                "P-value": test.pvalue,

                "Reject H0": test.pvalue < 0.05

            })

    granger_df = pd.DataFrame(results)

    granger_df.sort_values("P-value")

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
