import pandas as pd
from statmodels.tsa.vector_ar.vecm import(VECM,select_order,select_coint_rank)


def granger_vecm(
        results,
        significance_level=0.05):
 
    gc = []

    for caused in results.names:

        for causing in results.names:

            if caused == causing:
                    continue

            test = results.test_granger_causality(
                caused=caused,
                causing=causing
            )

            gc.append({

                "Cause": causing,

                "Effect": caused,

                "Statistic": test.test_statistic,

                "P-value": test.pvalue,

                "Reject H0": test.pvalue < 0.05

            })

    granger_df = pd.DataFrame(gc)

    granger_df.sort_values("P-value")

def display_vecm_granger(
        results,
        significance_level=0.05):

    granger = granger_vecm(
        results,
        significance_level
    )

    significant = granger[
        granger["Granger Causality"]
    ]

    print()
    print("=" * 100)
    print("VECM GRANGER CAUSALITY")
    print("=" * 100)

    display(significant)

    return significant


def export_vecm_granger(
        results,
        excel_name="VECM_Granger"):

    granger = granger_vecm(
        results
    )

    granger.to_excel(
        f"{excel_name}.xlsx",
        index=False
    )

    return results


def vecm_granger_matrix(
        results):

    granger = granger_vecm(
        results
    )

    matrix = granger.pivot(
        index="Cause",
        columns="Effect",
        values="P-value"
    )

    return matrix
