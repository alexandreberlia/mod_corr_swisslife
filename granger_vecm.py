import pandas as pd
from statsmodels.tsa.vector_ar.vecm import(VECM,select_order,select_coint_rank)


def granger_vecm(
        results,
        significance_level):
 
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

    granger_df=pd.DataFrame(gc)
    significant=granger_df.sort_values("P-value",ascending=True)
    return significant

def display_vecm_granger(
        results,
        significance_level):
    granger_vecm(
        results,
        significance_level
    )

    print()
    print("=" * 100)
    print("VECM GRANGER CAUSALITY")
    print("=" * 100)

    return significant


def export_vecm_granger(
        results,
        significance_level,
        excel_name="VECM_Granger"):

    granger_table=granger_vecm(
        results,
        significance_level
    )

    granger_table.to_excel(
        f"{excel_name}.xlsx",
        index=False
    )

    return results


def vecm_granger_matrix(
        results, significance_level):

    granger_table=granger_vecm(
        results,
        significance_level
    )

    matrix = granger_table.pivot(
        index="Cause",
        columns="Effect",
        values="P-value"
    )

    return matrix
