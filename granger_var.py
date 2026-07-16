import pandas as pd


def granger_var(
        model,
        significance_level=0.05):

    variables = model.names

    results = []

    for caused in variables:

        for causing in variables:

            if caused == causing:
                continue

            test = model.test_causality(
                caused,
                [causing],
                kind="f"
            )

            results.append({

                "Cause": causing,

                "Effect": caused,

                "F Statistic": test.test_statistic,

                "P-value": test.pvalue,

                "Reject H0": (
                    test.pvalue < significance_level
                )

            })

    return pd.DataFrame(results)


def display_granger_var(
        model,
        significance_level=0.05):
    """
    Display significant Granger causalities.
    """

    results = granger_var(
        model,
        significance_level
    )

    significant = results[
        results["Reject H0"]
    ]

    print()
    print("=" * 100)
    print("GRANGER CAUSALITY")
    print("=" * 100)
    print()

    if len(significant) == 0:

        print(
            "No significant Granger causality detected."
        )

    else:

        print(significant)

    return significant

def export_granger_results(
        model,
        excel_name="Granger_Causality"):
    
    results = granger_var(
        model
    )

    results.to_excel(
        f"{excel_name}.xlsx",
        index=False
    )

    return results

def granger_heatmap_table(
        model,
        significance_level=0.05):
    """
    Granger causality matrix.
    """

    variables = model.names

    matrix = pd.DataFrame(
        index=variables,
        columns=variables
    )

    for caused in variables:

        for causing in variables:

            if caused == causing:

                matrix.loc[
                    causing,
                    caused
                ] = "-"

            else:

                test = model.test_causality(
                    caused,
                    [causing],
                    kind="f"
                )

                matrix.loc[
                    causing,
                    caused
                ] = round(
                    test.pvalue,
                    4
                )

    return matrix


