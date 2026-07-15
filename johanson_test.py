import pandas as pd

from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_test(
        data,
        det_order=0,
        k_ar_diff=1,
        block_name="Unnamed Block"):

    data = data.dropna()

    results = coint_johansen(
        data,
        det_order=det_order,
        k_ar_diff=k_ar_diff
    )

    print()
    print("=" * 100)
    print(f"JOHANSEN TEST : {block_name}")
    print("=" * 100)
    print()

    return results


def johansen_trace_table(results):

    table = pd.DataFrame({
        "Rank": range(len(results.lr1)),
        "Trace Statistic": results.lr1,
        "Critical Value 90%": results.cvt[:, 0],
        "Critical Value 95%": results.cvt[:, 1],
        "Critical Value 99%": results.cvt[:, 2]
    })

    return table

def estimate_cointegration_rank(
        results,
        alpha=0.05):
    """
    Determine Johansen rank.
    """

    rank = 0

    for i in range(len(results.lr1)):

        statistic = results.lr1[i]

        if alpha == 0.10:
            critical = results.cvt[i, 0]

        elif alpha == 0.05:
            critical = results.cvt[i, 1]

        elif alpha == 0.01:
            critical = results.cvt[i, 2]

        else:
            raise ValueError(
                "alpha must be 0.10, 0.05 or 0.01"
            )

        if statistic > critical:
            rank += 1

    return rank

def display_cointegration_vectors(
        results,
        variable_names):
    """
    Display beta vectors.
    """

    beta = pd.DataFrame(
        results.evec,
        index=variable_names
    )

    print()
    print("=" * 100)
    print("COINTEGRATION VECTORS")
    print("=" * 100)
    print()

    print(beta)

    return beta

def johansen_summary(
        data,
        k_ar_diff,
        block_name="Unnamed Block",
        alpha=0.05):
    
    test = johansen_test(
        data=data,
        k_ar_diff=k_ar_diff,
        block_name=block_name
    )

    table = johansen_trace_table(test)

    rank = estimate_cointegration_rank(
        test,
        alpha=alpha
    )

    print()
    print(table)

    print()
    print(f"Estimated Cointegration Rank = {rank}")

    display_cointegration_vectors(
        test,
        data.columns
    )

    return {
        "test": test,
        "table": table,
        "rank": rank
    }
