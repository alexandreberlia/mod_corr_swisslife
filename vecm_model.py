from statsmodels.tsa.vector_ar.vecm import VECM
import pandas as pd


def estimate_vecm(
        data,
        k_ar_diff,
        coint_rank=1,
        deterministic="ci",
        block_name="Unnamed Block"
):

    data = data.dropna()

    model = VECM(
        data,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic
    )

    results = model.fit()

    print()
    print("=" * 100)
    print(f"VECM ESTIMATION : {block_name}")
    print("=" * 100)
    print()

    print(f"Nombre de variables : {data.shape[1]}")
    print(f"Nombre d'observations : {len(data)}")
    print(f"k_ar_diff : {k_ar_diff}")
    print(f"Cointegration Rank : {coint_rank}")

    return results

def display_short_run_coefficients(results):

    print()
    print("=" * 100)
    print("SHORT RUN COEFFICIENTS")
    print("=" * 100)

    print(
        pd.DataFrame(
            results.gamma.reshape(
                results.gamma.shape[0],
                -1
            )
        )
    )

def display_cointegration_relation(results):

    print()
    print("=" * 100)
    print("COINTEGRATION RELATION")
    print("=" * 100)

    beta = pd.DataFrame(
        results.beta,
        index=results.names
    )

    print(beta)

def display_error_correction_terms(results):

    print()
    print("=" * 100)
    print("ERROR CORRECTION TERMS")
    print("=" * 100)

    alpha = pd.DataFrame(
        results.alpha,
        index=results.names
    )

    print(alpha)

def export_vecm_results(
        results,
        excel_name="VECM_Results"
):

    beta = pd.DataFrame(
        results.beta,
        index=results.names
    )

    alpha = pd.DataFrame(
        results.alpha,
        index=results.names
    )

    with pd.ExcelWriter(
            f"{excel_name}.xlsx"
    ) as writer:

        beta.to_excel(
            writer,
            sheet_name="Cointegration"
        )

        alpha.to_excel(
            writer,
            sheet_name="Adjustment"
        )

    print(
        f"{excel_name}.xlsx successfully saved."
    )

def estimate_all_vecm_blocks(dict_of_df):

    models = {}

    consumer_block = build_consumer_block(
        dict_of_df
    )

    models["Consumer"] = estimate_vecm(
        consumer_block,
        k_ar_diff=5,
        coint_rank=1,
        block_name="Consumer Block"
    )

    return models
