from statsmodels.tsa.vector_ar.vecm import VECM
import pandas as pd


def estimate_vecm(
        data,
        k_ar_diff,
        coint_rank,
        deterministic="ci",
        exog=dummy,
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

    return beta

def display_error_correction_terms(results):

    print()
    print("=" * 100)
    print("ERROR CORRECTION TERMS")
    print("=" * 100)

    alpha = pd.DataFrame(
        results.alpha,
        index=results.names
    )

    return alpha

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
        
    n = len(results.names)

    all_gamma=[]
    for i in range(results.k_ar - 1):

            start = i * n
            end = (i + 1) * n

            gamma_i = pd.DataFrame(
                results.gamma[:, start:end],
                index=results.names,
                columns=results.names)
            gamma_i["Gamma"]=f"Gamma_{i+1}"
            all_gamma.append(gamma_i)
    gamma_matrix=pd.concat(all_gamma)

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
        gamma_matrix.to_excel(
                writer,
                sheet_name="Coefficient de court terme"
        )

    print(
        f"{excel_name}.xlsx successfully saved."
    )

import pandas as pd


def display_gamma_matrices(results):

    n = len(results.names)

    all_gamma=[]
    for i in range(results.k_ar - 1):

            start = i * n
            end = (i + 1) * n

            gamma_i = pd.DataFrame(
                results.gamma[:, start:end],
                index=results.names,
                columns=results.names)
            gamma_i["Gamma"]=f"Gamma_{i+1}"
            all_gamma.append(gamma_i)
    gamma_matrix=pd.concat(all_gamma)
    return gamma_matrix
