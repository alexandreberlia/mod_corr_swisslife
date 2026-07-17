from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd


def estimate_var(
        data,
        lag,
        block_name="Unnamed Block",
        deterministic="c",
        exog=dummy
):

    data = data.dropna()

    model = VAR(data)

    results = model.fit(
        maxlags=lag,
        trend=deterministic
    )

    print()
    print("=" * 100)
    print(f"VAR ESTIMATION : {block_name}")
    print("=" * 100)
    print()

    print(f"Nombre de variables : {results.neqs}")
    print(f"Nombre d'observations : {results.nobs}")
    print(f"Lag retenu : {results.k_ar}")

    return results


def var_coefficients(results):
    return results.params


def display_var_equations(results):

    print()
    print("=" * 100)
    print("VAR EQUATIONS")
    print("=" * 100)

    for variable in results.names:

        print()
        print(f"Variable dépendante : {variable}")
        print("-" * 100)

        print(results.params[variable])


def export_var_coefficients(
        results,
        excel_name="VAR_Coefficients"
):

    coeffs = results.params

    coeffs.to_excel(
        f"{excel_name}.xlsx"
    )

    return coeffs


def estimate_all_blocks(dict_of_df):
    models = {}

    growth_block = build_growth_block(dict_of_df)

    models["Growth"] = estimate_var(
        growth_block,
        lag=15,
        block_name="Growth Block"
    )

    inflation_block = build_inflation_block(dict_of_df)

    models["Inflation"] = estimate_var(
        inflation_block,
        lag=6,
        block_name="Inflation Block"
    )

    employment_block = build_employment_block(dict_of_df)

    models["Employment"] = estimate_var(
        employment_block,
        lag=10,
        block_name="Employment Block"
    )

    macro_policy_block = build_macro_policy_block(dict_of_df)

    models["Macro Policy"] = estimate_var(
        macro_policy_block,
        lag=15,
        block_name="Macro Policy Block"
    )

    macro_core_block = build_macro_core_block(dict_of_df)

    models["Macro Core"] = estimate_var(
        macro_core_block,
        lag=15,
        block_name="Macro Core Block"
    )

    return models
