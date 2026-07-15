from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd
from economic_blocks import*

def select_optimal_lag(
        data: pd.DataFrame,
        block_name: str = "Unnamed",
        maxlags: int = 30,
        deterministic: str = "c"
    ) -> pd.DataFrame:

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    data = data.dropna()

    if len(data) <= maxlags:
        raise ValueError(
            f"Insufficient observations ({len(data)}) "
            f"for maxlags={maxlags}."
        )

    model = VAR(data)

    lag_selection = model.select_order(
        maxlags=maxlags,
        trend=deterministic
    )

    results = pd.DataFrame({
        f"Optimal Lag for {block_name}": [
            lag_selection.aic,
            lag_selection.bic,
            lag_selection.hqic,
            lag_selection.fpe
        ]
    })

    return results

criterion=pd.DataFrame({"Criterion": [
            "AIC",
            "BIC",
            "HQIC",
            "FPE"
        ]})

def optimal_lag(dict_of_df):
        optimal_lag.df=pd.concat([criterion,select_optimal_lag(build_growth_block(dict_of_df), block_name="Growth Block"),
                                 select_optimal_lag(build_inflation_block(dict_of_df), block_name="Inflation Block"),
                                 select_optimal_lag(build_employment_block(dict_of_df), block_name="Employment Block"),
                                 select_optimal_lag(build_macro_policy_block(dict_of_df), block_name="Macro Policy Block"),
                                 select_optimal_lag(build_macro_core_block(dict_of_df), block_name="Macro Core Block")],axis=1)
        return optimal_lag.df
