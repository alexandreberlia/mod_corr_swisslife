from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd


def select_optimal_lag(
        data: pd.DataFrame,
        block_name: str = "Unnamed",
        maxlags: int = 15,
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
        "Criterion": [
            "AIC",
            "BIC",
            "HQIC",
            "FPE"
        ],
        f"Optimal Lag for {block_name}": [
            lag_selection.aic,
            lag_selection.bic,
            lag_selection.hqic,
            lag_selection.fpe
        ]
    })

    return results
