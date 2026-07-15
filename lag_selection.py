from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd


def select_optimal_lag(
        data: pd.DataFrame,
        maxlags: int = 15,
        deterministic: str = "c"
    ) -> pd.DataFrame:
    """
    Sélectionne le nombre optimal de retards d'un modèle VAR/VECM.

    Parameters
    ----------
    data : pd.DataFrame
        Séries temporelles utilisées dans le système.

    maxlags : int, default=15
        Nombre maximal de retards testés.

    deterministic : str, default="c"
        Composantes déterministes :
        - "n"  : aucune constante ni tendance
        - "c"  : constante
        - "ct" : constante + tendance
        - "ctt": constante + tendance quadratique

    Returns
    -------
    pd.DataFrame
        Tableau contenant le retard optimal selon :
        - AIC
        - BIC
        - HQIC
        - FPE
    """

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
        "Optimal Lag": [
            lag_selection.aic,
            lag_selection.bic,
            lag_selection.hqic,
            lag_selection.fpe
        ]
    })

    return results
