import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# CALCUL DES IRF POUR VECM
# ============================================================

from scipy.linalg import cholesky
import numpy as np


def compute_irf_vecm(vecm_results, periods=24):
    """
    Calcule les IRF d'un VECM.

    Returns
    -------
    dict contenant :

    irfs :
        Réponses impulsionnelles standards.

    orth_irfs :
        Réponses impulsionnelles orthogonalisées
        (décomposition de Cholesky).

    names :
        Noms des variables.
    """

    # représentation MA du VECM
    irfs = vecm_results.ma_rep(maxn=periods)

    # covariance des résidus
    sigma_u = vecm_results.sigma_u

    # Cholesky inférieur
    P = cholesky(sigma_u, lower=True)

    orth_irfs = np.zeros_like(irfs)

    for h in range(irfs.shape[0]):
        orth_irfs[h] = irfs[h] @ P

    return {
        "irfs": irfs,
        "orth_irfs": orth_irfs,
        "names": list(vecm_results.names)
    }

# ============================================================
# TRACER TOUTES LES IRF DU VECM
# ============================================================

def plot_irf_vecm(
    vecm_results,
    periods=24,
    figsize_per_subplot=(4.5, 3.3),
    max_label_length=30,
    show_grid=True
):

    irf_analysis = compute_irf_vecm(
        vecm_results,
        periods=periods
    )

    orth_irfs = irf_analysis["orth_irfs"]

    variables = irf_analysis["names"]

    n = len(variables)

    horizons = np.arange(periods + 1)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=n,
        figsize=(
            figsize_per_subplot[0] * n,
            figsize_per_subplot[1] * n
        ),
        squeeze=False,
        sharex=True
    )

    for response_idx, response in enumerate(variables):

        for impulse_idx, impulse in enumerate(variables):

            ax = axes[response_idx, impulse_idx]

            values = orth_irfs[
                :,
                response_idx,
                impulse_idx
            ]

            ax.plot(
                horizons,
                values,
                color="blue",
                linewidth=1.6
            )

            ax.axhline(
                y=0,
                color="black",
                linewidth=0.8
            )

            if response_idx == 0:
                ax.set_title(
                    "Choc :\n" +
                    format_label(
                        impulse,
                        max_label_length
                    ),
                    fontsize=9
                )

            if impulse_idx == 0:
                ax.set_ylabel(
                    "Réponse :\n" +
                    format_label(
                        response,
                        max_label_length
                    ),
                    fontsize=9
                )

            if response_idx == n - 1:
                ax.set_xlabel("Horizon")

            if show_grid:
                ax.grid(alpha=0.25)

    fig.suptitle(
        "Réponses impulsionnelles orthogonalisées du VECM",
        fontsize=15,
        y=1.01
    )

    fig.tight_layout()

    plt.show()

    return irf_analysis


def plot_single_irf_vecm(
    vecm_results,
    impulse,
    response,
    periods=24,
    figsize=(10, 5)
):

    irf_analysis = compute_irf_vecm(
        vecm_results,
        periods
    )

    orth_irfs = irf_analysis["orth_irfs"]

    variables = irf_analysis["names"]

    impulse_idx = variables.index(impulse)
    response_idx = variables.index(response)

    values = orth_irfs[
        :,
        response_idx,
        impulse_idx
    ]

    horizons = np.arange(periods + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        horizons,
        values,
        color="blue",
        linewidth=2
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_title(
        f"Réponse de {response}\n"
        f"à un choc de {impulse}"
    )

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Réponse")

    ax.grid(alpha=0.25)

    plt.show()

    return pd.Series(
        values,
        index=horizons,
        name=f"{response} <- {impulse}"
    )


def irf_dataframe_vecm(
    vecm_results,
    periods=24
):

    irf_analysis = compute_irf_vecm(
        vecm_results,
        periods
    )

    orth_irfs = irf_analysis["orth_irfs"]

    variables = irf_analysis["names"]

    results = []

    for impulse_idx, impulse in enumerate(variables):

        for response_idx, response in enumerate(variables):

            for horizon in range(periods + 1):

                results.append({
                    "Impulse": impulse,
                    "Response": response,
                    "Horizon": horizon,
                    "Value": orth_irfs[
                        horizon,
                        response_idx,
                        impulse_idx
                    ]
                })

    return pd.DataFrame(results)


def export_irf_vecm(
        vecm_results,
        periods=24,
        excel_name="IRF_Results_VECM"):

    df = irf_dataframe_vecm(
        vecm_results,
        periods
    )

    df.to_excel(
        f"{excel_name}.xlsx",
        index=False
    )

    return df

