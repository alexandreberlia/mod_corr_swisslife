import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. OUTIL DE MISE EN FORME DES TITRES
# ============================================================



    
def format_label(label, max_length=35):
    """
    Coupe les titres longs sur plusieurs lignes.
    """
    return "\n".join(
        textwrap.wrap(
            str(label),
            width=max_length
        )
    )



# ============================================================
# 2. CALCUL DES IRF
# ============================================================

def compute_irf_var(var_results, periods=24):
    """
    Calcule l'objet d'analyse des réponses impulsionnelles.

    Parameters
    ----------
    var_results :
        Résultat ajusté d'un modèle VAR statsmodels.

    periods : int
        Nombre d'horizons après le choc.

    Returns
    -------
    IRAnalysis
        Objet contenant notamment :
        - irfs : IRF non orthogonalisées ;
        - orth_irfs : IRF orthogonalisées par Cholesky.
    """

    return var_results.irf(periods)


# ============================================================
# 3. TRACER TOUTES LES IRF ORTHOGONALISÉES
# ============================================================

def plot_irf_var(
    var_results,
    periods=24,
    figsize_per_subplot=(4.5, 3.3),
    max_label_length=30,
    show_grid=True
):
    """
    Trace toutes les IRF orthogonalisées du VAR.

    Organisation du graphique
    --------------------------
    - Colonnes : chocs orthogonaux ;
    - Lignes : variables répondantes ;
    - Chaque cellule : réponse d'une variable à un choc.

    Attention
    ---------
    Avec n variables, la fonction crée n x n graphiques.
    """

    irf_analysis = var_results.irf(periods)

    # Dimension :
    # horizon x variable réponse x variable choc
    orth_irfs = irf_analysis.orth_irfs

    variables = list(var_results.names)
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

            # Titres des colonnes
            if response_idx == 0:
                ax.set_title(
                    "Choc :\n"
                    + format_label(
                        impulse,
                        max_label_length
                    ),
                    fontsize=9,
                    pad=10
                )

            # Titres des lignes
            if impulse_idx == 0:
                ax.set_ylabel(
                    "Réponse :\n"
                    + format_label(
                        response,
                        max_label_length
                    ),
                    fontsize=9,
                    labelpad=10
                )

            # Label de l'axe horizontal uniquement
            # sur la dernière ligne
            if response_idx == n - 1:
                ax.set_xlabel(
                    "Horizon",
                    fontsize=9
                )

            ax.tick_params(
                axis="both",
                labelsize=8
            )

            if show_grid:
                ax.grid(
                    alpha=0.25
                )

    fig.suptitle(
        "Réponses impulsionnelles orthogonalisées du VAR",
        fontsize=15,
        y=1.01
    )

    fig.tight_layout()

    plt.show()

    return irf_analysis


# ============================================================
# 4. TRACER UNE SEULE IRF
# ============================================================

def plot_single_irf_var(
    var_results,
    impulse,
    response,
    periods=24,
    figsize=(11, 5)
):
    """
    Trace la réponse d'une seule variable à un seul choc.

    Parameters
    ----------
    impulse : str
        Nom exact de la variable à l'origine du choc.

    response : str
        Nom exact de la variable qui répond au choc.
    """

    irf_analysis = var_results.irf(periods)
    orth_irfs = irf_analysis.orth_irfs

    variables = list(var_results.names)

    if impulse not in variables:
        raise ValueError(
            f"Le choc {impulse!r} n'existe pas.\n"
            f"Variables disponibles :\n{variables}"
        )

    if response not in variables:
        raise ValueError(
            f"La réponse {response!r} n'existe pas.\n"
            f"Variables disponibles :\n{variables}"
        )

    impulse_idx = variables.index(impulse)
    response_idx = variables.index(response)

    values = orth_irfs[
        :,
        response_idx,
        impulse_idx
    ]

    horizons = np.arange(periods + 1)

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.plot(
        horizons,
        values,
        color="blue",
        linewidth=2,
        label="IRF orthogonalisée"
    )

    ax.axhline(
        y=0,
        color="black",
        linewidth=1
    )

    ax.set_title(
        f"Réponse de :\n{format_label(response, 70)}\n\n"
        f"à un choc de :\n{format_label(impulse, 70)}",
        fontsize=12
    )

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Réponse")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    plt.show()

    return pd.Series(
        values,
        index=horizons,
        name=f"{response} <- choc {impulse}"
    )


# ============================================================
# 5. EFFET D'UN CHOC SUR TOUTES LES VARIABLES
# ============================================================

def plot_impulse_on_all_responses(
    var_results,
    impulse,
    periods=24,
    ncols=3,
    subplot_size=(5, 3.5),
    max_label_length=40
):
    """
    Trace l'effet d'un choc orthogonal donné sur toutes
    les variables du VAR.
    """

    irf_analysis = var_results.irf(periods)
    orth_irfs = irf_analysis.orth_irfs

    variables = list(var_results.names)

    if impulse not in variables:
        raise ValueError(
            f"Le choc {impulse!r} n'existe pas.\n"
            f"Variables disponibles :\n{variables}"
        )

    impulse_idx = variables.index(impulse)

    n = len(variables)
    nrows = int(np.ceil(n / ncols))

    horizons = np.arange(periods + 1)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            subplot_size[0] * ncols,
            subplot_size[1] * nrows
        ),
        squeeze=False
    )

    axes = axes.flatten()

    for response_idx, response in enumerate(variables):

        ax = axes[response_idx]

        values = orth_irfs[
            :,
            response_idx,
            impulse_idx
        ]

        ax.plot(
            horizons,
            values,
            color="blue",
            linewidth=1.7
        )

        ax.axhline(
            y=0,
            color="black",
            linewidth=0.8
        )

        ax.set_title(
            "Réponse de :\n"
            + format_label(
                response,
                max_label_length
            ),
            fontsize=9
        )

        ax.set_xlabel("Horizon")
        ax.set_ylabel("Réponse")
        ax.grid(alpha=0.25)

    # Supprimer les graphiques inutilisés
    for index in range(n, len(axes)):
        fig.delaxes(axes[index])

    fig.suptitle(
        "Réponses à un choc orthogonal de :\n"
        + format_label(
            impulse,
            max_label_length
        ),
        fontsize=14,
        y=1.01
    )

    fig.tight_layout()

    plt.show()

    return irf_analysis


# ============================================================
# 6. RÉPONSE D'UNE VARIABLE À TOUS LES CHOCS
# ============================================================

def plot_all_impulses_on_response(
    var_results,
    response,
    periods=24,
    ncols=3,
    subplot_size=(5, 3.5),
    max_label_length=40
):
    """
    Trace la réponse d'une variable donnée à tous les
    chocs orthogonaux du VAR.
    """

    irf_analysis = var_results.irf(periods)
    orth_irfs = irf_analysis.orth_irfs

    variables = list(var_results.names)

    if response not in variables:
        raise ValueError(
            f"La réponse {response!r} n'existe pas.\n"
            f"Variables disponibles :\n{variables}"
        )

    response_idx = variables.index(response)

    n = len(variables)
    nrows = int(np.ceil(n / ncols))

    horizons = np.arange(periods + 1)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            subplot_size[0] * ncols,
            subplot_size[1] * nrows
        ),
        squeeze=False
    )

    axes = axes.flatten()

    for impulse_idx, impulse in enumerate(variables):

        ax = axes[impulse_idx]

        values = orth_irfs[
            :,
            response_idx,
            impulse_idx
        ]

        ax.plot(
            horizons,
            values,
            color="blue",
            linewidth=1.7
        )

        ax.axhline(
            y=0,
            color="black",
            linewidth=0.8
        )

        ax.set_title(
            "Choc de :\n"
            + format_label(
                impulse,
                max_label_length
            ),
            fontsize=9
        )

        ax.set_xlabel("Horizon")
        ax.set_ylabel("Réponse")
        ax.grid(alpha=0.25)

    # Supprimer les graphiques inutilisés
    for index in range(n, len(axes)):
        fig.delaxes(axes[index])

    fig.suptitle(
        "Réponse de :\n"
        + format_label(
            response,
            max_label_length
        )
        + "\nà tous les chocs orthogonaux",
        fontsize=14,
        y=1.02
    )

    fig.tight_layout()

    plt.show()

    return irf_analysis


# ============================================================
# 7. CONVERTIR LES IRF EN DATAFRAME LONGUE
# ============================================================

def irf_dataframe_var(
    var_results,
    periods=24
):
    """
    Convertit toutes les IRF orthogonalisées en DataFrame.

    Colonnes produites
    ------------------
    Impulse :
        Variable à l'origine du choc.

    Response :
        Variable qui répond.

    Horizon :
        Horizon après le choc.

    Value :
        Valeur de l'IRF orthogonalisée.
    """

    irf_analysis = var_results.irf(periods)
    orth_irfs = irf_analysis.orth_irfs

    variables = list(var_results.names)

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


# ============================================================
# 8. DATAFRAME D'UNE SEULE IRF
# ============================================================

def single_irf_dataframe_var(
    var_results,
    impulse,
    response,
    periods=24
):
    """
    Extrait dans une DataFrame une seule combinaison
    choc-réponse.
    """

    df_irf = irf_dataframe_var(
        var_results=var_results,
        periods=periods
    )

    selection = df_irf[
        (df_irf["Impulse"] == impulse)
        & (df_irf["Response"] == response)
    ].copy()

    if selection.empty:
        raise ValueError(
            "Aucune IRF trouvée pour cette combinaison.\n"
            f"Choc demandé : {impulse}\n"
            f"Réponse demandée : {response}"
        )

    return selection.reset_index(drop=True)


# ============================================================
# 9. EXPORT EXCEL
# ============================================================

def export_irf_var(
    var_results,
    periods=24,
    excel_name="IRF_Results.xlsx"
):
    """
    Exporte toutes les IRF orthogonalisées dans Excel.
    """

    df_irf = irf_dataframe_var(
        var_results=var_results,
        periods=periods
    )

    if not excel_name.lower().endswith(".xlsx"):
        excel_name = f"{excel_name}.xlsx"

    df_irf.to_excel(
        excel_name,
        index=False
    )

    print(
        f"Les IRF ont été exportées dans : {excel_name}"
    )

    return df_irf


# ============================================================
# 10. AFFICHER LES NOMS EXACTS DES VARIABLES
# ============================================================

def afficher_variables_var(var_results):
    """
    Affiche les noms et les positions des variables du VAR.
    """

    variables = list(var_results.names)

    for indice, variable in enumerate(variables):
        print(f"{indice} : {variable}")

    return variables

