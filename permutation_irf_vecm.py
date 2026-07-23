from itertools import permutations
from math import factorial
from vecm_model import estimate_vecm
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from irf_vecm import compute_irf_vecm
# ============================================================
# ESTIMER UN VECM POUR PLUSIEURS ORDRES DE CHOLESKY
# ============================================================
def format_label(label, max_length=30):
    """
    Coupe les intitulés longs sur plusieurs lignes.
    """
    return fill(
        str(label),
        width=max_length
    )
def estimate_vecm_permutations(
    data,
    k_ar_diff,
    coint_rank,
    deterministic="ci",
    exog=None,
    variables=None,
    orders=None,
    verbose=True
):
    """
    Estime un VECM pour plusieurs ordres de colonnes.

    Parameters
    ----------
    data : pandas.DataFrame
        Données en niveau. Les dates peuvent être placées
        dans l'index.

    k_ar_diff : int
        Nombre de retards des différences.

    coint_rank : int
        Rang de cointégration.

    deterministic : str
        Composantes déterministes du VECM.

    exog : array-like ou DataFrame, optionnel
        Variables exogènes communes aux estimations.

    variables : list[str], optionnel
        Variables endogènes à utiliser.
        Si None, toutes les colonnes de data sont utilisées.

    orders : list[list[str]], optionnel
        Ordres précis à tester.

        Si None, toutes les permutations possibles
        des variables sont estimées.

    max_permutations : int
        Nombre maximal de permutations autorisées.

    verbose : bool
        Affiche la progression.

    Returns
    -------
    dict
        Contient :
        - models : résultats VECM par ordre ;
        - orders : ordres estimés ;
        - failed_orders : ordres ayant échoué ;
        - reference_order : ordre de référence.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data doit être un pandas.DataFrame."
        )

    if variables is None:
        variables = list(data.columns)
    else:
        variables = list(variables)

    missing_variables = [
        variable
        for variable in variables
        if variable not in data.columns
    ]

    if missing_variables:
        raise ValueError(
            "Variables absentes du DataFrame :\n"
            f"{missing_variables}"
        )

    if len(variables) != len(set(variables)):
        raise ValueError(
            "La liste variables contient des doublons."
        )

    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna()

    if data.empty:
        raise ValueError(
            "Aucune observation disponible après "
            "suppression des valeurs manquantes."
        )

    # Construction des ordres à tester
    if orders is None:

        number_of_orders = factorial(
            len(variables)
        )
        orders_to_estimate = [
            list(order)
            for order in permutations(variables)
        ]

    else:

        orders_to_estimate = [
            list(order)
            for order in orders
        ]

        for order in orders_to_estimate:

            if len(order) != len(set(order)):
                raise ValueError(
                    f"L'ordre suivant contient des doublons :\n"
                    f"{order}"
                )

            if set(order) != set(variables):
                raise ValueError(
                    "Chaque ordre doit contenir exactement "
                    "les mêmes variables que variables.\n\n"
                    f"Ordre incorrect :\n{order}"
                )

    models = {}
    estimated_orders = {}
    failed_orders = {}

    total = len(orders_to_estimate)

    for order_number, current_order in enumerate(
        orders_to_estimate,
        start=1
    ):

        order_name = " -> ".join(
            current_order
        )

        if verbose:
            print(
                f"[{order_number}/{total}] "
                f"Ordre : {order_name}"
            )

        permuted_data = data.loc[
            :,
            current_order
        ].copy()

        try:

            results = estimate_vecm(
                data=permuted_data,
                k_ar_diff=k_ar_diff,
                coint_rank=coint_rank,
                deterministic=deterministic,
                exog=exog,
                block_name=order_name
            )

            models[order_name] = results
            estimated_orders[order_name] = current_order

        except Exception as error:

            failed_orders[order_name] = str(error)

            if verbose:
                print(
                    f"Échec pour l'ordre : {order_name}"
                )
                print(
                    f"Erreur : {error}"
                )

    if len(models) == 0:
        raise RuntimeError(
            "Aucun VECM n'a pu être estimé."
        )

    return {
        "models": models,
        "orders": estimated_orders,
        "failed_orders": failed_orders,
        "reference_order": variables,
        "data": data
    }

# ============================================================
# CALCULER ET RÉORDONNER LES IRF DE TOUTES LES PERMUTATIONS
# ============================================================

def compute_irf_from_permutations(
    permutation_results,
    periods=24,
    reference_order=None
):
    """
    Calcule les IRF orthogonalisées de tous les VECM estimés.

    Les réponses et les chocs sont ensuite remis dans un ordre
    commun afin de permettre leur comparaison.

    Parameters
    ----------
    permutation_results : dict
        Résultat de estimate_vecm_permutations.

    periods : int
        Nombre d'horizons.

    reference_order : list[str], optionnel
        Ordre commun des variables dans les résultats.

    Returns
    -------
    dict
        irfs :
            Dictionnaire contenant une matrice
            horizon x réponse x choc pour chaque ordre.

        reference_order :
            Ordre commun des variables.
    """

    models = permutation_results["models"]

    if reference_order is None:
        reference_order = permutation_results[
            "reference_order"
        ].copy()
    else:
        reference_order = list(reference_order)

    permutation_irfs = {}

    for order_name, vecm_results in models.items():

        current_irf = compute_irf_vecm(
            vecm_results=vecm_results,
            periods=periods
        )

        current_names = current_irf["names"]
        current_values = current_irf["orth_irfs"]

        if set(current_names) != set(reference_order):
            raise ValueError(
                f"Les variables de l'ordre {order_name} "
                "ne correspondent pas à l'ordre de référence."
            )

        # Indices nécessaires pour replacer les variables
        # dans l'ordre de référence
        reorder_indices = [
            current_names.index(variable)
            for variable in reference_order
        ]

        # Réordonner les lignes, donc les réponses
        reordered_values = current_values[
            :,
            reorder_indices,
            :
        ]

        # Réordonner les colonnes, donc les chocs
        reordered_values = reordered_values[
            :,
            :,
            reorder_indices
        ]

        permutation_irfs[order_name] = (
            reordered_values
        )

    return {
        "irfs": permutation_irfs,
        "reference_order": reference_order,
        "periods": periods
    }

# ============================================================
# CROISER LES IRF ET CONSTRUIRE UNE IRF GLOBALE
# ============================================================

def aggregate_irf_permutations(
    permutation_irf_results,
    lower_quantile=0.10,
    upper_quantile=0.90,
    aggregation="median"
):
    """
    Agrège les IRF obtenues avec les différents ordres
    de Cholesky.

    Parameters
    ----------
    permutation_irf_results : dict
        Résultat de compute_irf_from_permutations.

    lower_quantile : float
        Quantile inférieur de l'enveloppe.

    upper_quantile : float
        Quantile supérieur de l'enveloppe.

    aggregation : {"median", "mean"}
        Méthode utilisée pour l'IRF globale.

    Returns
    -------
    dict
        global_irf :
            IRF globale agrégée.

        median_irf :
            Médiane des ordres.

        mean_irf :
            Moyenne des ordres.

        lower_irf, upper_irf :
            Enveloppe de sensibilité à l'ordre.

        all_irfs :
            Tableau contenant toutes les IRF.
    """

    if aggregation not in {
        "median",
        "mean"
    }:
        raise ValueError(
            "aggregation doit être 'median' ou 'mean'."
        )

    if not (
        0 <= lower_quantile
        < upper_quantile
        <= 1
    ):
        raise ValueError(
            "Les quantiles doivent vérifier : "
            "0 <= lower < upper <= 1."
        )

    irf_dictionary = permutation_irf_results[
        "irfs"
    ]

    order_names = list(
        irf_dictionary.keys()
    )

    # Dimension :
    # ordre x horizon x réponse x choc
    all_irfs = np.stack(
        [
            irf_dictionary[order_name]
            for order_name in order_names
        ],
        axis=0
    )

    mean_irf = np.mean(
        all_irfs,
        axis=0
    )

    median_irf = np.median(
        all_irfs,
        axis=0
    )

    lower_irf = np.quantile(
        all_irfs,
        lower_quantile,
        axis=0
    )

    upper_irf = np.quantile(
        all_irfs,
        upper_quantile,
        axis=0
    )

    minimum_irf = np.min(
        all_irfs,
        axis=0
    )

    maximum_irf = np.max(
        all_irfs,
        axis=0
    )

    std_irf = np.std(
        all_irfs,
        axis=0,
        ddof=0
    )

    # Part des ordres donnant une réponse positive
    positive_share = np.mean(
        all_irfs > 0,
        axis=0
    )

    if aggregation == "median":
        global_irf = median_irf
    else:
        global_irf = mean_irf

    return {
        "global_irf": global_irf,
        "mean_irf": mean_irf,
        "median_irf": median_irf,
        "lower_irf": lower_irf,
        "upper_irf": upper_irf,
        "minimum_irf": minimum_irf,
        "maximum_irf": maximum_irf,
        "std_irf": std_irf,
        "positive_share": positive_share,
        "all_irfs": all_irfs,
        "order_names": order_names,
        "reference_order": permutation_irf_results[
            "reference_order"
        ],
        "periods": permutation_irf_results[
            "periods"
        ],
        "aggregation": aggregation,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile
    }

# ============================================================
# TRACER UNE IRF POUR TOUS LES ORDRES
# ============================================================

def plot_single_global_irf(
    aggregated_results,
    impulse,
    response,
    figsize=(12, 6),
    show_individual_orders=True
):
    """
    Trace les différentes IRF de Cholesky et l'IRF globale
    pour une combinaison choc-réponse.

    Courbes grises :
        IRF des différents ordres de Cholesky.

    Zone bleue :
        Enveloppe entre les quantiles choisis.

    Courbe bleue :
        IRF globale, médiane ou moyenne.
    """

    variables = aggregated_results[
        "reference_order"
    ]

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

    impulse_idx = variables.index(
        impulse
    )

    response_idx = variables.index(
        response
    )

    all_irfs = aggregated_results[
        "all_irfs"
    ]

    global_values = aggregated_results[
        "global_irf"
    ][
        :,
        response_idx,
        impulse_idx
    ]

    lower_values = aggregated_results[
        "lower_irf"
    ][
        :,
        response_idx,
        impulse_idx
    ]

    upper_values = aggregated_results[
        "upper_irf"
    ][
        :,
        response_idx,
        impulse_idx
    ]

    horizons = np.arange(
        aggregated_results["periods"] + 1
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    if show_individual_orders:

        for order_idx in range(
            all_irfs.shape[0]
        ):

            label = None

            if order_idx == 0:
                label = (
                    "IRF selon les ordres "
                    "de Cholesky"
                )

            ax.plot(
                horizons,
                all_irfs[
                    order_idx,
                    :,
                    response_idx,
                    impulse_idx
                ],
                color="grey",
                alpha=0.25,
                linewidth=1,
                label=label
            )

    lower_q = aggregated_results[
        "lower_quantile"
    ]

    upper_q = aggregated_results[
        "upper_quantile"
    ]

    ax.fill_between(
        horizons,
        lower_values,
        upper_values,
        color="royalblue",
        alpha=0.18,
        label=(
            "Sensibilité aux ordres "
            f"Q{lower_q:.0%}-Q{upper_q:.0%}"
        )
    )

    aggregation = aggregated_results[
        "aggregation"
    ]

    aggregation_label = (
        "Médiane"
        if aggregation == "median"
        else "Moyenne"
    )

    ax.plot(
        horizons,
        global_values,
        color="blue",
        linewidth=2.5,
        label=f"IRF globale, {aggregation_label}"
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

    return pd.DataFrame({
        "Horizon": horizons,
        "IRF_globale": global_values,
        "Borne_inferieure": lower_values,
        "Borne_superieure": upper_values
    })

# ============================================================
# TRACER TOUTES LES IRF GLOBALES
# ============================================================

def plot_all_global_irfs(
    aggregated_results,
    figsize_per_subplot=(4.8, 3.5),
    max_label_length=30,
    show_individual_orders=True,
    show_grid=True
):
    """
    Trace toutes les combinaisons choc-réponse.

    Colonnes :
        Variables à l'origine du choc.

    Lignes :
        Variables répondantes.
    """

    variables = aggregated_results[
        "reference_order"
    ]

    all_irfs = aggregated_results[
        "all_irfs"
    ]

    global_irf = aggregated_results[
        "global_irf"
    ]

    lower_irf = aggregated_results[
        "lower_irf"
    ]

    upper_irf = aggregated_results[
        "upper_irf"
    ]

    periods = aggregated_results[
        "periods"
    ]

    horizons = np.arange(
        periods + 1
    )

    n = len(variables)

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

            ax = axes[
                response_idx,
                impulse_idx
            ]

            if show_individual_orders:

                for order_idx in range(
                    all_irfs.shape[0]
                ):

                    ax.plot(
                        horizons,
                        all_irfs[
                            order_idx,
                            :,
                            response_idx,
                            impulse_idx
                        ],
                        color="grey",
                        alpha=0.16,
                        linewidth=0.8
                    )

            ax.fill_between(
                horizons,
                lower_irf[
                    :,
                    response_idx,
                    impulse_idx
                ],
                upper_irf[
                    :,
                    response_idx,
                    impulse_idx
                ],
                color="royalblue",
                alpha=0.16
            )

            ax.plot(
                horizons,
                global_irf[
                    :,
                    response_idx,
                    impulse_idx
                ],
                color="blue",
                linewidth=2
            )

            ax.axhline(
                y=0,
                color="black",
                linewidth=0.8
            )

            if response_idx == 0:

                ax.set_title(
                    "Choc :\n"
                    + format_label(
                        impulse,
                        max_label_length
                    ),
                    fontsize=9
                )

            if impulse_idx == 0:

                ax.set_ylabel(
                    "Réponse :\n"
                    + format_label(
                        response,
                        max_label_length
                    ),
                    fontsize=9
                )

            if response_idx == n - 1:
                ax.set_xlabel("Horizon")

            if show_grid:
                ax.grid(alpha=0.20)

            ax.tick_params(
                axis="both",
                labelsize=8
            )

    aggregation = aggregated_results[
        "aggregation"
    ]

    aggregation_label = (
        "médiane"
        if aggregation == "median"
        else "moyenne"
    )

    fig.suptitle(
        "IRF globales du VECM\n"
        f"Agrégation par {aggregation_label} "
        "des ordres de Cholesky",
        fontsize=15,
        y=1.01
    )

    fig.tight_layout()
    plt.show()

    return fig, axes

# ============================================================
# DATAFRAME DES IRF GLOBALES
# ============================================================

def global_irf_dataframe(
    aggregated_results
):
    """
    Convertit les IRF globales et les indicateurs de
    sensibilité à l'ordre en DataFrame longue.
    """

    variables = aggregated_results[
        "reference_order"
    ]

    global_irf = aggregated_results[
        "global_irf"
    ]

    mean_irf = aggregated_results[
        "mean_irf"
    ]

    median_irf = aggregated_results[
        "median_irf"
    ]

    lower_irf = aggregated_results[
        "lower_irf"
    ]

    upper_irf = aggregated_results[
        "upper_irf"
    ]

    minimum_irf = aggregated_results[
        "minimum_irf"
    ]

    maximum_irf = aggregated_results[
        "maximum_irf"
    ]

    std_irf = aggregated_results[
        "std_irf"
    ]

    positive_share = aggregated_results[
        "positive_share"
    ]

    periods = aggregated_results[
        "periods"
    ]

    records = []

    for impulse_idx, impulse in enumerate(variables):

        for response_idx, response in enumerate(variables):

            for horizon in range(periods + 1):

                records.append({
                    "Impulse": impulse,
                    "Response": response,
                    "Horizon": horizon,
                    "Global_IRF": global_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Mean_IRF": mean_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Median_IRF": median_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Lower_IRF": lower_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Upper_IRF": upper_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Minimum_IRF": minimum_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Maximum_IRF": maximum_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Std_Across_Orders": std_irf[
                        horizon,
                        response_idx,
                        impulse_idx
                    ],
                    "Positive_Order_Share": positive_share[
                        horizon,
                        response_idx,
                        impulse_idx
                    ]
                })

    return pd.DataFrame(records)

# ============================================================
# EXPORT DES RÉSULTATS
# ============================================================

def export_global_irfs(
    aggregated_results,
    excel_name="Global_IRF_VECM.xlsx"
):
    """
    Exporte les IRF globales dans Excel.
    """

    if not excel_name.lower().endswith(".xlsx"):
        excel_name = f"{excel_name}.xlsx"

    df_global = global_irf_dataframe(
        aggregated_results
    )

    df_orders = pd.DataFrame({
        "Cholesky_Order":
            aggregated_results["order_names"]
    })

    with pd.ExcelWriter(
        excel_name,
        engine="openpyxl"
    ) as writer:

        df_global.to_excel(
            writer,
            sheet_name="Global_IRF",
            index=False
        )

        df_orders.to_excel(
            writer,
            sheet_name="Estimated_Orders",
            index=False
        )

    print(
        f"Résultats exportés dans : {excel_name}"
    )

    return df_global
