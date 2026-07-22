from itertools import permutations
from math import factorial
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR


def format_label(label, max_length=30):
    """
    Coupe les intitulés longs sur plusieurs lignes.
    """
    return fill(
        str(label),
        width=max_length
    )
def estimate_var_permutations(
    data,
    lags,
    trend="c",
    exog=None,
    variables=None,
    orders=None,
    max_permutations=720,
    verbose=True
):
    """
    Estime un VAR pour plusieurs ordres de variables.

    Chaque ordre produit la même forme réduite du VAR,
    à une permutation près, mais une identification de
    Cholesky différente.

    Parameters
    ----------
    data : pandas.DataFrame
        Données stationnaires utilisées dans le VAR.

    lags : int
        Nombre de retards du VAR.

    trend : {"n", "c", "ct", "ctt"}
        Terme déterministe :
        - "n"   : aucun terme déterministe ;
        - "c"   : constante ;
        - "ct"  : constante et tendance ;
        - "ctt" : constante, tendance linéaire et quadratique.

    exog : array-like ou DataFrame, optionnel
        Variables exogènes communes aux estimations.

    variables : list[str], optionnel
        Variables endogènes à utiliser.

    orders : list[list[str]], optionnel
        Ordres précis à tester.

        Si None, toutes les permutations sont utilisées.

    max_permutations : int ou None
        Nombre maximal de permutations autorisées.
        Utiliser None pour supprimer la limite.

    verbose : bool
        Affiche la progression.

    Returns
    -------
    dict
        Dictionnaire contenant les modèles estimés,
        les ordres réussis et les ordres ayant échoué.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data doit être un pandas.DataFrame."
        )

    if not isinstance(lags, int) or lags < 1:
        raise ValueError(
            "lags doit être un entier supérieur ou égal à 1."
        )

    data = data.copy()

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

    if len(variables) < 2:
        raise ValueError(
            "Un VAR nécessite au moins deux variables."
        )

    # Ne conserver que les variables endogènes demandées
    clean_data = data.loc[:, variables].copy()

    for column in clean_data.columns:
        clean_data[column] = pd.to_numeric(
            clean_data[column],
            errors="coerce"
        )

    # Traitement des variables exogènes
    clean_exog = None

    if exog is not None:

        if isinstance(exog, pd.DataFrame):
            clean_exog = exog.copy()

            for column in clean_exog.columns:
                clean_exog[column] = pd.to_numeric(
                    clean_exog[column],
                    errors="coerce"
                )

            clean_data, clean_exog = clean_data.align(
                clean_exog,
                join="inner",
                axis=0
            )

            valid_rows = (
                clean_data.notna().all(axis=1)
                & clean_exog.notna().all(axis=1)
            )

            clean_data = clean_data.loc[valid_rows]
            clean_exog = clean_exog.loc[valid_rows]

        else:
            clean_exog = np.asarray(exog)

            if len(clean_exog) != len(clean_data):
                raise ValueError(
                    "exog doit avoir le même nombre "
                    "d'observations que data."
                )

            valid_rows = clean_data.notna().all(axis=1)

            clean_data = clean_data.loc[valid_rows]
            clean_exog = clean_exog[
                valid_rows.to_numpy()
            ]

    else:
        clean_data = clean_data.dropna()

    if clean_data.empty:
        raise ValueError(
            "Aucune observation disponible après "
            "le nettoyage des données."
        )

    if len(clean_data) <= lags:
        raise ValueError(
            "Le nombre d'observations est insuffisant "
            "par rapport au nombre de retards."
        )

    # Construction des ordres
    if orders is None:

        number_of_orders = factorial(
            len(variables)
        )

        if (
            max_permutations is not None
            and number_of_orders > max_permutations
        ):
            raise ValueError(
                f"{len(variables)} variables donnent "
                f"{number_of_orders} permutations.\n"
                f"La limite est fixée à {max_permutations}.\n"
                "Fournis une liste orders spécifique ou augmente "
                "max_permutations."
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

        if (
            max_permutations is not None
            and len(orders_to_estimate) > max_permutations
        ):
            raise ValueError(
                "Le nombre d'ordres fournis dépasse "
                f"max_permutations={max_permutations}."
            )

        for order in orders_to_estimate:

            if len(order) != len(set(order)):
                raise ValueError(
                    "L'ordre suivant contient des doublons :\n"
                    f"{order}"
                )

            if set(order) != set(variables):
                raise ValueError(
                    "Chaque ordre doit contenir exactement "
                    "les variables sélectionnées.\n\n"
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
        order_name = " -> ".join(current_order)

        if verbose:
            print(
                f"[{order_number}/{total}] "
                f"Ordre : {order_name}"
            )

        permuted_data = clean_data.loc[
            :,
            current_order
        ].copy()

        try:
            model = VAR(
                endog=permuted_data,
                exog=clean_exog
            )

            results = model.fit(
                maxlags=lags,
                ic=None,
                trend=trend
            )

            models[order_name] = results
            estimated_orders[order_name] = current_order

        except Exception as error:
            failed_orders[order_name] = str(error)

            if verbose:
                print(
                    f"Échec pour l'ordre : {order_name}"
                )
                print(f"Erreur : {error}")

    if len(models) == 0:
        raise RuntimeError(
            "Aucun VAR n'a pu être estimé."
        )

    if verbose:
        print()
        print(
            f"{len(models)} VAR estimés avec succès."
        )
        print(
            f"{len(failed_orders)} ordres en échec."
        )

    return {
        "models": models,
        "orders": estimated_orders,
        "failed_orders": failed_orders,
        "reference_order": variables,
        "data": clean_data,
        "lags": lags,
        "trend": trend
    }

def compute_irf_var(
    var_results,
    periods=24
):
    """
    Calcule les IRF orthogonalisées d'un VAR estimé.

    Parameters
    ----------
    var_results : VARResults
        Résultat d'un modèle VAR estimé.

    periods : int
        Nombre d'horizons après le choc.

    Returns
    -------
    dict
        Contient les IRF orthogonalisées,
        les IRF non orthogonalisées et les noms.
    """

    if not isinstance(periods, int) or periods < 1:
        raise ValueError(
            "periods doit être un entier positif."
        )

    irf_analysis = var_results.irf(
        periods=periods
    )

    names = list(var_results.names)

    orth_irfs = np.asarray(
        irf_analysis.orth_irfs,
        dtype=float
    )

    irfs = np.asarray(
        irf_analysis.irfs,
        dtype=float
    )

    expected_shape = (
        periods + 1,
        len(names),
        len(names)
    )

    if orth_irfs.shape != expected_shape:
        raise ValueError(
            "Dimension inattendue pour orth_irfs.\n"
            f"Dimension obtenue : {orth_irfs.shape}\n"
            f"Dimension attendue : {expected_shape}"
        )

    return {
        "orth_irfs": orth_irfs,
        "irfs": irfs,
        "names": names,
        "periods": periods,
        "irf_analysis": irf_analysis
    }

def compute_var_irf_from_permutations(
    permutation_results,
    periods=24,
    reference_order=None,
    verbose=True
):
    """
    Calcule les IRF orthogonalisées de tous les VAR estimés.

    Les axes des réponses et des impulsions sont replacés
    dans un ordre commun.

    Parameters
    ----------
    permutation_results : dict
        Résultat de estimate_var_permutations.

    periods : int
        Nombre d'horizons.

    reference_order : list[str], optionnel
        Ordre commun utilisé dans les résultats.

    verbose : bool
        Affiche la progression.

    Returns
    -------
    dict
        IRF réordonnées pour chaque ordre de Cholesky.
    """

    models = permutation_results["models"]

    if reference_order is None:
        reference_order = list(
            permutation_results["reference_order"]
        )
    else:
        reference_order = list(reference_order)

    if set(reference_order) != set(
        permutation_results["reference_order"]
    ):
        raise ValueError(
            "reference_order doit contenir exactement "
            "les variables des modèles estimés."
        )

    permutation_irfs = {}
    failed_irfs = {}

    total = len(models)

    for model_number, (
        order_name,
        var_results
    ) in enumerate(
        models.items(),
        start=1
    ):
        if verbose:
            print(
                f"[{model_number}/{total}] "
                f"IRF : {order_name}"
            )

        try:
            current_irf = compute_irf_var(
                var_results=var_results,
                periods=periods
            )

            current_names = current_irf["names"]
            current_values = current_irf[
                "orth_irfs"
            ]

            if set(current_names) != set(reference_order):
                raise ValueError(
                    "Les variables de l'ordre "
                    f"{order_name} ne correspondent pas "
                    "à l'ordre de référence."
                )

            reorder_indices = [
                current_names.index(variable)
                for variable in reference_order
            ]

            # Axe 1 : variables répondantes
            reordered_values = current_values[
                :,
                reorder_indices,
                :
            ]

            # Axe 2 : variables à l'origine du choc
            reordered_values = reordered_values[
                :,
                :,
                reorder_indices
            ]

            permutation_irfs[order_name] = (
                reordered_values
            )

        except Exception as error:
            failed_irfs[order_name] = str(error)

            if verbose:
                print(
                    f"Échec du calcul des IRF : {order_name}"
                )
                print(f"Erreur : {error}")

    if not permutation_irfs:
        raise RuntimeError(
            "Aucune IRF orthogonalisée n'a pu être calculée."
        )

    return {
        "irfs": permutation_irfs,
        "reference_order": reference_order,
        "periods": periods,
        "failed_irfs": failed_irfs,
        "model_type": "VAR"
    }

def aggregate_var_irf_permutations(
    permutation_irf_results,
    lower_quantile=0.10,
    upper_quantile=0.90,
    aggregation="median"
):
    """
    Agrège les IRF obtenues sous différents ordres
    de Cholesky pour un modèle VAR.
    """

    if aggregation not in {"median", "mean"}:
        raise ValueError(
            "aggregation doit être 'median' ou 'mean'."
        )

    if not (
        0 <= lower_quantile
        < upper_quantile
        <= 1
    ):
        raise ValueError(
            "Les quantiles doivent vérifier "
            "0 <= lower < upper <= 1."
        )

    irf_dictionary = permutation_irf_results[
        "irfs"
    ]

    if not irf_dictionary:
        raise ValueError(
            "Le dictionnaire des IRF est vide."
        )

    order_names = list(irf_dictionary.keys())

    shapes = {
        name: np.asarray(values).shape
        for name, values in irf_dictionary.items()
    }

    unique_shapes = set(shapes.values())

    if len(unique_shapes) != 1:
        raise ValueError(
            "Toutes les IRF n'ont pas la même dimension.\n"
            f"Dimensions observées : {shapes}"
        )

    # ordre x horizon x réponse x choc
    all_irfs = np.stack(
        [
            np.asarray(
                irf_dictionary[order_name],
                dtype=float
            )
            for order_name in order_names
        ],
        axis=0
    )

    mean_irf = np.nanmean(
        all_irfs,
        axis=0
    )

    median_irf = np.nanmedian(
        all_irfs,
        axis=0
    )

    lower_irf = np.nanquantile(
        all_irfs,
        lower_quantile,
        axis=0
    )

    upper_irf = np.nanquantile(
        all_irfs,
        upper_quantile,
        axis=0
    )

    minimum_irf = np.nanmin(
        all_irfs,
        axis=0
    )

    maximum_irf = np.nanmax(
        all_irfs,
        axis=0
    )

    std_irf = np.nanstd(
        all_irfs,
        axis=0,
        ddof=0
    )

    positive_share = np.nanmean(
        all_irfs > 0,
        axis=0
    )

    negative_share = np.nanmean(
        all_irfs < 0,
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
        "negative_share": negative_share,
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
        "upper_quantile": upper_quantile,
        "model_type": "VAR"
    }

def plot_single_global_var_irf(
    aggregated_results,
    impulse,
    response,
    figsize=(12, 6),
    show_individual_orders=True
):
    """
    Trace les différentes IRF obtenues avec les ordres
    de Cholesky et l'IRF globale du VAR.
    """

    variables = aggregated_results[
        "reference_order"
    ]

    if impulse not in variables:
        raise ValueError(
            f"Le choc {impulse!r} n'existe pas.\n"
            f"Variables disponibles : {variables}"
        )

    if response not in variables:
        raise ValueError(
            f"La réponse {response!r} n'existe pas.\n"
            f"Variables disponibles : {variables}"
        )

    impulse_idx = variables.index(impulse)
    response_idx = variables.index(response)

    all_irfs = aggregated_results["all_irfs"]

    global_values = aggregated_results[
        "global_irf"
    ][:, response_idx, impulse_idx]

    lower_values = aggregated_results[
        "lower_irf"
    ][:, response_idx, impulse_idx]

    upper_values = aggregated_results[
        "upper_irf"
    ][:, response_idx, impulse_idx]

    horizons = np.arange(
        global_values.shape[0]
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
        label=(
            f"IRF globale, {aggregation_label}"
        )
    )

    ax.axhline(
        y=0,
        color="black",
        linewidth=1
    )

    ax.set_title(
        f"VAR, réponse de :\n"
        f"{format_label(response, 70)}\n\n"
        f"à un choc de :\n"
        f"{format_label(impulse, 70)}",
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

def plot_all_global_var_irfs(
    aggregated_results,
    figsize_per_subplot=(4.8, 3.5),
    max_label_length=30,
    show_individual_orders=True,
    show_grid=True
):
    """
    Trace toutes les IRF globales du VAR.

    Colonnes :
        variables à l'origine du choc.

    Lignes :
        variables répondantes.
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

    horizons = np.arange(
        global_irf.shape[0]
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

    for response_idx, response in enumerate(
        variables
    ):
        for impulse_idx, impulse in enumerate(
            variables
        ):
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
        "IRF globales du VAR\n"
        f"Agrégation par {aggregation_label} "
        "des ordres de Cholesky",
        fontsize=15,
        y=1.01
    )

    fig.tight_layout()
    plt.show()

    return fig, axes

def global_var_irf_dataframe(
    aggregated_results
):
    """
    Convertit les IRF globales du VAR et les indicateurs
    de sensibilité à l'ordre en DataFrame long.
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

    negative_share = aggregated_results[
        "negative_share"
    ]

    number_of_horizons = global_irf.shape[0]

    records = []

    for impulse_idx, impulse in enumerate(
        variables
    ):
        for response_idx, response in enumerate(
            variables
        ):
            for horizon in range(
                number_of_horizons
            ):
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
                    "Positive_Order_Share":
                        positive_share[
                            horizon,
                            response_idx,
                            impulse_idx
                        ],
                    "Negative_Order_Share":
                        negative_share[
                            horizon,
                            response_idx,
                            impulse_idx
                        ]
                })

    return pd.DataFrame(records)

def export_global_var_irfs(
    aggregated_results,
    excel_name="Global_IRF_VAR.xlsx"
):
    """
    Exporte les IRF globales du VAR dans Excel.
    """

    if not excel_name.lower().endswith(".xlsx"):
        excel_name = f"{excel_name}.xlsx"

    df_global = global_var_irf_dataframe(
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
