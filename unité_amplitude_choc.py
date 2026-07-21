import numpy as np
import pandas as pd


def inferer_unite_bloomberg(nom):

    nom_upper = str(nom).upper().strip()

    tokens_pourcentage = [
        "YOY",       # variation annuelle
        "CYOY",      # variation annuelle
        "XYOY",      # variation annuelle
        "CQOQ",      # variation trimestrielle
        "CHY%",      # variation en %
        "MOM",       # variation mensuelle
        "FED FUNDS",
        "USURTOT",
        "PIQQ"
    ]

    if any(token in nom_upper for token in tokens_pourcentage):
        return "pourcentage / points de pourcentage"

    if "NFP TCH" in nom_upper:
        return "milliers d'emplois"

    if "ADP CHNG" in nom_upper:
        return "milliers d'emplois"

    tokens_pmi = [
        "PMI",
        "NAPM"
    ]

    if any(token in nom_upper for token in tokens_pmi):
        return "points d'indice PMI"

    tokens_indices = [
        "CONCCONF",
        "CONSSENT",
        "CFNAI",
        "LEI",
        "RCHSINDX",
        "CHPMINDX",
        "EMPRGBCI",
        "OUTFGAF"
    ]

    if any(token in nom_upper for token in tokens_indices):
        return "points d'indice"

    if "NHSPATOT" in nom_upper:
        return "milliers de logements, rythme annualisé"

    if "ETSLTOTL" in nom_upper:
        return "millions de logements, rythme annualisé"

    if "SAARTOTL" in nom_upper:
        return "millions de véhicules, rythme annualisé"


    if "XAU" in nom_upper:
        return "dollars US par once troy"

    if "CL1" in nom_upper:
        return "dollars US par baril"

    if "LMCADS03" in nom_upper:
        return "dollars US par tonne métrique"

    if "BITCOIN" in nom_upper or "ETHEREUM" in nom_upper:
        return "dollars US"

    # Indices actions
    tokens_indices_actions = [
        "SPX",
        "MXWO",
        "CCMP",
        "RTY",
        "S5FINL",
        "S5INFT",
        "S5HLTH",
        "S5COND",
        "S5TELS",
        "S5INDU",
        "S5CONS",
        "S5ENRS",
        "S5UTIL",
        "S5MATR",
        "S5RLST"
    ]

    if any(token in nom_upper for token in tokens_indices_actions):
        return "points d'indice"

    if "INDEX" in nom_upper:
        return "points d'indice ou unité spécifique Bloomberg"

    return "unité non identifiée"

def tableau_unites_irf(
    data,
    vecm_result=None,
    transformations=None,
    unites_personnalisees=None
):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data doit être une DataFrame pandas.")

    transformations = transformations or {}
    unites_personnalisees = unites_personnalisees or {}

    variables = data.columns.tolist()
    ecart_type_donnees = data.std(ddof=1)
    if vecm_result is not None:
        sigma_u = np.asarray(vecm_result.sigma_u)

        if sigma_u.shape != (len(variables), len(variables)):
            raise ValueError(
                f"sigma_u a la dimension {sigma_u.shape}, "
                f"mais {len(variables)} variables sont présentes."
            )

        ecart_type_choc = np.sqrt(np.diag(sigma_u))

    else:
        ecart_type_choc = np.full(len(variables), np.nan)

    lignes = []

    for i, variable in enumerate(variables):

        transformation = transformations.get(variable, "niveau")

        unite_originale = unites_personnalisees.get(
            variable,
            inferer_unite_bloomberg(variable)
        )

        # Détermination de l'unité dans le modèle
        if transformation == "niveau":
            unite_modele = unite_originale
            interpretation = (
                f"Un choc d'un écart-type correspond à "
                f"{ecart_type_choc.4f} {unite_originale}"
            )

        elif transformation == "difference":
            unite_modele = f"variation en {unite_originale}"
            interpretation = (
                f"Un choc d'un écart-type correspond à une variation de "
                f"{ecart_type_choc.4f} {unite_originale}"
            )

        elif transformation == "log":
            unite_modele = "logarithme du niveau"
            variation_pct = 100 * (np.exp(ecart_type_choc[i]) - 1)

            interpretation = (
                f"Un choc d'un écart-type correspond approximativement "
                f"à une variation de {variation_pct:.2f} % du niveau"
            )

        elif transformation == "log_difference":
            unite_modele = "variation logarithmique"
            variation_pct = 100 * (np.exp(ecart_type_choc[i]) - 1)

            interpretation = (
                f"Un choc d'un écart-type correspond à un rendement "
                f"d'environ {variation_pct:.2f} %"
            )

        elif transformation == "log_difference_100":
            unite_modele = "pourcentage"
            interpretation = (
                f"Un choc d'un écart-type correspond à une variation "
                f"d'environ {ecart_type_choc.2f} %"
            )

        elif transformation == "standardise":
            unite_modele = "écart-type"
            interpretation = (
                f"Un choc résiduel d'un écart-type vaut "
                f"{ecart_type_choc.4f} unité standardisée"
            )

        else:
            unite_modele = "transformation inconnue"
            interpretation = "Transformation à préciser"

        lignes.append({
            "variable": variable,
            "transformation": transformation,
            "unite_originale_probable": unite_originale,
            "unite_dans_le_modele": unite_modele,
            "ecart_type_historique": ecart_type_donnees.loc[variable],
            "ecart_type_innovation_vecm": ecart_type_choc[i],
            "interpretation_choc_1_sigma": interpretation
        })

    return pd.DataFrame(lignes).set_index("variable")
