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


def resume_unites_irf(
    model_result,
    data,
    transformations=None,
    unites=None,
    type_modele="auto"
):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data doit être une DataFrame pandas.")

    transformations = transformations or {}
    unites = unites or {}

    variables = list(data.columns)
    n = len(variables)
    if type_modele == "auto":
        nom_classe = model_result.__class__.__name__.upper()

        if "VECM" in nom_classe:
            type_modele = "VECM"
        elif "VAR" in nom_classe:
            type_modele = "VAR"
        else:
            type_modele = "modèle multivarié"
    sigma_u = np.asarray(model_result.sigma_u, dtype=float)

    if sigma_u.shape != (n, n):
        raise ValueError(
            f"sigma_u a la dimension {sigma_u.shape}, "
            f"mais les données contiennent {n} variables."
        )

    # Écart-type des innovations réduites
    std_residus = np.sqrt(np.diag(sigma_u))

    # Matrice d'impact contemporain de Cholesky
    P = np.linalg.cholesky(sigma_u)

    # Impact propre du choc de Cholesky
    impact_propre = np.diag(P)

    # Écart-type historique des données du modèle
    std_historique = data.std(ddof=1).to_numpy()
    lignes = []

    for i, variable in enumerate(variables):

        transformation = transformations.get(variable, "niveau")   
        unite_originale = unites.get(
            variable,
            inferer_unite_bloomberg(variable)
        )


        sigma_reduit = std_residus[i]
        choc_cholesky = impact_propre[i]

        if transformation == "niveau":

            unite_modele = unite_originale

            interpretation_reduite = (
                f"1 sigma résiduel = {sigma_reduit:.4f} "
                f"{unite_originale}"
            )

            interpretation_cholesky = (
                f"Impact propre du choc orthogonal = "
                f"{choc_cholesky:.4f} {unite_originale}"
            )

        elif transformation == "difference":

            unite_modele = f"variation en {unite_originale}"

            interpretation_reduite = (
                f"1 sigma résiduel = variation de "
                f"{sigma_reduit:.4f} {unite_originale}"
            )

            interpretation_cholesky = (
                f"Impact propre du choc orthogonal = variation de "
                f"{choc_cholesky:.4f} {unite_originale}"
            )

        elif transformation == "log":

            unite_modele = "logarithme du niveau"

            sigma_pct = 100 * (np.exp(sigma_reduit) - 1)
            cholesky_pct = 100 * (np.exp(choc_cholesky) - 1)

            interpretation_reduite = (
                f"1 sigma résiduel correspond à environ "
                f"{sigma_pct:.2f} % du niveau"
            )

            interpretation_cholesky = (
                f"Impact propre du choc orthogonal correspond à environ "
                f"{cholesky_pct:.2f} % du niveau"
            )

        elif transformation == "log_difference":

            unite_modele = "rendement logarithmique"

            sigma_pct = 100 * (np.exp(sigma_reduit) - 1)
            cholesky_pct = 100 * (np.exp(choc_cholesky) - 1)

            interpretation_reduite = (
                f"1 sigma résiduel correspond à un rendement "
                f"d'environ {sigma_pct:.2f} %"
            )

            interpretation_cholesky = (
                f"Impact propre du choc orthogonal correspond à un "
                f"rendement d'environ {cholesky_pct:.2f} %"
            )

        elif transformation == "log_difference_100":

            unite_modele = "pourcentage"

            interpretation_reduite = (
                f"1 sigma résiduel correspond à environ "
                f"{sigma_reduit:.2f} %"
            )

            interpretation_cholesky = (
                f"Impact propre du choc orthogonal correspond à environ "
                f"{choc_cholesky:.2f} %"
            )

        elif transformation == "standardise":

            unite_modele = "écart-type"

            interpretation_reduite = (
                f"1 sigma résiduel = "
                f"{sigma_reduit:.4f} unité standardisée"
            )

            interpretation_cholesky = (
                f"Impact propre du choc orthogonal = "
                f"{choc_cholesky:.4f} unité standardisée"
            )

        else:

            unite_modele = "transformation inconnue"
            interpretation_reduite = "Transformation à préciser"
            interpretation_cholesky = "Transformation à préciser"

        lignes.append({
            "variable": variable,
            "modele": type_modele,
            "transformation": transformation,
            "unite_originale": unite_originale,
            "unite_dans_le_modele": unite_modele,
            "ecart_type_historique": std_historique[i],
            "ecart_type_residuel": sigma_reduit,
            "impact_propre_cholesky": choc_cholesky,
            "interpretation_sigma_residuel": interpretation_reduite,
            "interpretation_choc_cholesky": interpretation_cholesky
        })

    resume = pd.DataFrame(lignes).set_index("variable")

    P_df = pd.DataFrame(
        P,
        index=variables,
        columns=[f"choc_{variable}" for variable in variables]
    )

    return resume, P_df
