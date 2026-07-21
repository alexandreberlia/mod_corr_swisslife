import numpy as np
import pandas as pd


import re
import unicodedata


def normaliser_nom(nom):
    nom = str(nom).strip().upper()
    nom = unicodedata.normalize(
        "NFKD",
        nom
    ).encode(
        "ascii",
        "ignore"
    ).decode("ascii")

    nom = re.sub(r"\s+", " ", nom)

    return nom


def inferer_unite_variable(nom):

    nom_upper = normaliser_nom(nom)

    # ==================================================
    # 1. PIB
    # ==================================================

    if (
        "GDP US CHAINED DOLLARS QOQ" in nom_upper
        or (
            "GDP" in nom_upper
            and "QOQ" in nom_upper
        )
    ):
        return (
            "pourcentage de variation trimestrielle annualisée"
        )

    if (
        "GDP US CHAINED DOLLARS YOY" in nom_upper
        or (
            "GDP" in nom_upper
            and "YOY" in nom_upper
            and "NOMINAL" not in nom_upper
        )
    ):
        return "pourcentage de variation annuelle"

    if (
        "GDP" in nom_upper
        and "NOMINAL" in nom_upper
        and "YOY" in nom_upper
    ):
        return "pourcentage de variation annuelle"

    if (
        "GDP PRICE INDEX" in nom_upper
        and "QOQ" in nom_upper
    ):
        return (
            "pourcentage de variation trimestrielle annualisée"
        )

    # ==================================================
    # 2. Emploi
    # ==================================================

    if "UNEMPLOYMENT RATE" in nom_upper:
        return "point de pourcentage"

    if "EMPLOYEES ON NONFARM PAYROL" in nom_upper:
        return "milliers d'emplois"

    if "ADP NATIONAL EMPLOYMENT REPORT" in nom_upper:
        return "milliers d'emplois"

    # ==================================================
    # 3. Inflation et salaires
    # ==================================================

    if "AVG HOURLY EARNINGS" in nom_upper:
        return (
            "pourcentage de variation ou dollar par heure "
            "selon la série source"
        )

    if "CPI URBAN CONSUMERS" in nom_upper:
        return "pourcentage de variation annuelle"

    if "PPI FINISHED GOODS" in nom_upper:
        return "pourcentage de variation annuelle"

    if (
        "PERSONAL CONSUMPTION EXPEND" in nom_upper
        and "(INFLATION)" in nom_upper
    ):
        return "pourcentage de variation annuelle"

    # ==================================================
    # 4. Production et activité économique
    # ==================================================

    if (
        "INDUSTRIAL PRODUCTION" in nom_upper
        and "YOY" in nom_upper
    ):
        return "pourcentage de variation annuelle"

    if "CAPACITY UTILIZATION" in nom_upper:
        return "point de pourcentage"

    if "DURABLE GOODS NEW ORDERS" in nom_upper:
        return (
            "pourcentage de variation mensuelle "
            "ou millions de dollars selon la série source"
        )

    if "CHICAGO FED NATIONAL ACTIVITY" in nom_upper:
        return "point d'indice standardisé"

    if "CONFERENCE BOARD US LEADING IN" in nom_upper:
        return "point d'indice"

    # ==================================================
    # 5. PMI et conditions d'activité
    # ==================================================

    if "PMI" in nom_upper:
        return "point d'indice PMI"

    if "PHILADELPHIA FED BUSINESS" in nom_upper:
        return "point d'indice de diffusion"

    if "EMPIRE STATE MANUFACTURING" in nom_upper:
        return "point d'indice de diffusion"

    if "RICHMOND MANUFACTURING SURVEY" in nom_upper:
        return "point d'indice de diffusion"

    if "MARKET NEWS INTERNATIONAL CHIC" in nom_upper:
        return "point d'indice PMI"

    # ==================================================
    # 6. Immobilier
    # ==================================================

    if "PRIVATE HOUSING AUTHORIZED" in nom_upper:
        return (
            "milliers de logements, rythme annualisé"
        )

    if "NAR TOTAL EXISTING HOMES" in nom_upper:
        return (
            "millions de logements, rythme annualisé"
        )

    # ==================================================
    # 7. Consommation et revenu des ménages
    # ==================================================

    if (
        "RETAIL SALES" in nom_upper
        and "MOM" in nom_upper
    ):
        return "pourcentage de variation mensuelle"

    if (
        "RETAIL SALES" in nom_upper
        and "YOY" in nom_upper
    ):
        return "pourcentage de variation annuelle"

    if "AUTO SALES TOTAL ANNUALIZED" in nom_upper:
        return (
            "millions de véhicules, rythme annualisé"
        )

    if "PERSONAL INCOME YOY" in nom_upper:
        return "pourcentage de variation annuelle"

    if (
        "PERSONAL CONSUMPTION EXPEND %" in nom_upper
        or (
            "PERSONAL CONSUMPTION EXPEND" in nom_upper
            and "%" in str(nom)
        )
    ):
        return "pourcentage de variation"

    if (
        "PERSONAL CONSUMPTION EXPEND" in nom_upper
        and "(HOUSEHOLD)" in nom_upper
    ):
        return (
            "milliards de dollars ou indice, "
            "selon la série source"
        )

    # ==================================================
    # 8. Confiance des consommateurs
    # ==================================================

    if "CONFERENCE BOARD CONSUMER CONF" in nom_upper:
        return "point d'indice de confiance"

    if "UNIVERSITY OF MICHIGAN CONSUME" in nom_upper:
        return "point d'indice de confiance"

    # ==================================================
    # 9. Indices actions
    # ==================================================

    indices_actions = [
        "S&P 500",
        "MSCI WORLD",
        "NASDAQ",
        "RUSSELL 2000"
    ]

    if any(
        indice in nom_upper
        for indice in indices_actions
    ):
        return "point d'indice"

    # ==================================================
    # 10. Taux d'intérêt
    # ==================================================

    if "FED FUNDS" in nom_upper:
        return "point de pourcentage"

    # ==================================================
    # 11. Matières premières
    # ==================================================

    if "GOLD SPOT" in nom_upper:
        return "dollar US par once troy"

    if (
        "GENERIC 1ST" in nom_upper
        and "CL" in nom_upper
    ):
        return "dollar US par baril"

    if "IRON ORE SPOT PRICE" in nom_upper:
        return "dollar US par tonne métrique"

    if "LME COPPER" in nom_upper:
        return "dollar US par tonne métrique"

    # ==================================================
    # 12. Cryptomonnaies
    # ==================================================

    if (
        "BBG BTC" in nom_upper
        or "BITCOIN" in nom_upper
    ):
        return "dollar US par bitcoin ou point d'indice"

    if (
        "BBG ETH" in nom_upper
        or "ETHEREUM" in nom_upper
    ):
        return "dollar US par ether ou point d'indice"

    # ==================================================
    # 13. Sous-indices sectoriels du S&P 500
    # ==================================================

    if "SUB S&P500 INDICATOR" in nom_upper:
        return "point d'indice sectoriel"

    # ==================================================
    # 14. Règles génériques
    # ==================================================

    if "YOY" in nom_upper:
        return "pourcentage de variation annuelle"

    if "QOQ" in nom_upper:
        return "pourcentage de variation trimestrielle"

    if "MOM" in nom_upper:
        return "pourcentage de variation mensuelle"

    if "%" in str(nom):
        return "pourcentage ou point de pourcentage"

    if "STOCK INDEX" in nom_upper:
        return "point d'indice"

    return "unité à vérifier"

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
            inferer_unite_variable(variable)
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

    return resume
