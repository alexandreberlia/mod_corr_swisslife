"""
cointegration_bloom
===================

Une fonction : ``tableau_cointegration(blocs)`` -> DataFrame.

Une ligne par paire de variables d'un MEME bloc (les blocs partagent un
calendrier ; des paires inter-blocs n'auraient presque aucune date commune).

Trois regles de decision :

1. Seules les paires I(1)/I(1) sont testees. Si une variable est I(0), le
   residu de la regression est asymptotiquement cette variable elle-meme,
   donc stationnaire : le test rejette presque toujours. Ce serait un
   artefact mecanique, pas de la cointegration.

2. Engle-Granger n'est pas symetrique : regresser 1 sur 2 ne donne pas le
   meme resultat que 2 sur 1. On teste les deux sens et on retient la
   p-value la PLUS GRANDE : la paire n'est declaree cointegree que si les
   deux sens concluent.

3. Correction de Benjamini-Hochberg. Avec 58 series on teste des centaines
   de paires ; a 5 %, une paire sur vingt ressort "significative" par pur
   hasard. La colonne "Cointegre" s'appuie sur la p-value corrigee.

Exemple
-------
    from data_loader_analysis_bloom import load_dataset
    from cointegration_bloom import tableau_cointegration

    frames = load_dataset("export.csv", n_series=58).frames
    coint = tableau_cointegration(frames)
    coint[coint["Cointegre"] == "Oui"]
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from stationnarite_bloom import tableau_stationnarite

__all__ = ["tableau_cointegration", "corriger_fdr"]

MIN_OBS = 50


def _normaliser(blocs) -> dict[str, pd.DataFrame]:
    """Accepte un DataFrame seul ou un dict {nom: DataFrame}."""
    if isinstance(blocs, pd.DataFrame):
        return {"bloc": blocs}
    return dict(blocs)


def corriger_fdr(pvalues: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg. p ajustee = min, sur les rangs superieurs, de
    p * m / rang. Les NaN (paires non testees) restent NaN.
    """
    valides = pvalues.dropna().sort_values()
    m = len(valides)
    if m == 0:
        return pvalues

    ajustees = valides * m / np.arange(1, m + 1)
    ajustees = ajustees[::-1].cummin()[::-1].clip(upper=1.0)
    return ajustees.reindex(pvalues.index)


def tableau_cointegration(
    blocs,
    stationnarite: pd.DataFrame | None = None,
    alpha: float = 0.05,
    trend: str = "c",
    decimales: int = 4,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    blocs : un DataFrame, ou un dict {nom du bloc: DataFrame} (= dict_of_df).
    stationnarite : le tableau de stationnarite_bloom. Calcule si absent.
    alpha : seuil applique a la p-value corrigee.
    trend : "c" (constante) ou "ct" (constante + tendance) dans la relation.

    Returns
    -------
    DataFrame : Bloc, Variable 1, Variable 2, Ordres, N, p (1 vers 2),
    p (2 vers 1), p retenue, p corrigee, Cointegre (Oui/Non), Statut.
    """
    blocs = _normaliser(blocs)

    if stationnarite is None:
        stationnarite = tableau_stationnarite(blocs, alpha=alpha)
    ordres = dict(zip(stationnarite["Variable"], stationnarite["Ordre"]))

    lignes = []
    for nom_bloc, bloc in blocs.items():
        for v1, v2 in combinations(bloc.columns, 2):
            o1 = ordres.get(v1, "inconnu")
            o2 = ordres.get(v2, "inconnu")
            ligne = {
                "Bloc": nom_bloc, "Variable 1": v1, "Variable 2": v2,
                "Ordres": f"{o1} / {o2}", "N": pd.NA,
                "p (1 vers 2)": np.nan, "p (2 vers 1)": np.nan,
                "p retenue": np.nan, "Statut": "teste",
            }

            if o1 != "I(1)" or o2 != "I(1)":
                ligne["Statut"] = "non teste : pas I(1)/I(1)"
                lignes.append(ligne)
                continue

            paire = bloc[[v1, v2]].apply(pd.to_numeric, errors="coerce").dropna()
            ligne["N"] = len(paire)

            if len(paire) < MIN_OBS:
                ligne["Statut"] = f"non teste : moins de {MIN_OBS} observations"
                lignes.append(ligne)
                continue

            try:
                p12 = coint(paire[v1], paire[v2], trend=trend, autolag="aic")[1]
                p21 = coint(paire[v2], paire[v1], trend=trend, autolag="aic")[1]
                ligne["p (1 vers 2)"] = p12
                ligne["p (2 vers 1)"] = p21
                ligne["p retenue"] = max(p12, p21)  # le sens le plus defavorable
            except Exception as erreur:
                ligne["Statut"] = f"echec : {erreur}"

            lignes.append(ligne)

    tableau = pd.DataFrame(lignes)
    if tableau.empty:
        return tableau

    tableau["p corrigee"] = corriger_fdr(tableau["p retenue"])
    tableau["Cointegre"] = np.where(
        tableau["p corrigee"].isna(), "-",
        np.where(tableau["p corrigee"] < alpha, "Oui", "Non"),
    )

    for colonne in ["p (1 vers 2)", "p (2 vers 1)", "p retenue", "p corrigee"]:
        tableau[colonne] = tableau[colonne].round(decimales)
    tableau["N"] = tableau["N"].astype("Int64")

    return (
        tableau[
            ["Bloc", "Variable 1", "Variable 2", "Ordres", "N", "p (1 vers 2)",
             "p (2 vers 1)", "p retenue", "p corrigee", "Cointegre", "Statut"]
        ]
        .sort_values(["Cointegre", "p corrigee"], ascending=[False, True])
        .reset_index(drop=True)
    )
