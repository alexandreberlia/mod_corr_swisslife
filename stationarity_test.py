"""
stationnarite_bloom
===================

Une fonction : ``tableau_stationnarite(blocs)`` -> DataFrame.

Une ligne par variable, avec les deux tests et le verdict.

Regle de decision (volontairement simple, modifiable en un endroit) :
    - ADF   H0 = racine unitaire   -> p < alpha  => stationnaire
    - KPSS  H0 = stationnarite     -> p > alpha  => stationnaire
    - I(0) si le NIVEAU est stationnaire
    - I(1) si le niveau ne l'est pas mais la DIFFERENCE PREMIERE l'est
    - I(2)+ sinon

On ne teste jamais au-dela de la difference premiere : differencier une
serie deja stationnaire cree un artefact (moyenne mobile non inversible)
qui fait rejeter KPSS a tort, et c'est exactement ce qui produisait des
verdicts "I(2)" sur du bruit blanc.

Exemple
-------
    from data_loader_analysis_bloom import load_dataset
    from stationnarite_bloom import tableau_stationnarite

    frames = load_dataset("export.csv", n_series=58).frames
    stat = tableau_stationnarite(frames)
    stat[stat["Stationnaire"] == "Non"]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

__all__ = ["tableau_stationnarite", "tester_serie"]

MIN_OBS = 30


def _normaliser(blocs) -> dict[str, pd.DataFrame]:
    """Accepte un DataFrame seul ou un dict {nom: DataFrame}."""
    if isinstance(blocs, pd.DataFrame):
        return {"bloc": blocs}
    return dict(blocs)


def tester_serie(serie: pd.Series, alpha: float = 0.05) -> dict:
    """
    ADF + KPSS sur le niveau, puis sur la difference premiere si besoin.

    Renvoie un dict plat : c'est ce dict qui devient une ligne du tableau.
    Toute modification de la regle de decision se fait ici, nulle part ailleurs.
    """
    valeurs = pd.to_numeric(serie, errors="coerce").dropna()
    resultat = {
        "N": len(valeurs),
        "ADF stat": np.nan, "ADF p": np.nan,
        "KPSS stat": np.nan, "KPSS p": np.nan,
        "Stationnaire": "-", "Ordre": "indetermine", "Remarque": "",
    }

    if len(valeurs) < MIN_OBS:
        resultat["Remarque"] = f"moins de {MIN_OBS} observations"
        return resultat

    # --- niveau ---------------------------------------------------------
    adf_stat, adf_p = _adf(valeurs)
    kpss_stat, kpss_p = _kpss(valeurs)
    resultat.update({"ADF stat": adf_stat, "ADF p": adf_p,
                     "KPSS stat": kpss_stat, "KPSS p": kpss_p})

    if np.isnan(adf_p) or np.isnan(kpss_p):
        resultat["Remarque"] = "test en echec"
        return resultat

    adf_dit_stationnaire = adf_p < alpha
    kpss_dit_stationnaire = kpss_p > alpha

    if adf_dit_stationnaire and kpss_dit_stationnaire:
        resultat.update(Stationnaire="Oui", Ordre="I(0)")
        return resultat

    if adf_dit_stationnaire != kpss_dit_stationnaire:
        resultat["Remarque"] = "ADF et KPSS en desaccord"

    # --- difference premiere -------------------------------------------
    diff = valeurs.diff().dropna()
    if len(diff) < MIN_OBS:
        resultat["Stationnaire"] = "Non"
        return resultat

    adf_p_d = _adf(diff)[1]
    kpss_p_d = _kpss(diff)[1]
    diff_stationnaire = (adf_p_d < alpha) and (kpss_p_d > alpha)

    resultat["Stationnaire"] = "Non"
    resultat["Ordre"] = "I(1)" if diff_stationnaire else "I(2)+"
    return resultat


def _adf(serie: pd.Series, regression: str = "c") -> tuple[float, float]:
    try:
        sortie = adfuller(serie, regression=regression, autolag="AIC")
        return float(sortie[0]), float(sortie[1])
    except Exception:
        return np.nan, np.nan


def _kpss(serie: pd.Series, regression: str = "c") -> tuple[float, float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # p-value bornee a [0.01, 0.10]
            sortie = kpss(serie, regression=regression, nlags="auto")
        return float(sortie[0]), float(sortie[1])
    except Exception:
        return np.nan, np.nan


def tableau_stationnarite(blocs, alpha: float = 0.05, decimales: int = 4) -> pd.DataFrame:
    """
    Parameters
    ----------
    blocs : un DataFrame, ou un dict {nom du bloc: DataFrame} (= dict_of_df).
    alpha : seuil des deux tests.

    Returns
    -------
    DataFrame : Bloc, Variable, N, ADF stat, ADF p, KPSS stat, KPSS p,
    Stationnaire (Oui/Non), Ordre (I(0)/I(1)/I(2)+), Remarque.
    """
    lignes = []
    for nom_bloc, bloc in _normaliser(blocs).items():
        for variable in bloc.columns:
            ligne = {"Bloc": nom_bloc, "Variable": variable}
            ligne.update(tester_serie(bloc[variable], alpha=alpha))
            lignes.append(ligne)

    tableau = pd.DataFrame(lignes)
    for colonne in ["ADF stat", "ADF p", "KPSS stat", "KPSS p"]:
        tableau[colonne] = tableau[colonne].round(decimales)
    return tableau[
        ["Bloc", "Variable", "N", "ADF stat", "ADF p", "KPSS stat", "KPSS p",
         "Stationnaire", "Ordre", "Remarque"]
    ]
