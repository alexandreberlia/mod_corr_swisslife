"""
run_analysis_bloom
==================

Fait le lien entre le chargement des donnees Bloomberg et les tests
econometriques.

Le point de raccord : ``integration_order`` renvoie un dict, pas une chaine.
C'est ``stationarity_block`` qui expose l'ordre sous forme de texte, dans sa
colonne "Ordre". C'est cette colonne qui alimente ``cointegration_block`` :

    orders = dict(zip(stat["Variable"], stat["Ordre"]))

Exemple
-------
    from run_analysis_bloom import analyse

    res = analyse("export.csv", n_series=58, names_file="corrected_file.csv")

    res["stationnarite"]                 # un rapport par variable
    res["cointegration"]                 # une ligne par paire
    res["cointegration"].query("`Rejet FDR 5%`")
    res["suspects"]                      # verdicts d'ordre contradictoires
    res["johansen"]["df1"]["beta_normalise"]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader_analysis_bloom import load_dataset
from econometrics_bloom import (
    cointegration_block,
    johansen_report,
    stationarity_block,
)

__all__ = ["analyse", "verdicts_suspects"]


def verdicts_suspects(stationnarite: pd.DataFrame) -> pd.DataFrame:
    """
    Lignes ou le verdict sur le niveau contredit l'ordre retenu.

    Symptome de la descente de Pantula interrompue par un conflit ADF/KPSS :
    le niveau est declare stationnaire mais l'ordre final ne l'est pas.
    """
    if stationnarite.empty or "Verdict niveau (c)" not in stationnarite.columns:
        return stationnarite.head(0)

    niveau_stationnaire = (
        stationnarite["Verdict niveau (c)"].str.startswith("I(0)", na=False)
        | stationnarite["Verdict niveau (ct)"].str.startswith("I(0)", na=False)
    )
    return stationnarite.loc[
        niveau_stationnaire & (stationnarite["Ordre"] != "I(0)"),
        ["Block", "Variable", "N", "Ordre", "Verdict niveau (c)",
         "Verdict niveau (ct)", "Notes"],
    ]


def analyse(
    path: str | Path,
    freq: str | None = None,
    alpha: float = 0.05,
    johansen: bool = True,
    det_order: int = 0,
    k_ar_diff: int = 2,
    **loader_kwargs,
) -> dict:
    """
    Chaine complete : chargement -> stationnarite -> cointegration.

    Parameters
    ----------
    path : le CSV Bloomberg.
    freq : temporalite des tests. None (defaut) = frequence d'origine, ce qui
        est le bon choix : rien n'est interpole, les p-values portent sur des
        observations reelles.
    johansen : lancer aussi le test sur systeme complet, bloc par bloc.
    **loader_kwargs : passes a load_dataset (n_series, names_file, sep, ...).

    Returns
    -------
    dict avec les cles "data", "stationnarite", "cointegration", "suspects",
    "johansen".
    """
    data = load_dataset(path, **loader_kwargs)
    if freq:
        data = data.resample(freq)

    rapports_stat: list[pd.DataFrame] = []
    rapports_coint: list[pd.DataFrame] = []
    rapports_johansen: dict[str, dict] = {}

    for nom, block in data.frames.items():
        stat = stationarity_block(block, block_name=nom, alpha=alpha)
        rapports_stat.append(stat)

        if block.shape[1] < 2:  # cointegration_block leve ValueError sinon
            continue

        orders = dict(zip(stat["Variable"], stat["Ordre"]))
        rapports_coint.append(
            cointegration_block(block, orders, block_name=nom, alpha=alpha)
        )

        if johansen and sum(o == "I(1)" for o in orders.values()) >= 2:
            rapports_johansen[nom] = johansen_report(
                block, orders, det_order=det_order, k_ar_diff=k_ar_diff
            )

    stationnarite = (
        pd.concat(rapports_stat, ignore_index=True) if rapports_stat else pd.DataFrame()
    )
    cointegration = (
        pd.concat(rapports_coint, ignore_index=True)
        if rapports_coint
        else pd.DataFrame()
    )

    return {
        "data": data,
        "stationnarite": stationnarite,
        "cointegration": cointegration,
        "suspects": verdicts_suspects(stationnarite),
        "johansen": rapports_johansen,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    res = analyse(sys.argv[1])
    print(res["stationnarite"]["Ordre"].value_counts().to_string())
    print()
    if not res["cointegration"].empty:
        print(res["cointegration"]["Statut"].value_counts().to_string())
        print()
        retenues = res["cointegration"]
        if "Rejet FDR 5%" in retenues.columns:
            retenues = retenues[retenues["Rejet FDR 5%"] == True]  # noqa: E712
            print(f"{len(retenues)} paire(s) cointegrees apres correction FDR")
    print(f"{len(res['suspects'])} verdict(s) d'ordre a verifier")
