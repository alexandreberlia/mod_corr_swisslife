"""
run_analysis_bloom
==================

Enchaine chargement -> stationnarite -> cointegration, et met les resultats
en forme.

``analyse()`` renvoie deux tableaux lisibles :

* ``res["stationnarite"]`` — une ligne par variable : statistiques et
  p-values ADF/KPSS, accord des deux tests, ordre d'integration, verdict
  "Stationnaire" Oui/Non, et avec quelles variables elle est cointegree.
* ``res["cointegration"]`` — une ligne par paire : p-values dans les deux
  sens, verdict "Cointegre" Oui/Non, significativite apres correction FDR.

Les booleens sont convertis en Oui/Non, les valeurs manquantes en "-", les
p-values arrondies. Les tableaux bruts restent accessibles sous
``res["stationnarite_detail"]`` et ``res["cointegration_detail"]``.

Exemple
-------
    from run_analysis_bloom import analyse

    res = analyse("export.csv", n_series=58, names_file="corrected_file.csv")

    res["stationnarite"]
    res["cointegration"]
    res["cointegration"].query("Cointegre == 'Oui'")
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

__all__ = [
    "analyse",
    "rapport_stationnarite",
    "rapport_cointegration",
    "verdicts_suspects",
    "afficher",
]

MANQUANT = "-"


# --------------------------------------------------------------------------- #
# Mise en forme                                                                #
# --------------------------------------------------------------------------- #
def _oui_non(value) -> str:
    """True -> 'Oui', False -> 'Non', NaN -> '-'."""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return MANQUANT
    return "Oui" if bool(value) else "Non"


def _arrondi(series: pd.Series, decimals: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(decimals)


def rapport_cointegration(detail: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Une ligne par paire, colonnes lisibles."""
    if detail.empty:
        return detail

    out = pd.DataFrame(
        {
            "Bloc": detail["Block"],
            "Variable 1": detail["Variable 1"],
            "Variable 2": detail["Variable 2"],
            "Ordres": detail["Ordre 1"].astype(str) + " / " + detail["Ordre 2"].astype(str),
            "N": detail["Observations"],
            "p (1 vers 2)": _arrondi(detail.get("p-value (1~2)"), decimals),
            "p (2 vers 1)": _arrondi(detail.get("p-value (2~1)"), decimals),
            "Sens concordants": detail.get("Accord des deux sens", pd.NA).map(_oui_non),
            "Cointegre": detail["Cointegre"].map(_oui_non),
            "Significatif FDR": detail.get("Rejet FDR 5%", pd.NA).map(_oui_non),
            "Statut": detail["Statut"],
        }
    )
    out.loc[out["Statut"] != "teste", ["Cointegre", "Sens concordants", "Significatif FDR"]] = MANQUANT
    out["N"] = pd.to_numeric(out["N"], errors="coerce").astype("Int64")
    out = out.sort_values(
        ["Cointegre", "p (1 vers 2)"], ascending=[False, True]
    ).reset_index(drop=True)

    # Seules les colonnes de texte recoivent "-" : les p-values restent
    # numeriques, sinon tout filtrage ou tri dessus casserait.
    texte = out.select_dtypes(include="object").columns
    out[texte] = out[texte].fillna(MANQUANT)
    return out


def rapport_stationnarite(
    detail: pd.DataFrame,
    cointegration: pd.DataFrame | None = None,
    decimals: int = 4,
    max_partenaires: int = 3,
) -> pd.DataFrame:
    """
    Une ligne par variable : les deux tests, leur accord, le verdict.

    Les statistiques affichees sont celles de la specification a constante
    ("c"). La p-value ADF avec tendance ("ct") est ajoutee car une serie
    trend-stationnaire n'est detectee que par celle-ci.
    """
    if detail.empty:
        return detail

    verdict = detail["Verdict niveau (c)"].fillna("")
    accord = verdict.str.contains("accord", case=False, na=False)

    out = pd.DataFrame(
        {
            "Bloc": detail["Block"],
            "Variable": detail["Variable"],
            "N": detail["N"],
            "ADF stat": _arrondi(detail.get("ADF stat (c)"), 3),
            "ADF p": _arrondi(detail.get("ADF p (c)"), decimals),
            "ADF p (tendance)": _arrondi(detail.get("ADF p (ct)"), decimals),
            "KPSS stat": _arrondi(detail.get("KPSS stat (c)"), 3),
            "KPSS p": _arrondi(detail.get("KPSS p (c)"), decimals),
            "Tests d'accord": accord.map(_oui_non),
            "Ordre": detail["Ordre"],
            "Stationnaire": detail["Ordre"].map(
                lambda o: "Oui" if o == "I(0)" else ("Non" if str(o).startswith("I(") else MANQUANT)
            ),
        }
    )

    partenaires: dict[str, list[str]] = {}
    if cointegration is not None and not cointegration.empty:
        retenues = cointegration[cointegration["Cointegre"] == "Oui"]
        for _, row in retenues.iterrows():
            partenaires.setdefault(row["Variable 1"], []).append(row["Variable 2"])
            partenaires.setdefault(row["Variable 2"], []).append(row["Variable 1"])

    def _resume(nom: str) -> str:
        noms = partenaires.get(nom, [])
        if not noms:
            return MANQUANT
        visibles = " | ".join(noms[:max_partenaires])
        reste = len(noms) - max_partenaires
        return visibles + (f" (+{reste})" if reste > 0 else "")

    out["Nb cointegrations"] = out["Variable"].map(lambda n: len(partenaires.get(n, [])))
    out["Cointegree avec"] = out["Variable"].map(_resume)
    out["Remarque"] = (
        detail["Notes"].fillna("").str.slice(0, 70).replace("", MANQUANT)
    )
    return out.reset_index(drop=True)


def verdicts_suspects(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Lignes ou le verdict sur le niveau contredit l'ordre retenu.

    Symptome de la descente de Pantula interrompue par un conflit ADF/KPSS :
    le niveau est declare stationnaire mais l'ordre final ne l'est pas.
    """
    if detail.empty or "Verdict niveau (c)" not in detail.columns:
        return detail.head(0)

    niveau_stationnaire = detail["Verdict niveau (c)"].str.startswith(
        "I(0)", na=False
    ) | detail["Verdict niveau (ct)"].str.startswith("I(0)", na=False)
    return detail.loc[
        niveau_stationnaire & (detail["Ordre"] != "I(0)"),
        ["Block", "Variable", "N", "Ordre", "Verdict niveau (c)",
         "Verdict niveau (ct)", "Notes"],
    ].reset_index(drop=True)


def afficher(res: dict, n: int = 20) -> None:
    """Affichage console des deux tableaux, colonnes non tronquees."""
    with pd.option_context(
        "display.width", 250, "display.max_columns", 40, "display.max_colwidth", 45
    ):
        print("=== STATIONNARITE ===")
        print(res["stationnarite"].head(n).to_string(index=False))
        print("\n=== COINTEGRATION ===")
        print(res["cointegration"].head(n).to_string(index=False))


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def analyse(
    path: str | Path,
    freq: str | None = None,
    alpha: float = 0.05,
    johansen: bool = True,
    det_order: int = 0,
    k_ar_diff: int = 2,
    decimals: int = 4,
    **loader_kwargs,
) -> dict:
    """
    Chaine complete : chargement -> stationnarite -> cointegration.

    Parameters
    ----------
    path : le CSV Bloomberg.
    freq : temporalite des tests. None (defaut) = frequence d'origine, donc
        aucune interpolation : les p-values portent sur des observations
        reelles. "W", "ME", "QE" pour convertir.
    alpha : seuil commun aux tests.
    johansen : lancer aussi le test sur systeme complet, bloc par bloc.
    decimals : arrondi des p-values dans les rapports.
    **loader_kwargs : passes a load_dataset (n_series, names_file, sep, ...).

    Returns
    -------
    dict : "stationnarite" et "cointegration" (tableaux lisibles),
    "stationnarite_detail" et "cointegration_detail" (colonnes brutes),
    "suspects", "johansen", "data".
    """
    data = load_dataset(path, **loader_kwargs)
    if freq:
        data = data.resample(freq)

    stats: list[pd.DataFrame] = []
    coints: list[pd.DataFrame] = []
    rapports_johansen: dict[str, dict] = {}

    for nom, block in data.frames.items():
        stat = stationarity_block(block, block_name=nom, alpha=alpha)
        stats.append(stat)

        if block.shape[1] < 2:  # cointegration_block leve ValueError sinon
            continue

        orders = dict(zip(stat["Variable"], stat["Ordre"]))
        coints.append(cointegration_block(block, orders, block_name=nom, alpha=alpha))

        if johansen and sum(o == "I(1)" for o in orders.values()) >= 2:
            rapports_johansen[nom] = johansen_report(
                block, orders, det_order=det_order, k_ar_diff=k_ar_diff
            )

    stat_detail = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()
    coint_detail = pd.concat(coints, ignore_index=True) if coints else pd.DataFrame()

    coint_lisible = rapport_cointegration(coint_detail, decimals=decimals)
    stat_lisible = rapport_stationnarite(stat_detail, coint_lisible, decimals=decimals)

    return {
        "stationnarite": stat_lisible,
        "cointegration": coint_lisible,
        "stationnarite_detail": stat_detail,
        "cointegration_detail": coint_detail,
        "suspects": verdicts_suspects(stat_detail),
        "johansen": rapports_johansen,
        "data": data,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    afficher(analyse(sys.argv[1]))
