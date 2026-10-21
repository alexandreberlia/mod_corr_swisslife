"""
Vos règles de lecture technique, au même endroit pour l'analyse ET le modèle.

C'est le SEUL fichier à modifier pour changer un seuil, un sens ou un nom de colonne :
analyse_regimes.py et data_prep.py lisent tous les deux cette liste.

Champs de chaque signal :
  id        : nom court, sans espace ni accent (sert de nom de variable dans le modèle)
  nom       : nom lisible pour les tableaux
  col       : nom de l'indicateur dans votre dictionnaire `panels`
  seuil     : valeur à franchir
  cross     : "up" = franchissement à la hausse, "down" = à la baisse
  sens      : +1 = annonce une hausse du titre, -1 = une baisse
  condition : condition supplémentaire sur le cours (optionnelle)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAUX = [
    {"id": "rsi_up45",    "nom": "RSI ↑45",         "col": "RSI",        "seuil": 45,     "cross": "up",   "sens": +1},
    {"id": "rsi_dn65",    "nom": "RSI ↓65",         "col": "RSI",        "seuil": 65,     "cross": "down", "sens": -1},
    {"id": "bb_up0",      "nom": "BB% ↑0",          "col": "BB_pct",     "seuil": 0.0,    "cross": "up",   "sens": +1},
    {"id": "bb_dn095",    "nom": "BB% ↓0,95",       "col": "BB_pct",     "seuil": 0.95,   "cross": "down", "sens": -1},
    {"id": "rev5_up3",    "nom": "rev_5 ↑0,03",     "col": "rev_5",      "seuil": 0.03,   "cross": "up",   "sens": +1},
    {"id": "rev5_dn25",   "nom": "rev_5 ↓-0,025",   "col": "rev_5",      "seuil": -0.025, "cross": "down", "sens": -1},
    {"id": "stoch_up15",  "nom": "Stoch ↑15",       "col": "stoch_long", "seuil": 15,     "cross": "up",   "sens": +1},
    {"id": "stoch_dn75",  "nom": "Stoch ↓75",       "col": "stoch_long", "seuil": 75,     "cross": "down", "sens": -1},
    {"id": "macd_up",     "nom": "MACD hist ↑-2,5", "col": "MACD_hist",  "seuil": -2.5,   "cross": "up",   "sens": +1},
    {"id": "macd_dn",     "nom": "MACD hist ↓2,5",  "col": "MACD_hist",  "seuil": 2.5,    "cross": "down", "sens": -1},
    {"id": "er_up",       "nom": "ER ↑0,35 & cours > J-10", "col": "ER", "seuil": 0.35,   "cross": "up",   "sens": +1,
     "condition": "cours_sup_j10"},
    {"id": "er_dn",       "nom": "ER ↓0,35 & cours < J-10", "col": "ER", "seuil": 0.35,   "cross": "down", "sens": -1,
     "condition": "cours_inf_j10"},
]

HORIZON = 4                      # séances
SEUIL_BAS, SEUIL_HAUT = 25, 75   # régimes du score : bas < 25, élevé > 75


def franchissement(s: pd.Series, seuil: float, sens: str) -> pd.Series:
    """True le jour où la série franchit le seuil. s : indexée (date, ticker).

    hausse : valeur de la veille < seuil <= valeur du jour
    baisse : valeur de la veille > seuil >= valeur du jour
    La veille est prise à l'intérieur de chaque ticker (jamais d'un titre à l'autre).
    """
    veille = s.groupby(level="ticker").shift(1)
    if sens == "up":
        return (veille < seuil) & (s >= seuil)
    return (veille > seuil) & (s <= seuil)


def calcul_signaux(df: pd.DataFrame, signaux: list[dict] = SIGNAUX,
                   par: str = "nom", verbose: bool = True) -> pd.DataFrame:
    """Une colonne booléenne par signal (True le jour du franchissement).

    df  : (date, ticker), avec "Close" et les indicateurs
    par : "nom" ou "id", pour nommer les colonnes
    """
    close = df["Close"]
    close_j10 = close.groupby(level="ticker").shift(10)
    conditions = {"cours_sup_j10": close > close_j10, "cours_inf_j10": close < close_j10}

    out = pd.DataFrame(index=df.index)
    for sig in signaux:
        if sig["col"] not in df.columns:
            if verbose:
                print(f"  (colonne {sig['col']!r} absente : signal {sig['nom']!r} ignoré)")
            continue
        ev = franchissement(df[sig["col"]], sig["seuil"], sig["cross"])
        if sig.get("condition"):
            ev &= conditions[sig["condition"]]
        out[sig[par]] = ev.fillna(False).astype(bool)
    return out


def memoire_signaux(evenements: pd.DataFrame, demi_vie: float = 2.0,
                    max_jours: int = 10) -> pd.DataFrame:
    """Transforme chaque événement en une variable « fraîcheur du signal ».

    Le jour du franchissement : 1. Les jours suivants, la valeur décroît
    (divisée par 2 toutes les `demi_vie` séances). Après `max_jours` séances
    sans nouveau franchissement : 0.

    Le modèle sait ainsi qu'un signal a eu lieu, ET depuis combien de temps :
    un franchissement d'il y a 2 séances pèse encore sur un horizon de 4 séances.
    """
    out = pd.DataFrame(index=evenements.index)
    for c in evenements.columns:
        ev = evenements[c]
        # position de chaque ligne dans son ticker, et position du dernier événement
        pos = ev.groupby(level="ticker").cumcount()
        last = pos.where(ev).groupby(level="ticker").ffill()
        jours = pos - last
        val = np.power(0.5, jours / demi_vie)
        out[c] = val.where(jours <= max_jours, 0.0).fillna(0.0)
    return out


def regime(score: pd.Series, bas: float = SEUIL_BAS, haut: float = SEUIL_HAUT) -> pd.Series:
    """Libellé du régime : "bas", "neutre" ou "élevé" (NaN si score inconnu)."""
    return pd.Series(np.select([score < bas, score > haut], ["bas", "élevé"], "neutre"),
                     index=score.index).where(score.notna())


if __name__ == "__main__":
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=6), ["A"]],
                                     names=["date", "ticker"])
    s = pd.Series([40, 44, 46, 50, 44, 45], index=idx, dtype=float)
    assert franchissement(s, 45, "up").tolist() == [False, False, True, False, False, True]
    assert franchissement(s, 45, "down").tolist() == [False, False, False, False, True, False]

    ev = pd.DataFrame({"x": [False, True, False, False, False, True]}, index=idx)
    m = memoire_signaux(ev, demi_vie=2, max_jours=3)["x"].round(3).tolist()
    assert m == [0.0, 1.0, 0.707, 0.5, 0.354, 1.0], m
    print("signaux.py OK")
