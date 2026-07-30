"""
leadlag.py — Estimation du décalage (avance/retard) entre deux séries,
avec distribution de probabilité sur le décalage estimé.

Convention de signe unique dans tout le module
---------------------------------------------
lag k > 0  <=>  x est EN AVANCE de k périodes sur y
                (on corrèle x[t-k] avec y[t])

lag k < 0  <=>  x est en retard sur y

Trois problèmes traités explicitement
-------------------------------------
1. CORRÉLATION FALLACIEUSE.
   Deux séries autocorrélées exhibent des corrélations croisées élevées même
   indépendantes. La variance de la CCF empirique est gonflée par :

       Σk rho_x(k) rho_y(k)

   (formule de Bartlett).

   Correctif : pré-blanchiment de Box-Jenkins.
   On ajuste un AR(p) sur x, on filtre LES DEUX séries par ce même filtre,
   puis on calcule la CCF sur les résidus.

2. TESTS MULTIPLES.
   Chercher le max de |rho| sur plusieurs décalages puis tester ce max comme
   s'il avait été choisi a priori gonfle massivement le taux de faux positifs.

   Correctif : bootstrap de la statistique

       max_k |rho(k)|

3. INCERTITUDE SUR argmax.
   Le décalage estimé est lui-même une variable aléatoire.

   Correctif : bootstrap stationnaire (Politis-Romano) sur les paires (x, y),
   ce qui permet d'obtenir une distribution empirique de :

       P(avance = k)

Dépendances : numpy, pandas, statsmodels, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _to_period_index(s: pd.Series, freq: str | None):
    """
    Convertit un index temporel en PeriodIndex.

    Renvoie :
        (serie, freq)
    ou :
        (None, None)
    """
    idx = s.index

    if isinstance(idx, pd.PeriodIndex):
        return s, idx.freqstr

    if isinstance(idx, pd.DatetimeIndex):
        f = freq or pd.infer_freq(idx)

        if f is None:
            raise ValueError(
                "Index de dates à pas irrégulier : impossible d'inférer "
                "la fréquence. Passez explicitement freq='Q', 'M', 'A'..."
            )

        return s.to_period(f), s.to_period(f).index.freqstr

    return None, None


def _align(x, y, freq: str | None = None, strict: bool = True):
    """
    Aligne deux séries SUR LES DATES et sur une grille temporelle régulière.

    Les périodes manquantes sont conservées sous forme de NaN plutôt que
    supprimées.

    Returns
    -------
    sx, sy : pd.Series
    mode   : "date" ou "position"
    freq   : fréquence retenue
    """
    sx = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, float))
    sy = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, float))

    sx = sx.astype(float).copy()
    sy = sy.astype(float).copy()

    px, fx = _to_period_index(sx, freq)
    py, fy = _to_period_index(sy, freq)

    if px is None or py is None:
        msg = (
            "ALIGNEMENT POSITIONNEL : au moins une des deux séries n'a pas "
            "d'index temporel. Les décalages seront comptés en LIGNES."
        )

        if strict and (px is not None or py is not None):
            raise ValueError(
                msg +
                " (une seule des deux séries est datée : mélange interdit)"
            )

        warnings.warn(msg)

        n = min(len(sx), len(sy))

        if len(sx) != len(sy):
            raise ValueError(
                f"Longueurs différentes ({len(sx)} vs {len(sy)}) "
                "sans index temporel."
            )

        idx = pd.RangeIndex(n)

        return (
            pd.Series(sx.to_numpy()[:n], index=idx),
            pd.Series(sy.to_numpy()[:n], index=idx),
            "position",
            None
        )

    if fx != fy:
        raise ValueError(
            f"Fréquences incompatibles : x='{fx}', y='{fy}'. "
            "Agrégerez au préalable la série la plus fine."
        )

    for s, nm in ((px, "x"), (py, "y")):
        if s.index.has_duplicates:
            raise ValueError(
                f"Index de {nm} comporte des dates dupliquées."
            )

    lo = max(px.index.min(), py.index.min())
    hi = min(px.index.max(), py.index.max())

    if lo > hi:
        raise ValueError(
            "Les deux séries ne se recouvrent sur aucune période."
        )

    grid = pd.period_range(lo, hi, freq=fx)

    ax = px.reindex(grid)
    ay = py.reindex(grid)

    n_ok = int((ax.notna() & ay.notna()).sum())

    if n_ok < 30:
        raise ValueError(
            f"Seulement {n_ok} périodes communes renseignées."
        )

    n_gap = len(grid) - n_ok

    if n_gap:
        warnings.warn(
            f"{n_gap} période(s) sur {len(grid)} incomplètes entre "
            f"{lo} et {hi}."
        )

    return ax, ay, "date", fx


def _standardise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / np.nanstd(a, ddof=1)


def optimal_block_length(a: np.ndarray) -> float:
    """
    Longueur de bloc moyenne pour le bootstrap stationnaire.
    """
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]

    n = len(a)

    a = _standardise(a)

    rho = np.corrcoef(a[:-1], a[1:])[0, 1]
    rho = np.clip(abs(rho), 0.0, 0.95)

    return max(
        2.0,
        min(
            n / 4.0,
            n ** (1 / 3)
            * (1 + 2 * rho)
            / (1 - rho + 1e-6) ** 0.5
        )
    )
