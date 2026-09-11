"""
regimes.py — subdivision de l'entraînement par régime d'activité macro.

FORMAT D'ENTRÉE
Un DataFrame à 3 colonnes de DATES (une par régime), au jour le jour. Les mois
sont complets mais ne se suivent pas : janvier 1989 peut être en "high" et
février 1989 en "low".

    high        medium      low
    1989-01-02  1989-02-01  1974-07-01
    1989-01-03  1989-02-02  1974-07-02
    ...

ÉPISODE
Bloc de dates CONSÉCUTIVES d'un même régime. C'est l'unité qui compte
statistiquement : trois fenêtres tirées de l'épisode 1989-1991 partagent le même
contexte de taux, de valorisation et de composition sectorielle. Elles ne sont
PAS trois observations indépendantes. On rapporte donc systématiquement le
nombre d'épisodes à côté du nombre de fenêtres, et une erreur-type groupée par
épisode en plus de la version naïve.

DEUX RISQUES DE LOOK-AHEAD À CONNAÎTRE

  LATENCE     Le PIB du T1 sort fin avril, le CFNAI vers le 25 du mois suivant.
              Un score de mars calé sur des données publiées en avril utilise une
              information indisponible. -> `decalage_mois` (défaut 1).

  RÉVISIONS   Les séries FRED téléchargées aujourd'hui sont les valeurs RÉVISÉES,
              pas celles connues à l'époque. Le PIB 2008 d'aujourd'hui n'est pas
              celui qu'on lisait fin 2008 ; le CFNAI est révisé chaque mois sur
              tout l'historique. Un score bâti dessus "sait" rétrospectivement où
              étaient les récessions, et sépare donc les régimes plus nettement
              qu'il n'était possible en temps réel.
              Le décalage ne corrige PAS ce point. Seules les séries de vintages
              (ALFRED) le feraient. À garder en tête en lisant les résultats.
"""

import numpy as np
import pandas as pd


REGIMES = ("high", "medium", "low")


# ============================================================================
# Chargement
# ============================================================================

def charger_regimes(df: pd.DataFrame, index_prix: pd.DatetimeIndex = None,
                    decalage_mois: int = 1, colonnes: dict = None) -> pd.Series:
    """3 colonnes de dates -> Series date : régime.

    colonnes : {"high": "H", ...} si tes en-têtes diffèrent.
    decalage_mois : retard de publication. Le régime appliqué au mois M est celui
                    calculé pour le mois M - decalage_mois. Corrige la LATENCE
                    (PIB du T1 publié fin avril, CFNAI vers le 25 du mois suivant),
                    PAS les RÉVISIONS — voir l'avertissement en tête de module.

    Le travail se fait au MOIS, puisque les mois sont complets par construction :
    cela évite les collisions de fins de mois qu'un décalage jour par jour
    produirait (31 janvier et 28 janvier tombent tous deux sur le 28 février).
    """
    cols = colonnes or {r: r for r in REGIMES}
    paires = []
    for reg, col in cols.items():
        if col not in df.columns:
            raise KeyError(f"colonne '{col}' absente. Colonnes : {list(df.columns)}")
        d = pd.DatetimeIndex(pd.to_datetime(df[col].dropna()).unique())
        if len(d):
            paires.append(pd.Series(reg, index=d))
    if not paires:
        raise ValueError("aucune date exploitable dans le DataFrame")

    s = pd.concat(paires).sort_index()
    dup = s.index.duplicated(keep=False)
    if dup.any():
        exemples = sorted({str(x.date()) for x in s.index[dup]})[:3]
        raise ValueError(f"{dup.sum()} dates présentes dans plusieurs colonnes "
                         f"(ex. {', '.join(exemples)})")

    # --- passage au mois : une étiquette par mois (la majoritaire) ---
    par_mois = (s.groupby(s.index.to_period("M"))
                 .agg(lambda x: x.value_counts().index[0]))

    if decalage_mois:
        par_mois.index = par_mois.index + decalage_mois

    if index_prix is None:
        return par_mois.rename("regime")

    # --- propagation sur le calendrier boursier ---
    out = pd.Series(par_mois.reindex(index_prix.to_period("M")).to_numpy(),
                    index=index_prix, name="regime")
    return out.ffill()


def resume_regimes(serie: pd.Series) -> pd.DataFrame:
    """Vue d'ensemble : séances, épisodes, durée, couverture temporelle."""
    lignes = []
    for r in REGIMES:
        eps = episodes(serie, r)
        n = int((serie == r).sum())
        lignes.append({
            "regime": r, "seances": n,
            "%_total": n / serie.notna().sum() * 100 if serie.notna().any() else 0,
            "episodes": len(eps),
            "duree_moy_j": np.mean([e[2] for e in eps]) if eps else 0,
            "duree_min_j": min([e[2] for e in eps]) if eps else 0,
            "duree_max_j": max([e[2] for e in eps]) if eps else 0,
            "premier": eps[0][0].date() if eps else None,
            "dernier": eps[-1][1].date() if eps else None,
        })
    return pd.DataFrame(lignes)


# ============================================================================
# Épisodes
# ============================================================================

def episodes(serie: pd.Series, regime: str, duree_min: int = 1) -> list:
    """Blocs de dates CONSÉCUTIVES du régime. Renvoie [(debut, fin, n_seances)]."""
    m = (serie == regime).to_numpy()
    if not m.any():
        return []
    idx = serie.index
    bords = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    debuts, fins = np.where(bords == 1)[0], np.where(bords == -1)[0] - 1
    return [(idx[a], idx[b], b - a + 1)
            for a, b in zip(debuts, fins) if b - a + 1 >= duree_min]


def split_episodes(eps: list, part_train: float = 0.70) -> tuple:
    """Découpe les épisodes d'un régime en train / test.

    Les épisodes sont triés chronologiquement et les PLUS RÉCENTS vont au test.
    On conserve ainsi la logique "on teste sur du postérieur", tout en
    garantissant que CHAQUE régime dispose d'une période de test — ce qu'un
    découpage chronologique global ne garantit pas (si les 30 % récents sont
    entièrement "high", les sous-algos medium et low ne seraient jamais testés).
    """
    if not eps:
        return [], []
    eps = sorted(eps, key=lambda e: e[0])
    # découpe sur le volume de séances, pas sur le nombre d'épisodes
    total = sum(e[2] for e in eps)
    cible, cum, k = total * part_train, 0, 0
    for i, e in enumerate(eps):
        cum += e[2]
        k = i + 1
        if cum >= cible:
            break
    k = min(max(k, 1), len(eps) - 1) if len(eps) > 1 else len(eps)
    return eps[:k], eps[k:]


# ============================================================================
# Fenêtres
# ============================================================================

def fenetres_regime(index: pd.DatetimeIndex, h: int, eps: list,
                    marge_init: int = 300, recouvrement: bool = False,
                    pas: int = None) -> tuple:
    """Fenêtres de h séances tenant ENTIÈREMENT dans un épisode homogène.

    Renvoie (fenetres, ids_episode) — l'id sert à grouper les observations par
    épisode pour l'erreur-type. Deux fenêtres du même épisode ne sont pas
    indépendantes.

    recouvrement=True autorise un pas inférieur à h (utile aux horizons longs où
    les fenêtres disjointes sont trop peu nombreuses). L'écart-type est alors
    sous-estimé : le protocole le signale.
    """
    pos = {d: i for i, d in enumerate(index)}
    saut = pas if (recouvrement and pas) else (max(h // 3, 1) if recouvrement else h)

    fen, ids = [], []
    for k, (d0, d1, _) in enumerate(eps):
        if d0 not in pos or d1 not in pos:
            continue
        a, b = pos[d0], pos[d1]
        a = max(a, marge_init)
        p = a
        while p + h <= b:
            fen.append((index[p], index[p + h]))
            ids.append(k)
            p += saut
    return fen, ids


def erreur_type_groupee(valeurs: np.ndarray, ids: np.ndarray) -> float:
    """Erreur-type groupée par épisode (cluster-robust).

    L'erreur naïve std/sqrt(n) suppose n observations indépendantes. Si les n
    fenêtres se répartissent sur g épisodes seulement, la vraie taille
    d'échantillon est plus proche de g. On agrège donc par épisode.
    """
    v, ids = np.asarray(valeurs, float), np.asarray(ids)
    ok = ~np.isnan(v)
    v, ids = v[ok], ids[ok]
    if len(v) < 2:
        return np.nan
    moyennes = np.array([v[ids == g].mean() for g in np.unique(ids)])
    g = len(moyennes)
    if g < 2:
        return np.nan
    return float(moyennes.std(ddof=1) / np.sqrt(g))
