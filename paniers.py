"""
paniers.py — définition des paniers d'indicateurs (entrée / sortie).

MÉCANIQUE
    conditions dures  ET strict, 4 max. Barrières de cohérence : "la conviction
                      est-elle vérifiée, oui ou non". Au-delà de ~7 conditions la
                      couverture s'effondre (mesuré : 0 titre médian à 8 conditions
                      sur un univers de 120).
    score             chaque condition devient CONTINUE : au lieu de vrai/faux, on
                      mesure à quel point elle est satisfaite. Sous-score dans
                      [-1, +1], nul exactement au seuil. C'est ce score qui CLASSE
                      les titres ayant passé les barrières.

SOUS-SCORES
    ">"     r >= s : (r-s)/(1-s)      -> 0 au seuil, +1 au maximum
            r <  s : (r-s)/s          -> 0 au seuil, -1 au minimum
    "<"     symétrique
    "bande" +1 au centre, 0 aux bords, négatif au-delà
            -> traduit "ni trop ni trop peu" (cours AU NIVEAU de la MM)

SEUILS
    nombre  seuil absolu sur la valeur brute
    "csNN"  NNe percentile CROSS-SECTIONNEL, recalculé chaque date -> stationnaire

TOUT EST MODIFIABLE dans le bloc CONVICTIONS en bas de fichier.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================================
# Sous-scores continus
# ============================================================================

def _rang(x: pd.DataFrame, min_titres: int = 10) -> pd.DataFrame:
    """Rang percentile cross-sectionnel (0-1), à chaque date."""
    ok = x.notna().sum(axis=1) >= min_titres
    return x.rank(axis=1, pct=True).where(ok, np.nan)


def _seuil_pct(seuil) -> float:
    """'cs60' -> 0.60. Un nombre reste tel quel (seuil absolu)."""
    if isinstance(seuil, str) and seuil.startswith("cs"):
        return float(seuil[2:]) / 100.0
    return float(seuil)


def sous_score(x: pd.DataFrame, op: str, seuil, seuil_haut=None,
               min_titres: int = 10) -> pd.DataFrame:
    """Degré de satisfaction d'une condition, dans [-1, +1].

    op : ">" | "<" | "bande"
    Les seuils "csNN" sont interprétés sur le RANG percentile ; un seuil
    numérique est comparé à la valeur brute (converti en indicatrice ±1).
    """
    if isinstance(seuil, str) and seuil.startswith("cs"):
        r = _rang(x, min_titres)
        s = _seuil_pct(seuil)

        if op == ">":
            return pd.DataFrame(
                np.where(r >= s,
                         (r - s) / max(1 - s, 1e-9),
                         (r - s) / max(s, 1e-9)),
                index=x.index, columns=x.columns).where(r.notna())

        if op == "<":
            return pd.DataFrame(
                np.where(r <= s,
                         (s - r) / max(s, 1e-9),
                         (s - r) / max(1 - s, 1e-9)),
                index=x.index, columns=x.columns).where(r.notna())

        if op == "bande":
            s2 = _seuil_pct(seuil_haut)
            centre, demi = (s + s2) / 2, max((s2 - s) / 2, 1e-9)
            return (1 - (r - centre).abs() / demi).clip(-1, 1).where(r.notna())

        raise ValueError(f"Opérateur inconnu : {op}")

    # --- seuil absolu : pas de gradation possible, on renvoie ±1 ---
    s = float(seuil)
    if op == ">":
        out = np.sign(x - s)
    elif op == "<":
        out = np.sign(s - x)
    elif op == "bande":
        s2 = float(seuil_haut)
        centre, demi = (s + s2) / 2, max((s2 - s) / 2, 1e-9)
        out = (1 - (x - centre).abs() / demi).clip(-1, 1)
    else:
        raise ValueError(f"Opérateur inconnu : {op}")
    return pd.DataFrame(out, index=x.index, columns=x.columns).where(x.notna())


def condition_dure(x: pd.DataFrame, op: str, seuil, seuil_haut=None,
                   min_titres: int = 10) -> pd.DataFrame:
    """Barrière binaire. Même grammaire de seuils que sous_score."""
    if isinstance(seuil, str) and seuil.startswith("cs"):
        r = _rang(x, min_titres)
        s = _seuil_pct(seuil)
        if op == ">":
            return (r > s).where(r.notna(), False)
        if op == "<":
            return (r < s).where(r.notna(), False)
        if op == "bande":
            s2 = _seuil_pct(seuil_haut)
            return ((r >= s) & (r <= s2)).where(r.notna(), False)
    else:
        s = float(seuil)
        if op == ">":
            return (x > s).fillna(False)
        if op == "<":
            return (x < s).fillna(False)
        if op == "bande":
            return ((x >= s) & (x <= float(seuil_haut))).fillna(False)
    raise ValueError(f"Opérateur inconnu : {op}")


# ============================================================================
# Panier
# ============================================================================

@dataclass
class Panier:
    """Un panier d'indicateurs, côté entrée OU côté sortie.

    dures  : {feature: (op, seuil[, seuil_haut])}  -> ET strict, 4 max
    score  : {feature: (op, seuil[, seuil_haut], poids)} -> classement continu
    """
    nom: str
    dures: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    sens: str = "entree"          # "entree" | "sortie"

    # ---------- indicateurs utilisés ----------

    @property
    def indicateurs(self) -> set:
        return set(self.dures) | set(self.score)

    def specs_score(self) -> dict:
        """{feature: (op, seuil, seuil_haut, poids)} normalisé."""
        out = {}
        for f, spec in self.score.items():
            if len(spec) == 4:
                op, s, s2, w = spec
            elif len(spec) == 3:
                op, s, w = spec
                s2 = None
            else:
                raise ValueError(f"score[{f}] mal formé : {spec}")
            out[f] = (op, s, s2, float(w))
        return out

    # ---------- calculs ----------

    def masque(self, panels: dict, min_titres: int = 10) -> pd.DataFrame:
        ref = panels[next(iter(panels))]
        m = pd.DataFrame(True, index=ref.index, columns=ref.columns)
        for f, spec in self.dures.items():
            if f not in panels:
                raise KeyError(f"'{f}' absent des panels")
            op, s = spec[0], spec[1]
            s2 = spec[2] if len(spec) > 2 else None
            m &= condition_dure(panels[f], op, s, s2, min_titres)
        return m & ref.notna()

    def calculer(self, panels: dict, min_titres: int = 10,
                 appliquer_masque: bool = True) -> pd.DataFrame:
        """Score agrégé dans [-1, +1]. NaN là où les barrières ne passent pas."""
        specs = self.specs_score()
        if not specs:
            base = pd.DataFrame(0.0, index=panels["close"].index,
                                columns=panels["close"].columns)
            return base.where(self.masque(panels, min_titres)) if appliquer_masque else base

        acc, total = None, 0.0
        for f, (op, s, s2, w) in specs.items():
            if f not in panels:
                raise KeyError(f"'{f}' absent des panels")
            ss = sous_score(panels[f], op, s, s2, min_titres) * w
            acc = ss if acc is None else acc.add(ss, fill_value=0.0)
            total += abs(w)

        sc = acc / total
        return sc.where(self.masque(panels, min_titres)) if appliquer_masque else sc

    def couverture(self, panels: dict, min_titres: int = 10) -> pd.Series:
        return self.masque(panels, min_titres).sum(axis=1)


# ============================================================================
# CONVICTIONS — bloc à éditer
# ============================================================================
# Conviction : "volatilité en expansion, tendance haussière, cours AU NIVEAU de
# la MM mais qui monte (capter la hausse, pas arriver après), volumes en hausse".
#
# Les conditions dures sont volontairement PEU SÉVÈRES (signe plutôt que
# percentile élevé) : ce sont des barrières de cohérence, pas de sélectivité.
# La sélectivité vient du score.

PANIERS_ENTREE = [
    Panier(
        nom="E1 proximite MM",
        dures={
            "pente_ema50":     (">", 0),          # la MM monte effectivement
            "atr_expansion":   (">", 0),          # la vol augmente effectivement
            "vol_expansion":   (">", 0),          # le volume augmente effectivement
            "proximite_ema50": (">", "cs40"),     # pas parmi les plus éloignés
        },
        score={
            "mom_glissant":    (">",     "cs60",           1.0),
            "pente_ema50":     (">",     "cs55",           0.8),
            "proximite_ema50": ("bande", "cs55", "cs95",   1.0),
            "atr_expansion":   (">",     "cs60",           0.7),
            "vol_regime":      ("bande", "cs40", "cs85",   0.5),
            "vol_expansion":   (">",     "cs55",           0.6),
            "rvol_log":        (">",     "cs50",           0.4),
        },
    ),
    Panier(
        nom="E2 juste au-dessus",
        dures={
            "pente_ema50":   (">", 0),
            "atr_expansion": (">", 0),
            "vol_expansion": (">", 0),
        },
        score={
            "mom_glissant":    (">",     "cs60",           1.0),
            "juste_au_dessus": ("bande", "cs60", "cs95",   1.0),
            "pente_ema50":     (">",     "cs55",           0.8),
            "atr_expansion":   (">",     "cs60",           0.7),
            "vol_expansion":   (">",     "cs55",           0.6),
        },
    ),
    Panier(
        nom="E3 sans volume",
        dures={
            "pente_ema50":     (">", 0),
            "atr_expansion":   (">", 0),
            "proximite_ema50": (">", "cs40"),
        },
        score={
            "mom_glissant":    (">",     "cs60",           1.0),
            "pente_ema50":     (">",     "cs55",           0.8),
            "proximite_ema50": ("bande", "cs55", "cs95",   1.0),
            "atr_expansion":   (">",     "cs60",           0.7),
            "vol_regime":      ("bande", "cs40", "cs85",   0.5),
        },
    ),
]

# Sortie : SOUPLE, aucune condition dure. Le déclenchement se règle par
# `seuil_sortie` (voir bootstrap.py), pas par les seuils individuels.
PANIERS_SORTIE = [
    Panier(
        nom="S1 retournement momentum",
        sens="sortie",
        score={
            "mom_glissant":  ("<", "cs40", 1.0),     # le momentum se retourne
            "pente_ema50":   ("<", "cs35", 0.9),     # la MM s'aplatit
            "vol_expansion": ("<", "cs30", 0.4),     # le volume se tarit
        },
    ),
    Panier(
        nom="S2 exces d'extension",
        sens="sortie",
        score={
            "ext_ema50":     (">", "cs85", 1.0),     # trop loin de la MM
            "atr_expansion": (">", "cs85", 0.6),     # expansion excessive
        },
    ),
    Panier(
        nom="S3 mixte",
        sens="sortie",
        score={
            "mom_glissant":  ("<", "cs40", 1.0),
            "pente_ema50":   ("<", "cs35", 0.8),
            "ext_ema50":     (">", "cs85", 0.8),
            "atr_expansion": (">", "cs85", 0.5),
            "vol_expansion": ("<", "cs30", 0.4),
        },
    ),
]
