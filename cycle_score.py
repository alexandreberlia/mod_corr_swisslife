"""cycle_score.py — Score cyclique 0-100 de l'etat de l'economie.

Prolonge le decoupage en 4 phases (niveau x momentum) par un score continu.

Filiation methodologique
------------------------
- CFNAI (Fed de Chicago) : premiere composante principale d'un grand nombre de
  series, standardisee (moyenne 0, ecart-type 1). Seuils calibres sur les
  recessions NBER : -0.70 (entree probable en recession), +0.20 (sortie),
  +0.70 (surchauffe/pressions inflationnistes). On reprend l'ACP et l'idee de
  seuils calibres empiriquement plutot que fixes a priori.
- CLI de l'OCDE : normalisation puis recalage sur une moyenne de long terme
  fixee a 100, « amplitude adjusted ». On reprend le principe du recalage sur
  une echelle bornee et lisible, mais sur 0-100 plutot que autour de 100.
- Conference Board LEI : distinction leading / coincident. On la reprend en
  separant DEUX sous-scores plutot qu'un seul agregat, parce que le decoupage
  en phases repose precisement sur le croisement de deux dimensions.

Architecture
------------
    SCORE_NIVEAU    ou en est l'activite par rapport a sa tendance   (coincident)
    SCORE_MOMENTUM  dans quelle direction elle va                    (avance)
    SCORE_GLOBAL    moyenne ponderee des deux

Chaque sous-score est construit ainsi :
    1. selection de composantes stationnaires et validees comme predictives
    2. z-score de chaque composante, signe aligne (un z eleve = economie forte)
    3. agregation par premiere composante principale (a la CFNAI)
    4. conversion en 0-100 par la fonction de repartition normale :
           score = 100 * Phi(z)
       Ce choix n'est pas cosmetique. Un min-max serait ecrase par les valeurs
       extremes de 2008 et 2020 ; Phi(z) donne un PERCENTILE HISTORIQUE, donc
       « 40 » se lit « plus faible que 60 % des trimestres depuis 1970 ».
       Un mouvement de 45 a 55 traverse la zone dense de la distribution et
       represente un vrai changement d'etat ; de 85 a 95, on est deja dans la
       queue et l'ecart est moins significatif.

Les seuils de phase ne sont PAS fixes a priori : ils sont calibres sur la
distribution empirique du score dans chaque phase du decoupage de reference.

Dependances : numpy, pandas, scipy
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Composantes
# ---------------------------------------------------------------------------
# signe +1 : une hausse de la serie = economie plus forte
# signe -1 : une hausse = economie plus faible (chomage, inscriptions)

COMPOSANTES_NIVEAU = {
    "OUTFAG Index": +1,           # Commandes à l'industrie
    "CFNAI Index": +1,     #Indice national d'activité de la Fed de Chicago
    "RCHSINDX Index": +1,      #Enquête manufacturière de la Fed de Richmond
    "CONSSENT Index": +1,    #Indice de sentiment des consommateurs
    "NAPMPMI Index": +1,   # Indice ISM manufacturier
    "Housing Permit": +1,   # permis de construire — la variable reelle la plus avancee
    "NHSPATOT Index": +1,     #Ventes de logements neufs
    "LEI YOY Index": +1      #Indicateur avancé de l'économie sur un an
}

COMPOSANTES_MOMENTUM = {
    "CHPMINDX Index": +1,      # Indice PMI de Chicago
    "NAPMPMI Index": +1,      # Indice ISM manufacturier
    "CFNAI Index": +1,     # permis de construire — la variable reelle la plus avancee
    "OUTFGAF Index": +1,             # Commandes à l'industrie
    "PCE CHNC Index": +1,
}


def _z(s: pd.Series, sens: int, win: int | None = None) -> pd.Series:
    """Z-score, oriente. `win` active un z-score glissant (evite le biais de
    reconstruction retrospective : a une date donnee on n'utilise que le passe)."""
    x = s.astype(float) * sens
    if win:
        m = x.rolling(win, min_periods=max(20, win // 3)).mean()
        sd = x.rolling(win, min_periods=max(20, win // 3)).std(ddof=1)
    else:
        m, sd = x.mean(), x.std(ddof=1)
    return (x - m) / sd.replace(0, np.nan) if isinstance(sd, pd.Series) else (x - m) / sd


def _acp_premier_axe(Z: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Premiere composante principale sur donnees standardisees (methode CFNAI).

    Les poids sont le premier vecteur propre de la matrice de correlation. Le
    signe est fixe pour que la composante soit positivement correlee a la moyenne
    simple des composantes — sinon l'ACP peut renvoyer l'axe inverse.
    """
    M = Z.dropna(how="all")
    C = M.corr()
    if C.isna().any().any():
        C = C.fillna(0.0)
        np.fill_diagonal(C.values, 1.0)
    vals, vecs = np.linalg.eigh(C.to_numpy())
    w = vecs[:, np.argmax(vals)]
    if np.corrcoef(np.nan_to_num(M.to_numpy()) @ w,
                   np.nan_to_num(M.mean(axis=1).to_numpy()))[0, 1] < 0:
        w = -w
    poids = pd.Series(w / np.abs(w).sum(), index=M.columns)
    # Moyenne ponderee sur les composantes DISPONIBLES a chaque date : une serie
    # qui demarre tard ne penalise pas les dates anterieures.
    num = (M * poids).sum(axis=1, min_count=1)
    den = M.notna().mul(poids.abs(), axis=1).sum(axis=1)
    comp = num / den.replace(0, np.nan)
    return comp, poids


def _vers_100(z: pd.Series) -> pd.Series:
    """z -> 0-100 par la fonction de repartition normale : un percentile."""
    zz = (z - z.mean()) / z.std(ddof=1)
    return pd.Series(100.0 * stats.norm.cdf(zz.to_numpy()), index=z.index)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_score(panel: pd.DataFrame,
                comp_niveau: dict = None, comp_momentum: dict = None,
                poids_niveau: float = 0.5, rolling: int | None = None,
                couverture_min: float = 0.60) -> pd.DataFrame:
    """Construit les trois scores.

    Parameters
    ----------
    rolling : int | None
        None = z-score plein echantillon (analyse retrospective).
        40   = z-score glissant sur 10 ans, seule version honnete pour un
               backtest, puisqu'a chaque date on n'utilise que le passe.
    couverture_min : part MINIMALE du poids total qui doit etre disponible pour
        qu'un score soit publie. Point critique : les series macro n'ont pas
        toutes la meme date de fin (« bord droit en dents de scie »). Sans ce
        garde-fou, les derniers trimestres seraient calcules sur deux ou trois
        composantes seulement et compares a un historique construit sur dix —
        ce qui produit des valeurs erratiques indiscernables d'un vrai
        mouvement cyclique. Les colonnes couv_* rapportent cette part.
    """
    cn = comp_niveau or COMPOSANTES_NIVEAU
    cm = comp_momentum or COMPOSANTES_MOMENTUM
    out, detail = {}, {}
    for nom, comps in (("niveau", cn), ("momentum", cm)):
        dispo = {k: v for k, v in comps.items() if k in panel.columns}
        if not dispo:
            raise ValueError(f"Aucune composante '{nom}' presente dans le panel.")
        Z = pd.DataFrame({k: _z(panel[k], v, rolling) for k, v in dispo.items()})
        comp, poids = _acp_premier_axe(Z)
        couv = Z.notna().mul(poids.abs(), axis=1).sum(axis=1) / poids.abs().sum()
        comp = comp.where(couv >= couverture_min)
        out[f"score_{nom}"] = _vers_100(comp)
        out[f"z_{nom}"] = (comp - comp.mean()) / comp.std(ddof=1)
        out[f"couv_{nom}"] = couv.round(2)
        detail[nom] = poids
    df = pd.DataFrame(out)
    df["score_global"] = (poids_niveau * df["score_niveau"]
                          + (1 - poids_niveau) * df["score_momentum"])
    df.attrs["poids"] = detail
    return df


def calibrer_coupure(score_niveau, score_momentum, phases) -> dict:
    """Cherche les coupures qui reproduisent le mieux le decoupage de reference.

    Fixer la coupure a 50 est arbitraire : rien ne garantit que la mediane du
    score coincide avec la frontiere du decoupage Hamilton. On balaie donc les
    deux coupures et on retient le couple qui maximise la concordance.
    """
    d = pd.concat([score_niveau.rename("n"), score_momentum.rename("m"),
                   phases.rename("ph")], axis=1).dropna()
    d = d[d.ph != "Choc Covid"]
    best = (50.0, 50.0, -1.0)
    for cn in np.arange(25, 76, 2.5):
        for cm in np.arange(25, 76, 2.5):
            pred = np.where(d.n >= cn,
                            np.where(d.m >= cm, "Explosion", "Ralentissement"),
                            np.where(d.m >= cm, "Reprise", "Decrochage"))
            acc = float((pred == d.ph.to_numpy()).mean())
            if acc > best[2]:
                best = (float(cn), float(cm), acc)
    return dict(coupure_niveau=best[0], coupure_momentum=best[1],
                concordance=round(best[2], 3), n=len(d))


def calibrer_seuils(score: pd.Series, phases: pd.Series) -> pd.DataFrame:
    """Distribution empirique du score dans chaque phase de reference.

    Les seuils sortent des DONNEES, ils ne sont pas poses a priori : c'est la
    demarche du CFNAI, dont les seuils -0.70 / +0.20 / +0.70 ont ete calibres
    sur les recessions NBER, non choisis pour leur elegance.
    """
    d = pd.concat([score.rename("s"), phases.rename("ph")], axis=1).dropna()
    t = d.groupby("ph")["s"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return t[["count", "min", "10%", "25%", "50%", "75%", "90%", "max"]].round(1)


def classer(score_niveau: float, score_momentum: float, seuil: float = 50.0) -> str:
    """Phase deduite du croisement des deux sous-scores — meme logique que le
    decoupage binaire d'origine, mais avec des scores continus."""
    haut = score_niveau >= seuil
    accel = score_momentum >= seuil
    if haut and accel:
        return "Explosion"
    if haut and not accel:
        return "Ralentissement"
    if not haut and accel:
        return "Reprise"
    return "Decrochage"


def libelle(score: float) -> str:
    """Lecture qualitative. Bornes fondees sur les percentiles de la loi normale."""
    if score >= 85:
        return "Surchauffe (top 15 % historique)"
    if score >= 70:
        return "Expansion soutenue"
    if score >= 55:
        return "Expansion moderee"
    if score >= 45:
        return "Neutre / proche de la tendance"
    if score >= 30:
        return "Ralentissement"
    if score >= 15:
        return "Contraction"
    return "Contraction severe (bas 15 % historique)"


# ---------------------------------------------------------------------------
# Poids fixes definis par l'utilisateur
# ---------------------------------------------------------------------------
# Chaque composante est decrite par (signe, poids) :
#   signe : +1 si une hausse de la serie = economie plus forte, -1 sinon
#   poids : importance relative. Les poids sont RENORMALISES pour sommer a 1,
#           donc seules leurs proportions comptent — (2, 1, 1) equivaut a
#           (0.5, 0.25, 0.25).
#
# Pourquoi cette option : l'ACP pondere par la VARIANCE PARTAGEE, ce qui
# penalise mecaniquement toute variable porteuse d'information orthogonale
# (chomage 0.029, pente des taux 0.036 dans la version ACP). Or c'est
# precisement cette orthogonalite qui fait leur interet. Des poids explicites
# rendent le choix editorial visible et discutable, au lieu de le deleguer a
# une decomposition spectrale.
#
# C'est aussi la position du Conference Board pour le LEI : ponderation par
# l'inverse de la volatilite plutot que par ACP, afin qu'aucune composante ne
# domine du seul fait de son amplitude.

POIDS_NIVEAU_DEFAUT = {
    "OUTFAG Index": (+1, 0.1509),           # Commandes à l'industrie
    "CFNAI Index": (+1, 0.1491),   #Indice national d'activité de la Fed de Chicago
    "RCHSINDX Index": (+1, 0.1477),      #Enquête manufacturière de la Fed de Richmond
    "CONSSENT Index": (+1, 0.0698),   #Indice de sentiment des consommateurs
    "NAPMPMI Index": (+1,0.0680),   # Indice ISM manufacturier
    "Housing Permit": (+1, 0.1363),  # permis de construire — la variable reelle la plus avancee
    "NHSPATOT Index": (+1, 0.0698),    #Ventes de logements neufs
    "LEI YOY Index": (+1, 0.1420)     #Indicateur avancé de l'économie sur un an
}

POIDS_MOMENTUM_DEFAUT = {
    "CHPMINDX Index": (+1,0.182378),      # Indice PMI de Chicago
    "NAPMPMI Index": (+1,0.176532),      # Indice ISM manufacturier
    "CFNAI Index": (+1,0.411570),     # permis de construire — la variable reelle la plus avancee
    "OUTFGAF Index": (+1,0.113310),           # Commandes à l'industrie
    "PCE CHNC Index": (+1, 0.116210),
}


def _agreger_poids_fixes(panel: pd.DataFrame, spec: dict,
                         rolling: int | None, couverture_min: float):
    """Agregation a poids imposes.

    spec : {nom_colonne: (signe, poids)}. Les colonnes absentes du panel sont
    signalees puis ignorees — un silence ici masquerait une faute de frappe
    dans un nom de serie.
    """
    manquantes = [k for k in spec if k not in panel.columns]
    if manquantes:
        warnings.warn("Composantes absentes du panel, ignorees : "
                      + ", ".join(manquantes))
    dispo = {k: v for k, v in spec.items() if k in panel.columns}
    if not dispo:
        raise ValueError("Aucune composante presente dans le panel.")

    Z = pd.DataFrame({k: _z(panel[k], s, rolling) for k, (s, _) in dispo.items()})
    w = pd.Series({k: float(p) for k, (_, p) in dispo.items()})
    if (w < 0).any():
        raise ValueError("Les poids doivent etre positifs ; le sens s'exprime "
                         "par le signe (+1/-1), pas par un poids negatif.")
    if w.sum() <= 0:
        raise ValueError("La somme des poids doit etre strictement positive.")
    w = w / w.sum()

    num = (Z * w).sum(axis=1, min_count=1)
    den = Z.notna().mul(w, axis=1).sum(axis=1)
    couv = den / w.sum()
    comp = (num / den.replace(0, np.nan)).where(couv >= couverture_min)
    return comp, w, couv.round(2)


def build_score_poids_fixes(panel: pd.DataFrame,
                            poids_niveau_spec: dict = None,
                            poids_momentum_spec: dict = None,
                            poids_niveau: float = 0.5,
                            rolling: int | None = None,
                            couverture_min: float = 0.60) -> pd.DataFrame:
    """Score cyclique 0-100 a ponderation IMPOSEE (aucune ACP).

    Parameters
    ----------
    poids_niveau_spec, poids_momentum_spec : dict
        {nom_colonne: (signe, poids)}. Voir POIDS_NIVEAU_DEFAUT pour le format.
        Les poids sont renormalises : seules leurs proportions comptent.
    poids_niveau : float
        Part du sous-score de niveau dans le score global.
    rolling : int | None
        None = z-scores plein echantillon (retrospectif).
        40  = fenetre glissante de 10 ans. Contrairement a la version ACP, ce
        reglage suffit ici a rendre le score utilisable en temps reel : les
        poids etant imposes, ils n'incorporent aucune information future.
    couverture_min : float
        Part minimale du poids disponible pour publier un score.

    Returns
    -------
    DataFrame : score_niveau, score_momentum, score_global, z_*, couv_*
        .attrs["poids"]  : poids normalises effectivement appliques
        .attrs["contrib"]: contribution de chaque composante, par date
    """
    specs = {"niveau": poids_niveau_spec or POIDS_NIVEAU_DEFAUT,
             "momentum": poids_momentum_spec or POIDS_MOMENTUM_DEFAUT}
    out, poids, contrib = {}, {}, {}
    for nom, spec in specs.items():
        comp, w, couv = _agreger_poids_fixes(panel, spec, rolling, couverture_min)
        out[f"score_{nom}"] = _vers_100(comp)
        out[f"z_{nom}"] = (comp - comp.mean()) / comp.std(ddof=1)
        out[f"couv_{nom}"] = couv
        poids[nom] = w.round(4)
        dispo = {k: v for k, v in spec.items() if k in panel.columns}
        Z = pd.DataFrame({k: _z(panel[k], s, rolling) for k, (s, _) in dispo.items()})
        contrib[nom] = (Z * w).round(4)
    df = pd.DataFrame(out)
    df["score_global"] = (poids_niveau * df["score_niveau"]
                          + (1 - poids_niveau) * df["score_momentum"])
    df.attrs["poids"] = poids
    df.attrs["contrib"] = contrib
    return df


def decomposer(scores: pd.DataFrame, date, bloc: str = "niveau") -> pd.DataFrame:
    """Contribution de chaque composante au score d'une date donnee.

    Repond a « pourquoi le score a-t-il baisse ? » : chaque ligne donne le
    z-score de la composante, son poids, et le produit des deux. La somme des
    contributions reconstitue le score brut.
    """
    c = scores.attrs.get("contrib", {}).get(bloc)
    if c is None:
        raise ValueError("Scores construits sans contributions "
                         "(utilisez build_score_poids_fixes).")
    d = pd.Period(date, freq="Q") if not isinstance(date, pd.Period) else date
    if d not in c.index:
        raise KeyError(f"{d} absent. Plage : {c.index.min()} -> {c.index.max()}")
    w = scores.attrs["poids"][bloc]
    row = c.loc[d]
    t = pd.DataFrame({"poids": w, "contribution": row})
    t["z_composante"] = (t.contribution / t.poids.replace(0, np.nan)).round(2)
    t["part_%"] = (100 * t.contribution / t.contribution.sum()).round(1)
    return t.sort_values("contribution").round(4)


# ---------------------------------------------------------------------------
# Configuration a trois blocs (avance / coincident / retarde)
# ---------------------------------------------------------------------------
# Reprend la tripartition du Conference Board (leading / coincident / lagging).
# Le bloc RETARDE — inflation et taux directeur — n'est pas un bloc d'activite :
# il monte APRES l'acceleration et redescend APRES le retournement. Il ne mesure
# donc pas ou en est le cycle mais A QUEL POINT IL EST AVANCE. Un bloc retarde
# eleve alors que le bloc avance flechit est la signature classique de fin de
# cycle : la banque centrale resserre encore alors que les commandes ralentissent
# deja.
#
# TRANSFORMATIONS. Inflation et Fed funds sont non stationnaires en niveau (leur
# glissement annuel porte quarante ans de desinflation). On les prend en
# VARIATION SUR 4 TRIMESTRES : l'acceleration de l'inflation, le resserrement
# cumule sur un an. Le champ transform le precise.
#
# SIGNES. Dans un score CYCLIQUE, une inflation qui accelere et une banque
# centrale qui resserre signalent une economie qui tourne au-dessus de son
# potentiel : signe +1. Ce n'est pas un jugement de bien-etre — c'est une
# mesure de position dans le cycle. Si l'objectif etait un score de « sante
# economique », le signe serait a inverser.

def _appliquer_transform(s: pd.Series, transform: str | None) -> pd.Series:
    if transform in (None, "niveau"):
        return s
    if transform == "d4":
        return s.diff(4)
    if transform == "d1":
        return s.diff()
    if transform == "rdt4":
        return 100.0 * (s / s.shift(4) - 1.0)
    raise ValueError(f"transform inconnu : {transform!r}")


# format : nom_colonne -> (signe, poids, transform)
BLOC_AVANCE = {
    "NAPMPMI Index":  (+1, 0.17, None),    # ISM manufacturier 0.05
    "CHPMINDX Index": (+1, 0.17, None),    # PMI de Chicago 0.05
    "OUTFGAF Index":  (+1, 0.17, None),    # nouvelles commandes 0.05
    "CONSSENT Index": (+1, 0.17, None),    # sentiment des menages (Michigan) 0.05
    "CONCCONF Index": (+1, 0.32, None),    # confiance des menages 0.1
}

BLOC_COINCIDENT = {
    "GDP CYOY Index": (+1, 0.222, None),    # PIB, glissement annuel 0.1
    "IP  YOY Index":  (+1, 0.111, None),    # production industrielle 0.05
    "USURTOT Index":  (-1, 0.333, None),    # chomage (inverse) 0.15
    "PCE CHNC Index": (+1, 0.122, None),    # consommation des menages 0.1
    "SAARTOTL Index": (+1, 0.222, None),    # ventes automobiles 0.05
}

BLOC_RETARDE = {
    "CPI XYOY Index": (+1, 0.2, "d4"),    # acceleration de l'inflation sous-jacente 0.05
    "PCE CYOY Index": (+1, 0.2, "d4"),    # idem, deflateur PCE 0.05
    "FED FUNDS":      (+1, 0.6, "d4"),    # resserrement cumule sur un an 0.15
}


def _agreger_3champs(panel, spec, rolling, couverture_min):
    manquantes = [k for k in spec if k not in panel.columns]
    if manquantes:
        warnings.warn("Absentes du panel, ignorees : " + ", ".join(manquantes))
    dispo = {k: v for k, v in spec.items() if k in panel.columns}
    if not dispo:
        raise ValueError("Aucune composante presente dans le panel.")
    Z = pd.DataFrame({k: _z(_appliquer_transform(panel[k], tr), sg, rolling)
                      for k, (sg, _, tr) in dispo.items()})
    w = pd.Series({k: float(p) for k, (_, p, _) in dispo.items()})
    w = w / w.sum()
    num = (Z * w).sum(axis=1, min_count=1)
    den = Z.notna().mul(w, axis=1).sum(axis=1)
    couv = den / w.sum()
    comp = (num / den.replace(0, np.nan)).where(couv >= couverture_min)
    return comp, w, couv.round(2), Z


def build_score_3blocs(panel: pd.DataFrame,
                       avance: dict = None, coincident: dict = None,
                       retarde: dict = None,
                       poids_globaux=(0.30, 0.45, 0.25),
                       rolling: int | None = None,
                       couverture_min: float = 0.60) -> pd.DataFrame:
    """Score 0-100 a trois blocs : avance, coincident, retarde.

    Parameters
    ----------
    avance, coincident, retarde : dict
        {nom_colonne: (signe, poids, transform)}. transform vaut None, "d4",
        "d1" ou "rdt4". Les poids sont renormalises par bloc.
    poids_globaux : tuple
        Ponderation (avance, coincident, retarde) dans le score global. Le bloc
        retarde recoit un poids faible : il documente la position dans le cycle,
        il ne mesure pas son intensite.

    Returns
    -------
    DataFrame : score_avance, score_coincident, score_retarde, score_global,
                z_*, couv_*.  .attrs["poids"] et .attrs["contrib"].
    """
    specs = {"avance": avance or BLOC_AVANCE,
             "coincident": coincident or BLOC_COINCIDENT,
             "retarde": retarde or BLOC_RETARDE}
    out, poids, contrib = {}, {}, {}
    for nom, sp in specs.items():
        comp, w, couv, Z = _agreger_3champs(panel, sp, rolling, couverture_min)
        out[f"score_{nom}"] = _vers_100(comp)
        out[f"z_{nom}"] = (comp - comp.mean()) / comp.std(ddof=1)
        out[f"couv_{nom}"] = couv
        poids[nom] = w.round(4)
        contrib[nom] = (Z * w).round(4)
    df = pd.DataFrame(out)
    pa, pc, pr = poids_globaux
    tot = pa + pc + pr
    df["score_global"] = (pa * df.score_avance + pc * df.score_coincident
                          + pr * df.score_retarde) / tot
    df.attrs["poids"] = poids
    df.attrs["contrib"] = contrib
    return df


def phase_3blocs(score_avance: float, score_coincident: float,
                 seuil: float = 50.0) -> str:
    """Phase deduite du croisement coincident (niveau) x avance (direction)."""
    haut = score_coincident >= seuil
    accel = score_avance >= seuil
    return {(True, True): "Explosion", (True, False): "Ralentissement",
            (False, True): "Reprise", (False, False): "Decrochage"}[(haut, accel)]
