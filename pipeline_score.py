"""pipeline_score.py — Banc d'essai du scoring niveau / momentum.

    python pipeline_score.py

Construit deux scores separes — un pour la POSITION (au-dessus ou en dessous du
potentiel), un pour la DIRECTION (acceleration ou deceleration) — et les
confronte aux binaires de la datation. Permet de tester des compositions et des
ponderations sans toucher a cycle_score.

POURQUOI DEUX SCORES ET NON UN
-------------------------------
Un score global unique ne peut pas distinguer Reprise de Ralentissement : ce
sont les deux coins opposes de la grille, avec des valeurs globales quasi
identiques. L'information est dans le COUPLE, pas dans la moyenne.

SUR LA PONDERATION — ce que les tests ont montre
-------------------------------------------------
La question pese moins qu'il n'y parait. Mesure precedemment : la correlation
entre un score a poids fixes et un score par ACP atteint 0,985. Un facteur
cyclique commun domine (premier axe = 40 % de la variance), si bien que toute
ponderation raisonnable capte a peu pres la meme chose.

  EQUIPONDERE      Chaque composante est deja standardisee en z-score, donc de
                   variance 1. Ponderer par l'inverse de la volatilite APRES
                   standardisation revient donc exactement a l'equiponderation
                   — la ponderation par la vol n'apporte rien ici, contrairement
                   a un portefeuille ou les actifs ont des vols differentes.

  POIDS FIXES      Choisis a priori sur des arguments economiques. Robuste,
                   verifiable, ne depend d'aucune estimation.

  ACP              Pondere par la variance PARTAGEE : ecrase toute variable
                   porteuse d'information orthogonale. Le chomage y tombe a
                   0,03, la pente a 0,04 — non parce qu'ils comptent peu, mais
                   parce qu'ils bougent differemment des autres.

  DYNAMIQUE        A eviter avec ces donnees. Reestimer les poids dans le temps
                   sur ~50 episodes revient a ajuster du bruit : le test fait
                   sur le S&P a montre qu'une ponderation optimisee en
                   echantillon faisait MOINS BIEN hors echantillon qu'une
                   ponderation fixe (-0,144 contre -0,172).

Recommandation : poids fixes, peu de composantes, choisies a priori. Utiliser
l'importance de Shapley comme controle de coherence, pas comme regle
d'attribution.

CE QU'IL FAUT REGARDER
----------------------
  AUC          capacite a separer les deux etats du binaire. 0,5 = hasard.
  concordance  part des trimestres ou le score du bon cote de 50 correspond au
               binaire de la datation.
  seuil opt.   la coupure qui maximise la concordance — rarement 50 exactement.

Dependances : numpy, pandas, scikit-learn
"""

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


# ---------------------------------------------------------------------------
# Compositions : nom -> (signe, poids, transformation)
# ---------------------------------------------------------------------------
# signe : +1 si une hausse de la serie signale une conjoncture forte, -1 sinon
# poids : relatif, renormalise a l'interieur du bloc
# transf : None, "d4" (variation sur 4 trimestres), "yoy"

BLOC_NIVEAU = {
    "CFNAI Index":    (+1, 2.5, None),   # activite agregee
    "IP  YOY Index":  (+1, 2.0, None),   # production industrielle
    "USURTOT Index":  (-1, 2.0, None),   # chomage : signe negatif
    "NFP TCH Index":  (+1, 1.5, None),   # creations d'emplois
    "PCE CHNC Index": (+1, 1.0, None),   # consommation
}

BLOC_MOMENTUM = {
    "LEI YOY Index":  (+1, 2.5, None),   # indice avance
    "OUTFGAF Index":  (+1, 2.0, None),   # commandes Philly Fed
    "NAPMPMI Index":  (+1, 1.5, None),   # ISM
    "NHSPATOT Index": (+1, 1.5, None),   # mises en chantier
    "T10Y3M":         (+1, 1.0, None),   # pente des taux
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _appliquer_ponderation(spec: dict, panel: pd.DataFrame, mode: str) -> dict:
    """Variante de ponderation, appliquee AU SPEC — jamais au calcul."""
    if mode == "fixe":
        return dict(spec)
    if mode == "egale":
        return {k: (sg, 1.0, tr) for k, (sg, _, tr) in spec.items()}
    if mode == "vol":
        out = {}
        for k, (sg, _, tr) in spec.items():
            if k not in panel.columns:
                continue
            sd = panel[k].std()
            if sd and sd > 0:
                out[k] = (sg, 1.0 / sd, tr)
        return out
    raise ValueError("ponderation : 'fixe', 'egale' ou 'vol'")


def construire_paire(panel: pd.DataFrame, spec_niveau: dict, spec_momentum: dict,
                     ponderation: str = "fixe", poids_niveau: float = 0.5,
                     rolling: int | None = None,
                     couverture_min: float = 0.6) -> pd.DataFrame:
    """Les deux scores en un appel, via build_score_poids_fixes.

    C'est LA fonction de production pour l'architecture niveau/momentum a poids
    imposes : elle renvoie les deux sous-scores, le score global, les
    couvertures et les contributions. Verifie identique au bit pres a une
    construction manuelle.

    LIMITE : `build_score_poids_fixes` attend un spec a DEUX champs
    {nom: (signe, poids)} et ne gere pas les transformations. Si une composante
    exige un `d4` ou un `rdt4` — inflation, taux directeur — passez par
    `construire()` ci-dessous, qui s'appuie sur l'agregation a trois champs.
    """
    from cycle_score import build_score_poids_fixes

    sn = _appliquer_ponderation(spec_niveau, panel, ponderation)
    sm = _appliquer_ponderation(spec_momentum, panel, ponderation)
    for lab, sp in (("niveau", sn), ("momentum", sm)):
        mauvais = [k for k, v in sp.items() if len(v) == 3 and v[2] is not None]
        if mauvais:
            raise ValueError(
                f"transformations non supportees ici ({lab}) : {mauvais}. "
                "Utilisez construire() pour ces composantes.")
    to2 = lambda sp: {k: (v[0], v[1]) for k, v in sp.items()}
    return build_score_poids_fixes(panel, poids_niveau_spec=to2(sn),
                                   poids_momentum_spec=to2(sm),
                                   poids_niveau=poids_niveau, rolling=rolling,
                                   couverture_min=couverture_min)


def construire(panel: pd.DataFrame, spec: dict, ponderation: str = "fixe",
               rolling: int | None = None,
               couverture_min: float = 0.6) -> pd.DataFrame:
    """UN seul bloc, avec support des transformations (d4, rdt4).

    S'appuie sur `_agreger_3champs` de cycle_score. A utiliser quand on teste un
    bloc isole, ou quand une composante demande une transformation. Sinon,
    `construire_paire` est plus direct.
    """
    from cycle_score import _agreger_3champs, _vers_100

    sp = _appliquer_ponderation(spec, panel, ponderation)
    comp, w, couv, Z = _agreger_3champs(panel, sp, rolling, couverture_min)
    out = pd.DataFrame({"z": comp, "score": _vers_100(comp), "couverture": couv})
    out.attrs["poids"] = w.round(3).to_dict()
    out.attrs["contrib"] = (Z * w).round(4)
    return out


# ---------------------------------------------------------------------------
# Evaluation contre les binaires de la datation
# ---------------------------------------------------------------------------

def evaluer(score: pd.Series, binaire: pd.Series, nom: str = "") -> dict:
    """AUC, concordance a 50, et seuil optimal."""
    from sklearn.metrics import roc_auc_score

    d = pd.concat([score.rename("s"), binaire.rename("b")], axis=1).dropna()
    if len(d) < 40 or d.b.nunique() < 2:
        return dict(nom=nom, n=len(d), auc=np.nan)
    auc = roc_auc_score(d.b, d.s)
    conc50 = ((d.s >= 50).astype(int) == d.b).mean()
    grille = np.arange(20, 81, 1.0)
    scores = [((d.s >= t).astype(int) == d.b).mean() for t in grille]
    j = int(np.argmax(scores))
    return dict(nom=nom, n=len(d), auc=round(auc, 3),
                concordance_50=round(100 * conc50),
                seuil_opt=round(float(grille[j]), 1),
                concordance_opt=round(100 * scores[j]))


def comparer_ponderations(panel, spec, binaire, nom="") -> pd.DataFrame:
    out = []
    for p in ["fixe", "egale", "vol"]:
        S = construire(panel, spec, ponderation=p)
        r = evaluer(S["score"], binaire, f"{nom} / {p}")
        r["poids"] = S.attrs["poids"]
        out.append(r)
    T = pd.DataFrame(out)
    # correlation entre les trois versions : si elle est tres elevee, le choix
    # de ponderation ne change presque rien — c'est le cas usuel ici
    ss = {p: construire(panel, spec, ponderation=p)["score"] for p in ["fixe", "egale", "vol"]}
    C = pd.DataFrame(ss).corr()
    T.attrs["correlations"] = C.round(3)
    return T


def balayer_composante(panel, spec, binaire, nom="") -> pd.DataFrame:
    """Retire chaque composante a tour de role : laquelle compte vraiment ?"""
    base = evaluer(construire(panel, spec)["score"], binaire, "complet")
    out = [dict(retiree="(aucune)", **{k: base[k] for k in ("auc", "concordance_opt")})]
    for c in list(spec):
        s2 = {k: v for k, v in spec.items() if k != c}
        if not s2:
            continue
        try:
            r = evaluer(construire(panel, s2)["score"], binaire, c)
            out.append(dict(retiree=c, auc=r["auc"],
                            concordance_opt=r["concordance_opt"],
                            delta_auc=round(r["auc"] - base["auc"], 3)))
        except Exception:
            pass
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    panel = pd.read_csv("panel_trimestriel.csv", index_col=0)
    panel.index = pd.PeriodIndex(panel.index, freq="Q")
    ph = pd.read_csv("phases_et_covariables_v2.csv")
    ph.index = pd.PeriodIndex(ph["trimestre"].str.replace("T", "Q"), freq="Q")

    print("=" * 74)
    print("SCORES NIVEAU ET MOMENTUM")
    print("=" * 74)
    for lab, spec, cible in [("NIVEAU", BLOC_NIVEAU, ph["niveau"]),
                             ("MOMENTUM", BLOC_MOMENTUM, ph["momentum"])]:
        S = construire(panel, spec)
        print(f"\n--- {lab} ---")
        print("  poids :", S.attrs["poids"])
        print("  ", evaluer(S["score"], cible, lab))
        print("\n  ponderations comparees :")
        T = comparer_ponderations(panel, spec, cible, lab)
        print(T[["nom", "n", "auc", "concordance_50", "seuil_opt",
                 "concordance_opt"]].to_string(index=False))
        print("\n  correlation entre ponderations :")
        print(T.attrs["correlations"].to_string())
        print("\n  effet du retrait de chaque composante :")
        print(balayer_composante(panel, spec, cible, lab).to_string(index=False))

    print("\n" + "=" * 74)
    print("GRILLE CROISEE — les 4 phases reconstituees par les deux scores")
    print("=" * 74)
    Sn = construire(panel, BLOC_NIVEAU)["score"]
    Sm = construire(panel, BLOC_MOMENTUM)["score"]
    d = pd.concat([Sn.rename("niv"), Sm.rename("mom"), ph["phase"]], axis=1).dropna()
    d["predite"] = np.where(d.niv >= 50,
                            np.where(d.mom >= 50, "Explosion", "Ralentissement"),
                            np.where(d.mom >= 50, "Reprise", "Decrochage"))
    print(pd.crosstab(d.phase, d.predite).to_string())
    print(f"\nconcordance globale : {100 * (d.phase == d.predite).mean():.0f} %")
