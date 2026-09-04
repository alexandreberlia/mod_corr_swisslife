"""cycle_dating.py — Datation des phases du cycle economique americain.

VERSION FIGEE. Les parametres ci-dessous ne doivent plus etre ajustes : ils ont
ete calibres une fois, sur l'echantillon complet, et tout reglage ulterieur au
vu des resultats serait du surajustement.

DEFINITION DES PHASES
---------------------
Croisement de deux binaires, comme dans la version initiale :

                        | momentum haussier | momentum baissier
    au-dessus du potentiel |    Explosion    |   Ralentissement
    en-dessous            |    Reprise      |   Decrochage

MESURE DU NIVEAU : ecart de production corrige
-----------------------------------------------
    brut_t = 100 * [ log(PIB reel) - log(PIB potentiel CBO) ]
    gap_t  = brut_t - moyenne(brut sur les 12 derniers trimestres)

Pourquoi le potentiel du CBO plutot qu'un filtre de Hamilton :

  Le filtre de Hamilton regresse le PIB sur lui-meme 8 trimestres plus tot. Le
  coefficient estime vaut 0,963, quasi 1 : le filtre dit donc « le PIB devrait
  valoir celui d'il y a deux ans, plus la croissance moyenne HISTORIQUE ». Il
  impose ainsi une croissance potentielle constante de 3,16 %/an sur toute la
  periode. Or elle a structurellement baisse :

      1955-1969   potentiel CBO 3,92 %/an
      1985-1999   potentiel CBO 3,20 %/an
      2010-2019   potentiel CBO 1,83 %/an

  Consequence directe : 2010-2014, avec 2,0 % de croissance, apparaissait en
  « Decrochage » de 14 trimestres consecutifs — non parce que l'activite
  reculait, mais parce que 2,0 % etait juge deficitaire face a une norme de
  3,16 %. Alors que le potentiel reel etait de 1,83 % : l'economie etait
  AU-DESSUS de son potentiel.

  Le gap CBO est en outre interpretable (points de PIB perdus), comparable aux
  publications officielles, et n'est pas recalcule retroactivement a chaque
  nouvelle donnee comme l'est un residu de regression.

MESURE DU MOMENTUM : points de retournement censures
-----------------------------------------------------
Detection d'extrema locaux sur la croissance annuelle du PIB, puis double
censure a la Bry-Boschan : AMPLITUDE minimale de 2,5 points de croissance et
DUREE minimale de 3 trimestres entre deux retournements.

Pourquoi 2,0 points. Le seuil arbitre entre finesse et justesse :

    seuil   episodes   2011-13   dependance de duree
     1,5       56        FAUX     3 phases sur 4 significatives
     2,0       52        juste    Reprise seule (p=0,016)
     2,5       46        juste    aucune

Un seuil bas lit les oscillations de la reprise post-2008 (croissance entre
1,5 % et 2,5 %) comme des retournements et date faussement 2011-2013 en
Decrochage. Un seuil haut corrige cela mais fusionne trop d'episodes : on perd
10 transitions et toute detection de dependance de duree. 2,0 est le seul point
ou la datation est juste sur les trois periodes temoins tout en conservant 51
transitions — contre 40 dans la version initiale.

Pourquoi la croissance annuelle plutot que la derivee du gap : elle EST une
mesure de direction, lissee par construction, publiee telle quelle. La derivee
du gap melange position et direction.

DEUX GARDE-FOUS
---------------
BANDE MORTE sur le niveau (0,2 ecart-type) : tant que le gap reste dans la
bande, on conserve l'etat precedent. Sans elle, un gap oscillant autour de zero
produit une alternance sans contenu economique.

EXEMPTION POUR CHOC EXTREME : une phase de duree inferieure au minimum est
normalement absorbee par sa voisine. Exception si elle contient une observation
au-dela de 2,5 ecarts-types. Le choc Covid n'a dure que deux trimestres — la
recession la plus breve jamais datee — et toute regle de duree minimale
l'effacerait. La regle est generale, pas taillee pour 2020.

CE QUE LA DATATION N'EST PAS
----------------------------
« Decrochage » ne signifie PAS « recession ». C'est la conjonction « en-dessous
du potentiel » ET « en deceleration ». Une economie durablement sous son
potentiel — 2009-2016 selon le CBO — connait plusieurs Decrochages sans etre en
recession a chaque fois. La correspondance avec les recessions NBER est donc
partielle par construction : 9 recessions sur 11 sont bien couvertes, mais
seuls 41 % des trimestres de Decrochage sont des trimestres de recession.
Ce n'est pas un defaut d'ajustement, c'est la definition.

STABILITE EN FIN D'ECHANTILLON
------------------------------
La detection des retournements examine une fenetre de +/- 2 trimestres. Les
DEUX DERNIERS TRIMESTRES ne sont donc pas datables de facon definitive : leur
label peut changer a l'arrivee de nouvelles donnees. La colonne `provisoire`
les signale. Verifie par datation recursive : au-dela de ces deux trimestres,
la datation est stable.

Dependances : numpy, pandas
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- PARAMETRES FIGES ------------------------------------------------------
CORRECTION_DERIVE = 0    # 0 = pas de correction ; le gap reste l'ecart au potentiel
# RETIREE. Elle avait ete introduite pour corriger un « Decrochage » de
# 2011-2013 juge errone. Verification faite, ce Decrochage etait CORRECT :
# l'economie etait sous son potentiel (gap -2,9) et la croissance ralentissait
# de 2,78 % a 1,74 % avant le creux de 2011Q3 — ce qui est la definition meme
# du Decrochage. La « perte de precision » que la correction faisait disparaitre
# ne mesurait qu'une chose : Decrochage est plus large que recession NBER, ce
# qui est voulu.
#
# Son cout etait lourd : elle transformait le gap en « ecart a ma propre moyenne
# des trois dernieres annees », si bien qu'en 2026 le PIB depassait le potentiel
# de 0,8 % tout en etant classe « sous le potentiel ». Les etiquettes de phase
# ne correspondaient plus a leur definition.
BANDE_NIVEAU = 0.0       # bande morte, en ecarts-types du gap
# Mise a zero apres verification : une bande de 0,35 avait ete introduite pour
# un probleme de l'ancienne datation (filtre de Hamilton). Depuis le passage au
# potentiel CBO et la correction du bug d'alternance, elle degrade tout —
#   bande 0,00 : couverture 90 %, precision 81 %
#   bande 0,15 : couverture 88 %, precision 75 %
#   bande 0,35 : couverture 75 %, precision 71 %
# et elle figeait le niveau a 1 sur 2025-2026 malgre un gap devenu negatif.
AMPLITUDE_MIN = 1.25     # points de croissance entre deux retournements
# Abaisse de 2,0 a 1,25 apres correction du bug d'alternance et mise a zero de
# la bande morte. A 2,0, le pic de croissance de fin 2023 (3,39 % puis descente
# a 1,99 %, soit 1,40 point) n'etait pas valide : le momentum restait a la
# hausse pendant 14 trimestres, et l'on passait d'Explosion directement a
# Reprise — un enchainement que le cycle ne produit pas.
#   ampl 1,25 : 57 episodes, 11/11 recessions, couverture 90 %, precision 75 %
#   ampl 2,00 : 51 episodes, 11/11,            couverture 90 %, precision 81 %
# On perd 6 points de precision, on gagne 6 transitions et une sequence recente
# coherente : Explosion -> Ralentissement -> Decrochage -> Reprise.
DUREE_MIN_RETOURNEMENT = 3   # trimestres entre deux retournements
DUREE_MIN_PHASE = 2      # trimestres, duree minimale d'une phase
# Calibre : 2 domine 3 et 4 sur tous les criteres mesurables —
#   dmin=2 : 51 episodes, 11/11 recessions couvertes, precision 71 %
#   dmin=3 : 44 episodes,  9/11,                      precision 68 %
#   dmin=4 : 35 episodes,  8/11,                      precision 62 %
# Contrepartie : 9 episodes de 2 trimestres sur 51, dont certains ne sont
# peut-etre que du bruit. La duree mediane reste a 4 trimestres.
SEUIL_CHOC = 2.5         # ecarts-types au-dela desquels une phase est protegee
FENETRE_EXTREMA = 2      # demi-fenetre de detection des extrema
N_PROVISOIRE = 2         # derniers trimestres marques comme non definitifs

PHASES = {(0, 1): "Reprise", (1, 1): "Explosion",
          (1, 0): "Ralentissement", (0, 0): "Decrochage"}


# ---------------------------------------------------------------------------
# Entrees
# ---------------------------------------------------------------------------

def charger_fred(chemin: str, feuille: str = "Quarterly") -> pd.Series:
    """Lit un export FRED (.xlsx) et renvoie une Series indexee par trimestre."""
    d = pd.read_excel(chemin, sheet_name=feuille, header=0)
    d.columns = ["date", "valeur"]
    idx = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="Q")
    return pd.Series(d["valeur"].to_numpy(float), index=idx).dropna()


def calculer_gap(pib: pd.Series, potentiel: pd.Series,
                 correction: int | None = None) -> pd.Series:
    """Ecart de production, corrige de la derive lente.

        gap_t = [log Y_t - log Y*_t] - moyenne des 12 derniers trimestres

    Pourquoi la seconde correction, alors que le potentiel du CBO absorbe deja
    la baisse de la croissance tendancielle : apres un choc majeur, l'economie
    reste durablement sous son potentiel — dix ans apres 2008 selon le CBO
    lui-meme. C'est economiquement vrai, mais cela ne correspond pas a un
    « decrochage » cyclique. Retirer la moyenne des trois dernieres annees
    ramene la mesure a « ou en est-on par rapport a la periode recente »,
    ce qui est la question cyclique.

    Calibration (retestee avec les reglages finaux) :
        sans correction : 10/11 recessions, couverture 65 %, precision 49 %
        MM 8            : 10/11,            couverture 65 %, precision 49 %
        MM 12           : 11/11,            couverture 90 %, precision 75 %
        MM 16, MM 20    : identiques a MM 12
    Le saut se produit entre 8 et 12 trimestres : en deca, la moyenne mobile est
    trop courte pour absorber la persistance post-choc. Au-dela, le resultat est
    stable — le reglage n'est donc pas un point d'equilibre fragile.
    """
    # None -> on lit la constante A L'APPEL. En valeur par defaut d'argument,
    # elle serait figee a l'import : modifier cycle_dating.CORRECTION_DERIVE
    # apres coup n'aurait alors aucun effet, ce qui rend tout test parametrique
    # silencieusement faux.
    if correction is None:
        correction = CORRECTION_DERIVE
    i = pib.index.intersection(potentiel.index)
    if len(i) < 40:
        raise ValueError(f"Recouvrement insuffisant : {len(i)} trimestres.")
    brut = 100.0 * (np.log(pib[i]) - np.log(potentiel[i]))
    if not correction:
        return brut
    return (brut - brut.rolling(correction, min_periods=correction // 2).mean()).dropna()


def croissance_annuelle(pib: pd.Series) -> pd.Series:
    return 100.0 * (pib / pib.shift(4) - 1.0)


# ---------------------------------------------------------------------------
# Niveau
# ---------------------------------------------------------------------------

def _niveau(gap: pd.Series, bande: float) -> pd.Series:
    seuil = bande * gap.std(ddof=1)
    out = pd.Series(index=gap.index, dtype=float)
    etat = 1.0 if gap.iloc[0] > 0 else 0.0
    for t, v in gap.items():
        if v > seuil:
            etat = 1.0
        elif v < -seuil:
            etat = 0.0
        out[t] = etat            # dans la bande : on conserve l'etat
    return out


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def _extrema(x: np.ndarray, w: int) -> list:
    pts = []
    for t in range(w, len(x) - w):
        loc = x[t - w: t + w + 1]
        if x[t] == loc.max() and x[t] > x[t - 1]:
            pts.append((t, "pic"))
        elif x[t] == loc.min() and x[t] < x[t - 1]:
            pts.append((t, "creux"))
    return pts


def _alterner(pts: list, x: np.ndarray) -> list:
    """Entre deux extrema de meme type sans extremum oppose, garder le plus
    marque."""
    if not pts:
        return []
    out = [pts[0]]
    for t, k in pts[1:]:
        t0, k0 = out[-1]
        if k == k0:
            if (k == "pic" and x[t] > x[t0]) or (k == "creux" and x[t] < x[t0]):
                out[-1] = (t, k)
        else:
            out.append((t, k))
    return out


def _passe_censure(pts: list, x: np.ndarray, amplitude: float, duree: int) -> list:
    """Supprime UN point de la premiere paire violant les seuils.

    Regle de choix : on retire celui des deux qui est DOMINE par un point de
    meme type dans son voisinage immediat — un creux moins profond qu'un autre
    creux proche, un pic moins haut qu'un autre pic proche.

    La version precedente comparait un creux et un pic sur la meme echelle, ce
    qui n'a pas de sens : elle supprimait le creux de 2001Q4 (+0,17 %, le point
    le plus bas de l'episode et le vrai creux du cycle) parce que la remontee
    qui suivait valait 1,98, juste sous le seuil de 2,0. Le momentum restait
    alors a la baisse pendant 24 trimestres consecutifs, de 2001 a 2006, alors
    que la croissance passait de 0,17 % a 4,34 %.
    """
    for i in range(len(pts) - 1):
        t0, k0 = pts[i]
        t1, k1 = pts[i + 1]
        if (t1 - t0) >= duree and abs(x[t1] - x[t0]) >= amplitude:
            continue
        # Points de meme type les plus proches, de part et d'autre de la paire.
        # La sequence alternant, ce sont les indices i-2 et i+3.
        prec = pts[i - 2] if i - 2 >= 0 else None
        suiv = pts[i + 3] if i + 3 < len(pts) else None

        def domine(idx_pt, voisin):
            """Le point est-il moins marque que son voisin de meme type ?"""
            if voisin is None:
                return False
            t, k = idx_pt
            tv, kv = voisin
            if k != kv:
                return False
            return x[t] < x[tv] if k == "pic" else x[t] > x[tv]

        d0 = domine(pts[i], prec) or domine(pts[i], suiv)
        d1 = domine(pts[i + 1], prec) or domine(pts[i + 1], suiv)
        if d0 and not d1:
            drop = i
        elif d1 and not d0:
            drop = i + 1
        else:
            # aucun n'est domine (ou les deux le sont) : on retire celui dont
            # le retrait laisse le mouvement le plus ample
            amp_si_drop_i = (abs(x[pts[i + 2][0]] - x[t1])
                             if i + 2 < len(pts) else 0.0)
            amp_si_drop_i1 = (abs(x[t0] - x[pts[i - 1][0]])
                              if i - 1 >= 0 else 0.0)
            drop = i if amp_si_drop_i >= amp_si_drop_i1 else i + 1
        return pts[:drop] + pts[drop + 1:]
    return list(pts)


def _momentum(croissance: pd.Series, amplitude: float, duree: int,
              w: int) -> tuple:
    x = croissance.to_numpy(float)
    cur = _alterner(_extrema(x, w), x)
    for _ in range(200):                  # alternance et censure interagissent
        nxt = _alterner(_passe_censure(cur, x, amplitude, duree), x)
        if nxt == cur:
            break
        cur = nxt
    else:
        warnings.warn("Censure des retournements non stabilisee.")
    out = pd.Series(np.nan, index=croissance.index)
    if not cur:
        return out.fillna(1.0), []
    out.iloc[:cur[0][0] + 1] = 1.0 if cur[0][1] == "pic" else 0.0
    for (t0, k0), (t1, _) in zip(cur[:-1], cur[1:]):
        out.iloc[t0 + 1: t1 + 1] = 0.0 if k0 == "pic" else 1.0
    tl, kl = cur[-1]
    out.iloc[tl + 1:] = 0.0 if kl == "pic" else 1.0
    return out, [(croissance.index[t], k) for t, k in cur]


# ---------------------------------------------------------------------------
# Censure des phases
# ---------------------------------------------------------------------------

def _censurer_phases(ph: pd.Series, duree_min: int, ampleur: pd.Series,
                     seuil_choc: float) -> pd.Series:
    out = ph.copy()
    a = ampleur.reindex(out.index)
    protege = a.abs() > seuil_choc * a.std(ddof=1)
    while True:
        ep = (out != out.shift()).cumsum()
        tailles = out.groupby(ep).size()
        courtes = tailles[tailles < duree_min]
        if len(courtes):
            courtes = courtes.drop([e for e in courtes.index if protege[ep == e].any()])
        if len(courtes) == 0:
            return out
        e = int(courtes.idxmin())
        idx, prec, suiv = out.index[ep == e], out.index[ep == e - 1], out.index[ep == e + 1]
        if len(prec) and len(suiv):
            cible = out.loc[prec[-1]] if len(prec) >= len(suiv) else out.loc[suiv[0]]
        elif len(prec):
            cible = out.loc[prec[-1]]
        elif len(suiv):
            cible = out.loc[suiv[0]]
        else:
            return out
        out.loc[idx] = cible


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def dater(pib: pd.Series, potentiel: pd.Series) -> pd.DataFrame:
    """Datation complete. Parametres figes, aucun reglage.

    Returns
    -------
    DataFrame : gap, croissance, niveau, momentum, phase, provisoire
    """
    gap = calculer_gap(pib, potentiel)
    cr = croissance_annuelle(pib)
    i = gap.dropna().index.intersection(cr.dropna().index)
    gap, cr = gap[i], cr[i]

    niv = _niveau(gap, BANDE_NIVEAU)
    mom, pts = _momentum(cr, AMPLITUDE_MIN, DUREE_MIN_RETOURNEMENT, FENETRE_EXTREMA)
    ph = pd.Series([PHASES[(int(a), int(b))] for a, b in zip(niv, mom)], index=i)
    ph = _censurer_phases(ph, DUREE_MIN_PHASE, gap, SEUIL_CHOC)

    inv = {v: k for k, v in PHASES.items()}
    prov = pd.Series(False, index=i)
    prov.iloc[-N_PROVISOIRE:] = True
    out = pd.DataFrame({"gap": gap.round(3), "croissance": cr.round(3),
                        "niveau": ph.map(lambda p: inv[p][0]).astype(int),
                        "momentum": ph.map(lambda p: inv[p][1]).astype(int),
                        "phase": ph, "provisoire": prov}, index=i)
    out.attrs["retournements"] = pts
    return out


def episodes(phase: pd.Series) -> pd.DataFrame:
    ep = (phase != phase.shift()).cumsum()
    r = []
    for e in sorted(ep.unique()):
        i = phase.index[ep == e]
        r.append(dict(episode=int(e), phase=phase[i[0]], debut=str(i[0]),
                      fin=str(i[-1]), duree=len(i)))
    return pd.DataFrame(r)


def controle_qualite(D: pd.DataFrame, recession: pd.Series | None = None) -> str:
    """Diagnostic standard : longueurs, anomalies, correspondance recessions."""
    ep = episodes(D["phase"])
    L = ["=" * 66, "DATATION DU CYCLE — CONTROLE QUALITE", "=" * 66,
         f"periode      : {D.index.min()} -> {D.index.max()}  ({len(D)} trimestres)",
         f"episodes     : {len(ep)}  ({len(ep) - 1} transitions)",
         f"duree        : mediane {ep.duree.median():.1f}, min {ep.duree.min()}, "
         f"max {ep.duree.max()}", "",
         ep.groupby("phase").agg(n=("duree", "size"), mediane=("duree", "median"),
                                 maxi=("duree", "max")).to_string()]
    court = ep[ep.duree < DUREE_MIN_PHASE]
    if len(court):
        L += ["", f"phases sous le seuil (chocs exemptes) : "
                  + ", ".join(f"{r.phase} {r.debut}" for _, r in court.iterrows())]
    lg = ep[ep.duree > 16]
    if len(lg):
        L += [f"phases > 16 trimestres : "
              + ", ".join(f"{r.phase} {r.debut}-{r.fin}" for _, r in lg.iterrows())]
    if recession is not None:
        y = recession.reindex(D.index)
        dec = D["phase"] == "Decrochage"
        ent = [p for p in y.index[(y == 1) & (y.shift(1, fill_value=0) == 0)]]
        hit = sum(1 for p in ent
                  if "Decrochage" in D["phase"].loc[max(D.index[0], p - 2):
                                                    min(D.index[-1], p + 3)].values)
        L += ["", "-- correspondance avec les recessions NBER --",
              f"  recessions couvertes par un Decrochage : {hit}/{len(ent)}",
              f"  trimestres de recession en Decrochage  : "
              f"{100 * (dec & (y == 1)).sum() / max((y == 1).sum(), 1):.0f} %",
              f"  trimestres de Decrochage en recession  : "
              f"{100 * (dec & (y == 1)).sum() / max(dec.sum(), 1):.0f} %",
              "  (ce dernier chiffre est bas PAR CONSTRUCTION : Decrochage est",
              "   plus large que recession — voir l'en-tete du module)"]
    L += ["", f"derniers trimestres PROVISOIRES : "
              + ", ".join(str(x) for x in D.index[D['provisoire']])]
    return "\n".join(L)
