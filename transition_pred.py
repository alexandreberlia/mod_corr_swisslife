"""transitions.py — Quelle sera la prochaine phase, et quand.

Decomposition du probleme
-------------------------
    P(etre en phase D dans j trimestres)
        = P(changer avant j)  x  P(destination = D | changement)

Les deux termes sont estimes separement, pour une raison de comptage :

  QUAND      modele de duree (cloglog) sur 40 transitions. C'est le module
             prevision.hazard_phase.
  VERS QUOI  matrice de transition empirique. Ici l'echantillon est bien plus
             mince — 8 a 11 sorties par phase, et parfois 2 seulement dans une
             branche minoritaire.

Pourquoi ne PAS estimer un modele a risques concurrents complet
---------------------------------------------------------------
Un cloglog multinomial exigerait un jeu de coefficients par destination. Or la
matrice de transition observee est en grande partie DETERMINISTE :

    Explosion   -> Ralentissement  10/10
    Decrochage  -> Reprise           8/8
    Reprise     -> Explosion 8/11, Ralentissement 2/11
    Ralentissement -> Decrochage 7/11, Explosion 2/11

Deux transitions sur quatre n'ont aucune variabilite a expliquer, et les
branches minoritaires comptent 2 evenements. Un modele parametrique y serait
entierement determine par ces deux points. On s'en tient donc a la frequence
historique, avec un intervalle de credibilite qui dit honnetement ce que 2
evenements permettent d'affirmer.

Incertitude sur les destinations
--------------------------------
Frequence brute + posterieur bayesien Dirichlet(1,...,1), c'est-a-dire regle de
Laplace : (k+1)/(n+K). Deux vertus. Une destination jamais observee recoit une
probabilite faible mais NON NULLE — l'absence dans 11 episodes ne prouve pas
l'impossibilite. Et l'intervalle de credibilite s'elargit mecaniquement quand
les effectifs sont minces, ce qu'une frequence brute masquerait.

Dependances : numpy, pandas, scipy
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def episodes(phases: pd.Series) -> pd.Series:
    return (phases != phases.shift()).cumsum()


def anciennete(phases: pd.Series) -> pd.Series:
    ep = episodes(phases)
    return ep.groupby(ep).cumcount() + 1


# ---------------------------------------------------------------------------
# Matrice de transition
# ---------------------------------------------------------------------------

def matrice_transition(phases: pd.Series, exclure: tuple = ("Choc Covid",),
                       niveau: float = 0.90) -> dict:
    """Compte les transitions observees et en deduit les probabilites.

    Returns
    -------
    dict : 'comptes', 'probabilites' (posterieur de Laplace), 'intervalles'
    """
    p = phases[~phases.isin(exclure)].dropna()
    ep = episodes(p)
    suite = [(p[ep == e].iloc[0], p[ep == e + 1].iloc[0])
             for e in sorted(ep.unique())[:-1] if (ep == e + 1).any()]
    if not suite:
        raise ValueError("Aucune transition observee.")
    dep = sorted({a for a, _ in suite})
    arr = sorted({b for _, b in suite})
    C = pd.DataFrame(0, index=dep, columns=arr, dtype=int)
    for a, b in suite:
        C.loc[a, b] += 1

    P, IC = C.astype(float).copy(), {}
    a_lo = (1 - niveau) / 2
    for d in dep:
        k = C.loc[d].to_numpy(float)
        n, K = k.sum(), len(arr)
        P.loc[d] = (k + 1.0) / (n + K)           # posterieur de Laplace
        for j, dest in enumerate(arr):
            # marginale Beta du Dirichlet
            lo, hi = stats.beta.ppf([a_lo, 1 - a_lo], k[j] + 1, n - k[j] + K - 1)
            IC[(d, dest)] = (round(float(lo), 3), round(float(hi), 3))
    return dict(comptes=C, probabilites=P.round(3), intervalles=IC,
                n_transitions=len(suite))


# ---------------------------------------------------------------------------
# Prevision complete
# ---------------------------------------------------------------------------

def prevoir_phases(modele_hasard: dict, matrice: dict, phase: str,
                   t_actuel: int, H: int = 6, x: dict | None = None,
                   absorbant: bool = True) -> pd.DataFrame:
    """Repartition de probabilite sur les phases, a chaque horizon.

    Parameters
    ----------
    modele_hasard : sortie de prevision.hazard_phase
    matrice : sortie de matrice_transition
    phase, t_actuel : etat courant et anciennete (en trimestres)
    x : valeurs des covariables du modele de hasard, si celui-ci en comporte
    absorbant : si True, on ne modelise PAS les transitions ulterieures. Une
        fois sortie, la trajectoire s'arrete a la premiere destination. C'est
        volontaire : chainer les transitions supposerait le processus
        markovien, hypothese que la dependance de duree contredit — le hasard
        depend de l'anciennete, pas seulement de la phase.

    Returns
    -------
    DataFrame : une ligne par horizon, une colonne par phase.
    """
    if phase not in modele_hasard:
        raise KeyError(f"Phase '{phase}' non estimee. "
                       f"Disponibles : {list(modele_hasard)}")
    P = matrice["probabilites"]
    if phase not in P.index:
        raise KeyError(f"Phase '{phase}' absente de la matrice de transition.")
    dests = list(P.columns)
    m = modele_hasard[phase]
    b, noms = np.asarray(m["params"]), m["noms"]

    lignes, surv = [], 1.0
    cumul = {d: 0.0 for d in dests}
    for j in range(1, H + 1):
        t = t_actuel + j - 1
        eta = b[0] + b[1] * np.log(t)
        for k, nom in enumerate(noms[2:], start=2):
            if x is None or nom not in x:
                raise ValueError(f"Valeur manquante pour la covariable '{nom}'.")
            eta += b[k] * x[nom]
        hz = 1.0 - np.exp(-np.exp(np.clip(eta, -20, 20)))
        sortie_j = surv * hz            # probabilite de sortir CE trimestre-la
        surv *= (1.0 - hz)
        for d in dests:
            cumul[d] += sortie_j * float(P.loc[phase, d])
        ligne = {"horizon": j, "anciennete": t, "hasard": round(hz, 3),
                 f"reste_{phase}": round(surv, 3)}
        ligne.update({d: round(cumul[d], 3) for d in dests})
        ligne["phase_probable"] = (phase if surv >= max(cumul.values())
                                   else max(cumul, key=cumul.get))
        lignes.append(ligne)
    out = pd.DataFrame(lignes)
    if not absorbant:
        warnings.warn("absorbant=False n'est pas implemente : le chainage des "
                      "transitions supposerait un processus markovien, "
                      "incompatible avec la dependance de duree estimee.")
    return out


def resume(prev: pd.DataFrame, phase: str, matrice: dict,
           horizon_cible: int | None = None) -> str:
    """Formule en clair la prevision a un horizon donne."""
    h = horizon_cible or len(prev)
    r = prev[prev.horizon == h].iloc[0]
    dests = [c for c in prev.columns
             if c not in ("horizon", "anciennete", "hasard", "phase_probable")
             and not c.startswith("reste_")]
    reste = float(r[f"reste_{phase}"])
    best = max(dests, key=lambda d: float(r[d]))
    ic = matrice["intervalles"].get((phase, best), (np.nan, np.nan))
    return (f"A {h} trimestre(s) : {100*(1-reste):.0f} % de chance d'avoir quitte "
            f"'{phase}'. Destination la plus probable : '{best}' "
            f"({100*float(r[best]):.0f} % en absolu ; part conditionnelle "
            f"{100*matrice['probabilites'].loc[phase, best]:.0f} %, "
            f"IC90 [{100*ic[0]:.0f} ; {100*ic[1]:.0f}] %).")


# ---------------------------------------------------------------------------
# Exploitation d'un indicateur avance
# ---------------------------------------------------------------------------

def signal_indicateur(serie: pd.Series, avance: int, coef: float,
                      seuils: tuple = (0.25, 0.75)) -> dict:
    """Traduit la derniere valeur d'un indicateur avance en signal exploitable.

    L'avance k signifie : la valeur observee en t informe sur la transition en
    t+k. Le signal est donc CONNU d'avance sur k trimestres — c'est precisement
    ce qui le rend utilisable.

    `coef` est le coefficient estime sur la transition. Son SIGNE fixe la
    lecture : negatif = une valeur elevee eloigne la transition.
    """
    s = serie.dropna()
    if len(s) < 20:
        raise ValueError(f"Serie trop courte ({len(s)} obs).")
    val = float(s.iloc[-1])
    pct = float((s < val).mean())
    lo, hi = seuils
    if coef < 0:
        etat = ("signal de transition" if pct <= lo
                else "signal de prolongation" if pct >= hi else "neutre")
    else:
        etat = ("signal de prolongation" if pct <= lo
                else "signal de transition" if pct >= hi else "neutre")
    return dict(date=str(s.index[-1]), valeur=round(val, 2),
                percentile=round(100 * pct, 1), avance=avance,
                informe_sur=str(s.index[-1] + avance), signal=etat,
                sens=("valeur elevee = transition eloignee" if coef < 0
                      else "valeur elevee = transition proche"))
