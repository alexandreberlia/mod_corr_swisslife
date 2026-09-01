"""relance.py — Relance des analyses sur la datation revisee.

    python relance.py

Ce qui doit etre refait, et pourquoi
------------------------------------
  lead_rank      les phases servent au decoupage par regime ET au bootstrap
                 par episode  -> A REFAIRE
  leadlag_probit l'indicatrice de transition est construite sur les phases
                 -> A REFAIRE
  cycle_model    le modele de duree est entierement fonde sur les phases
                 -> A REFAIRE
  cycle_score    la construction du score n'utilise pas les phases ; seule la
                 CALIBRATION des seuils par phase en depend
                 -> RECALIBRER SEULEMENT
  diagnostics    crible de stationnarite, independant des phases
                 -> INCHANGE
  sensitivity    regression y sur x avec controles, sans phases
                 -> INCHANGE

Duree indicative : 15 a 25 minutes selon la machine, l'essentiel etant les
bootstraps de lead_rank.
"""

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

# ===========================================================================
# 0. CHARGEMENT
# ===========================================================================

def charger_phases(chemin="phases_et_covariables_v2.csv"):
    """Le fichier revise. La colonne 'trimestre' est au format 1950T2."""
    d = pd.read_csv(chemin)
    d.index = pd.PeriodIndex(d["trimestre"].str.replace("T", "Q"), freq="Q")
    return d.drop(columns=["trimestre"])


phases = charger_phases()
panel = pd.read_csv("panel_trimestriel.csv", index_col=0)
panel.index = pd.PeriodIndex(panel.index, freq="Q")

print(f"phases : {phases.index.min()} -> {phases.index.max()} "
      f"({len(phases)} trimestres, {phases.phase.nunique()} phases)")
print(f"panel  : {panel.shape[1]} series\n")

# Crible de stationnarite — inchange, mais necessaire en amont.
from diagnostics import screen_panel

crible = screen_panel(panel)
ok = crible.query("verdict == 'stationnaire'").serie.tolist()
print(f"series stationnaires retenues : {len(ok)}\n")


# ===========================================================================
# 1. CLASSEMENT DES INDICATEURS  (le plus rentable, commencer par la)
# ===========================================================================
print("=" * 70)
print("1. CLASSEMENT DES INDICATEURS")
print("=" * 70)

from lead_rank import rank_leads

CIBLES = {
    "chomage": (panel["USURTOT Index"].diff(4),
                {"USURTOT Index", "NBER", "GDP CQOQ Index"}),
    "pib": (panel["GDP CYOY Index"],
            {"GDP CYOY Index", "GDP CQOQ Index", "NBER"}),
    "sentiment": (panel["NAPMPMI Index"],
                  {"NAPMPMI Index", "CHPMINDX Index", "NAPMNMI Index",
                   "NBER", "GDP CQOQ Index"}),
}

for nom, (y, exclure) in CIBLES.items():
    cand = [c for c in ok if c not in exclure]
    for axe in ["global", "momentum", "niveau"]:
        R = rank_leads(phases, panel[cand], y.rename(nom),
                       grouping=axe, horizons=range(1, 9),
                       n_boot=400, min_obs=100)
        R.to_csv(f"classement_{nom}_{axe}.csv", index=False)
        print(f"  {nom:<10} / {axe:<9} : {R.FDR.sum():2d} retenues sur {len(R)}")

# Ne pas descendre a grouping='phase' : 4 cellules de ~57 trimestres, plus
# rien ne survit a la correction pour tests multiples.


# ===========================================================================
# 2. PROBIT SUR LES TRANSITIONS  (fournit les covariables du modele de duree)
# ===========================================================================
print("\n" + "=" * 70)
print("2. PROBIT SUR LES TRANSITIONS")
print("=" * 70)

from leadlag import leadlag_probit

ep = (phases.phase != phases.phase.shift()).cumsum()

# Deux cibles binaires distinctes. La premiere melange des transitions de
# nature opposee et donne systematiquement un resultat nul ; la seconde isole
# les degradations, seules annoncees par les variables financieres.
sortie = pd.Series(0, index=phases.index)
vers_bas = pd.Series(0, index=phases.index)
for e in ep.unique():
    idx = phases.index[ep == e]
    if e == ep.max():
        continue
    sortie.loc[idx[-1]] = 1
    if phases.phase[ep == e + 1].iloc[0] == "Decrochage":
        vers_bas.loc[idx[-1]] = 1

print(f"  transitions : {int(sortie.sum())} | dont vers Decrochage : "
      f"{int(vers_bas.sum())}\n")

res = []
for c in ok:
    n_ev = int(vers_bas.reindex(panel[c].dropna().index).sum())
    if n_ev < 5:          # sous 5 evenements apres alignement, inexploitable
        continue
    try:
        r = leadlag_probit(panel[c], vers_bas, max_lag=8, min_lag=1, n_boot=200)
    except Exception:
        continue
    j = list(r.lags).index(r.best_lag)
    res.append(dict(variable=c, n_ev=n_ev, avance=r.best_lag,
                    pseudo_R2=round(r.best_stat, 3),
                    coef=round(r.detail["coef"][j], 4),
                    p_glob=round(r.p_global, 4)))
P = pd.DataFrame(res).sort_values("p_glob")
P.to_csv("probit_transitions_v2.csv", index=False)
print(P.head(8).to_string(index=False))
print("\n  Verifier le SIGNE : negatif attendu (indicateur faible -> degradation).")
print("  Les p-values ne sont pas corrigees du balayage : diviser le seuil par "
      f"{len(P)} pour Bonferroni.")


# ===========================================================================
# 3. MODELE DE DUREE
# ===========================================================================
print("\n" + "=" * 70)
print("3. MODELE DE DUREE")
print("=" * 70)

from cycle_model import CycleModel

m = CycleModel().fit(phases.phase)
print(m.summary())

# Covariable : la meilleure du probit, decalee de son avance.
if len(P):
    best = P.iloc[0]
    ex = pd.DataFrame({"x": panel[best.variable].shift(int(best.avance))})
    try:
        m2 = CycleModel(mutualise=True).fit(phases.phase, exog=ex)
        print(f"\n  avec covariable {best.variable} (avance {int(best.avance)}) :")
        pl = m2.pool_
        if pl:
            j = pl["idx_gamma"][0]
            print(f"    coef {pl['params'][j]:+.4f} (se {pl['se'][j]:.4f}, "
                  f"p={pl['pvalues'][j]:.3f}) sur {pl['evenements']} evenements")
            print("   ", m2.test_homogeneite().to_string(index=False))
    except Exception as e:
        print(f"  covariable non estimable : {e}")

# Backtest + recalibration. Le critere de succes est chiffre : la tranche haute
# de calibration doit depasser 44 %, valeur mesuree sans covariable sur
# l'ancienne datation.
bt = m.backtest(phases.phase, debut="1975Q1", H=4)
m.calibrate(bt)
bt.to_csv("backtest_duree_v2.csv", index=False)
h = bt[bt.h == 4]
print(f"\n  backtest : {len(h)} previsions a 4 trimestres")
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, 1.01)]:
    s = h[(h.p_change >= lo) & (h.p_change < hi)]
    if len(s) > 5:
        print(f"    annonce [{lo:.1f};{hi:.1f}) n={len(s):3d} -> "
              f"observe {100 * s.change_reel.mean():.0f} %")
print("\n  " + m.explain(H=4))


# ===========================================================================
# 4. SCORE — RECALIBRATION DES SEUILS SEULEMENT
# ===========================================================================
print("\n" + "=" * 70)
print("4. SCORE : RECALIBRATION DES SEUILS")
print("=" * 70)

from cycle_score import build_score_3blocs, calibrer_seuils

# La construction du score ne depend pas des phases : inutile de la refaire.
S = build_score_3blocs(panel)
S.to_csv("cycle_score_v2.csv")
J = S.join(phases.phase, how="inner").dropna(subset=["score_global"])
print(calibrer_seuils(J.score_global, J.phase).to_string())
print("\n  Ce sont les nouveaux reperes par phase. Les anciens (Decrochage 15,")
print("  Explosion 72) ne valent plus.")

print("\n" + "=" * 70)
print("TERMINE. Fichiers ecrits :")
print("  classement_{cible}_{axe}.csv   9 fichiers")
print("  probit_transitions_v2.csv")
print("  backtest_duree_v2.csv, cycle_score_v2.csv")
print("=" * 70)
