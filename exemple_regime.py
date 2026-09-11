"""
exemple_regimes.py — trois sous-algos, un par régime d'activité macro.

    python exemple_regimes.py

ENTRÉE ATTENDUE : un CSV à 3 colonnes de dates (une par régime), au jour le jour.
    high,medium,low
    1989-01-02,1989-02-01,1974-07-01
    ...
Les mois sont complets mais ne se suivent pas.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import pandas as pd
import yfinance as yf

from indicateurs import Indicateurs
from features import features_completes
from portefeuille import ParamsPF, construire_panels
from paniers import PANIERS_ENTREE, PANIERS_SORTIE
from bootstrap import ParamsBS
from regimes import charger_regimes, resume_regimes
from protocole_regimes import (lancer_tous_regimes, resume_regimes_final,
                               paniers_finaux)

# ============================================================== RÉGLAGES
FICHIER_SCORE  = "scoring_macro.csv"     # 3 colonnes de dates
COLONNES       = {"high": "high", "medium": "medium", "low": "low"}
DECALAGE_MOIS  = 1        # latence de publication (PIB ~1 trim., CFNAI ~1 mois)
PART_TRAIN     = 0.70     # part des ÉPISODES (en séances) réservée à l'entraînement
HORIZONS       = {"court": 15, "moyen": 63}   # "long": 126 si les épisodes le permettent

P = ParamsBS(capital=10_000.0, n_long=5, stop_atr=2.5, trail_atr=None,
             seuil_entree=0.0, seuil_sortie=0.0, sensibilite=0.0,
             cost_bps=10.0, min_titres=20)

TICKERS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","JPM","BAC","WFC","GS",
           "JNJ","PFE","UNH","MRK","ABT","XOM","CVX","COP","PG","KO",
           "PEP","WMT","HD","MCD","NKE","CAT","HON","GE","MMM","UPS",
           "IBM","INTC","CSCO","ORCL","TXN","QCOM","ADBE","T","VZ","CMCSA"]

# ------------------------------------------------------------ 1. données
brut = yf.download(TICKERS, start="2005-01-01", auto_adjust=True,
                   group_by="ticker", progress=False)
prix = {t: brut[t].dropna() for t in TICKERS if t in brut.columns.get_level_values(0)}
panels = construire_panels(prix, Indicateurs, ParamsPF(min_titres=P.min_titres),
                           generateur=features_completes)
close = panels["close"]
print(f"{len(prix)} titres, {len(close)} séances "
      f"({close.index[0].date()} → {close.index[-1].date()})")

# ------------------------------------------------------------ 2. régimes
df_score = pd.read_csv(FICHIER_SCORE)
serie = charger_regimes(df_score, index_prix=close.index,
                        decalage_mois=DECALAGE_MOIS, colonnes=COLONNES)

print("\nRÉGIMES SUR LA PÉRIODE COUVERTE PAR LES PRIX")
print(resume_regimes(serie).to_string(index=False))
print(f"""
  AVERTISSEMENTS
  - Décalage appliqué : {DECALAGE_MOIS} mois (latence de publication).
  - Les séries FRED téléchargées aujourd'hui sont RÉVISÉES. Le classement en
    régimes est donc plus net qu'il ne l'était en temps réel. Le décalage ne
    corrige pas ce point (seuls les vintages ALFRED le feraient).
  - L'unité d'indépendance est l'ÉPISODE, pas la fenêtre. Sous 6 épisodes par
    régime, ne rien conclure d'un classement entre paniers.""")

# --------------------------------------------- 3. trois sous-algos + global
t0 = time.time()
res = lancer_tous_regimes(PANIERS_ENTREE, PANIERS_SORTIE, panels, serie,
                          horizons=HORIZONS, p=P, part_train=PART_TRAIN,
                          comparer_global=True, verbose=True)
print(f"\n({time.time() - t0:.0f}s)")

print()
print(resume_regimes_final(res))

# ------------------------------------------------- 4. les paniers de production
finaux = paniers_finaux(res)
print("\n" + "=" * 100)
print("PANIERS DE PRODUCTION — un par régime")
print("=" * 100)
for r, (pe, ps) in finaux.items():
    print(f"\n### RÉGIME {r.upper()}")
    print(f"  ENTRÉE — conditions dures : {dict(pe.dures)}")
    for f, (op, s, s2, w) in sorted(pe.specs_score().items(), key=lambda x: -x[1][3]):
        print(f"    {f:<20} {op:<6} {(f'{s} → {s2}' if s2 else str(s)):<16} "
              f"poids {w:.3f}")
    print(f"  SORTIE")
    for f, (op, s, s2, w) in sorted(ps.specs_score().items(), key=lambda x: -x[1][3]):
        print(f"    {f:<20} {op:<6} {(f'{s} → {s2}' if s2 else str(s)):<16} "
              f"poids {w:.3f}")

# ------------------------------------------------- 5. le panier du jour
regime_actuel = serie.iloc[-1]
print("\n" + "=" * 100)
print(f"RÉGIME ACTUEL : {regime_actuel.upper()}  (au {serie.index[-1].date()})")
if regime_actuel in finaux:
    pe, ps = finaux[regime_actuel]
    print(f"-> panier à appliquer : {pe.nom} / {ps.nom}")
    print("""
Pour rejouer ce panier jour par jour :
    from simulation import simuler_panier
    sim = simuler_panier(prix, "2023-01-01", Indicateurs, pe, ps, p_bs=P)
    print(sim.rapport())""")
else:
    print("-> aucun panier disponible pour ce régime (épisodes trop courts)")

res["test"].to_csv("validation_regimes.csv", index=False)
print("\nExporté : validation_regimes.csv")
