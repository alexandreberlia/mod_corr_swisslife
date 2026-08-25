"""
exemple_protocole.py — les 7 étapes, de bout en bout, sur données réelles.

    python exemple_protocole.py

Tout se règle dans le bloc RÉGLAGES ci-dessous et dans paniers.py.
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
from protocole import lancer, resume_final

# ============================================================== RÉGLAGES
PART_TRAIN = 0.70                  # 70 % entraînement / 30 % test
HORIZONS   = {"court": 15, "moyen": 63, "long": 126}

P = ParamsBS(
    capital=10_000.0,
    n_long=5,
    stop_atr=2.5,                  # stop-loss, en ATR
    trail_atr=None,                # None = stop fixe. Mettre 5.0 pour un trailing.

    seuil_entree=0.0,              # score d'entrée minimal (curseur de sélectivité)
    seuil_sortie=0.0,              # +0.3 = sortie rare | -0.3 = sortie nerveuse
    sensibilite=0.0,               # sortie asymétrique. 0 = DÉSACTIVÉE (comparaison)

    freq_decision=1,               # décisions quotidiennes
    cost_bps=10.0,
    min_titres=20,
    couverture_min=5.0,
)

TICKERS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","CRM",
           "AMD","INTC","CSCO","ADBE","QCOM","TXN","IBM","NOW","INTU","MU",
           "JPM","BAC","WFC","GS","MS","V","MA","AXP","BLK","SCHW",
           "JNJ","PFE","UNH","ABBV","MRK","LLY","TMO","ABT","BMY","AMGN",
           "XOM","CVX","COP","SLB","EOG","PG","KO","PEP","WMT","COST",
           "HD","MCD","NKE","SBUX","TGT","CAT","HON","GE","BA","MMM",
           "UPS","RTX","LMT","DE","UNP","LOW","CVS","T","VZ","CMCSA"]

# ------------------------------------------------------------ 1. données
brut = yf.download(TICKERS, start="2013-01-01", auto_adjust=True,
                   group_by="ticker", progress=False)
prix = {t: brut[t].dropna() for t in TICKERS if t in brut.columns.get_level_values(0)}
print(f"{len(prix)} titres, {len(next(iter(prix.values())))} séances")

panels = construire_panels(prix, Indicateurs, ParamsPF(min_titres=P.min_titres),
                           generateur=features_completes)
close = panels["close"]
split = close.index[int(len(close) * PART_TRAIN)]

# ------------------------------------------- couverture des barrières dures
# Diagnostic AVANT de lancer : si la couverture est nulle, rien ne servira.
print("\nCOUVERTURE DES PANIERS D'ENTRÉE (barrières dures, période train)")
for pk in PANIERS_ENTREE:
    c = pk.couverture(panels, P.min_titres).loc[:split]
    flag = "" if c.mean() >= P.couverture_min else "   <-- TROP FAIBLE"
    print(f"  {pk.nom:<24} moy {c.mean():5.1f} | méd {c.median():4.0f} | "
          f"min {c.min():3.0f}{flag}")
print("\n  Rappel : au-delà de ~7 conditions dures, la couverture médiane tombe à 0.")
print("  Plafond fixé à 4. La sélectivité doit venir du score, pas des barrières.")

# ------------------------------------------------------- 2 à 7. protocole
t0 = time.time()
res = lancer(PANIERS_ENTREE, PANIERS_SORTIE, panels, split,
             horizons=HORIZONS, p=P, seuil_corr=0.8, max_dures=4, verbose=True)
print(f"\n(protocole exécuté en {time.time() - t0:.0f}s)")

print()
print(resume_final(res))

# --------------------------------------------- le panier final, en clair
pe = res["fusion"]["panier_entree"]
ps = res["fusion"]["panier_sortie"]
print("\n" + "=" * 100)
print("PANIER FINAL")
print("=" * 100)
print(f"\nENTRÉE — conditions dures : {dict(pe.dures)}")
for f, (op, s, s2, w) in sorted(pe.specs_score().items(), key=lambda x: -x[1][3]):
    borne = f"{s} → {s2}" if s2 else f"{s}"
    print(f"  {f:<20} {op:<6} {borne:<16} poids {w:.3f}")
print(f"\nSORTIE — souple (aucune condition dure)")
for f, (op, s, s2, w) in sorted(ps.specs_score().items(), key=lambda x: -x[1][3]):
    borne = f"{s} → {s2}" if s2 else f"{s}"
    print(f"  {f:<20} {op:<6} {borne:<16} poids {w:.3f}")

res["train"]["tableau"].drop(columns=["detail"], errors="ignore")\
    .to_csv("bootstrap_train.csv", index=False)
res["test"].to_csv("validation_test.csv", index=False)
print("\nExporté : bootstrap_train.csv, validation_test.csv")
