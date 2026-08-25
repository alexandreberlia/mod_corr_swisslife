"""
lancer_simulation.py — "si j'avais lancé l'algo le JJ/MM/AAAA, où en serais-je ?"

    python lancer_simulation.py

Deux modes, selon MODE ci-dessous :

  "protocole"  (recommandé) exécute le protocole en 7 étapes sur la période
               d'entraînement, récupère le PANIER FUSIONNÉ, puis le rejoue jour
               par jour sur la période de test. C'est le chemin cohérent avec
               paniers.py / bootstrap.py / fusion.py.

  "panier"     part directement d'un couple de paniers déclarés dans paniers.py,
               sans passer par le protocole. Plus rapide, utile pour tester une
               conviction sans sélection préalable.

Dans les deux cas : stop-loss en ATR, décision sur la clôture de t, exécution à
l'ouverture de t+1, positions restantes valorisées au dernier cours SANS ordre
de vente (donc sans frais de sortie).
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
from protocole import lancer
from simulation import simuler_panier

# ============================================================== RÉGLAGES
MODE        = "protocole"      # "protocole" | "panier"
DATE_DEBUT  = "2023-01-01"     # 1re date où une position peut être ouverte
CAPITAL     = 10_000.0

# mode "panier" seulement : index dans PANIERS_ENTREE / PANIERS_SORTIE
I_ENTREE, I_SORTIE = 0, 0

P = ParamsBS(
    capital=CAPITAL,
    n_long=5,
    stop_atr=2.5,
    trail_atr=None,            # 5.0 pour un trailing chandelier
    seuil_entree=0.0,
    seuil_sortie=0.0,
    sensibilite=0.0,           # sortie asymétrique. 0 = désactivée
    cost_bps=10.0,
    min_titres=20,
)

TICKERS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","CRM",
           "AMD","INTC","CSCO","ADBE","QCOM","TXN","IBM","NOW","INTU","MU",
           "JPM","BAC","WFC","GS","MS","V","MA","AXP","BLK","SCHW",
           "JNJ","PFE","UNH","ABBV","MRK","LLY","TMO","ABT","BMY","AMGN",
           "XOM","CVX","COP","SLB","EOG","PG","KO","PEP","WMT","COST",
           "HD","MCD","NKE","SBUX","TGT","CAT","HON","GE","BA","MMM",
           "UPS","RTX","LMT","DE","UNP","LOW","CVS","T","VZ","CMCSA"]

# ------------------------------------------------------------ 1. données
# Historique BIEN ANTÉRIEUR à DATE_DEBUT : il amorce les indicateurs
# (mom_12_1 exige 252 séances) et, en mode "protocole", sert d'entraînement.
DEBUT_HISTO = str(pd.Timestamp(DATE_DEBUT) - pd.DateOffset(years=8))[:10]

brut = yf.download(TICKERS, start=DEBUT_HISTO, auto_adjust=True,
                   group_by="ticker", progress=False)
prix = {t: brut[t].dropna() for t in TICKERS if t in brut.columns.get_level_values(0)}
print(f"{len(prix)} titres depuis {DEBUT_HISTO}")

# ---------------------------------------- 2. obtenir le couple de paniers
if MODE == "protocole":
    print("\nProtocole en 7 étapes sur l'entraînement "
          f"(jusqu'au {DATE_DEBUT})…")
    panels = construire_panels(prix, Indicateurs,
                               ParamsPF(min_titres=P.min_titres),
                               generateur=features_completes)
    t0 = time.time()
    res = lancer(PANIERS_ENTREE, PANIERS_SORTIE, panels,
                 pd.Timestamp(DATE_DEBUT), p=P, verbose=False)
    if "ERREUR" in res:
        raise SystemExit(res["ERREUR"])
    print(f"({time.time() - t0:.0f}s)")

    print("\nCouples retenus par horizon :")
    for g, m in res["train"]["meilleurs"].items():
        print(f"  {g:<7} {m['entree']:<22} / {m['sortie']:<26} "
              f"P&L {m['pnl']:+.2f} %")

    panier_e = res["fusion"]["panier_entree"]
    panier_s = res["fusion"]["panier_sortie"]

    print("\nPANIER FUSIONNÉ — entrée")
    print(f"  conditions dures : {dict(panier_e.dures)}")
    for f, (op, s, s2, w) in sorted(panier_e.specs_score().items(),
                                    key=lambda x: -x[1][3]):
        print(f"    {f:<20} {op:<6} {(f'{s} → {s2}' if s2 else str(s)):<16} "
              f"poids {w:.3f}")
    print("PANIER FUSIONNÉ — sortie")
    for f, (op, s, s2, w) in sorted(panier_s.specs_score().items(),
                                    key=lambda x: -x[1][3]):
        print(f"    {f:<20} {op:<6} {(f'{s} → {s2}' if s2 else str(s)):<16} "
              f"poids {w:.3f}")
else:
    panier_e = PANIERS_ENTREE[I_ENTREE]
    panier_s = PANIERS_SORTIE[I_SORTIE]
    print(f"\nPaniers : {panier_e.nom} / {panier_s.nom}")

# ------------------------------------------------ 3. simulation jour par jour
print("\n" + "=" * 72)
sim = simuler_panier(prix, DATE_DEBUT, Indicateurs, panier_e, panier_s,
                     p_bs=P, capital=CAPITAL)
print()
print(sim.rapport())

# ------------------------------------------------------------ 4. export
sim.equity.to_csv("equity.csv")
if len(sim.trades):
    sim.trades.to_csv("trades.csv", index=False)
if len(sim.positions):
    sim.positions.to_csv("positions_ouvertes.csv", index=False)
print("\nExporté : equity.csv, trades.csv, positions_ouvertes.csv")

if MODE == "protocole":
    print("\nRAPPEL : en mode 'protocole', la période simulée est celle de TEST —")
    print("le panier n'a jamais vu ces données. C'est ce qui rend le P&L lisible.")
