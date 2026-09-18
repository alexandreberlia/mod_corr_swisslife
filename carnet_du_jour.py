"""
carnet_du_jour.py — LE SCRIPT DE PRODUCTION.

    python carnet_du_jour.py

"Sur la base de la clôture d'hier, qu'est-ce que j'achète et je vends ce matin ?"

Le script rejoue la stratégie depuis DATE_DEPART pour reconstituer les positions
détenues, puis prend la décision de la dernière clôture connue — celle qui
s'exécute à la prochaine ouverture.

Trois modes de choix du panier, via MODE :
  "regimes"   le panier dépend du régime macro du jour (nécessite scoring_macro.csv)
  "protocole" le protocole en 7 étapes sélectionne le panier (lent, ~2 min)
  "manuel"    un couple déclaré dans paniers.py (rapide)
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
from carnet import carnet_du_jour, rapport_carnet

# ============================================================== RÉGLAGES
MODE         = "manuel"          # "manuel" | "protocole" | "regimes"
DATE_DEPART  = "2024-01-02"      # quand le portefeuille a été ouvert
CAPITAL      = 10_000.0
I_ENTREE, I_SORTIE = 0, 0        # mode "manuel" : index dans PANIERS_*

FICHIER_SCORE = "scoring_macro.csv"     # mode "regimes"

P = ParamsBS(capital=CAPITAL, n_long=5, stop_atr=2.5, trail_atr=None,
             seuil_entree=0.0, seuil_sortie=0.0, sensibilite=0.0,
             cost_bps=10.0, min_titres=20)

TICKERS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","CRM",
           "AMD","INTC","CSCO","ADBE","QCOM","TXN","IBM","NOW","INTU","MU",
           "JPM","BAC","WFC","GS","MS","V","MA","AXP","BLK","SCHW",
           "JNJ","PFE","UNH","ABBV","MRK","LLY","TMO","ABT","BMY","AMGN",
           "XOM","CVX","COP","SLB","EOG","PG","KO","PEP","WMT","COST",
           "HD","MCD","NKE","SBUX","TGT","CAT","HON","GE","BA","MMM",
           "UPS","RTX","LMT","DE","UNP","LOW","CVS","T","VZ","CMCSA"]

# ------------------------------------------------------------ 1. données
# Historique large : il amorce les indicateurs (mom_12_1 exige 252 séances) et,
# en mode "protocole"/"regimes", sert à l'entraînement.
DEBUT = str(pd.Timestamp(DATE_DEPART) - pd.DateOffset(years=8))[:10]
brut = yf.download(TICKERS, start=DEBUT, auto_adjust=True,
                   group_by="ticker", progress=False)
prix = {t: brut[t].dropna() for t in TICKERS if t in brut.columns.get_level_values(0)}
derniere = max(df.index[-1] for df in prix.values())
print(f"{len(prix)} titres · dernière clôture disponible : {derniere.date()}")

age = (pd.Timestamp.today().normalize() - derniere).days
if age > 4:
    print(f"  ATTENTION : données vieilles de {age} jours. Vérifier la source.")

# ---------------------------------------------- 2. choisir le couple de paniers
if MODE == "manuel":
    pe, ps = PANIERS_ENTREE[I_ENTREE], PANIERS_SORTIE[I_SORTIE]

elif MODE == "protocole":
    from protocole import lancer
    print("\nProtocole en 7 étapes (entraînement jusqu'à DATE_DEPART)…")
    panels = construire_panels(prix, Indicateurs, ParamsPF(min_titres=P.min_titres),
                               generateur=features_completes)
    t0 = time.time()
    res = lancer(PANIERS_ENTREE, PANIERS_SORTIE, panels,
                 pd.Timestamp(DATE_DEPART), p=P, verbose=False)
    if "ERREUR" in res:
        raise SystemExit(res["ERREUR"])
    pe, ps = res["fusion"]["panier_entree"], res["fusion"]["panier_sortie"]
    print(f"({time.time() - t0:.0f}s) panier fusionné retenu")

elif MODE == "regimes":
    from regimes import charger_regimes
    from protocole_regimes import lancer_tous_regimes, paniers_finaux
    print("\nProtocole par régimes macro…")
    panels = construire_panels(prix, Indicateurs, ParamsPF(min_titres=P.min_titres),
                               generateur=features_completes)
    serie = charger_regimes(pd.read_csv(FICHIER_SCORE),
                            index_prix=panels["close"].index, decalage_mois=1)
    res = lancer_tous_regimes(PANIERS_ENTREE, PANIERS_SORTIE, panels, serie,
                              horizons={"court": 15, "moyen": 63}, p=P, verbose=False)
    finaux = paniers_finaux(res)
    regime = serie.iloc[-1]
    print(f"Régime actuel : {regime.upper()}")
    if regime not in finaux:
        raise SystemExit(f"aucun panier pour le régime '{regime}'")
    pe, ps = finaux[regime]
else:
    raise SystemExit(f"MODE inconnu : {MODE}")

print(f"Paniers : {pe.nom}  /  {ps.nom}")

# ------------------------------------------------------- 3. carnet d'ordres
c = carnet_du_jour(prix, Indicateurs, pe, ps, DATE_DEPART, p=P, capital=CAPITAL)
print()
print(rapport_carnet(c))

# ------------------------------------------------------------ 4. export
if len(c["achats"]):
    c["achats"].to_csv("ordres_achat.csv", index=False)
if len(c["ventes"]):
    c["ventes"].to_csv("ordres_vente.csv", index=False)
c["positions"].to_csv("positions.csv", index=False)
print("Exporté : ordres_achat.csv, ordres_vente.csv, positions.csv")
