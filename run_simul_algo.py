"""
lancer_simulation.py — "si j'avais lancé l'algo le JJ/MM/AAAA, où en serais-je ?"

    python lancer_simulation.py
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

from indicateurs import Indicateurs
from portefeuille import ParamsPF
from simulation import simuler, test_causalite

# ============================================================ À RENSEIGNER
DATE_DEBUT = "2025-01-01"      # première date où une position peut être ouverte
CAPITAL    = 10_000.0
FREQ_RECAL = 63                # recalibrage des poids tous les N jours (~trimestre)

TICKERS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","JPM","BAC","XOM","CVX",
           "JNJ","PFE","PG","KO","CAT","HON","UNH","V","MA","HD"]
SECTEURS = pd.Series({
    "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","GOOGL":"Tech","META":"Tech",
    "AMZN":"Conso","PG":"Conso","KO":"Conso","HD":"Conso",
    "JPM":"Finance","BAC":"Finance","V":"Finance","MA":"Finance",
    "XOM":"Energie","CVX":"Energie",
    "JNJ":"Sante","PFE":"Sante","UNH":"Sante",
    "CAT":"Indus","HON":"Indus",
})

# ------------------------------------------------------------ 1. données
# On télécharge BIEN AVANT DATE_DEBUT : cet historique sert uniquement à amorcer
# les indicateurs (mom_12_1 exige 252 séances, la calibration IC davantage).
# Il n'est jamais tradé.
DEBUT_HISTO = str(pd.Timestamp(DATE_DEBUT) - pd.DateOffset(years=4))[:10]

brut = yf.download(TICKERS, start=DEBUT_HISTO, auto_adjust=True,
                   group_by="ticker", progress=False)
prix = {t: brut[t].dropna() for t in TICKERS
        if t in brut.columns.get_level_values(0)}
print(f"{len(prix)} titres, {len(next(iter(prix.values())))} séances "
      f"depuis {DEBUT_HISTO}\n")

# ------------------------------------------------------------ 2. réglages
# Règle : n_long <= (1 - rang_entree) x taille_univers, avec marge d'éligibilité.
p = ParamsPF(
    n_long=5, rang_entree=0.70, rang_sortie=0.40,
    min_titres=8, dollar_vol_min=5e6,
    er_rank_min=0.35, adx_min=12.0,
    stop_atr=2.5, trail_atr=5.0,
    freq_rebal=5, cost_bps=10.0,
)

# ------------------------------------------------------------ 3. simulation
sim = simuler(prix, DATE_DEBUT, Indicateurs, p, SECTEURS,
              capital=CAPITAL, freq_recalib=FREQ_RECAL)

print()
print(sim.rapport())

# ------------------------------------------------------ 4. contrôle causalité
print("\n\n=== CONTRÔLE DE CAUSALITÉ ===")
print("Le carnet doit être identique qu'on dispose ou non des données futures.")
poids = list(sim.poids_utilises.values())[-1]
for d in sim.equity.index[::max(1, len(sim.equity)//4)][:4]:
    r = test_causalite(prix, Indicateurs, p, SECTEURS, d, poids)
    print(f"  {d.date()} : {'identique' if r else '*** DIVERGENCE ***'}")

# ------------------------------------------------------ 5. export optionnel
sim.equity.to_csv("equity.csv")
if len(sim.trades):
    sim.trades.to_csv("trades.csv", index=False)
if len(sim.positions):
    sim.positions.to_csv("positions_ouvertes.csv", index=False)
print("\nExporté : equity.csv, trades.csv, positions_ouvertes.csv")
