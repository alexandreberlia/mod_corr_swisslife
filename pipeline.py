"""
pipeline_complet.py — chaîne de bout en bout.
Nécessite : indicateurs.py (ta classe), features.py, portefeuille.py
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

from indicateurs import Indicateurs
from features import features_orientees, calibrer
from portefeuille import ParamsPF, Portefeuille, construire_panels, stats

# ------------------------------------------------------------------ 1. univers
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

brut = yf.download(TICKERS, start="2015-01-01", auto_adjust=True,
                   group_by="ticker", progress=False)
prix = {t: brut[t].dropna() for t in TICKERS
        if t in brut.columns.get_level_values(0)}
print(f"{len(prix)} titres chargés")

# ------------------------------------------------- 2. réglages et panels
# Règle : n_long <= (1 - rang_entree) x taille_univers, avec marge pour l'éligibilité.
p = ParamsPF(
    n_long=5, rang_entree=0.70, rang_sortie=0.40,
    min_titres=8,                       # au plus la moitié de l'univers
    dollar_vol_min=5e6, er_rank_min=0.35, adx_min=12.0,
    stop_atr=2.5, trail_atr=5.0,
    freq_rebal=5, cost_bps=10.0,
)

# generateur=features_orientees est OBLIGATOIRE si les poids viennent de l'IC
panels = construire_panels(prix, Indicateurs, p, generateur=features_orientees)
close = panels["close"]

# --------------------------------- 3. calibration IN-SAMPLE uniquement
# Calibrer sur toute la période puis backtester dessus = look-ahead pur.
split = close.index[int(len(close) * 0.60)]
poids, rapport, corr = calibrer(panels, close, split=split, horizon=20, n_max=5)

print(f"\n=== IC in-sample (jusqu'au {split.date()}) ===")
print(rapport[["feature","IC_moy","ic_bas","ic_haut","t_NW","significatif"]]
      .round(3).to_string(index=False))
print("\n=== POIDS CALIBRÉS ===")
for k, v in poids.items():
    print(f"  {k:<12} {v:+.3f}")

if not poids:
    raise SystemExit("Aucune feature significative — élargir l'univers ou l'historique.")

# --------------------------------------------- 4. backtest et carnet
p.poids = poids
pf = Portefeuille(panels, p, SECTEURS)
res = pf.backtest(10_000)

print("\n=== PERFORMANCE ===")
print(stats(res, close=close).round(2).to_string())

oos = res["equity"].loc[split:]
bh_oos = close.pct_change().loc[split:].mean(axis=1).add(1).cumprod()
print(f"\nOUT-OF-SAMPLE ({split.date()} -> fin) :")
print(f"  stratégie   : {(oos.iloc[-1]/oos.iloc[0]-1)*100:+.1f}%")
print(f"  buy & hold  : {(bh_oos.iloc[-1]-1)*100:+.1f}%")
print("  -> seul cet écart a une valeur probante.")

print("\n=== MOTIFS DE SORTIE ===")
print(res["trades"].motif.value_counts().to_string())

print("\n=== ORDRES DU JOUR ===")
print(pf.book(equity=10_000).to_string(index=False))
