"""
exemple_selection_combi.py — générer des combinaisons, les tester en
CLASSIFICATION BINAIRE (hausse/baisse), garder la meilleure par horizon.

    python exemple_selection_combi.py
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

from indicateurs import Indicateurs
from features import (features_completes, definir_combinaisons, evaluer_binaire,
                      classer_combinaisons, resume_selection, FAMILLES, HORIZONS_DEFAUT)
from portefeuille import ParamsPF, construire_panels

# ---------------------------------------------------------------- 1. univers
# Le bruit décroît en 1/sqrt(n). Sous ~100 titres, rien n'est détectable.
TICKERS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","CRM",
           "AMD","INTC","CSCO","ADBE","QCOM","TXN","IBM","NOW","INTU","MU",
           "JPM","BAC","WFC","GS","MS","V","MA","AXP","BLK","SCHW",
           "JNJ","PFE","UNH","ABBV","MRK","LLY","TMO","ABT","BMY","AMGN",
           "XOM","CVX","COP","SLB","EOG","PG","KO","PEP","WMT","COST",
           "HD","MCD","NKE","SBUX","TGT","CAT","HON","GE","BA","MMM",
           "UPS","RTX","LMT","DE","UNP","LOW","CVS","T","VZ","CMCSA"]

brut = yf.download(TICKERS, start="2014-01-01", auto_adjust=True,
                   group_by="ticker", progress=False)
prix = {t: brut[t].dropna() for t in TICKERS if t in brut.columns.get_level_values(0)}
print(f"{len(prix)} titres chargés")

p = ParamsPF(min_titres=25)
panels = construire_panels(prix, Indicateurs, p, generateur=features_completes)
close = panels["close"]

print("\nFAMILLES D'INDICATEURS DISPONIBLES")
for fam, feats in FAMILLES.items():
    print(f"  {fam:<12} {feats}")

# ------------------------------------------------- 2. TES CONVICTIONS
# "volatilité en expansion + tendance haussière + cours AU NIVEAU de la MM qui
#  monte (pas bien au-dessus : capter la hausse, pas arriver après) + volumes
#  en augmentation"
CONVICTIONS = {
    "vol-expansion + MM qui monte": {
        "score": {
            "mom_glissant":    1.0,   # log-rendements glissants positifs
            "proximite_ema50": 1.0,   # -|prix-MM|/ATR : proche de la MM
            "pente_ema50":     0.8,   # ... mais MM en progression
            "atr_expansion":   0.6,   # volatilité qui augmente
            "vol_expansion":   0.5,   # volumes qui augmentent
        },
        "conditions": {
            "pente_ema50":   (">", 0),    # MM effectivement haussière
            "atr_expansion": (">", 0),    # vol effectivement en hausse
            "vol_expansion": (">", 0),    # volume effectivement en hausse
        },
        "familles": {"momentum": "mom_glissant", "regime": "proximite_ema50",
                     "volatilite": "atr_expansion", "volume": "vol_expansion"},
    },
    "variante juste_au_dessus (+0.3 ATR)": {
        "score": {"mom_glissant": 1.0, "juste_au_dessus": 1.0,
                  "atr_expansion": 0.6, "vol_expansion": 0.5},
        "conditions": {"pente_ema50": (">", 0)},
        "familles": {"momentum": "mom_glissant", "regime": "juste_au_dessus",
                     "volatilite": "atr_expansion", "volume": "vol_expansion"},
    },
    "variante sans condition (score seul)": {
        "score": {"mom_glissant": 1.0, "proximite_ema50": 1.0, "pente_ema50": 0.8,
                  "atr_expansion": 0.6, "vol_expansion": 0.5},
        "familles": {"momentum": "mom_glissant", "regime": "proximite_ema50",
                     "volatilite": "atr_expansion", "volume": "vol_expansion"},
    },
}

# ------------------------------------ 3. FONCTION 1 : générer les combinaisons
# Une feature EXACTEMENT par famille obligatoire -> aucune combinaison n'est
# "trois momentums déguisés". La famille volume est optionnelle.
combis = definir_combinaisons(
    obligatoires=("momentum", "regime", "volatilite"),
    optionnelles=("volume",),
    convictions=CONVICTIONS,
    max_combis=200,
)
print(f"\n{len(combis)} combinaisons à tester "
      f"(dont {len(CONVICTIONS)} convictions manuelles)")

# --------------------------- 4. FONCTION 2 : évaluer UNE combinaison en détail
print("\n" + "=" * 100)
print("ÉVALUATION BINAIRE — ta conviction principale")
print("=" * 100)
ev = evaluer_binaire(combis["vol-expansion + MM qui monte"], panels, close,
                     horizons=(5, 10, 21, 63, 252), q=0.10, min_titres=25)
print(ev[["horizon", "taux_base_%", "prec_hausse_%", "prec_baisse_%", "lift_hausse",
          "exactitude_%", "hasard_%", "edge_pt", "MCC", "ecart_repart_pt",
          "t_NW", "significatif"]].round(2).to_string(index=False))

# ------------------------------- 5. FONCTION 3 : classer et garder la meilleure
print("\n" + "=" * 100)
print("CLASSEMENT — meilleure combinaison par horizon")
print("=" * 100)
res = classer_combinaisons(combis, panels, close,
                           horizons=HORIZONS_DEFAUT, critere="MCC",
                           q=0.10, reference="zero", min_titres=25)
print(resume_selection(res, top=8))

print("\nRETENUES :")
for groupe, nom in res["meilleures"].items():
    print(f"  {groupe:<8} -> {nom}")
    print(f"           familles : "
          f"{combis[nom].get('familles', {})}")

# ------------------------------------------------------- 6. contrôle marché
# reference='mediane' : "hausse" = mieux que la médiane du panel ce jour-là.
# Si l'edge s'effondre, la combinaison ne faisait que suivre le marché.
print("\n" + "=" * 100)
print("CONTRÔLE — même test, mais 'hausse' = surperformer la médiane du panel")
print("=" * 100)
res_rel = classer_combinaisons(combis, panels, close, critere="MCC",
                               q=0.10, reference="mediane", min_titres=25,
                               verbose=False)
for groupe, nom in res_rel["meilleures"].items():
    print(f"  {groupe:<8} -> {nom}")
print("\nSi les combinaisons gagnantes changent complètement entre les deux")
print("références, l'edge absolu venait surtout du bêta marché.")

res["tableau"].to_csv("resultats_combinaisons.csv", index=False)
print("\nDétail complet exporté : resultats_combinaisons.csv")
