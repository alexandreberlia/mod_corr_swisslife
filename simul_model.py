from indicateurs import Indicateurs
from portefeuille_1 import ParamsPF, construire_panels
from bootstrap import ParamsBS
from features import features_completes
from data_prep import load_macro, prepare_dataset
from windows import make_loaders
from train import TrainConfig, train, predict, load_model
from resultats import rapport
from Portefeuille import (choisir_seuils_portefeuille, backtest_portefeuille,
                         rapport_portefeuille, decisions_portefeuille_du_jour)
from prevision import figer_seance

import json
import pandas as pd
import yfinance as yf


# ============================ RÉGLAGES ============================
ENTRAINER = True            # True = entraîne + choisit les seuils + backtest (une fois)
                            # False = décisions du jour seulement (modèle et seuils déjà sauvés)
TICKERS = ["XLK"]
DATE = "2005-01-01"
HORIZON = 22
FENETRE = 20
FICHIER_MODELE = "modele_proba.pt"
FICHIER_SEUILS = "seuils.json"

cours_now = {"XLK": 182.5}  # cours du moment (votre fonction "dernier cours dispo")
mon_portefeuille = {}       # ex. {"XLK": {"date": "2026-10-20", "prix": 180.1}}

reglages = dict(train_start="2005-01-01", train_end="2016-12-31", val_end="2019-12-31",
                horizon=HORIZON, target="direction", signaux=True,
                indicator_field_level=0,
                regime_col="score_niveau",   # le score sert de régime (contexte)
                # score_min=25,              # décommentez pour n'apprendre que si score > 25
                )
# ==================================================================


def construire_indicateurs(prices_):
    """Indicateurs au format attendu + ATR, à partir d'un dictionnaire de prix."""
    P = ParamsBS(capital=1_000_000, n_long=40, stop_atr=3, trail_atr=0, seuil_entree=0.3,
                 seuil_sortie=0.0, sensibilite=0.0, cost_bps=10.0, min_titres=1)
    pan = construire_panels(prices_, Indicateurs, ParamsPF(min_titres=P.min_titres),
                            generateur=features_completes)
    ind = {"rsi": pan["rsi"],
           "bb_pct": pan["bollinger_pct_b"],
           "rev_5": pan["rev_5"],
           "stoch": pan["stoch"],
           "er": pan["er"],
           "hist": pan["hist"]}               # PAS le score ici : il passe par macro
    return pd.concat(ind, axis=1, names=["Indicator", "Ticker"]), pan["atr"]


if __name__ == "__main__":
    # --- Données brutes (clôtures) ---
    score = load_macro("score_updated.csv", date_col="periode_M")[["score_niveau"]]

    brut = yf.download(tickers=TICKERS, start=DATE, group_by="Ticker",
                       auto_adjust=True, progress=False)
    prices = {}
    for ticker in TICKERS:
        if ticker not in brut.columns.get_level_values("Ticker"):
            print(f"{ticker} absent du téléchargement")
            continue
        df = brut[ticker].copy()
        df.columns.name = None
        df = df.dropna(how="all", subset=["Open", "High", "Low", "Close", "Volume"])
        if df.empty:
            print(f"{ticker} ne contient aucune donnée")
            continue
        prices[ticker] = df

    def options(atr):
        return dict(couts_bps=5, entree="cloture",
                    stop_atr=2, atr=atr,          # stop désactivé : stop_atr=None
                    max_positions=len(prices),     # 1 titre -> 1 place (sinon ligne à 1/10)
                    max_entrees_jour=2,
                    duree_max=HORIZON)             # = horizon de la cible

    # ================= BLOC 1 : une fois, sur les CLÔTURES =================
    if ENTRAINER:
        indicateurs, atr = construire_indicateurs(prices)
        OPTIONS = options(atr)

        data = prepare_dataset(prices, indicateurs, score, **reglages)
        loaders = make_loaders(data, window=FENETRE, batch_size=256)
        cfg = TrainConfig(task="proba", d_model=16, dropout=0.2, patience=8)
        model, history = train(loaders, cfg, save_path=FICHIER_MODELE)
        rapport(FICHIER_MODELE, loaders, split="val", out_dir="resultats_proba", data=data)

        grille = choisir_seuils_portefeuille(predict(model, loaders["val"]), prices,
                                             min_trades=10, **OPTIONS)
        print(grille.head(10).round(3).to_string())
        seuil_achat = float(grille.iloc[0]["seuil_achat"])
        ss = grille.iloc[0]["seuil_sortie"]
        seuil_sortie = None if ss == "aucun (temps seul)" else float(ss)
        print("seuil_achat =", seuil_achat, "| seuil_sortie =", seuil_sortie)
        with open(FICHIER_SEUILS, "w") as f:
            json.dump({"seuil_achat": seuil_achat, "seuil_sortie": seuil_sortie}, f)

        bt = backtest_portefeuille(predict(model, loaders["test"]), prices,
                                   seuil_achat, seuil_sortie, **OPTIONS)
        print(rapport_portefeuille(bt, out_dir="backtest_etf_tech").round(3).to_string())

    # ============ BLOC 2 : chaque jour, au COURS DU MOMENT ============
    prices_jour = figer_seance(prices, cours_now)
    indicateurs_jour, atr_jour = construire_indicateurs(prices_jour)
    OPTIONS_JOUR = options(atr_jour)

    data_jour = prepare_dataset(prices_jour, indicateurs_jour, score, **reglages)
    model, _ = load_model(FICHIER_MODELE)
    with open(FICHIER_SEUILS) as f:
        s = json.load(f)

    dec = decisions_portefeuille_du_jour(model, data_jour, FENETRE, mon_portefeuille,
                                         s["seuil_achat"], s["seuil_sortie"],
                                         prices=prices_jour, **OPTIONS_JOUR)
    print("À VENDRE :\n", dec["vendre"].round(3).to_string())
    print("À GARDER :\n", dec["garder"].round(3).to_string())
    print("À ACHETER :\n", dec["acheter"].round(3).to_string())
    print(dec["toutes les probabilités"].round(3).to_string())
