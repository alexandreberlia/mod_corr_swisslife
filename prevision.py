"""
Utiliser un modèle entraîné : probabilité de hausse d'un titre, aujourd'hui.

Deux temps :

  A. UNE FOIS, juste après l'entraînement du modèle, dans le même script :
         sauver_preparation("preparation_h30.pkl", data, window=20, reglages=reglages)
     -> enregistre la normalisation, les variables et les réglages de prepare_dataset,
        indispensables pour préparer de nouvelles données exactement comme à l'entraînement.

  B. CHAQUE FOIS que vous voulez une prévision :
         res = proba_hausse("modele_h30.pt", "preparation_h30.pkl",
                            prices_aapl, panels_aapl, macro, ticker="AAPL")

Ce que donne le modèle : UNE probabilité, celle que la clôture dans H séances
(H = l'horizon d'entraînement, ici 30) soit SUPÉRIEURE à la clôture du dernier jour
disponible. Ce n'est pas une probabilité pour chacune des 30 séances.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# A. Sauvegarde de la préparation (après l'entraînement)
# ---------------------------------------------------------------------------
def sauver_preparation(path: str, data, window: int, reglages: dict,
                       tickers_statiques: list[str] | None = None) -> None:
    """Enregistre tout ce qu'il faut pour préparer de nouvelles données.

    data      : la sortie de prepare_dataset utilisée pour entraîner le modèle
    window    : la taille de fenêtre utilisée dans make_loaders
    reglages  : le dictionnaire des paramètres passés à prepare_dataset
                (horizon, target, signaux, regime_col, score_min, ...)
    tickers_statiques : si vous avez entraîné avec static=ticker_onehot(data.tickers),
                passez data.tickers ici
    """
    info = {"scaler": data.scaler, "feature_cols": data.feature_cols,
            "regime_cols": data.regime_cols, "signal_cols": data.signal_cols,
            "window": window, "reglages": dict(reglages),
            "tickers_statiques": tickers_statiques}
    with open(path, "wb") as f:
        pickle.dump(info, f)
    print(f"Préparation enregistrée dans {path} : horizon {reglages.get('horizon')}, "
          f"fenêtre {window}, {len(data.feature_cols)} variables.")


def charger_preparation(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# B. Prévision
# ---------------------------------------------------------------------------
def proba_hausse(model_path: str, prep_path: str, prices, panels, macro, ticker: str,
                 historique: int = 60) -> dict:
    """Probabilité de hausse à H séances pour `ticker`, au dernier jour disponible.

    prices, panels : vos données À JOUR jusqu'à aujourd'hui, au même format qu'à
                     l'entraînement (au moins ~6 mois d'historique pour remplir les fenêtres)
    macro          : load_macro(...) à jour
    historique     : nombre de séances passées pour lesquelles recalculer la probabilité
                     (pour voir son évolution récente)
    """
    import contextlib
    import io

    from torch.utils.data import DataLoader

    from data_prep import PreparedData, prepare_dataset
    from train import load_model, predict
    from windows import WindowDataset

    prep = charger_preparation(prep_path)
    model, ckpt = load_model(model_path)
    H = prep["reglages"].get("horizon", 1)

    # 1. Préparation des nouvelles données, avec la normalisation de l'entraînement
    with contextlib.redirect_stdout(io.StringIO()):
        data = prepare_dataset(prices, panels, macro, scaler=prep["scaler"], **prep["reglages"])
    if list(data.feature_cols) != list(ckpt["feature_cols"]):
        raise ValueError("Les variables ne correspondent pas à celles du modèle :\n"
                         f"  modèle  : {ckpt['feature_cols']}\n  données : {data.feature_cols}\n"
                         "Vérifiez que panels contient les mêmes indicateurs, avec les mêmes noms.")
    if list(data.regime_cols) != list(ckpt.get("regime_cols", [])):
        raise ValueError("Les variables de contexte (score) ne correspondent pas au modèle.")

    df = data.df.xs(ticker, level="ticker", drop_level=False)
    if df.empty:
        raise ValueError(f"Aucune donnée préparée pour {ticker} : vérifiez son nom dans prices et panels.")
    derniere = df.index.get_level_values("date").max()

    # 2. Le régime du jour est-il couvert par le modèle ?
    statut = df.loc[(derniere, ticker), "split"]
    if statut == "hors_score":
        print(f"{ticker}, {derniere.date()} : le score macro est hors du domaine d'entraînement "
              f"(filtre score_min / score_max) -> le modèle ne donne pas de probabilité.")

    # 3. Prédiction sur les `historique` dernières séances (dont aujourd'hui)
    dates = df.index.get_level_values("date")
    a_predire = (dates >= dates.unique().sort_values()[-historique]) & (df["split"] != "hors_score")
    d2 = df.copy()
    d2["split"] = np.where(a_predire, "prevision", "historique")
    sub = PreparedData(df=d2, feature_cols=data.feature_cols, scaler=data.scaler,
                       tickers=[ticker], regime_cols=data.regime_cols,
                       signal_cols=data.signal_cols)
    static = None
    if getattr(model, "n_static", None):
        ts = prep.get("tickers_statiques")
        if not ts or ticker not in ts:
            raise ValueError(f"Le modèle a été entraîné avec l'identité des titres, et {ticker} "
                             "n'en faisait pas partie : il ne peut pas le prédire.")
        static = pd.DataFrame([[1.0 if t == ticker else 0.0 for t in ts]], index=[ticker], columns=ts)
    ds = WindowDataset(sub, "prevision", window=prep["window"], static=static)
    hist = predict(model, DataLoader(ds, batch_size=512, shuffle=False), with_weights=True)
    hist = hist.xs(ticker, level="ticker")

    # Valeur réalisée quand elle est connue (séances assez anciennes)
    hist["résultat connu"] = df.xs(ticker, level="ticker").loc[hist.index, "split"].ne("live")
    hist.loc[~hist["résultat connu"], "y"] = np.nan

    # 4. Explication du jour : signaux récents et score
    jour = {"ticker": ticker, "date des données": derniere.date(), "horizon (séances)": H}
    if derniere in hist.index:
        p = float(hist.loc[derniere, "proba"])
        jour["probabilité de hausse"] = p
        fr = df.loc[(derniere, ticker), data.signal_cols] if data.signal_cols else pd.Series(dtype=float)
        jour["signaux récents"] = ", ".join(f"{c[4:]} ({v:.2f})" for c, v in fr.items() if v >= 0.25) \
            or "aucun"
        for c in [c for c in data.regime_cols if c in data.scaler.mean.index]:
            jour[c] = float(df.loc[(derniere, ticker), c] * data.scaler.std[c] + data.scaler.mean[c])
    else:
        jour["probabilité de hausse"] = np.nan

    print(f"\n{ticker} : probabilité que la clôture dans {H} séances soit au-dessus de celle du "
          f"{derniere.date()} = {jour['probabilité de hausse']:.1%}")
    for k in ("signaux récents", prep['reglages'].get('regime_col')):
        if k and k in jour:
            v = jour[k]
            print(f"  {k} : {v:.1f}" if isinstance(v, float) else f"  {k} : {v}")
    return {"jour": jour, "historique": hist[["proba", "y", "résultat connu"]
                                            + [c for c in hist.columns if c.startswith("w_")]]}


def graphique(res: dict, prices, ticker: str, path: str | None = None):
    """Cours et probabilité de hausse sur les dernières séances."""
    import matplotlib.pyplot as plt

    from data_prep import to_long

    h = res["historique"]
    px = to_long(prices).xs(ticker, level="ticker")
    col = "Adj Close" if "Adj Close" in px.columns else "Close"
    px = px[col].reindex(h.index)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.3]})
    a1.plot(px.index, px.values, color="black", lw=1.2)
    a1.set_ylabel("cours")
    a1.set_title(f"{ticker} : cours et probabilité de hausse à {res['jour']['horizon (séances)']} séances")
    a2.plot(h.index, h["proba"], color="tab:blue", lw=1.5)
    a2.axhline(0.5, color="grey", lw=0.8, ls="--")
    a2.set_ylabel("probabilité de hausse")
    a2.set_ylim(max(0, h["proba"].min() - 0.05), min(1, h["proba"].max() + 0.05))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ---------------------------------------------------------------------------
# Test rapide : python prevision.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from data_prep import load_macro, prepare_dataset
    from signaux import SIGNAUX
    from train import TrainConfig, train
    from windows import make_loaders

    rng = np.random.default_rng(5)

    def simuler(tickers, debut, fin):
        dates = pd.bdate_range(debut, fin)
        prices, rsi = {}, {}
        for t in tickers:
            c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
            o = c * np.exp(rng.normal(0, 0.004, len(dates)))
            prices[t] = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01,
                                      "Low": np.minimum(o, c) * 0.99, "Close": c}, index=dates)
            d = pd.Series(c, index=dates).diff()
            rsi[t] = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean())
        return prices, {"RSI": pd.DataFrame(rsi)}

    months = pd.date_range("2004-01-01", "2026-10-01", freq="MS")
    macro = load_macro(pd.DataFrame({"date": months, "score": rng.uniform(20, 90, len(months))}))
    sig = [s for s in SIGNAUX if s["col"] == "RSI"]

    # Entraînement sur 5 titres (Apple n'en fait pas partie)
    prices, panels = simuler([f"T{i}" for i in range(5)], "2004-06-01", "2019-12-31")
    reglages = dict(train_start="2005-01-01", train_end="2014-12-31", val_end="2017-12-31",
                    horizon=30, target="direction", signaux=sig, regime_col="score",
                    score_min=25, variables_prix=False)
    data = prepare_dataset(prices, panels, macro, **reglages)
    loaders = make_loaders(data, window=20, batch_size=256)
    train(loaders, TrainConfig(task="proba", d_model=8, max_epochs=3, patience=3, verbose=False),
          save_path="modele_h30_demo.pt")
    sauver_preparation("preparation_h30_demo.pkl", data, window=20, reglages=reglages)

    # Prévision pour un titre nouveau, avec seulement ses données récentes
    p_aapl, pan_aapl = simuler(["AAPL"], "2025-06-01", "2026-10-26")
    res = proba_hausse("modele_h30_demo.pt", "preparation_h30_demo.pkl",
                       p_aapl, pan_aapl, macro, ticker="AAPL")
    graphique(res, p_aapl, "AAPL", "prevision_aapl_demo.png")
    print(res["historique"].tail(5).round(3).to_string())
    assert 0 < res["jour"]["probabilité de hausse"] < 1
    print("TEST OK")
