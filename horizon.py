"""
Quel horizon de détention donne le meilleur résultat à vos signaux ?

Pour chaque horizon H (en séances) :
  1. la cible devient « le cours est-il plus haut dans H séances ? » ;
  2. un modèle est entraîné (réseau VSN+LSTM, ou régression logistique en mode rapide) ;
  3. il est noté sur la VALIDATION uniquement :
       - AUC, avec un intervalle de confiance à 95 % (rééchantillonnage par mois),
       - Sharpe net de frais de la stratégie, après choix du seuil (choisir_seuils).
Le test n'est JAMAIS utilisé ici : il reste réservé au jugement final de l'horizon retenu.

Utilisation conseillée, en deux temps :

    from horizons import comparer_horizons
    # 1. balayage rapide de 1 à 20 séances (quelques secondes par horizon)
    rapide = comparer_horizons(prices, panels, macro, horizons=range(1, 21),
                               methode="logistique", **reglages)
    # 2. réseau de neurones sur les horizons les plus prometteurs
    reseau = comparer_horizons(prices, panels, macro, horizons=[3, 4, 5, 8, 10],
                               methode="reseau", **reglages)

où `reglages` reprend vos paramètres de prepare_dataset, par exemple :
    reglages = dict(train_start="2005-01-01", train_end="2016-12-31", val_end="2019-12-31",
                    signaux=True, regime_col="score", score_min=25)
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from data_prep import prepare_dataset
from decision import choisir_seuils, rendements_journaliers
from train import auc, proba_metrics


# ---------------------------------------------------------------------------
# Évaluation commune
# ---------------------------------------------------------------------------
def auc_intervalle(pred: pd.DataFrame, n: int = 300, seed: int = 0) -> tuple[float, float]:
    """IC à 95 % de l'AUC, en tirant des MOIS entiers au hasard (les jours d'un
    même mois sont corrélés : les tirer un par un donnerait un intervalle trop étroit)."""
    mois = [g for _, g in pred.groupby(pred.index.get_level_values("date").to_period("M"))]
    if len(mois) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        d = pd.concat([mois[i] for i in rng.integers(0, len(mois), len(mois))])
        vals.append(auc(d["proba"].to_numpy(), d["y"].to_numpy()))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _noter(pred_val: pd.DataFrame, rets: pd.DataFrame, horizon: int, couts_bps: float,
           max_positions: int | None, min_trades: int) -> dict:
    m = proba_metrics(pred_val["proba"].to_numpy(), pred_val["y"].to_numpy())
    lo, hi = auc_intervalle(pred_val)
    grille = choisir_seuils(pred_val, rets, horizon=horizon, couts_bps=couts_bps,
                            max_positions=max_positions, min_trades=min_trades)
    best = grille.iloc[0]
    return {"jours-titres validation": len(pred_val),
            "fréquence de hausse": m["base"],
            "AUC": m["auc"], "AUC IC95 bas": lo, "AUC IC95 haut": hi,
            "gain log-loss": m["gain_logloss"],
            "seuil retenu": best["seuil_achat"],
            "Sharpe net (validation)": best["Sharpe net"] if best["évaluable"] else np.nan,
            "Sharpe référence": best["Sharpe référence"],
            "trades (validation)": int(best["trades"]),
            "trades gagnants": best["trades gagnants"]}


# ---------------------------------------------------------------------------
# Modèle rapide : régression logistique sur les variables du jour
# ---------------------------------------------------------------------------
def _matrice(data, d: pd.DataFrame, regimes: list[str]) -> pd.DataFrame:
    """Variables du jour + interactions signal x régime (la lecture selon le score).

    regimes : indicatrices de régime à croiser avec les signaux. Elles sont choisies
    UNE fois, sur l'entraînement, pour que validation et entraînement aient
    exactement les mêmes colonnes (sinon scikit-learn refuse de prédire).
    """
    X = d[data.feature_cols].copy()
    for r in regimes:
        for c in data.signal_cols:
            X[f"{c}_x_{r}"] = d[c] * d[r]
    for c in [c for c in data.regime_cols if c not in ("regime_bas", "regime_eleve")]:
        X[c] = d[c]
    return X


def _predire_logistique(data, C: float = 0.1) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    df = data.df
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    # régimes qui varient sur l'ENTRAÎNEMENT (une indicatrice constante n'apprend rien)
    regimes = [r for r in ("regime_bas", "regime_eleve")
               if r in tr.columns and tr[r].nunique() > 1]
    X_tr = _matrice(data, tr, regimes)
    X_va = _matrice(data, va, regimes)[X_tr.columns]          # mêmes colonnes, même ordre
    m = LogisticRegression(C=C, max_iter=3000).fit(X_tr, tr["target"])
    return pd.DataFrame({"proba": m.predict_proba(X_va)[:, 1],
                         "y": va["target"].to_numpy()}, index=va.index)


# ---------------------------------------------------------------------------
# Comparaison des horizons
# ---------------------------------------------------------------------------
def comparer_horizons(prices, indicators, macro, horizons=(1, 2, 3, 4, 5, 7, 10, 15, 20),
                      methode: str = "logistique", window: int = 20, batch_size: int = 256,
                      cfg=None, couts_bps: float = 10.0, max_positions: int | None = None,
                      min_trades: int = 50, static=None, adapter_memoire: bool = False,
                      out_dir: str = "horizons", **prep_kwargs) -> pd.DataFrame:
    """Entraîne et note un modèle par horizon, sur la validation.

    methode         : "logistique" (rapide) ou "reseau" (VSN+LSTM, plusieurs minutes par horizon)
    cfg             : TrainConfig pour le réseau (par défaut task="proba", d_model=16)
    adapter_memoire : False = mêmes variables pour tous les horizons (seule la cible change :
                      comparaison la plus propre). True = la mémoire des signaux s'allonge
                      avec l'horizon (max_jours = max(10, H)).
    prep_kwargs     : vos paramètres de prepare_dataset (dates, signaux, regime_col, score_min...)
    """
    os.makedirs(out_dir, exist_ok=True)
    rets = rendements_journaliers(prices)
    prep_kwargs = {**prep_kwargs, "target": "direction"}
    if methode == "reseau":
        from train import TrainConfig, predict, train
        from windows import make_loaders
        cfg = cfg or TrainConfig(task="proba", d_model=16, dropout=0.2, patience=8, verbose=False)

    lignes = []
    for H in horizons:
        t0 = time.time()
        kw = dict(prep_kwargs)
        if adapter_memoire:
            kw["max_jours"] = max(10, H)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):          # messages répétitifs masqués
            data = prepare_dataset(prices, indicators, macro, horizon=H, **kw)
        if methode == "logistique":
            pv = _predire_logistique(data)
            extra = {}
        else:
            loaders = make_loaders(data, window=window, batch_size=batch_size, static=static)
            model, hist = train(loaders, cfg, save_path=os.path.join(out_dir, f"modele_h{H}.pt"))
            pv = predict(model, loaders["val"])
            extra = {"meilleure époque": int(hist["val_loss"].idxmin())}
        note = _noter(pv, rets, H, couts_bps, max_positions, min_trades)
        lignes.append({"horizon (séances)": H, **note, **extra})
        print(f"H = {H:>2} : AUC {note['AUC']:.3f} [{note['AUC IC95 bas']:.3f} ; "
              f"{note['AUC IC95 haut']:.3f}]  Sharpe net {note['Sharpe net (validation)']:.2f} "
              f"(référence {note['Sharpe référence']:.2f})  {time.time() - t0:.0f}s")

    res = pd.DataFrame(lignes).set_index("horizon (séances)")
    res["AUC significative"] = res["AUC IC95 bas"] > 0.5
    _exporter(res, out_dir, methode)
    _conclusion(res)
    return res


def _conclusion(res: pd.DataFrame) -> None:
    ok = res[res["AUC significative"]]
    print()
    if ok.empty:
        print("Aucun horizon n'a une AUC significativement supérieure à 0,5 sur la validation : "
              "les signaux n'ont pas d'effet détectable, quel que soit l'horizon.")
        return
    best = ok["AUC"].idxmax()
    print(f"Meilleure AUC significative : H = {best} séances "
          f"(AUC {ok.loc[best, 'AUC']:.3f}, IC95 bas {ok.loc[best, 'AUC IC95 bas']:.3f}).")
    voisins = [h for h in res.index if abs(h - best) <= 2 and h != best]
    if voisins:
        print("Horizons voisins :", ", ".join(f"H={h} -> {res.loc[h, 'AUC']:.3f}" for h in voisins),
              "\n  -> un bon horizon est entouré de voisins proches (plateau), pas isolé (pic).")
    sh = res["Sharpe net (validation)"]
    if sh.notna().any():
        hb = sh.idxmax()
        print(f"Meilleur Sharpe net : H = {hb} séances ({sh.max():.2f}, référence "
              f"{res.loc[hb, 'Sharpe référence']:.2f}).")


def _exporter(res: pd.DataFrame, out_dir: str, methode: str) -> None:
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            matplotlib.use("Agg")
    except ImportError:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res.round(4).to_excel(os.path.join(out_dir, f"comparaison_horizons_{methode}.xlsx"))
    h = res.index.to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.fill_between(h, res["AUC IC95 bas"], res["AUC IC95 haut"], color="tab:blue", alpha=0.18,
                    label="intervalle de confiance à 95 %")
    a1.plot(h, res["AUC"], "o-", color="tab:blue", label="AUC")
    a1.axhline(0.5, color="grey", lw=1, ls="--", label="hasard")
    a1.set_xlabel("horizon (séances)")
    a1.set_ylabel("AUC (validation)")
    a1.set_title("Qualité de la prédiction selon l'horizon")
    a1.legend(fontsize=8)
    a2.plot(h, res["Sharpe net (validation)"], "o-", color="tab:green", label="stratégie, net de frais")
    a2.plot(h, res["Sharpe référence"], "-", color="grey", label="référence équipondérée")
    a2.axhline(0, color="grey", lw=0.8)
    a2.set_xlabel("horizon (séances)")
    a2.set_ylabel("Sharpe (validation)")
    a2.set_title("Résultat de la stratégie selon l'horizon")
    a2.legend(fontsize=8)
    fig.suptitle(f"Comparaison des horizons ({methode})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"comparaison_horizons_{methode}.png"), dpi=130)
    plt.close(fig)
    print(f"\nRésultats écrits dans : {os.path.abspath(out_dir)}")


# ---------------------------------------------------------------------------
# Test : un effet planté qui dure 8 séances -> l'horizon ~8 doit ressortir
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from signaux import franchissement

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2004-06-01", "2021-12-31")
    T = [f"T{i}" for i in range(8)]
    n = len(dates)
    prices, rsi_cols, bb_cols = {}, {}, {}
    for t in T:
        a = np.zeros(n)
        for i in range(1, n):
            a[i] = 0.9 * a[i - 1] + rng.normal(0, 6)
        rsi = pd.Series(np.clip(50 + a, 0, 100), index=dates)
        s = rsi.copy()
        s.index = pd.MultiIndex.from_product([dates, [t]], names=["date", "ticker"])
        ev = franchissement(s, 45, "up").to_numpy()
        # Effet planté : +0,25 %/jour pendant les 8 séances qui suivent un franchissement
        drift = np.zeros(n)
        for k in np.flatnonzero(ev):
            drift[k + 1:k + 9] += 0.0025
        r = drift + rng.normal(0, 0.012, n)
        prices[t] = pd.DataFrame({"Close": 50 * np.exp(np.cumsum(r))}, index=dates)
        rsi_cols[t] = rsi
        bb_cols[t] = pd.Series(rng.uniform(0, 1, n), index=dates)
    panels = {"RSI": pd.DataFrame(rsi_cols), "BB_pct": pd.DataFrame(bb_cols)}
    from signaux import SIGNAUX
    sig = [s for s in SIGNAUX if s["col"] in ("RSI", "BB_pct")]
    reglages = dict(train_start="2005-01-01", train_end="2015-12-31", val_end="2018-12-31",
                    signaux=sig, variables_prix=False)

    res = comparer_horizons(prices, panels, None, horizons=[1, 2, 4, 6, 8, 10, 14, 20],
                            methode="logistique", out_dir="horizons_demo",
                            adapter_memoire=True, **reglages)
    best = res["AUC"].idxmax()
    assert 4 <= best <= 10, f"l'horizon trouvé ({best}) devrait être proche de 8"

    res2 = comparer_horizons(prices, panels, None, horizons=[2, 8, 20], methode="reseau",
                             out_dir="horizons_demo", adapter_memoire=True, **reglages)
    assert res2["AUC"].idxmax() == 8
    print("\nTEST OK")
