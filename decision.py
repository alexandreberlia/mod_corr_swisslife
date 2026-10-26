"""
Étape 6 : des probabilités à la décision.

Règle de décision (par défaut, achat seul) :
  - chaque soir, pour chaque titre, le modèle donne P(hausse dans H séances) ;
  - si P >= seuil_achat, on achète à la clôture et on garde H séances ;
  - (option) si P <= seuil_vente, on vend à découvert pour H séances ;
  - le capital est découpé en H « tranches » : chaque soir, une tranche (1/H du capital)
    est investie à parts égales dans les titres retenus, et la tranche investie il y a
    H séances est libérée. Les positions se chevauchent donc sans jamais dépasser 100 %.
    Si aucun titre n'est retenu un soir, cette tranche reste en liquidités.

Contenu :
  backtest(...)          rejoue la règle sur une période passée, frais compris
  choisir_seuils(...)    teste plusieurs seuils sur la VALIDATION
  decisions_du_jour(...) applique la règle aux dernières données disponibles

Règle d'or : seuils choisis sur la validation, appliqués UNE fois au test.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data_prep import to_long

JOURS_AN = 252


# ---------------------------------------------------------------------------
# Rendements
# ---------------------------------------------------------------------------
def rendements_journaliers(prices) -> pd.DataFrame:
    """Rendements simples clôture à clôture : DataFrame dates x titres."""
    px = to_long(prices)
    col = "Adj Close" if "Adj Close" in px.columns else "Close"
    close = px[col].unstack("ticker").sort_index()
    return close.pct_change(fill_method=None)


# ---------------------------------------------------------------------------
# Règle de décision
# ---------------------------------------------------------------------------
def selection(pred: pd.DataFrame, seuil_achat: float, seuil_vente: float | None = None,
              max_positions: int | None = None) -> pd.DataFrame:
    """Décision de chaque soir : +1 achat, -1 vente, 0 rien. DataFrame dates x titres.

    max_positions : au plus N achats (et N ventes) par soir, les plus convaincus.
    """
    proba = pred["proba"].unstack("ticker").sort_index()
    longs = proba >= seuil_achat
    shorts = (proba <= seuil_vente) if seuil_vente is not None else proba < -1
    if max_positions:
        rang_l = proba.where(longs).rank(axis=1, ascending=False)
        rang_s = proba.where(shorts).rank(axis=1, ascending=True)
        longs &= rang_l <= max_positions
        shorts &= rang_s <= max_positions
    return longs.astype(int) - shorts.astype(int)


def poids_portefeuille(decision: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Poids détenus à la clôture de chaque jour (méthode des tranches)."""
    n = decision.abs().sum(axis=1).replace(0, np.nan)
    cohorte = decision.div(n, axis=0).fillna(0.0)            # poids d'une tranche
    return cohorte.rolling(horizon, min_periods=1).sum() / horizon


def backtest(pred: pd.DataFrame, rets: pd.DataFrame, seuil_achat: float,
             seuil_vente: float | None = None, horizon: int = 4,
             couts_bps: float = 10.0, max_positions: int | None = None,
             stop_atr: float | None = None, prices=None, atr: pd.DataFrame | None = None,
             stop_sur: str = "low") -> dict:
    """Rejoue la stratégie sur la période couverte par `pred` (sortie de predict()).

    couts_bps : frais + écart achat/vente, en points de base par aller simple
                (10 = 0,10 % à chaque achat et à chaque vente).

    Stop de protection (optionnel) :
      stop_atr : distance du stop en multiples d'ATR, ex. 2. Pour un achat au cours P,
                 stop = P - 2 x ATR du jour d'achat (fixe pendant toute la détention).
                 Pour une vente à découvert : P + 2 x ATR.
      prices   : votre dictionnaire de prix (Open/High/Low/Close), obligatoire avec un stop
      atr      : DataFrame dates x titres de l'ATR (calculé avec votre fonction) ;
                 None = ATR 14 jours de Wilder calculé ici
      stop_sur : "low" (défaut) = déclenché si le plus bas du jour touche le stop ;
                 sortie au stop, ou à l'ouverture si elle est déjà sous le stop (gap).
                 "close" = déclenché seulement si la clôture passe sous le stop ;
                 sortie à cette clôture.
    """
    dec = selection(pred, seuil_achat, seuil_vente, max_positions)
    dates = rets.index[(rets.index >= dec.index.min()) & (rets.index <= dec.index.max())]
    dec = dec.reindex(index=dates, columns=rets.columns).fillna(0)
    r = rets.reindex(index=dates)

    # Référence : tous les titres de l'univers à parts égales, toujours investi
    eligibles = pred["proba"].unstack("ticker").reindex(index=dates, columns=rets.columns).notna()
    reference = r.where(eligibles.cumsum() > 0).mean(axis=1).fillna(0.0)

    if stop_atr is not None:
        if prices is None:
            raise ValueError("Avec un stop, passez aussi prices=... (pour les plus bas et l'ATR).")
        res = _backtest_stop(dec, prices, atr, horizon, stop_atr, couts_bps, stop_sur)
        res.update({"reference": reference, "decisions": dec})
        res["stats"] = statistiques(res, horizon)
        return res

    w = poids_portefeuille(dec, horizon)
    # Position prise à la clôture de t -> rendement du jour t+1
    brut = (w.shift(1) * r).sum(axis=1, min_count=1).fillna(0.0)
    rotation = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    frais = (rotation * couts_bps / 1e4).shift(1).fillna(0.0)
    net = brut - frais

    # Détail des trades : rendement sur les H séances qui suivent la décision
    fwd = (1 + r).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    trades = []
    for sens, lab in [(1, "achat"), (-1, "vente")]:
        m = dec == sens
        if m.values.any():
            t = fwd.where(m).stack().rename("rdt_H").to_frame()
            t["sens"] = lab
            t["gain"] = t["rdt_H"] * sens
            trades.append(t)
    trades = pd.concat(trades) if trades else pd.DataFrame(columns=["rdt_H", "sens", "gain"])
    trades.index = trades.index.set_names(["date", "ticker"])

    res = {"net": net, "brut": brut, "frais": frais, "reference": reference,
           "poids": w, "decisions": dec, "trades": trades}
    res["stats"] = statistiques(res, horizon)
    return res


def ohlc(prices) -> dict[str, pd.DataFrame]:
    """Open/High/Low/Close en DataFrames dates x titres, ajustés comme la clôture."""
    px = to_long(prices).copy()
    if "Adj Close" in px.columns:                 # prix bruts + clôture ajustée : on ajuste tout
        f = px["Adj Close"] / px["Close"]
        for c in ("Open", "High", "Low"):
            if c in px.columns:
                px[c] = px[c] * f
        px["Close"] = px["Adj Close"]
    return {c: px[c].unstack("ticker").sort_index()
            for c in ("Open", "High", "Low", "Close") if c in px.columns}


def atr_wilder(o: dict[str, pd.DataFrame], n: int = 14) -> pd.DataFrame:
    h, l, c = o["High"], o["Low"], o["Close"]
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _backtest_stop(dec: pd.DataFrame, prices, atr, horizon: int, k: float,
                   couts_bps: float, stop_sur: str) -> dict:
    """Simulation trade par trade : chaque position a son propre stop et sa propre sortie.

    Même découpage en tranches que sans stop : une position ouverte le jour d prend
    1 / (H x nombre de titres retenus ce jour-là) du capital. Après un stop, cette part
    reste en liquidités jusqu'à la fin de la tranche.
    """
    o = ohlc(prices)
    dates, cols = dec.index, dec.columns
    C = o["Close"].reindex(index=dates, columns=cols).to_numpy()
    if stop_sur == "low":
        if "Low" not in o or "High" not in o:
            raise ValueError("stop_sur='low' demande les colonnes High et Low dans prices.")
        L = o["Low"].reindex(index=dates, columns=cols).to_numpy()
        Hh = o["High"].reindex(index=dates, columns=cols).to_numpy()
        O = o["Open"].reindex(index=dates, columns=cols).to_numpy() if "Open" in o else None
    if atr is None:
        if "High" not in o or "Low" not in o:
            raise ValueError("Pour calculer l'ATR il faut High et Low ; sinon passez atr=...")
        atr = atr_wilder(o)
    A = atr.reindex(index=dates, columns=cols).to_numpy()

    D = dec.to_numpy()
    n_jour = np.abs(D).sum(axis=1)
    T = len(dates)
    brut, frais, expo = np.zeros(T), np.zeros(T), np.zeros(T)
    cout = couts_bps / 1e4
    lignes = []

    for i, j in np.argwhere(D != 0):
        sens = D[i, j]
        p0, a = C[i, j], A[i, j]
        if not (np.isfinite(p0) and np.isfinite(a)) or i + 1 >= T:
            continue
        w = 1.0 / (horizon * n_jour[i])
        stop = p0 - sens * k * a
        frais[i + 1] += w * cout                           # frais d'entrée
        prec, sortie, motif, px_sortie = p0, None, "échéance", p0
        fin = min(i + horizon, T - 1)
        for t in range(i + 1, fin + 1):
            ct = C[t, j]
            # le stop est-il touché ce jour-là ?
            if stop_sur == "low":
                extreme = L[t, j] if sens > 0 else Hh[t, j]
                touche = np.isfinite(extreme) and (extreme <= stop if sens > 0 else extreme >= stop)
                if touche:
                    ouv = O[t, j] if O is not None else np.nan
                    gap = np.isfinite(ouv) and (ouv < stop if sens > 0 else ouv > stop)
                    px = ouv if gap else stop
            else:
                touche = np.isfinite(ct) and (ct <= stop if sens > 0 else ct >= stop)
                px = ct
            if touche:
                brut[t] += w * sens * (px / prec - 1)
                expo[t] += w
                sortie, motif, px_sortie = t, "stop", px
                break
            if np.isfinite(ct):
                brut[t] += w * sens * (ct / prec - 1)
                prec = ct
                px_sortie = ct
            expo[t] += w
        if sortie is None:
            sortie = fin
            motif = "échéance" if fin == i + horizon else "fin de période"
        frais[sortie] += w * cout                          # frais de sortie
        lignes.append({"date": dates[i], "ticker": cols[j],
                       "sens": "achat" if sens > 0 else "vente",
                       "prix d'entrée": p0, "ATR": a, "stop": stop,
                       "date de sortie": dates[sortie], "motif de sortie": motif,
                       "séances détenues": sortie - i, "prix de sortie": px_sortie,
                       "rdt_H": px_sortie / p0 - 1,
                       "gain": sens * (px_sortie / p0 - 1)})

    trades = pd.DataFrame(lignes)
    if len(trades):
        trades = trades.set_index(["date", "ticker"])
    else:
        trades = pd.DataFrame(columns=["rdt_H", "sens", "gain", "motif de sortie"])
    brut = pd.Series(brut, index=dates)
    frais = pd.Series(frais, index=dates)
    return {"net": brut - frais, "brut": brut, "frais": frais,
            "poids": pd.Series(expo, index=dates).to_frame("exposition"),
            "trades": trades}


def _stats_serie(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    annees = len(r) / JOURS_AN
    cagr = eq.iloc[-1] ** (1 / annees) - 1 if annees > 0 and eq.iloc[-1] > 0 else np.nan
    vol = r.std() * np.sqrt(JOURS_AN)
    dd = (eq / eq.cummax() - 1).min()
    return {"rendement annualisé": cagr, "volatilité annualisée": vol,
            "Sharpe (sans taux sans risque)": (r.mean() * JOURS_AN) / vol if vol > 0 else np.nan,
            "perte maximale (max drawdown)": dd,
            "rendement total": eq.iloc[-1] - 1}


def statistiques(res: dict, horizon: int) -> pd.DataFrame:
    s = pd.DataFrame({"stratégie (net de frais)": _stats_serie(res["net"]),
                      "stratégie (avant frais)": _stats_serie(res["brut"]),
                      "référence équipondérée": _stats_serie(res["reference"])})
    t = res["trades"].dropna(subset=["rdt_H"])
    extra = {
        "nombre de trades": len(t),
        "trades par an": len(t) / (len(res["net"]) / JOURS_AN),
        f"trades gagnants (sur {horizon} séances)": (t["gain"] > 0).mean() if len(t) else np.nan,
        "gain moyen par trade": t["gain"].mean() if len(t) else np.nan,
        "exposition moyenne (part du capital investie)": res["poids"].abs().sum(axis=1).mean(),
        "frais totaux (en % du capital)": res["frais"].sum(),
    }
    if "motif de sortie" in t.columns and len(t):
        stp = t["motif de sortie"] == "stop"
        extra["trades sortis par le stop"] = stp.mean()
        extra["gain moyen des trades stoppés"] = t.loc[stp, "gain"].mean() if stp.any() else np.nan
        extra["gain moyen des trades à échéance"] = t.loc[~stp, "gain"].mean() if (~stp).any() else np.nan
        extra["durée moyenne de détention (séances)"] = t["séances détenues"].mean()
    for k, v in extra.items():
        s.loc[k, "stratégie (net de frais)"] = v
    return s


# ---------------------------------------------------------------------------
# Choix des seuils sur la validation
# ---------------------------------------------------------------------------
def choisir_seuils(pred_val: pd.DataFrame, rets: pd.DataFrame, horizon: int = 4,
                   grille_achat=None, grille_vente=None, couts_bps: float = 10.0,
                   min_trades: int = 50, max_positions: int | None = None,
                   **options_stop) -> pd.DataFrame:
    """Teste plusieurs seuils sur la validation et les classe par Sharpe net.

    options_stop : stop_atr=..., prices=..., atr=..., stop_sur=... comme dans backtest.

    min_trades : un seuil qui ne déclenche que 10 trades n'est pas évaluable ;
                 il est écarté, même si son résultat est spectaculaire.
    """
    if grille_achat is None:
        q = pred_val["proba"].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 0.95]).round(3)
        grille_achat = sorted(set(q.tolist()))
    grille_vente = grille_vente if grille_vente is not None else [None]
    lignes = []
    for sa in grille_achat:
        for sv in grille_vente:
            if sv is not None and sv >= sa:
                continue
            bt = backtest(pred_val, rets, sa, sv, horizon, couts_bps, max_positions,
                          **options_stop)
            st = bt["stats"]["stratégie (net de frais)"]
            lignes.append({"seuil_achat": sa, "seuil_vente": sv,
                           "Sharpe net": st["Sharpe (sans taux sans risque)"],
                           "Sharpe référence": bt["stats"].loc["Sharpe (sans taux sans risque)",
                                                               "référence équipondérée"],
                           "rendement annualisé net": st["rendement annualisé"],
                           "perte maximale": st["perte maximale (max drawdown)"],
                           "trades": int(st["nombre de trades"]),
                           "trades gagnants": st[f"trades gagnants (sur {horizon} séances)"],
                           "exposition moyenne": st["exposition moyenne (part du capital investie)"]})
    tab = pd.DataFrame(lignes)
    tab["évaluable"] = tab["trades"] >= min_trades
    tab = tab.sort_values(["évaluable", "Sharpe net"], ascending=[False, False]).reset_index(drop=True)
    meilleur = tab.iloc[0]
    if not meilleur["évaluable"] or meilleur["Sharpe net"] <= max(0.0, meilleur["Sharpe référence"]):
        print("ATTENTION : sur la validation, aucun seuil ne fait mieux que la référence "
              "(ou que zéro) après frais. Le modèle ne justifie pas de trader en l'état.")
    return tab


# ---------------------------------------------------------------------------
# Rapport du backtest
# ---------------------------------------------------------------------------
def rapport_backtest(res: dict, out_dir: str = "backtest", titre: str = "") -> pd.DataFrame:
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            matplotlib.use("Agg")
    except ImportError:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    eq = pd.DataFrame({"stratégie (net de frais)": (1 + res["net"]).cumprod(),
                       "stratégie (avant frais)": (1 + res["brut"]).cumprod(),
                       "référence équipondérée": (1 + res["reference"]).cumprod()})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2]})
    for c, st in zip(eq.columns, ["-", ":", "-"]):
        a1.plot(eq.index, eq[c], st, label=c,
                color={"référence équipondérée": "grey"}.get(c, "tab:blue"))
    a1.set_ylabel("valeur de 1 € investi")
    a1.set_title(f"Stratégie contre référence {titre}".strip())
    a1.legend()
    dd = eq["stratégie (net de frais)"] / eq["stratégie (net de frais)"].cummax() - 1
    a2.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.4)
    a2.set_ylabel("perte depuis\nle plus haut")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "courbe_de_capital.png"), dpi=130)
    plt.close(fig)

    stats = res["stats"]
    mensuel = pd.DataFrame({k: (1 + res[v]).resample("ME").prod() - 1
                            for k, v in [("stratégie nette", "net"), ("référence", "reference")]})
    mensuel.index = mensuel.index.strftime("%Y-%m")
    with pd.ExcelWriter(os.path.join(out_dir, "backtest.xlsx"), engine="openpyxl") as w:
        stats.to_excel(w, sheet_name="Statistiques")
        mensuel.to_excel(w, sheet_name="Rendements mensuels")
        res["trades"].reset_index().to_excel(w, sheet_name="Trades", index=False)
    print(f"Backtest écrit dans : {os.path.abspath(out_dir)}")
    return stats


# ---------------------------------------------------------------------------
# Décisions du jour
# ---------------------------------------------------------------------------
def decisions_du_jour(model, data, window: int, seuil_achat: float,
                      seuil_vente: float | None = None, static=None,
                      horizon: int = 4) -> pd.DataFrame:
    """Probabilité et décision pour chaque titre, à la dernière date disponible.

    `data` doit être préparé avec des prix à jour jusqu'à aujourd'hui : les dernières
    séances (dont le futur est inconnu) y sont étiquetées "live".
    """
    from torch.utils.data import DataLoader

    from train import predict
    from windows import WindowDataset

    if not (data.df["split"] == "live").any():
        raise ValueError("Aucune ligne 'live' : mettez à jour prix et indicateurs jusqu'à "
                         "aujourd'hui, puis relancez prepare_dataset.")
    ds = WindowDataset(data, "live", window=window, static=static)
    pred = predict(model, DataLoader(ds, batch_size=512, shuffle=False), with_weights=True)
    derniere = pred.index.get_level_values("date").max()
    jour = pred.xs(derniere, level="date").copy()
    jour["décision"] = np.where(jour["proba"] >= seuil_achat, "ACHETER",
                                np.where((seuil_vente is not None) & (jour["proba"] <= (seuil_vente or -1)),
                                         "VENDRE", "-"))
    jour["sortie prévue"] = np.where(jour["décision"] != "-", f"dans {horizon} séances", "")
    # Signaux encore « frais » (franchissement dans les ~4 dernières séances)
    sig_cols = getattr(data, "signal_cols", []) or []
    if sig_cols:
        fr = data.df.xs(derniere, level="date").reindex(jour.index)[sig_cols]
        jour["signaux récents"] = fr.apply(
            lambda r: ", ".join(f"{c[4:]} ({v:.2f})" for c, v in r.items() if v >= 0.25) or "aucun",
            axis=1)
    # contexte remis dans son échelle d'origine (le score tel que dans votre CSV)
    for c in [c for c in jour.columns if c.startswith("ctx_")]:
        brut = c[4:]
        if brut in data.scaler.mean.index:
            jour[c] = jour[c] * data.scaler.std[brut] + data.scaler.mean[brut]
    cols = ["proba", "décision", "sortie prévue"] + (["signaux récents"] if sig_cols else []) \
        + [c for c in jour.columns if c.startswith("ctx_")]
    out = jour[cols].sort_values("proba", ascending=False)
    out.attrs["date"] = derniere
    print(f"Décisions pour la séance suivant le {derniere.date()} "
          f"(seuil d'achat {seuil_achat:.3f}) :")
    return out


# ---------------------------------------------------------------------------
# Test de bout en bout sur données simulées : python decision.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.argv = ["test"]
    src = open("test_regime.py", encoding="utf-8").read().split("res = {}")[0]
    exec(src, globals())                       # données simulées avec un effet planté

    from data_prep import prepare_dataset
    from train import TrainConfig, predict, train
    from windows import make_loaders

    data = prepare_dataset(prices, panels, macro, **common, regime_col="score")
    print(data.df["split"].value_counts().to_dict())
    loaders = make_loaders(data, window=20, batch_size=256)
    model, _ = train(loaders, TrainConfig(task="proba", d_model=16, max_epochs=25, patience=5,
                                          lr=2e-3, verbose=False), save_path=None)
    rets = rendements_journaliers(prices)

    # 1. Seuils choisis sur la validation
    pv = predict(model, loaders["val"])
    grille = choisir_seuils(pv, rets, horizon=4, couts_bps=10)
    print("\nSeuils testés sur la validation :")
    print(grille.round(3).head(6).to_string(index=False))
    sa = float(grille.iloc[0]["seuil_achat"])

    # 2. Appliqués une seule fois au test
    pt = predict(model, loaders["test"])
    bt = backtest(pt, rets, seuil_achat=sa, horizon=4, couts_bps=10)
    stats = rapport_backtest(bt, out_dir="backtest_demo", titre="(test, données simulées)")
    print("\nTest :")
    print(stats.round(3).to_string())

    # Vérifications de cohérence
    w = bt["poids"]
    assert (w.abs().sum(axis=1) <= 1 + 1e-9).all(), "exposition > 100 %"
    assert (w >= -1e-12).all().all(), "position vendeuse en mode achat seul"
    s = stats["stratégie (net de frais)"]
    assert s["Sharpe (sans taux sans risque)"] > stats.loc["Sharpe (sans taux sans risque)",
                                                            "référence équipondérée"]

    # 3. Décisions du jour
    jour = decisions_du_jour(model, data, window=20, seuil_achat=sa)
    print(jour.round(3).to_string())
    print("\nTEST OK")
