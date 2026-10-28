"""
Stratégie « portefeuille » : places limitées, entrée et sortie sur signal.

Un seul modèle (probabilité de hausse), deux seuils :
  - ENTRÉE  si P(hausse) >= seuil_achat
  - SORTIE  si P(hausse) <= seuil_sortie (par défaut 0,50 : le modèle ne pense plus
            que le titre va monter). seuil_sortie=None = pas de sortie sur signal.

Règles de gestion (fixées à l'avance, à ne pas optimiser) :
  - départ 100 % en liquidités ;
  - au plus `max_positions` lignes en même temps (10), une seule ligne par titre ;
  - au plus `max_entrees_jour` achats par jour (2), en priorité les plus fortes probabilités ;
  - sortie au PREMIER de ces événements :
        signal de sortie (après au moins `duree_min` séances),
        durée maximale `duree_max` atteinte (sortie de secours),
        stop à `stop_atr` x ATR sous le prix d'achat (désactivé si None) ;
  - taille de chaque achat : 1 / max_positions de la valeur du portefeuille
    (taille="courant", gains réinvestis) ou du capital de départ (taille="initial").

Exécution (comme dans decision.py) :
  entree="cloture"   : décision et exécution au même cours (dernier cours disponible,
                       ex. cours de 16h avec figer_seance) ; dans le backtest, la clôture
  entree="ouverture" : décision le soir, exécution à l'ouverture suivante

Contenu :
  backtest_portefeuille(...)          simulation jour par jour
  choisir_seuils_portefeuille(...)    seuils choisis sur la VALIDATION
  rapport_portefeuille(...)           Excel + graphiques
  decisions_portefeuille_du_jour(...) quoi vendre, quoi garder, quoi acheter aujourd'hui
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from decision import JOURS_AN, _reference, _stats_serie, atr_wilder, ohlc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def backtest_portefeuille(pred: pd.DataFrame, prices, seuil_achat: float,
                          seuil_sortie: float | None = 0.5, max_positions: int = 10,
                          max_entrees_jour: int = 2, duree_max: int = 30, duree_min: int = 1,
                          couts_bps: float = 10.0, entree: str = "cloture",
                          stop_atr: float | None = None, atr: pd.DataFrame | None = None,
                          stop_sur: str = "low", taille: str = "courant", capital: float = 1.0,
                          **autres_options) -> dict:
    """Rejoue la stratégie jour par jour sur la période couverte par `pred` (sortie de predict()).

    Les jours sans probabilité pour un titre (score hors domaine, données manquantes) :
    pas d'achat possible, et une ligne détenue n'est pas vendue sur signal (elle reste
    soumise à la durée maximale et au stop).
    """
    o = ohlc(prices)
    proba = pred["proba"].unstack("ticker").sort_index()
    cl = o["Close"]
    dates = cl.index[(cl.index >= proba.index.min()) & (cl.index <= proba.index.max())]
    cols = list(proba.columns)
    T = len(dates)
    get = lambda c: o[c].reindex(index=dates, columns=cols).to_numpy() if c in o else None
    Craw, O, L, Hh = get("Close"), get("Open"), get("Low"), get("High")
    C = pd.DataFrame(Craw).ffill().to_numpy()               # valorisation : dernier cours connu
    P = proba.reindex(index=dates, columns=cols).to_numpy()
    if entree == "ouverture" and O is None:
        raise ValueError("entree='ouverture' demande la colonne Open dans prices.")
    if stop_atr is not None:
        if atr is None:
            atr = atr_wilder(o)
        A = atr.reindex(index=dates, columns=cols).to_numpy()
        if stop_sur == "low" and L is None:
            raise ValueError("stop_sur='low' demande la colonne Low dans prices.")
    c = couts_bps / 1e4

    cash = capital
    pos: dict[int, dict] = {}            # colonne -> {qty, i_entree, px, stop, montant}
    attente: list[tuple] = []            # ordres à exécuter à l'ouverture suivante
    equity, frais_j = np.zeros(T), np.zeros(T)
    n_pos, expo = np.zeros(T), np.zeros(T)
    trades = []

    def vendre(j: int, px: float, i: int, motif: str) -> None:
        nonlocal cash
        p = pos.pop(j)
        brut = p["qty"] * px
        cash += brut * (1 - c)
        frais_j[i] += brut * c
        trades.append({"ticker": cols[j], "date d'entrée": dates[p["i_entree"]],
                       "prix d'entrée": p["px"], "date de sortie": dates[i], "prix de sortie": px,
                       "motif de sortie": motif, "séances détenues": i - p["i_entree"],
                       "proba à l'entrée": p["proba"], "stop": p["stop"],
                       "gain": px / p["px"] - 1,
                       "résultat (€ pour 1 € de capital)": brut * (1 - c) - p["montant"]})

    def acheter(j: int, px: float, i: int, montant: float, i_decision: int) -> None:
        nonlocal cash
        montant = min(montant, cash / (1 + c))
        if montant <= 1e-12 or not np.isfinite(px) or px <= 0:
            return
        cash -= montant * (1 + c)
        frais_j[i] += montant * c
        stop = px - stop_atr * A[i_decision, j] if stop_atr is not None else np.nan
        pos[j] = {"qty": montant / px, "i_entree": i, "px": px, "stop": stop,
                  "montant": montant * (1 + c), "proba": P[i_decision, j]}

    for i in range(T):
        # 1. Ordres décidés la veille, exécutés à l'ouverture (ventes d'abord)
        if attente:
            for typ, j, info in sorted(attente, key=lambda x: x[0] != "vente"):
                px = O[i, j] if np.isfinite(O[i, j]) else C[i, j]
                if typ == "vente" and j in pos:
                    vendre(j, px, i, info)
                elif typ == "achat" and j not in pos:
                    acheter(j, px, i, info["montant"], info["i"])
            attente = []

        # 2. Stops pendant la séance
        if stop_atr is not None:
            for j in list(pos):
                p = pos[j]
                if p["i_entree"] == i and entree == "cloture":
                    continue                                   # acheté ce soir : pas encore exposé
                bas = L[i, j] if stop_sur == "low" else Craw[i, j]
                if np.isfinite(bas) and np.isfinite(p["stop"]) and bas <= p["stop"]:
                    ouv = O[i, j] if (O is not None and p["i_entree"] != i) else np.nan
                    px = ouv if (np.isfinite(ouv) and ouv < p["stop"]) else \
                        (p["stop"] if stop_sur == "low" else Craw[i, j])
                    vendre(j, px, i, "stop")

        # 3. Décisions au dernier cours de la séance
        sorties = []
        for j, p in pos.items():
            k = i - p["i_entree"] + (1 if entree == "ouverture" else 0)   # séances détenues
            pj = P[i, j]
            if k >= duree_max:
                sorties.append((j, "durée maximale"))
            elif seuil_sortie is not None and k >= duree_min and np.isfinite(pj) and pj <= seuil_sortie:
                sorties.append((j, "signal de sortie"))
        valeur = cash + sum(p["qty"] * C[i, j] for j, p in pos.items())
        vendus = {j for j, _ in sorties}
        if entree == "cloture":
            for j, motif in sorties:
                vendre(j, Craw[i, j] if np.isfinite(Craw[i, j]) else C[i, j], i, motif)
        else:
            attente += [("vente", j, motif) for j, motif in sorties]

        places = max_positions - (len(pos) - (len(vendus) if entree == "ouverture" else 0))
        n_achats = min(max_entrees_jour, max(places, 0))
        if n_achats > 0 and i < T - (1 if entree == "ouverture" else 0):
            cand = [(P[i, j], j) for j in range(len(cols))
                    if j not in pos and j not in vendus and np.isfinite(P[i, j])
                    and P[i, j] >= seuil_achat and np.isfinite(Craw[i, j])]
            cand.sort(reverse=True)
            montant = (valeur if taille == "courant" else capital) / max_positions
            for _, j in cand[:n_achats]:
                if entree == "cloture":
                    acheter(j, Craw[i, j], i, montant, i)
                else:
                    attente.append(("achat", j, {"montant": montant, "i": i}))

        # 4. Valeur du portefeuille à la clôture
        investi = sum(p["qty"] * C[i, j] for j, p in pos.items())
        equity[i] = cash + investi
        n_pos[i] = len(pos)
        expo[i] = investi / equity[i] if equity[i] > 0 else 0.0

    # Positions encore ouvertes à la fin : vendues au dernier cours (frais compris)
    for j in list(pos):
        vendre(j, C[T - 1, j], T - 1, "fin de période")
    equity[T - 1] = cash

    eq = pd.Series(equity, index=dates)
    net = eq.pct_change().fillna(eq.iloc[0] / capital - 1)
    prec = eq.shift(1).fillna(capital)
    frais = pd.Series(frais_j, index=dates) / prec
    rets = cl.pct_change(fill_method=None).reindex(index=dates, columns=cols)
    reference = _reference(rets, proba.reindex(index=dates, columns=cols).notna(), couts_bps)

    res = {"net": net, "brut": net + frais, "frais": frais, "reference": reference,
           "valeur": eq, "positions": pd.Series(n_pos, index=dates),
           "exposition": pd.Series(expo, index=dates),
           "trades": pd.DataFrame(trades)}
    res["stats"] = stats_portefeuille(res, max_positions)
    return res


def stats_portefeuille(res: dict, max_positions: int) -> pd.DataFrame:
    s = pd.DataFrame({"stratégie (net de frais)": _stats_serie(res["net"]),
                      "stratégie (avant frais)": _stats_serie(res["brut"]),
                      "référence équipondérée": _stats_serie(res["reference"])})
    t = res["trades"]
    extra = {"nombre de trades": len(t),
             "trades par an": len(t) / (len(res["net"]) / JOURS_AN)}
    if len(t):
        extra.update({
            "trades gagnants": (t["gain"] > 0).mean(),
            "gain moyen par trade": t["gain"].mean(),
            "gain moyen des trades gagnants": t.loc[t["gain"] > 0, "gain"].mean(),
            "perte moyenne des trades perdants": t.loc[t["gain"] <= 0, "gain"].mean(),
            "durée moyenne de détention (séances)": t["séances détenues"].mean(),
        })
        for motif, part in t["motif de sortie"].value_counts(normalize=True).items():
            extra[f"sorties : {motif}"] = part
    extra.update({
        "positions détenues en moyenne": res["positions"].mean(),
        f"jours avec les {max_positions} places prises": (res["positions"] >= max_positions).mean(),
        "exposition moyenne (part investie)": res["exposition"].mean(),
        "frais totaux (en % du capital)": res["frais"].sum(),
    })
    for k, v in extra.items():
        s.loc[k, "stratégie (net de frais)"] = v
    return s


# ---------------------------------------------------------------------------
# Choix des seuils sur la validation
# ---------------------------------------------------------------------------
def choisir_seuils_portefeuille(pred_val: pd.DataFrame, prices, grille_achat=None,
                                grille_sortie=None, min_trades: int = 30,
                                **options) -> pd.DataFrame:
    """Teste les couples (seuil d'achat, seuil de sortie) sur la VALIDATION.

    Ajoute toujours la variante « sortie au temps seulement » (seuil_sortie=None) :
    si la sortie sur signal ne fait pas nettement mieux, gardez la version simple.
    Gardez une grille GROSSIÈRE : plus on teste de couples, plus le meilleur est flatteur.
    grille_sortie : par défaut deux valeurs « naturelles » : 0,50, et la médiane des
                    probabilités du modèle sur la validation (un seuil fixe à 0,50 peut ne
                    jamais se déclencher si le modèle donne rarement moins de 0,50).
    options : max_positions, max_entrees_jour, duree_max, couts_bps, entree, stop_atr, atr...
    """
    if grille_achat is None:
        q = pred_val["proba"].quantile([0.7, 0.8, 0.9, 0.95]).round(3)
        grille_achat = sorted(set(q.tolist()))
    if grille_sortie is None:
        grille_sortie = sorted({0.5, round(float(pred_val["proba"].median()), 3)})
    sorties = list(grille_sortie) + ([None] if None not in grille_sortie else [])
    lignes = []
    for sa in grille_achat:
        for ss in sorties:
            if ss is not None and ss >= sa:
                continue
            bt = backtest_portefeuille(pred_val, prices, sa, ss, **options)
            st = bt["stats"]
            col = st["stratégie (net de frais)"]
            lignes.append({"seuil_achat": sa,
                           "seuil_sortie": "aucun (temps seul)" if ss is None else ss,
                           "Sharpe net": col["Sharpe (sans taux sans risque)"],
                           "Sharpe référence": st.loc["Sharpe (sans taux sans risque)",
                                                      "référence équipondérée"],
                           "rendement annualisé net": col["rendement annualisé"],
                           "perte maximale": col["perte maximale (max drawdown)"],
                           "trades": int(col["nombre de trades"]),
                           "trades gagnants": col.get("trades gagnants", np.nan),
                           "durée moyenne": col.get("durée moyenne de détention (séances)", np.nan),
                           "positions moyennes": col["positions détenues en moyenne"]})
    tab = pd.DataFrame(lignes)
    tab["évaluable"] = tab["trades"] >= min_trades
    tab = tab.sort_values(["évaluable", "Sharpe net"], ascending=[False, False]).reset_index(drop=True)
    best = tab.iloc[0]
    if not best["évaluable"] or best["Sharpe net"] <= max(0.0, best["Sharpe référence"]):
        print("ATTENTION : sur la validation, aucune combinaison ne fait mieux que la référence "
              "(ou que zéro) après frais.")
    return tab


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def rapport_portefeuille(res: dict, out_dir: str = "backtest_portefeuille", titre: str = "") -> pd.DataFrame:
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            matplotlib.use("Agg")
    except ImportError:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    eq = (1 + res["net"]).cumprod()
    ref = (1 + res["reference"]).cumprod()
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.1, 1.1]})
    a1.plot(eq.index, eq, color="tab:blue", label="stratégie (net de frais)")
    a1.plot(ref.index, ref, color="grey", label="référence (acheter et conserver)")
    a1.set_ylabel("valeur de 1 € investi")
    a1.set_title(f"Portefeuille à places limitées {titre}".strip())
    a1.legend()
    dd = eq / eq.cummax() - 1
    a2.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.4)
    a2.set_ylabel("perte depuis\nle plus haut")
    a3.fill_between(res["positions"].index, res["positions"].values, step="post", alpha=0.5)
    a3.set_ylabel("positions\ndétenues")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "courbe_portefeuille.png"), dpi=130)
    plt.close(fig)

    mensuel = pd.DataFrame({"stratégie nette": (1 + res["net"]).resample("ME").prod() - 1,
                            "référence": (1 + res["reference"]).resample("ME").prod() - 1})
    mensuel.index = mensuel.index.strftime("%Y-%m")
    with pd.ExcelWriter(os.path.join(out_dir, "backtest_portefeuille.xlsx"), engine="openpyxl") as w:
        res["stats"].to_excel(w, sheet_name="Statistiques")
        mensuel.to_excel(w, sheet_name="Rendements mensuels")
        res["trades"].to_excel(w, sheet_name="Trades", index=False)
        if len(res["trades"]):
            par_titre = res["trades"].groupby("ticker").agg(
                trades=("gain", "size"), gagnants=("gain", lambda g: (g > 0).mean()),
                gain_moyen=("gain", "mean"), duree_moyenne=("séances détenues", "mean"))
            par_titre.to_excel(w, sheet_name="Par titre")
    print(f"Backtest écrit dans : {os.path.abspath(out_dir)}")
    return res["stats"]


# ---------------------------------------------------------------------------
# Décisions du jour
# ---------------------------------------------------------------------------
def decisions_portefeuille_du_jour(model, data, window: int, portefeuille: dict,
                                   seuil_achat: float, seuil_sortie: float | None = 0.5,
                                   max_positions: int = 10, max_entrees_jour: int = 2,
                                   duree_max: int = 30, duree_min: int = 1, prices=None,
                                   stop_atr: float | None = None, atr: pd.DataFrame | None = None,
                                   static=None, entree: str = "cloture", **autres_options) -> dict:
    """Quoi vendre, quoi garder, quoi acheter aujourd'hui.

    portefeuille : vos lignes actuelles, sous la forme
                   {"MC.PA": {"date": "2026-10-12", "prix": 698.5}, "TTE.PA": {...}}
                   ("prix" sert au stop ; facultatif si le stop est désactivé)
                   ou simplement {"MC.PA": "2026-10-12", ...}
    """
    from decision import decisions_du_jour

    jour = decisions_du_jour(model, data, window=window, seuil_achat=seuil_achat, static=static,
                             horizon=duree_max, prices=prices, entree=entree)
    derniere = jour.attrs["date"]
    seances = data.df.index.get_level_values("date").unique().sort_values()
    if stop_atr is not None and atr is None and prices is not None:
        atr = atr_wilder(ohlc(prices))

    lignes = []
    for t, info in portefeuille.items():
        info = info if isinstance(info, dict) else {"date": info}
        d0 = pd.Timestamp(info["date"])
        k = int(((seances > d0) & (seances <= derniere)).sum())   # séances détenues
        p = jour.loc[t, "proba"] if t in jour.index else np.nan
        cours = jour.loc[t, "dernier cours"] if (t in jour.index and "dernier cours" in jour) else np.nan
        stop = np.nan
        if stop_atr is not None and "prix" in info and atr is not None and t in atr.columns:
            a = atr[t].loc[:d0].iloc[-1] if len(atr[t].loc[:d0]) else np.nan
            stop = info["prix"] - stop_atr * a
        if np.isfinite(stop) and np.isfinite(cours) and cours <= stop:
            action, motif = "VENDRE", "stop"
        elif k >= duree_max:
            action, motif = "VENDRE", "durée maximale"
        elif seuil_sortie is not None and k >= duree_min and np.isfinite(p) and p <= seuil_sortie:
            action, motif = "VENDRE", "signal de sortie"
        else:
            action = "GARDER"
            motif = "pas de probabilité aujourd'hui (score hors domaine ?)" if not np.isfinite(p) else ""
        lignes.append({"ticker": t, "action": action, "motif": motif, "proba": p,
                       "séances détenues": k, "dernier cours": cours, "stop": stop})
    lignes_df = pd.DataFrame(lignes).set_index("ticker") if lignes else \
        pd.DataFrame(columns=["action", "motif", "proba", "séances détenues", "dernier cours", "stop"])

    n_ventes = int((lignes_df["action"] == "VENDRE").sum()) if len(lignes_df) else 0
    places = max_positions - (len(portefeuille) - n_ventes)
    n_achats = max(0, min(max_entrees_jour, places))
    cand = jour[(~jour.index.isin(list(portefeuille))) & (jour["proba"] >= seuil_achat)]
    achats = cand.sort_values("proba", ascending=False).head(n_achats)
    a_garder = [c for c in ["proba", "dernier cours", "stop", "signaux récents"] if c in achats.columns]
    achats = achats[a_garder]

    print(f"\nPortefeuille : {len(portefeuille)} lignes, {n_ventes} à vendre, "
          f"{places} place(s) libre(s) après ventes, {len(achats)} achat(s) proposé(s) "
          f"(au plus {max_entrees_jour} par jour).")
    if len(cand) > len(achats):
        print(f"  {len(cand) - len(achats)} autre(s) titre(s) au-dessus du seuil, non retenu(s) "
              "faute de place ou au-delà du nombre d'achats par jour.")
    return {"vendre": lignes_df[lignes_df["action"] == "VENDRE"] if len(lignes_df) else lignes_df,
            "garder": lignes_df[lignes_df["action"] == "GARDER"] if len(lignes_df) else lignes_df,
            "acheter": achats, "toutes les probabilités": jour}
