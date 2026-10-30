"""
Stratégie « portefeuille » : places limitées, entrée et sortie sur signal.

Un seul modèle (probabilité de hausse), deux seuils :
  - ENTRÉE  si P(hausse) >= seuil_achat
  - SORTIE  si P(hausse) <= seuil_sortie (par défaut 0,50 : le modèle ne pense plus
            que le titre va monter). seuil_sortie=None = pas de sortie sur signal.

Règles de gestion (fixées à l'avance, à ne pas optimiser) :
  - départ 100 % en liquidités ;
  - au plus `max_positions` lignes en même temps (10) ;
  - au plus `max_par_titre` lignes par titre (2), la 2e au moins `ecart_min` (3) séances
    après la 1re ;
  - au plus `max_entrees_jour` achats par jour (2), en priorité les plus fortes probabilités ;
  - sortie au PREMIER de ces événements :
        signal de sortie (après au moins `duree_min` séances) : vend toutes les lignes du titre,
        stop à `stop_atr` x ATR sous le prix d'achat de la ligne (désactivé si None),
        durée maximale `duree_max` (désactivée par défaut : None) ;
  - taille de chaque ligne : 1 / max_positions de la valeur du portefeuille
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
                          max_entrees_jour: int = 2, max_par_titre: int = 2,
                          ecart_min: int = 3, recroisement: bool = False,
                          duree_max: int | None = None, duree_min: int = 1,
                          couts_bps: float = 10.0, entree: str = "cloture",
                          stop_atr: float | None = None, atr: pd.DataFrame | None = None,
                          stop_sur: str = "low", taille: str = "courant", capital: float = 1.0,
                          **autres_options) -> dict:
    """Rejoue la stratégie jour par jour sur la période couverte par `pred` (sortie de predict()).

    Sorties : signal (P <= seuil_sortie) et/ou stop. duree_max=None (défaut) = pas de
    sortie au temps. Un signal de sortie vend TOUTES les lignes du titre ; le stop est
    propre à chaque ligne (calculé sur son prix et son ATR d'entrée).

    Plusieurs lignes par titre : au plus `max_par_titre` (2), une nouvelle ligne au moins
    `ecart_min` séances après la précédente sur ce titre. recroisement=True : la nouvelle
    ligne exige en plus que la proba soit repassée sous seuil_achat entre-temps (un
    « nouveau » signal, et pas le même qui dure).

    Positions encore ouvertes à la fin du backtest : valorisées au dernier cours, sans
    frais de vente, et listées dans les trades avec le motif « encore ouverte ».
    """
    if seuil_sortie is None and stop_atr is None and duree_max is None:
        raise ValueError("Aucune règle de sortie : donnez seuil_sortie, stop_atr ou duree_max.")
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
    pos: dict[int, list[dict]] = {}      # colonne -> lignes ouvertes [{qty, i_entree, px, stop...}]
    sous_seuil: dict[int, bool] = {}     # proba repassée sous seuil_achat depuis la dernière entrée
    attente: list[tuple] = []            # ordres à exécuter à l'ouverture suivante
    equity, frais_j = np.zeros(T), np.zeros(T)
    n_pos, expo = np.zeros(T), np.zeros(T)
    trades = []
    n_lignes = lambda: sum(len(v) for v in pos.values())

    def vendre(j: int, ligne: dict, px: float, i: int, motif: str, frais: bool = True) -> None:
        nonlocal cash
        pos[j].remove(ligne)
        if not pos[j]:
            del pos[j]
        brut = ligne["qty"] * px
        cc = c if frais else 0.0
        cash += brut * (1 - cc)
        frais_j[i] += brut * cc
        trades.append({"ticker": cols[j], "ligne": ligne["n"], "date d'entrée": dates[ligne["i_entree"]],
                       "prix d'entrée": ligne["px"], "date de sortie": dates[i], "prix de sortie": px,
                       "motif de sortie": motif, "séances détenues": i - ligne["i_entree"],
                       "proba à l'entrée": ligne["proba"], "stop": ligne["stop"],
                       "gain": px / ligne["px"] - 1,
                       "résultat (€ pour 1 € de capital)": brut * (1 - cc) - ligne["montant"]})

    def acheter(j: int, px: float, i: int, montant: float, i_decision: int) -> None:
        nonlocal cash
        if len(pos.get(j, [])) >= max_par_titre:
            return
        montant = min(montant, cash / (1 + c))
        if montant <= 1e-12 or not np.isfinite(px) or px <= 0:
            return
        cash -= montant * (1 + c)
        frais_j[i] += montant * c
        stop = px - stop_atr * A[i_decision, j] if stop_atr is not None else np.nan
        pos.setdefault(j, []).append({"qty": montant / px, "i_entree": i, "px": px, "stop": stop,
                                      "montant": montant * (1 + c), "proba": P[i_decision, j],
                                      "n": len(pos.get(j, [])) + 1})
        sous_seuil[j] = False

    def peut_ajouter(j: int, i: int) -> bool:
        lignes = pos.get(j, [])
        if len(lignes) >= max_par_titre:
            return False
        if not lignes:
            return True
        derniere = max(l["i_entree"] for l in lignes)
        ecart = i - derniere + (1 if entree == "ouverture" else 0)
        return ecart >= ecart_min and (not recroisement or sous_seuil.get(j, False))

    for i in range(T):
        # 1. Ordres décidés la veille, exécutés à l'ouverture (ventes d'abord)
        if attente:
            for typ, j, info in sorted(attente, key=lambda x: x[0] != "vente"):
                px = O[i, j] if np.isfinite(O[i, j]) else C[i, j]
                if typ == "vente":
                    ligne, motif = info
                    if j in pos and ligne in pos[j]:
                        vendre(j, ligne, px, i, motif)
                else:
                    acheter(j, px, i, info["montant"], info["i"])
            attente = []

        # 2. Stops pendant la séance (ligne par ligne)
        if stop_atr is not None:
            for j in list(pos):
                for ligne in list(pos.get(j, [])):
                    if ligne["i_entree"] == i and entree == "cloture":
                        continue                               # achetée ce soir : pas encore exposée
                    bas = L[i, j] if stop_sur == "low" else Craw[i, j]
                    if np.isfinite(bas) and np.isfinite(ligne["stop"]) and bas <= ligne["stop"]:
                        ouv = O[i, j] if (O is not None and ligne["i_entree"] != i) else np.nan
                        px = ouv if (np.isfinite(ouv) and ouv < ligne["stop"]) else \
                            (ligne["stop"] if stop_sur == "low" else Craw[i, j])
                        vendre(j, ligne, px, i, "stop")

        # 3. Décisions au dernier cours de la séance
        for j in range(len(cols)):
            if np.isfinite(P[i, j]) and P[i, j] < seuil_achat:
                sous_seuil[j] = True
        sorties = []                                           # (j, ligne, motif)
        for j, lignes in pos.items():
            pj = P[i, j]
            signal = seuil_sortie is not None and np.isfinite(pj) and pj <= seuil_sortie
            for ligne in lignes:
                k = i - ligne["i_entree"] + (1 if entree == "ouverture" else 0)   # séances détenues
                if duree_max is not None and k >= duree_max:
                    sorties.append((j, ligne, "durée maximale"))
                elif signal and k >= duree_min:
                    sorties.append((j, ligne, "signal de sortie"))
        valeur = cash + sum(l["qty"] * C[i, j] for j, ls in pos.items() for l in ls)
        vendus = {j for j, _, _ in sorties}
        if entree == "cloture":
            for j, ligne, motif in sorties:
                vendre(j, ligne, Craw[i, j] if np.isfinite(Craw[i, j]) else C[i, j], i, motif)
        else:
            attente += [("vente", j, (ligne, motif)) for j, ligne, motif in sorties]

        places = max_positions - (n_lignes() - (len(sorties) if entree == "ouverture" else 0))
        n_achats = min(max_entrees_jour, max(places, 0))
        if n_achats > 0 and i < T - (1 if entree == "ouverture" else 0):
            cand = [(P[i, j], j) for j in range(len(cols))
                    if j not in vendus and np.isfinite(P[i, j]) and P[i, j] >= seuil_achat
                    and np.isfinite(Craw[i, j]) and peut_ajouter(j, i)]
            cand.sort(reverse=True)
            montant = (valeur if taille == "courant" else capital) / max_positions
            for _, j in cand[:n_achats]:
                if entree == "cloture":
                    acheter(j, Craw[i, j], i, montant, i)
                else:
                    attente.append(("achat", j, {"montant": montant, "i": i}))

        # 4. Valeur du portefeuille à la clôture
        investi = sum(l["qty"] * C[i, j] for j, ls in pos.items() for l in ls)
        equity[i] = cash + investi
        n_pos[i] = n_lignes()
        expo[i] = investi / equity[i] if equity[i] > 0 else 0.0

    # Positions encore ouvertes à la fin : valorisées au dernier cours, sans frais
    for j in list(pos):
        for ligne in list(pos[j]):
            vendre(j, ligne, C[T - 1, j], T - 1, "encore ouverte", frais=False)
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

    Ajoute la variante « pas de sortie sur signal » (seuil_sortie=None, libellé
    « aucun (stop seul) ») quand un stop ou une durée maximale existe.
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
    autre_sortie = options.get("stop_atr") is not None or options.get("duree_max") is not None
    sorties = [s for s in grille_sortie if s is not None]
    if autre_sortie:
        sorties.append(None)          # variante « pas de sortie sur signal » (stop seul)
    lignes = []
    for sa in grille_achat:
        for ss in sorties:
            if ss is not None and ss >= sa:
                continue
            bt = backtest_portefeuille(pred_val, prices, sa, ss, **options)
            st = bt["stats"]
            col = st["stratégie (net de frais)"]
            lignes.append({"seuil_achat": sa,
                           "seuil_sortie": "aucun (stop seul)" if ss is None else ss,
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
def _lignes_portefeuille(portefeuille: dict) -> list[tuple[str, dict]]:
    """Accepte {"CAT": [{"date":..., "prix":...}, {...}]}, {"CAT": {"date":..., "prix":...}}
    ou {"CAT": "2026-10-12"} ; renvoie [(titre, {"date", "prix"?}), ...]."""
    out = []
    for t, info in portefeuille.items():
        infos = info if isinstance(info, (list, tuple)) else [info]
        for x in infos:
            out.append((t, x if isinstance(x, dict) else {"date": x}))
    return out


def decisions_portefeuille_du_jour(model, data, window: int, portefeuille: dict,
                                   seuil_achat: float, seuil_sortie: float | None = 0.5,
                                   max_positions: int = 10, max_entrees_jour: int = 2,
                                   max_par_titre: int = 2, ecart_min: int = 3,
                                   duree_max: int | None = None, duree_min: int = 1, prices=None,
                                   stop_atr: float | None = None, atr: pd.DataFrame | None = None,
                                   static=None, entree: str = "cloture", **autres_options) -> dict:
    """Quoi vendre, quoi garder, quoi acheter aujourd'hui.

    portefeuille : vos lignes actuelles. Un titre peut avoir 1 ou 2 lignes :
                   {"CAT":   [{"date": "2026-10-12", "prix": 352.1},
                              {"date": "2026-10-20", "prix": 360.4}],
                    "MC.PA": {"date": "2026-10-15", "prix": 698.5}}
                   ("prix" sert au stop ; facultatif si le stop est désactivé)
                   ou simplement {"MC.PA": "2026-10-15"}
    (recroisement n'est pas vérifiable ici : il faudrait l'historique des probas.)
    """
    from decision import decisions_du_jour

    jour = decisions_du_jour(model, data, window=window, seuil_achat=seuil_achat, static=static,
                             horizon=duree_max or 0, prices=prices, entree=entree)
    jour = jour.drop(columns=["sortie prévue"], errors="ignore")
    derniere = jour.attrs["date"]
    seances = data.df.index.get_level_values("date").unique().sort_values()
    if stop_atr is not None and atr is None and prices is not None:
        atr = atr_wilder(ohlc(prices))

    lignes = []
    for t, info in _lignes_portefeuille(portefeuille):
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
        elif duree_max is not None and k >= duree_max:
            action, motif = "VENDRE", "durée maximale"
        elif seuil_sortie is not None and k >= duree_min and np.isfinite(p) and p <= seuil_sortie:
            action, motif = "VENDRE", "signal de sortie"
        else:
            action = "GARDER"
            motif = "pas de probabilité aujourd'hui (score hors domaine ?)" if not np.isfinite(p) else ""
        lignes.append({"ticker": t, "date d'achat": d0.date(), "action": action, "motif": motif,
                       "proba": p, "séances détenues": k, "prix d'achat": info.get("prix", np.nan),
                       "dernier cours": cours, "stop": stop})
    cols_l = ["ticker", "date d'achat", "action", "motif", "proba", "séances détenues",
              "prix d'achat", "dernier cours", "stop"]
    lignes_df = pd.DataFrame(lignes, columns=cols_l)

    # Lignes restantes par titre après les ventes, et séances depuis le dernier achat
    reste = lignes_df[lignes_df["action"] == "GARDER"]
    n_reste = reste.groupby("ticker").size()
    depuis = reste.groupby("ticker")["séances détenues"].min()
    vendus = set(lignes_df.loc[lignes_df["action"] == "VENDRE", "ticker"])
    n_ventes = int((lignes_df["action"] == "VENDRE").sum())
    places = max_positions - (len(lignes_df) - n_ventes)
    n_achats = max(0, min(max_entrees_jour, places))

    def eligible(t):
        if t in vendus:
            return False
        n = int(n_reste.get(t, 0))
        return n == 0 or (n < max_par_titre and int(depuis.get(t, 0)) >= ecart_min)

    cand = jour[(jour["proba"] >= seuil_achat) & jour.index.map(eligible).astype(bool)]
    achats = cand.sort_values("proba", ascending=False).head(n_achats).copy()
    achats["ligne"] = [("2e ligne" if n_reste.get(t, 0) else "nouvelle") for t in achats.index]
    a_garder = [c for c in ["ligne", "proba", "dernier cours", "stop", "signaux récents"]
                if c in achats.columns]
    achats = achats[a_garder]

    print(f"\nPortefeuille : {len(lignes_df)} ligne(s), {n_ventes} à vendre, "
          f"{places} place(s) libre(s) après ventes, {len(achats)} achat(s) proposé(s) "
          f"(au plus {max_entrees_jour} par jour, {max_par_titre} lignes par titre, "
          f"{ecart_min} séances d'écart).")
    if len(cand) > len(achats):
        print(f"  {len(cand) - len(achats)} autre(s) titre(s) au-dessus du seuil, non retenu(s) "
              "faute de place ou au-delà du nombre d'achats par jour.")
    lignes_df = lignes_df.set_index("ticker")
    return {"vendre": lignes_df[lignes_df["action"] == "VENDRE"],
            "garder": lignes_df[lignes_df["action"] == "GARDER"],
            "acheter": achats, "toutes les probabilités": jour}
