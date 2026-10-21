"""
Analyse préalable : les signaux techniques se comportent-ils différemment
selon le régime du score macro ?

Pour chaque signal (franchissement de seuil) et chaque régime du score :
  - combien de fois le signal s'est produit,
  - à quelle fréquence il a eu raison (hausse à 4 séances pour un signal
    haussier, baisse pour un signal baissier),
  - comparé à la fréquence « normale » de hausse / baisse dans ce régime.

L'analyse porte UNIQUEMENT sur la période d'entraînement (2005-2018 par défaut) :
regarder la validation ou le test ici reviendrait à choisir le modèle sur eux.

Utilisation (dans votre notebook) :

    from analyse_regimes import analyse
    res = analyse(prices, indicators, "score_macro.csv", score_col="score",
                  start="2005-01-01", end="2018-12-31")
    res["synthese"]          # tableau principal
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data_prep import align_macro, indicators_to_long, load_macro, read_macro_table, to_long

# Les règles (SIGNAUX), l'horizon et les seuils de régime sont dans signaux.py,
# partagé avec le modèle : modifiez-les là-bas.
from signaux import (HORIZON, SEUIL_BAS, SEUIL_HAUT, SIGNAUX,  # noqa: E402,F401
                     calcul_signaux, franchissement, regime)

N_MIN = 30             # en dessous, un taux de réussite n'est pas interprétable


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - demi, centre + demi


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
def analyse(prices: pd.DataFrame, indicators: pd.DataFrame, macro_csv,
            score_col: str = "score", date_col: str = "date",
            publication_lag_days: int = 30,
            start: str = "2005-01-01", end: str = "2018-12-31",
            indicator_field_level: int | None = None,
            out_dir: str | None = "analyse_regimes") -> dict[str, pd.DataFrame]:
    # Données au format (date, ticker)
    px = to_long(prices)
    ind = indicators_to_long(indicators, px.index.get_level_values("ticker").unique(),
                             indicator_field_level)
    close_col = "Adj Close" if "Adj Close" in px.columns else "Close"
    df = ind.select_dtypes("number").join(px[[close_col]].rename(columns={close_col: "Close"}),
                                         how="inner")

    # Cible : le cours a-t-il monté à 4 séances ?
    fwd = df["Close"].groupby(level="ticker").shift(-HORIZON) / df["Close"] - 1
    df["rdt_4j"] = fwd
    df["hausse_4j"] = (fwd > 0).astype(float).where(fwd.notna())

    # Régime du score, avec son délai de publication (pas de regard vers le futur)
    macro = load_macro(macro_csv, date_col=date_col,
                       publication_lag_days=publication_lag_days, add_changes=False)
    score = align_macro(df.index.get_level_values("date"), macro[[score_col]])[score_col]
    df["score"] = score.reindex(df.index.get_level_values("date")).values
    df["regime"] = regime(df["score"]).values

    sig = calcul_signaux(df)

    # Période d'analyse uniquement
    dates = df.index.get_level_values("date")
    keep = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)) \
        & df["hausse_4j"].notna() & df["regime"].notna()
    if keep.sum() == 0:
        raise ValueError("Aucune ligne exploitable : prix, indicateurs et score ne se recoupent "
                         "pas sur la période. Lancez diagnostic(prices, panels, fichier_macro, "
                         "score_col=...) pour voir à quelle étape les données disparaissent.")
    df, sig = df[keep], sig[keep]

    # Fréquences de référence par régime
    base = df.groupby("regime")["hausse_4j"].mean()

    lignes = []
    for s in SIGNAUX:
        nom = s["nom"]
        if nom not in sig.columns:
            continue
        for reg in ["bas", "neutre", "élevé", "tous"]:
            mask = sig[nom] if reg == "tous" else sig[nom] & (df["regime"] == reg)
            ev = df[mask]
            n = len(ev)
            succes = ev["hausse_4j"] if s["sens"] > 0 else 1 - ev["hausse_4j"]
            k = int(succes.sum())
            ref_h = df["hausse_4j"].mean() if reg == "tous" else base.get(reg, np.nan)
            ref = ref_h if s["sens"] > 0 else 1 - ref_h
            lo, hi = wilson(k, n)
            lignes.append({
                "signal": nom, "sens": "hausse" if s["sens"] > 0 else "baisse", "régime": reg,
                "nb signaux": n,
                "nb mois distincts": ev.index.get_level_values("date").to_period("M").nunique(),
                "taux de réussite": k / n if n else np.nan,
                "IC95 bas": lo, "IC95 haut": hi,
                "référence du régime": ref,
                "avantage (points)": (k / n - ref) * 100 if n else np.nan,
                "rdt moyen 4j (%)": ev["rdt_4j"].mean() * 100 if n else np.nan,
            })
    synthese = pd.DataFrame(lignes)
    synthese["significatif"] = (synthese["IC95 bas"] > synthese["référence du régime"]) | \
                               (synthese["IC95 haut"] < synthese["référence du régime"])

    avantage = synthese.pivot(index="signal", columns="régime", values="avantage (points)") \
        .reindex(index=[s["nom"] for s in SIGNAUX if s["nom"] in sig.columns],
                 columns=["bas", "neutre", "élevé", "tous"])
    effectifs = synthese.pivot(index="signal", columns="régime", values="nb signaux") \
        .reindex(index=avantage.index, columns=avantage.columns)

    # Stabilité : même calcul sur les deux moitiés de la période
    milieu = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    stab = []
    for s in SIGNAUX:
        if s["nom"] not in sig.columns:
            continue
        for moitie, m in [("1re moitié", dates[keep] < milieu), ("2e moitié", dates[keep] >= milieu)]:
            for reg in ["bas", "neutre", "élevé"]:
                mask = sig[s["nom"]].values & m & (df["regime"] == reg).values
                ev = df[mask]
                ref_h = df[m & (df["regime"] == reg).values]["hausse_4j"].mean()
                succes = ev["hausse_4j"] if s["sens"] > 0 else 1 - ev["hausse_4j"]
                ref = ref_h if s["sens"] > 0 else 1 - ref_h
                stab.append({"signal": s["nom"], "moitié": moitie, "régime": reg, "n": len(ev),
                             "avantage (points)": (succes.mean() - ref) * 100 if len(ev) else np.nan})
    stabilite = pd.DataFrame(stab).pivot_table(index="signal", columns=["régime", "moitié"],
                                               values="avantage (points)").reindex(avantage.index)

    regimes = df.groupby("regime").agg(
        jours_titres=("hausse_4j", "size"),
        mois=("hausse_4j", lambda x: x.index.get_level_values("date").to_period("M").nunique()),
        freq_hausse_4j=("hausse_4j", "mean"))

    res = {"synthese": synthese, "avantage": avantage, "effectifs": effectifs,
           "stabilite": stabilite, "regimes": regimes}

    print(f"Période analysée : {start} -> {end} | {len(df):,} jours-titres")
    print("\nRégimes du score (fréquence de hausse à 4 séances = référence) :")
    print(regimes.round(3).to_string())
    print("\nAvantage des signaux sur la référence du régime, en points de % :")
    print(avantage.round(1).to_string())
    print("\nNombre de signaux :")
    print(effectifs.fillna(0).astype(int).to_string())

    if out_dir:
        _export(res, out_dir)
    return res


def diagnostic(prices, indicators, macro_csv, score_col: str = "score",
               date_col: str = "date", start: str = "2005-01-01",
               end: str = "2018-12-31", publication_lag_days: int = 30) -> None:
    """Vérifie chaque étape de la préparation et indique où les données disparaissent."""
    ok = True

    def titre(t):
        print(f"\n=== {t} ===")

    def alerte(msg):
        nonlocal ok
        ok = False
        print("  !! " + msg)

    # 1. Prix
    titre("1. Prix")
    px = to_long(prices)
    tk_px = set(px.index.get_level_values("ticker"))
    d_px = px.index.get_level_values("date")
    print(f"  {len(tk_px)} titres, du {d_px.min().date()} au {d_px.max().date()}")
    print(f"  colonnes : {list(px.columns)}")
    print(f"  exemples de titres : {sorted(map(str, tk_px))[:5]}")
    if not {"Close", "Adj Close"} & set(px.columns):
        alerte("pas de colonne Close / Adj Close dans les prix.")

    # 2. Indicateurs
    titre("2. Indicateurs")
    ind = indicators_to_long(indicators, list(tk_px))
    tk_ind = set(ind.index.get_level_values("ticker"))
    d_ind = ind.index.get_level_values("date")
    print(f"  {len(tk_ind)} titres, du {d_ind.min().date()} au {d_ind.max().date()}")
    print(f"  indicateurs : {list(ind.columns)}")
    print(f"  exemples de titres : {sorted(map(str, tk_ind))[:5]}")
    print("  part de valeurs manquantes :",
          ", ".join(f"{c} {ind[c].isna().mean():.0%}" for c in ind.columns))
    non_num = [c for c in ind.columns if not pd.api.types.is_numeric_dtype(ind[c])]
    if non_num:
        alerte(f"indicateurs non numériques (ignorés) : {non_num}")
    attendus = {s["col"] for s in SIGNAUX}
    manquants = attendus - set(ind.columns)
    if manquants:
        alerte(f"indicateurs attendus par SIGNAUX mais absents : {sorted(manquants)} "
               "-> corrigez les noms dans la liste SIGNAUX.")

    # 3. Recoupement prix / indicateurs
    titre("3. Recoupement prix x indicateurs")
    communs = tk_px & tk_ind
    print(f"  titres communs : {len(communs)} / prix {len(tk_px)} / indicateurs {len(tk_ind)}")
    if not communs:
        alerte("aucun titre en commun : les noms diffèrent entre prices et panels "
               f"(ex. {sorted(map(str, tk_px))[:2]} contre {sorted(map(str, tk_ind))[:2]}).")
    joint = ind.join(px[[]], how="inner")
    print(f"  lignes (date, titre) communes : {len(joint):,}")
    if communs and len(joint) == 0:
        alerte("titres communs mais aucune date commune : vérifiez le format des dates "
               f"(prix : {d_px[:1].tolist()}, indicateurs : {d_ind[:1].tolist()}).")

    # 4. Score macro
    titre("4. Score macro")
    try:
        raw = read_macro_table(macro_csv, date_col)
    except Exception as e:
        alerte(str(e))
        print("\nDiagnostic interrompu.")
        return
    print(f"  colonnes lues : {list(raw.columns)}")
    print(raw.head(3).to_string())
    if score_col not in raw.columns:
        alerte(f"colonne {score_col!r} absente -> précisez score_col=... parmi {list(raw.columns)}")
        print("\nDiagnostic interrompu.")
        return
    sc = raw[score_col]
    if not pd.api.types.is_numeric_dtype(sc):
        alerte(f"la colonne {score_col!r} n'est pas numérique (exemples : {sc.head(3).tolist()}).")
        print("\nDiagnostic interrompu.")
        return
    else:
        print(f"  {score_col} : de {sc.min():.2f} à {sc.max():.2f}, "
              f"du {raw[date_col].min().date()} au {raw[date_col].max().date()}")
        if sc.max() <= 1:
            alerte("le score semble compris entre 0 et 1 : avec des seuils à 25 et 75, "
                   "tout tombera dans le régime « bas ». Multipliez par 100 ou changez "
                   "SEUIL_BAS / SEUIL_HAUT.")

    # 5. Score sur la période d'analyse
    titre(f"5. Score sur {start} -> {end}")
    macro = load_macro(raw, date_col=date_col, publication_lag_days=publication_lag_days,
                       add_changes=False)
    dates = joint.index.get_level_values("date")
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if len(dates) == 0:
        alerte(f"aucune donnée prix x indicateurs entre {start} et {end}.")
    else:
        al = align_macro(dates, macro[[score_col]])[score_col]
        print(f"  jours de bourse avec un score disponible : {al.notna().mean():.0%}")
        if al.notna().mean() == 0:
            alerte("aucun score disponible sur la période : le fichier macro commence-t-il "
                   "avant la période (en tenant compte du délai de publication) ?")
        else:
            print("  répartition des régimes (en jours) :",
                  regime(al).value_counts().to_dict())

    print("\nTout semble cohérent." if ok else "\nCorrigez les points marqués « !! » "
          "puis relancez analyse(...).")


def _export(res: dict, out_dir: str) -> None:
    import matplotlib
    if not _in_notebook():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(os.path.join(out_dir, "analyse_regimes.xlsx"), engine="openpyxl") as w:
        res["synthese"].round(4).to_excel(w, sheet_name="Synthèse", index=False)
        res["avantage"].round(2).to_excel(w, sheet_name="Avantage (points)")
        res["effectifs"].to_excel(w, sheet_name="Nombre de signaux")
        res["stabilite"].round(2).to_excel(w, sheet_name="Stabilité")
        res["regimes"].round(4).to_excel(w, sheet_name="Régimes")

    av, n = res["avantage"], res["effectifs"]
    fiable = av.where(n >= N_MIN)                 # cases trop peu fournies : grisées
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(av) + 1.8))
    lim = np.nanmax(np.abs(fiable.values)) if np.isfinite(fiable.values).any() else 1
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(fiable.values), cmap=cmap, vmin=-lim, vmax=lim,
                   aspect="auto")
    ax.set_xticks(range(av.shape[1]), [f"score {c}" if c != "tous" else "tous" for c in av.columns])
    ax.set_yticks(range(av.shape[0]), av.index)
    for i in range(av.shape[0]):
        for j in range(av.shape[1]):
            v, k = av.values[i, j], n.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}\n(n={int(k)})", ha="center", va="center", fontsize=8)
    ax.set_title("Avantage du signal sur la référence du régime (points de %)\n"
                 f"vert = a plus souvent raison que le hasard ; gris = moins de {N_MIN} signaux")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "avantage_par_regime.png"), dpi=130)
    plt.close(fig)
    print(f"\nFichiers écrits dans : {os.path.abspath(out_dir)}")


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tests : python analyse_regimes.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Détection des franchissements sur une petite série construite à la main
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=6), ["A"]],
                                     names=["date", "ticker"])
    s = pd.Series([40, 44, 46, 50, 44, 45], index=idx, dtype=float)
    assert franchissement(s, 45, "up").tolist() == [False, False, True, False, False, True]
    assert franchissement(s, 45, "down").tolist() == [False, False, False, False, True, False]
    # pas de franchissement « entre » deux titres
    idx2 = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A"), (pd.Timestamp("2020-01-01"), "B")],
                                     names=["date", "ticker"])
    assert not franchissement(pd.Series([40.0, 50.0], index=idx2), 45, "up").any()

    # 2. Pipeline complet sur données simulées, avec des indicateurs calculés depuis les prix
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2004-06-01", "2020-12-31")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    closes = pd.DataFrame({t: 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
                           for t in tickers}, index=dates)
    prices = pd.concat({"Close": closes}, axis=1)

    def indic(c: pd.Series) -> pd.DataFrame:
        d = c.diff()
        rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()
        ma, sd = c.rolling(20).mean(), c.rolling(20).std()
        low, high = c.rolling(14).min(), c.rolling(14).max()
        k = 100 * (c - low) / (high - low)
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        er = (c - c.shift(10)).abs() / c.diff().abs().rolling(10).sum()
        return pd.DataFrame({"RSI": 100 - 100 / (1 + rs), "BB_pct": (c - (ma - 2 * sd)) / (4 * sd),
                             "rev_5": c.pct_change(5), "stoch_long": k.rolling(3).mean(),
                             "MACD_hist": macd - macd.ewm(span=9).mean(), "ER": er})

    ind = pd.concat({t: indic(closes[t]) for t in tickers}, names=["ticker", "date"]) \
        .swaplevel().sort_index()
    months = pd.date_range("2004-01-01", "2020-12-01", freq="MS")
    sc = np.zeros(len(months))
    for i in range(1, len(months)):                       # score persistant, entre 0 et 100
        sc[i] = 0.9 * sc[i - 1] + rng.normal(0, 12)
    macro = pd.DataFrame({"date": months, "score": np.clip(50 + 1.6 * sc, 0, 100)})

    res = analyse(prices, ind, macro, score_col="score", out_dir="analyse_regimes_demo")
    syn = res["synthese"]
    assert set(syn["régime"]) == {"bas", "neutre", "élevé", "tous"}
    assert (syn.loc[syn["régime"] == "tous", "nb signaux"] > 0).all()
    # Sur des prix aléatoires, aucun signal ne doit avoir d'avantage réel
    assert syn.loc[syn["régime"] == "tous", "avantage (points)"].abs().max() < 5
    print("\nTests OK.")
