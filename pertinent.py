"""
Scanner de TOUS les indicateurs produits par construire_panels (niveaux bruts, pas seulement
les franchissements de seuils déclarés dans signaux.py).

    from scanner_indicateurs import scanner
    pan = construire_panels(prices, Indicateurs, ParamsPF(min_titres=1), generateur=features_completes)
    res = scanner(pan, prices, horizon=22,
                  train=("2005-01-01", "2016-12-31"), val=("2017-01-01", "2019-12-31"))
    print(res["resume"].round(3).to_string())

Pour chaque indicateur (DataFrame dates x titres du dictionnaire `pan`) :
  - IC : corrélation de rang (Spearman) entre l'indicateur du jour et le rendement des
    `horizon` séances suivantes, calculée MOIS PAR MOIS (tous titres confondus), puis
    moyennée. t-stat = IC moyen / écart-type * racine(nombre de mois).
  - part des mois où l'IC a le signe moyen ;
  - même IC sur la validation : le signe doit être le même ;
  - fréquence de hausse par quintile (feuille « Quintiles ») : une relation monotone est bon signe,
    une relation en U suggère un seuil plutôt qu'un niveau ;
  - indicateurs redondants (corrélation de rang > 0,8 entre eux) : en garder un par groupe.
Le TEST n'est jamais utilisé. Résultats écrits dans out_dir/scanner_indicateurs.xlsx.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _close(prices) -> pd.DataFrame:
    if isinstance(prices, dict):
        return pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
    return prices


def _ic_mensuel(x: pd.Series, y: pd.Series) -> pd.Series:
    d = pd.concat({"x": x, "y": y}, axis=1).dropna()
    if d.empty:
        return pd.Series(dtype=float)
    mois = d.index.get_level_values("date").to_period("M")
    return d.groupby(mois).apply(
        lambda g: g["x"].corr(g["y"], method="spearman") if len(g) >= 20 else np.nan).dropna()


def scanner(pan: dict, prices, horizon: int = 22, train=("2005-01-01", "2016-12-31"),
            val=("2017-01-01", "2019-12-31"), exclure: tuple = (), seuil_redondance: float = 0.8,
            out_dir: str | None = "scanner_indicateurs") -> dict:
    close = _close(prices)
    fwd = np.log(close.shift(-horizon) / close)                     # rendement futur (log)
    fwd_long = fwd.stack().rename("fwd")
    fwd_long.index.names = ["date", "ticker"]

    def periode(s, a, b):
        dts = s.index.get_level_values("date")
        return s[(dts >= pd.Timestamp(a)) & (dts <= pd.Timestamp(b))]

    lignes, quint, series_tr = [], {}, {}
    for nom, df in pan.items():
        if nom in exclure or not isinstance(df, pd.DataFrame):
            continue
        df = df.reindex(columns=[c for c in df.columns if c in close.columns])
        if df.empty:
            continue
        x = df.stack().rename(nom)
        x.index.names = ["date", "ticker"]
        x = x.replace([np.inf, -np.inf], np.nan)
        if x.nunique() < 3:
            continue
        x_tr, y_tr = periode(x, *train), periode(fwd_long, *train)
        x_va, y_va = periode(x, *val), periode(fwd_long, *val)
        ic_tr, ic_va = _ic_mensuel(x_tr, y_tr), _ic_mensuel(x_va, y_va)
        if len(ic_tr) < 12:
            continue
        m, s = ic_tr.mean(), ic_tr.std()
        lignes.append({
            "indicateur": nom,
            "IC moyen (train)": m,
            "t-stat (train)": m / s * np.sqrt(len(ic_tr)) if s > 0 else np.nan,
            "mois de même signe (train)": float((np.sign(ic_tr) == np.sign(m)).mean()),
            "IC moyen (val)": ic_va.mean() if len(ic_va) else np.nan,
            "mois de même signe (val)": float((np.sign(ic_va) == np.sign(m)).mean()) if len(ic_va) else np.nan,
            "observations (train)": int(pd.concat([x_tr, y_tr], axis=1).dropna().shape[0]),
        })
        # quintiles sur le train : fréquence de hausse et rendement moyen par quintile
        d = pd.concat({"x": x_tr, "y": y_tr}, axis=1).dropna()
        try:
            q = pd.qcut(d["x"].rank(method="first"), 5, labels=["Q1 (bas)", "Q2", "Q3", "Q4", "Q5 (haut)"])
            quint[nom] = d.groupby(q, observed=True)["y"].agg(
                hausse=lambda v: (v > 0).mean(), rendement_moyen="mean")
        except ValueError:
            pass
        series_tr[nom] = x_tr

    if not lignes:
        raise ValueError("Aucun indicateur exploitable : vérifiez que les colonnes de `pan` "
                         "portent les mêmes noms de titres que `prices`.")
    res = pd.DataFrame(lignes).set_index("indicateur")
    res["même signe en val"] = np.sign(res["IC moyen (train)"]) == np.sign(res["IC moyen (val)"])
    res["prometteur"] = (res["t-stat (train)"].abs() >= 2) & res["même signe en val"] & \
        (res["mois de même signe (train)"] >= 0.55)
    res = res.sort_values("t-stat (train)", key=lambda s: -s.abs())

    # redondance (corrélation de rang entre indicateurs, sur un échantillon du train)
    X = pd.concat(series_tr, axis=1)
    if len(X) > 200_000:
        X = X.sample(200_000, random_state=0)
    corr = X.rank().corr()
    paires = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns)
              for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= seuil_redondance]
    redond = pd.DataFrame(paires, columns=["indicateur A", "indicateur B", "corrélation"]) \
        .sort_values("corrélation", key=abs, ascending=False)

    quintiles = pd.concat(quint, names=["indicateur", "quintile"]) if quint else pd.DataFrame()
    out = {"resume": res, "quintiles": quintiles, "redondances": redond, "correlations": corr}

    base = float((periode(fwd_long, *train) > 0).mean())
    print(f"Fréquence normale de hausse à {horizon} séances (train) : {base:.3f}")
    print(f"{int(res['prometteur'].sum())} indicateur(s) prometteur(s) sur {len(res)} "
          "(|t| >= 2, même signe en validation, >= 55 % des mois du bon signe).")
    if len(redond):
        print(f"{len(redond)} paire(s) d'indicateurs redondants (|corr| >= {seuil_redondance}).")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "scanner_indicateurs.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            res.to_excel(w, sheet_name="Résumé")
            if len(quintiles):
                quintiles.to_excel(w, sheet_name="Quintiles")
            redond.to_excel(w, sheet_name="Redondances", index=False)
            corr.to_excel(w, sheet_name="Corrélations")
        print(f"Résultats : {os.path.abspath(path)}")
    return out
