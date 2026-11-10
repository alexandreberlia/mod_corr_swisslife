"""
Les signaux sont-ils moins lisibles quand le score macro est élevé ?

Trois tests, du plus fondamental au plus proche de la stratégie :

    from lisibilite_regime import signaux_par_regime, auc_par_regime, trades_par_regime

    # 1. Les signaux eux-mêmes (entraînement, puis validation)
    t1 = signaux_par_regime(data, macro, periode="train")
    t1v = signaux_par_regime(data, macro, periode="val")

    # 2. Le modèle : AUC par régime sur la validation
    t2 = auc_par_regime(predict(model, loaders["val"]), macro)

    # 3. La stratégie : résultat des trades selon le score à l'entrée
    t3 = trades_par_regime(bt["trades"], macro)

`macro` = la table passée à prepare_dataset (score brut, avant normalisation).
Le score retenu pour chaque jour est celui déjà publié ce jour-là (même règle que le modèle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_prep import align_macro


def _score_du_jour(dates, macro: pd.DataFrame, score_col: str) -> pd.Series:
    m = macro[[score_col]] if isinstance(macro, pd.DataFrame) else macro.to_frame(score_col)
    return align_macro(pd.DatetimeIndex(dates), m)[score_col]


def _regime(s: pd.Series, seuil: float) -> pd.Series:
    return pd.Series(np.where(s > seuil, f"score > {seuil:g}", f"score <= {seuil:g}"),
                     index=s.index).where(s.notna())


def _z_diff(p1, n1, p2, n2):
    if min(n1, n2) == 0:
        return np.nan
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else np.nan


# ---------------------------------------------------------------------------
def signaux_par_regime(data, macro, score_col: str = "score_niveau", seuil: float = 75,
                       periode: str = "train") -> pd.DataFrame:
    """Pour chaque signal : écart de la fréquence de hausse après déclenchement par rapport
    à la fréquence normale DU MÊME RÉGIME, score élevé contre le reste.
    « perte de lisibilité » = |écart| plus petit quand le score est élevé."""
    df = data.df[data.df["split"] == periode]
    cols = list(getattr(data, "signal_cols", [])) or [c for c in data.feature_cols if c.startswith("sig_")]
    dates = df.index.get_level_values("date")
    sc = _score_du_jour(dates.unique(), macro, score_col)
    reg = _regime(sc, seuil).reindex(dates).to_numpy()
    y = (df["target"] > 0).astype(float).to_numpy()
    haut, bas = f"score > {seuil:g}", f"score <= {seuil:g}"

    lignes = []
    for c in cols:
        ev = df[c].to_numpy() >= 0.999
        r = {"signal": c}
        for nom, m in (("élevé", reg == haut), ("reste", reg == bas)):
            base = y[m].mean() if m.any() else np.nan
            sel = m & ev
            n = int(sel.sum())
            f = y[sel].mean() if n else np.nan
            r[f"déclenchements ({nom})"] = n
            r[f"hausse après signal ({nom})"] = f
            r[f"normale ({nom})"] = base
            r[f"écart ({nom})"] = f - base if n else np.nan
        n1, n2 = r["déclenchements (élevé)"], r["déclenchements (reste)"]
        # le signal perd-il son effet en régime élevé ? (comparaison des deux écarts)
        r["z (écart élevé - écart reste)"] = _z_diff(
            r["hausse après signal (élevé)"] - r["normale (élevé)"] + 0.5, n1,
            r["hausse après signal (reste)"] - r["normale (reste)"] + 0.5, n2) if n1 and n2 else np.nan
        r["moins lisible en régime élevé"] = abs(r["écart (élevé)"]) < abs(r["écart (reste)"]) \
            if n1 and n2 else np.nan
        lignes.append(r)
    out = pd.DataFrame(lignes).set_index("signal")
    part = np.mean(reg == haut)
    print(f"[{periode}] part des jours avec {haut} : {part:.1%} | fréquence normale de hausse : "
          f"élevé {out['normale (élevé)'].iloc[0]:.3f}, reste {out['normale (reste)'].iloc[0]:.3f}")
    print("Lecture : un signal perd sa lisibilité si |écart (élevé)| < |écart (reste)| ; "
          "la différence est nette si |z| > 2. Peu de déclenchements en régime élevé = prudence.")
    return out


# ---------------------------------------------------------------------------
def auc_par_regime(pred: pd.DataFrame, macro, score_col: str = "score_niveau",
                   seuil: float = 75) -> pd.DataFrame:
    """AUC du modèle (colonne proba, sortie de predict) par régime, avec un intervalle
    par bootstrap sur les mois. Avec la cible 'return', utilise q0.5 et le signe de y."""
    from sklearn.metrics import roc_auc_score
    col = "proba" if "proba" in pred.columns else ("q0.5" if "q0.5" in pred.columns else "pred")
    dates = pred.index.get_level_values("date")
    sc = _score_du_jour(dates.unique(), macro, score_col)
    reg = _regime(sc, seuil).reindex(dates).to_numpy()
    y = (pred["y"] > 0).astype(int).to_numpy()
    p = pred[col].to_numpy()
    mois = dates.to_period("M").to_numpy()
    rng = np.random.default_rng(0)
    lignes = []
    for nom in (f"score > {seuil:g}", f"score <= {seuil:g}"):
        m = reg == nom
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            lignes.append({"régime": nom, "observations": int(m.sum())})
            continue
        a = roc_auc_score(y[m], p[m])
        um = np.unique(mois[m])
        boots = []
        for _ in range(200):
            tir = rng.choice(um, len(um), replace=True)
            sel = np.concatenate([np.where(m & (mois == t))[0] for t in tir])
            if len(np.unique(y[sel])) == 2:
                boots.append(roc_auc_score(y[sel], p[sel]))
        lignes.append({"régime": nom, "observations": int(m.sum()), "mois": len(um), "AUC": a,
                       "AUC IC 90 % bas": np.percentile(boots, 5),
                       "AUC IC 90 % haut": np.percentile(boots, 95)})
    out = pd.DataFrame(lignes).set_index("régime")
    print("Lecture : si l'intervalle de l'AUC en régime élevé contient 0,5, le modèle n'y "
          "distingue pas mieux que le hasard.")
    return out


# ---------------------------------------------------------------------------
def trades_par_regime(trades: pd.DataFrame, macro, score_col: str = "score_niveau",
                      seuil: float = 75) -> pd.DataFrame:
    """Résultat des trades du backtest selon le score au jour de l'entrée."""
    t = trades.copy()
    t = t[t["motif de sortie"] != "encore ouverte"] if "motif de sortie" in t else t
    d = pd.to_datetime(t["date d'entrée"])
    sc = _score_du_jour(d.unique(), macro, score_col)
    t["régime à l'entrée"] = _regime(sc, seuil).reindex(d).to_numpy()
    out = t.groupby("régime à l'entrée").agg(
        trades=("gain", "size"), part_gagnante=("gain", lambda g: (g > 0).mean()),
        gain_moyen=("gain", "mean"), gain_median=("gain", "median"))
    print("Lecture : comparez la part gagnante et le gain moyen ; avec moins de 30 trades dans "
          "un régime, la différence n'est pas fiable.")
    return out
