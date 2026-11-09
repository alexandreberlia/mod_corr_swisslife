"""
Tri rapide des signaux : lesquels apportent de l'information, avant d'entraîner le réseau.

    from pertinence_signaux import pertinence
    tab = pertinence(data)                    # data = prepare_dataset(..., **reglages)
    print(tab.round(3).to_string())

Trois mesures, calculées sur l'ENTRAÎNEMENT (et vérifiées sur la validation, jamais le test) :
  1. Étude d'événement : après chaque déclenchement (fraîcheur = 1), la fréquence de hausse
     à l'horizon est-elle différente de la fréquence normale ? (+ intervalle de confiance)
  2. Logistique : coefficient du signal quand tous les signaux sont pris ensemble.
  3. Ablation : hausse de la log-loss de validation quand on RETIRE ce signal du modèle
     logistique (en millièmes ; > 0 = le signal aidait).
Fonctionne avec target="direction" (hausse 0/1) ; avec "return", la cible est convertie en signe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss


def _wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def pertinence(data, cols: list[str] | None = None, min_declenchements: int = 200) -> pd.DataFrame:
    df = data.df
    cols = cols or list(getattr(data, "signal_cols", [])) or \
        [c for c in data.feature_cols if c.startswith("sig_")]
    if not cols:
        raise ValueError("Aucune colonne de signal trouvée (signaux=True dans prepare_dataset ?).")
    y_all = (df["target"] > 0).astype(float).where(df["target"].notna())
    tr = (df["split"] == "train") & y_all.notna()
    va = (df["split"] == "val") & y_all.notna()
    base_tr, base_va = y_all[tr].mean(), y_all[va].mean()

    # 1. Étude d'événement (jour du déclenchement)
    lignes = []
    for c in cols:
        ev_tr = tr & (df[c] >= 0.999)
        ev_va = va & (df[c] >= 0.999)
        n, k = int(ev_tr.sum()), float(y_all[ev_tr].sum())
        lo, hi = _wilson(k, n)
        f_va = y_all[ev_va].mean() if ev_va.any() else np.nan
        lignes.append({"signal": c, "déclenchements (train)": n,
                       "hausse après signal (train)": k / n if n else np.nan,
                       "IC 95 % bas": lo, "IC 95 % haut": hi,
                       "écart vs normale (train)": (k / n - base_tr) if n else np.nan,
                       "écart vs normale (val)": f_va - base_va,
                       "déclenchements (val)": int(ev_va.sum())})
    tab = pd.DataFrame(lignes).set_index("signal")
    tab["significatif"] = (tab["IC 95 % bas"] > base_tr) | (tab["IC 95 % haut"] < base_tr)
    tab["même sens en val"] = np.sign(tab["écart vs normale (train)"]) == \
        np.sign(tab["écart vs normale (val)"])
    tab["assez fréquent"] = tab["déclenchements (train)"] >= min_declenchements

    # 2. Logistique sur tous les signaux (fraîcheurs), 3. ablation sur la validation
    X_tr, X_va = df.loc[tr, cols].fillna(0), df.loc[va, cols].fillna(0)
    y_tr, y_va = y_all[tr], y_all[va]
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X_tr, y_tr)
    auc_ref = roc_auc_score(y_va, lr.predict_proba(X_va)[:, 1])
    ll_ref = log_loss(y_va, lr.predict_proba(X_va)[:, 1])
    tab["coef logistique"] = pd.Series(lr.coef_[0], index=cols)
    perte = {}
    for c in cols:
        autres = [x for x in cols if x != c]
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X_tr[autres], y_tr)
        perte[c] = 1000 * (log_loss(y_va, m.predict_proba(X_va[autres])[:, 1]) - ll_ref)
    tab["perte val si retiré (‰ log-loss)"] = pd.Series(perte)

    print(f"Fréquence normale de hausse : train {base_tr:.3f} | val {base_va:.3f}")
    print(f"AUC validation, logistique avec tous les signaux : {auc_ref:.3f}")
    print("Lecture : un signal utile est assez fréquent, significatif sur le train, de même sens "
          "en validation, et son retrait dégrade la validation (perte > 0).")
    tab["utile"] = tab["assez fréquent"] & tab["significatif"] & tab["même sens en val"] & \
        (tab["perte val si retiré (‰ log-loss)"] > 0)
    return tab.sort_values(["utile", "écart vs normale (train)"], ascending=[False, False],
                           key=lambda s: s.abs() if s.name != "utile" else s)
