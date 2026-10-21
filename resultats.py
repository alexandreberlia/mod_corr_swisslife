"""
Rassemble les résultats d'un modèle entraîné dans des fichiers lisibles.

Après l'entraînement, une seule ligne :

    from resultats import rapport
    rapport("best_model.pt", loaders)                 # sur la validation

Elle crée un dossier "resultats/" à côté de votre script ou notebook, avec :
    rapport.xlsx               classeur Excel, un onglet par type de résultat
    1_courbes_apprentissage.png
    2_ic_par_mois.png
    3_importance_variables.png
    4_importance_dans_le_temps.png
et affiche un résumé chiffré dans la console.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

from train import load_model, plot_history, predict


def _ic(d: pd.DataFrame, col: str) -> float:
    return d[col].corr(d["y"], method="spearman")


def rapport(model_path: str, loaders: dict, split: str = "val",
            out_dir: str = "resultats", data=None) -> dict[str, pd.DataFrame]:
    """Écrit les résultats du modèle `model_path` sur la période `split`.

    split : "val" pendant la mise au point ; "test" une seule fois, à la fin.
    data  : sortie de prepare_dataset ; nécessaire pour le tableau « lecture par
            régime » d'un modèle de probabilité.
    Renvoie aussi les tableaux, pour les afficher dans un notebook.
    """
    import matplotlib.pyplot as plt

    if os.environ.get("DISPLAY") is None and not _in_notebook():
        matplotlib.use("Agg")
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = load_model(model_path)
    if getattr(model, "task", "quantile") == "proba":
        return rapport_proba(model, ckpt, loaders, split, out_dir, data)
    history = pd.DataFrame(ckpt["history"]).set_index("epoch")
    pred = predict(model, loaders[split], with_weights=True)
    mid = "q0.5" if "q0.5" in pred.columns else ("pred" if "pred" in pred.columns
                                                  else pred.columns[1])
    has_q = {"q0.1", "q0.9"} <= set(pred.columns)
    wcols = [c for c in pred.columns if c.startswith("w_")]

    # ----- Tableaux -----
    resume = {
        "Période évaluée": split,
        "Du": str(pred.index.get_level_values("date").min().date()),
        "Au": str(pred.index.get_level_values("date").max().date()),
        "Nombre de prédictions": len(pred),
        "Meilleure époque": ckpt["best_epoch"],
        "IC global (Spearman)": round(_ic(pred, mid), 4),
        "Taux de bon signe": round(float(np.mean(np.sign(pred[mid]) == np.sign(pred["y"]))), 4),
    }
    if has_q:
        cov = ((pred["y"] >= pred["q0.1"]) & (pred["y"] <= pred["q0.9"])).mean()
        resume["Couverture [q0,1 ; q0,9] (cible 0,80)"] = round(float(cov), 4)
    by_month = pred.groupby(pd.Grouper(level="date", freq="ME"))
    ic_mois = by_month.apply(lambda d: _ic(d, mid)).rename("IC").to_frame()
    ic_mois.index = ic_mois.index.strftime("%Y-%m")
    resume["Part des mois avec IC > 0"] = round(float((ic_mois["IC"] > 0).mean()), 4)
    resume_df = pd.Series(resume, name="Valeur").to_frame()

    ic_titre = pred.groupby(level="ticker").apply(
        lambda d: pd.Series({"IC": _ic(d, mid), "Nombre de jours": len(d),
                             "Taux de bon signe": np.mean(np.sign(d[mid]) == np.sign(d["y"]))})
    ).sort_values("IC", ascending=False)

    importance = (pred[wcols].mean().sort_values(ascending=False)
                  .rename(lambda c: c[2:]).rename("Poids moyen").to_frame())
    importance_mois = by_month[wcols].mean().rename(columns=lambda c: c[2:])
    importance_mois.index = importance_mois.index.strftime("%Y-%m")

    preds_lisibles = pred.copy()
    for c in [c for c in pred.columns if c.startswith("q") or c in ("pred", "y")]:
        preds_lisibles[c] = np.expm1(pred[c]) * 100          # rendements log -> %
    preds_lisibles = preds_lisibles.rename(
        columns={c: f"{c} (%)" for c in pred.columns if c.startswith("q") or c in ("pred", "y")})

    tables = {"Résumé": resume_df, "Historique": history, "IC par mois": ic_mois,
              "IC par titre": ic_titre, "Importance variables": importance,
              "Importance par mois": importance_mois,
              "Prédictions": preds_lisibles.reset_index()}

    # ----- Excel -----
    xlsx = os.path.join(out_dir, "rapport.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for name, t in tables.items():
            t.to_excel(w, sheet_name=name, index=(name != "Prédictions"))
            ws = w.sheets[name]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 40)
            ws.freeze_panes = "B2"

    # ----- Graphiques -----
    plot_history(history, os.path.join(out_dir, "1_courbes_apprentissage.png"))

    fig, ax = plt.subplots(figsize=(11, 3.8))
    colors = np.where(ic_mois["IC"] >= 0, "tab:green", "tab:red")
    ax.bar(ic_mois.index, ic_mois["IC"], color=colors)
    ax.axhline(_ic(pred, mid), color="black", lw=1, ls="--", label="IC global")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"IC par mois ({split}) : vert = le modèle a eu raison en moyenne ce mois-là")
    ax.set_ylabel("IC (Spearman)")
    step = max(1, len(ic_mois) // 12)
    ax.set_xticks(range(0, len(ic_mois), step), ic_mois.index[::step], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_ic_par_mois.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(importance) + 1.2))
    imp = importance["Poids moyen"].iloc[::-1]
    ax.barh(imp.index, imp.values, color="tab:blue")
    ax.axvline(1 / len(imp), color="grey", ls="--", lw=1, label="poids si toutes égales")
    ax.set_xlabel("poids moyen du VSN")
    ax.set_title("Variables les plus utilisées par le modèle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_importance_variables.png"), dpi=130)

    top = importance.index[:6]
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in top:
        ax.plot(range(len(importance_mois)), importance_mois[c], label=c)
    ax.set_xticks(range(0, len(importance_mois), step), importance_mois.index[::step],
                  rotation=45, ha="right")
    ax.set_ylabel("poids moyen du VSN")
    ax.set_title("Évolution du poids des 6 variables principales")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_importance_dans_le_temps.png"), dpi=130)
    plt.close("all")

    # ----- Résumé console -----
    print(f"\nRésultats écrits dans : {os.path.abspath(out_dir)}")
    for k, v in resume.items():
        print(f"  {k:<42} {v}")
    print("  Variables principales :",
          ", ".join(f"{k} {v:.2f}" for k, v in importance["Poids moyen"].head(5).items()))
    return tables


def _regime_label(pred: pd.DataFrame) -> pd.Series | None:
    if {"ctx_regime_bas", "ctx_regime_eleve"} <= set(pred.columns):
        return pd.Series(np.where(pred["ctx_regime_eleve"] == 1, "élevé",
                                  np.where(pred["ctx_regime_bas"] == 1, "bas", "neutre")),
                         index=pred.index, name="régime")
    return None


def _write_excel(tables: dict, path: str, no_index=("Prédictions",)) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, t in tables.items():
            t.to_excel(w, sheet_name=name[:31], index=(name not in no_index))
            ws = w.sheets[name[:31]]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 40)
            ws.freeze_panes = "B2"


def rapport_proba(model, ckpt, loaders, split, out_dir, data=None) -> dict[str, pd.DataFrame]:
    """Rapport d'un modèle qui prédit une probabilité de hausse."""
    import matplotlib.pyplot as plt

    from signaux import SIGNAUX
    from train import auc, proba_metrics

    history = pd.DataFrame(ckpt["history"]).set_index("epoch")
    pred = predict(model, loaders[split], with_weights=True)
    p, y = pred["proba"].to_numpy(), pred["y"].to_numpy()
    m = proba_metrics(p, y)
    wcols = [c for c in pred.columns if c.startswith("w_")]
    reg = _regime_label(pred)

    by_month = pred.groupby(pd.Grouper(level="date", freq="ME"))
    auc_mois = by_month.apply(lambda d: auc(d["proba"].to_numpy(), d["y"].to_numpy())) \
        .rename("AUC").to_frame()
    auc_mois.index = auc_mois.index.strftime("%Y-%m")

    resume = pd.Series({
        "Période évaluée": split,
        "Du": str(pred.index.get_level_values("date").min().date()),
        "Au": str(pred.index.get_level_values("date").max().date()),
        "Nombre de prédictions": len(pred),
        "Meilleure époque": ckpt["best_epoch"],
        "AUC (0,5 = hasard)": round(m["auc"], 4),
        "Part des mois avec AUC > 0,5": round(float((auc_mois["AUC"] > 0.5).mean()), 4),
        "Fréquence réelle de hausse": round(m["base"], 4),
        "Probabilité moyenne prédite": round(float(p.mean()), 4),
        "Taux de bonne direction (seuil 50 %)": round(m["acc"], 4),
        "Gain de log-loss vs fréquence moyenne (> 0 = utile)": round(m["gain_logloss"], 5),
        "Score de Brier (plus bas = mieux)": round(m["brier"], 5),
    }, name="Valeur").to_frame()

    # Calibration : quand le modèle annonce x %, le titre monte-t-il x % du temps ?
    bornes = [0, .3, .4, .45, .5, .55, .6, .7, 1]
    tranche = pd.cut(pred["proba"], bornes, include_lowest=True)
    calib = pred.groupby(tranche, observed=True).agg(
        n=("y", "size"), proba_moyenne=("proba", "mean"), hausse_reelle=("y", "mean"))
    calib.index = calib.index.astype(str)

    auc_titre = pred.groupby(level="ticker").apply(
        lambda d: pd.Series({"AUC": auc(d["proba"].to_numpy(), d["y"].to_numpy()),
                             "Nombre de jours": len(d)})).sort_values("AUC", ascending=False)

    importance = (pred[wcols].mean().sort_values(ascending=False)
                  .rename(lambda c: c[2:]).rename("Poids moyen").to_frame())
    tables = {"Résumé": resume, "Historique": history, "Calibration": calib,
              "AUC par mois": auc_mois, "AUC par titre": auc_titre,
              "Importance variables": importance}
    if reg is not None:
        imp_reg = pred[wcols].groupby(reg).mean().T.rename(index=lambda c: c[2:])
        tables["Importance par régime"] = imp_reg.reindex(columns=["bas", "neutre", "élevé"])

    # Lecture par régime : après chaque signal, quelle probabilité le modèle donne-t-il ?
    lecture = None
    if data is not None and getattr(data, "signal_cols", None):
        evt = data.df.loc[pred.index, data.signal_cols] == 1.0      # jour du franchissement
        reg_ = reg if reg is not None else pd.Series("tous", index=pred.index)
        base = pred.groupby(reg_)["y"].mean()
        noms = {f"sig_{s['id']}": (s["nom"], s["sens"]) for s in SIGNAUX}
        lignes = []
        for c in data.signal_cols:
            nom, sens = noms.get(c, (c, 0))
            for r in ["bas", "neutre", "élevé"] if reg is not None else ["tous"]:
                mask = evt[c] & (reg_ == r)
                d = pred[mask.to_numpy()]
                lignes.append({"signal": nom, "sens attendu": {1: "hausse", -1: "baisse"}.get(sens, ""),
                               "régime": r, "n": len(d),
                               "proba prédite moyenne": d["proba"].mean() if len(d) else np.nan,
                               "hausse réalisée": d["y"].mean() if len(d) else np.nan,
                               "fréquence de hausse du régime": base.get(r, np.nan)})
        lecture = pd.DataFrame(lignes)
        lecture["écart prédit vs régime (points)"] = \
            (lecture["proba prédite moyenne"] - lecture["fréquence de hausse du régime"]) * 100
        tables["Lecture par régime"] = lecture

    lisibles = pred.copy()
    lisibles["proba"] = (lisibles["proba"] * 100).round(1)
    lisibles = lisibles.rename(columns={"proba": "proba hausse (%)", "y": "hausse réalisée (1/0)"})
    tables["Prédictions"] = lisibles.reset_index()

    _write_excel(tables, os.path.join(out_dir, "rapport.xlsx"))

    # ----- Graphiques -----
    plot_history(history, os.path.join(out_dir, "1_courbes_apprentissage.png"))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.2, 0.8], [0.2, 0.8], color="grey", ls="--", lw=1, label="calibration parfaite")
    ax.scatter(calib["proba_moyenne"], calib["hausse_reelle"], s=np.sqrt(calib["n"]) * 4,
               color="tab:blue", label="tranches de probabilité")
    ax.set_xlabel("probabilité annoncée par le modèle")
    ax.set_ylabel("fréquence de hausse réellement observée")
    ax.set_title("Calibration : les points doivent suivre la diagonale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_calibration.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    vals = auc_mois["AUC"] - 0.5
    ax.bar(auc_mois.index, vals, bottom=0.5, color=np.where(vals >= 0, "tab:green", "tab:red"))
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.axhline(m["auc"], color="black", ls="--", lw=1, label="AUC globale")
    step = max(1, len(auc_mois) // 12)
    ax.set_xticks(range(0, len(auc_mois), step), auc_mois.index[::step], rotation=45, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC par mois ({split}) : vert = mieux que le hasard ce mois-là")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_auc_par_mois.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(importance) + 1.2))
    imp = importance["Poids moyen"].iloc[::-1]
    ax.barh(imp.index, imp.values, color="tab:blue")
    ax.axvline(1 / len(imp), color="grey", ls="--", lw=1, label="poids si toutes égales")
    ax.set_xlabel("poids moyen du VSN")
    ax.set_title("Variables les plus utilisées par le modèle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_importance_variables.png"), dpi=130)

    if lecture is not None and reg is not None:
        piv = lecture.pivot(index="signal", columns="régime",
                            values="écart prédit vs régime (points)")[["bas", "neutre", "élevé"]]
        nn_ = lecture.pivot(index="signal", columns="régime", values="n")[["bas", "neutre", "élevé"]]
        order = [noms[c][0] for c in data.signal_cols if noms.get(c, (c,))[0] in piv.index]
        piv, nn_ = piv.reindex(order), nn_.reindex(order)
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(piv) + 1.8))
        lim = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(piv.values), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(3), ["score bas", "score neutre", "score élevé"])
        ax.set_yticks(range(len(piv)), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v, k = piv.values[i, j], nn_.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.1f}\n(n={int(k)})", ha="center", va="center", fontsize=8)
        ax.set_title("Lecture du modèle après chaque signal, selon le régime\n"
                     "probabilité prédite moins fréquence normale de hausse (points)")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "5_lecture_par_regime.png"), dpi=130)
    plt.close("all")

    print(f"\nRésultats écrits dans : {os.path.abspath(out_dir)}")
    print(resume.to_string())
    return tables


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    # Démonstration avec le modèle du test de train.py (données simulées)
    from data_prep import load_macro, prepare_dataset, ticker_onehot
    from windows import make_loaders

    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2012-01-01", "2024-12-31")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    sig = {}
    for t in tickers:
        s = rng.normal(0, 1, len(dates))
        r = 0.01 * (0.3 * np.roll(s, 1) + rng.normal(0, 1, len(dates)))
        raw[("Close", t)] = 100 * np.exp(np.cumsum(r))
        raw[("Volume", t)] = rng.integers(1e5, 1e6, len(dates))
        sig[t] = s
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    ind = pd.DataFrame({"signal": np.stack([sig[t] for t in tickers], 1).ravel(),
                        "RSI": rng.uniform(0, 100, len(idx))}, index=idx)
    months = pd.date_range("2011-06-01", "2024-12-01", freq="MS")
    macro = load_macro(pd.DataFrame({"date": months, "score": rng.normal(0, 1, len(months))}))
    pdata = prepare_dataset(raw, ind, macro, train_end="2020-12-31", val_end="2022-12-31")
    loaders = make_loaders(pdata, window=20, batch_size=128, static=ticker_onehot(pdata.tickers))

    rapport("test_model.pt", loaders, split="val", out_dir="resultats_demo")
