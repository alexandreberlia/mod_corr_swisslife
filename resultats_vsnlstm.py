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
            out_dir: str = "resultats") -> dict[str, pd.DataFrame]:
    """Écrit les résultats du modèle `model_path` sur la période `split`.

    split : "val" pendant la mise au point ; "test" une seule fois, à la fin.
    Renvoie aussi les tableaux, pour les afficher dans un notebook.
    """
    import matplotlib.pyplot as plt

    if os.environ.get("DISPLAY") is None and not _in_notebook():
        matplotlib.use("Agg")
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = load_model(model_path)
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
