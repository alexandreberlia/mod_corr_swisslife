"""diagnostics.py — Verifications a faire AVANT de croire une sortie de lead_rank.

Trois familles de tests, par ordre de gravite.

1. STATIONNARITE (le plus grave)
   Une regression entre deux series integrees produit des t eleves et des R2
   flatteurs sans aucun lien reel : c'est la regression fallacieuse de
   Granger-Newbold (1974). Aucune correction d'erreur-type n'y remedie — ni HAC,
   ni bootstrap — parce que le probleme n'est pas la variance de l'estimateur
   mais sa non-convergence. La seule parade est de transformer la serie.
   Verdict operationnel : ADF (H0 = racine unitaire) et KPSS (H0 = stationnaire)
   doivent CONCORDER. S'ils divergent, le cas est ambigu et il faut trancher a
   la main.

2. AUTOCORRELATION DES RESIDUS
   Attendue et NORMALE ici : a l'horizon k>1 les observations se chevauchent
   (y_{t+k} et y_{t+1+k} partagent k-1 trimestres), donc les residus suivent
   mecaniquement un MA(k-1). Ce n'est pas un defaut de specification, c'est une
   consequence du dispositif. Newey-West la neutralise a condition que la
   fenetre m soit >= k. Le test sert donc a VERIFIER QUE m EST ASSEZ GRAND,
   pas a rejeter le modele.

3. FORME FONCTIONNELLE ET HETEROSCEDASTICITE
   RESET de Ramsey : la relation est-elle vraiment lineaire ? White : la
   variance des residus depend-elle des regresseurs ? Le second est moins
   critique puisque HAC est deja robuste a l'heteroscedasticite.

Dependances : numpy, pandas, statsmodels, scipy
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Stationnarite
# ---------------------------------------------------------------------------

def stationarity(s: pd.Series, name: str = "") -> dict:
    """ADF + KPSS. Verdict fonde sur la CONCORDANCE des deux tests."""
    from statsmodels.tsa.stattools import adfuller, kpss
    x = pd.Series(s).dropna().to_numpy(float)
    if len(x) < 30:
        return dict(serie=name, n=len(x), adf_p=np.nan, kpss_p=np.nan,
                    verdict="echantillon trop court")
    try:
        adf_p = float(adfuller(x, autolag="AIC")[1])
    except Exception:
        adf_p = np.nan
    try:
        kpss_p = float(kpss(x, regression="c", nlags="auto")[1])
    except Exception:
        kpss_p = np.nan

    if np.isnan(adf_p) or np.isnan(kpss_p):
        v = "indetermine"
    elif adf_p > 0.10 and kpss_p < 0.05:
        v = "NON STATIONNAIRE"
    elif adf_p < 0.05 and kpss_p > 0.05:
        v = "stationnaire"
    elif adf_p < 0.05 and kpss_p < 0.05:
        v = "ambigu (peut-etre tendance deterministe)"
    else:
        v = "ambigu (faible puissance)"
    return dict(serie=name, n=len(x), adf_p=round(adf_p, 4),
                kpss_p=round(kpss_p, 4), verdict=v)


def screen_panel(panel: pd.DataFrame, min_obs: int = 40) -> pd.DataFrame:
    """Passe tout le panel au crible. A LANCER EN PREMIER, avant tout classement."""
    rows = [stationarity(panel[c], c) for c in panel.columns
            if panel[c].notna().sum() >= min_obs]
    out = pd.DataFrame(rows)
    ordre = {"NON STATIONNAIRE": 0, "ambigu (peut-etre tendance deterministe)": 1,
             "ambigu (faible puissance)": 2, "stationnaire": 3, "indetermine": 4}
    return out.assign(_o=out.verdict.map(ordre)).sort_values(
        ["_o", "adf_p"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def suggest_transform(s: pd.Series, name: str = "") -> str:
    """Transformation a appliquer, selon la nature de la serie.

    Regle simple : un NIVEAU de prix devient un rendement ; un niveau de taux ou
    d'indice diffus devient une variation ; une serie deja exprimee en glissement
    annuel ou en solde d'opinion se laisse telle quelle.
    """
    st = stationarity(s, name)
    if st["verdict"] == "stationnaire":
        return "aucune"
    x = pd.Series(s).dropna()
    if (x > 0).all() and x.max() / max(x.min(), 1e-9) > 5:
        return "rendement : 100*(s/s.shift(4)-1)"
    return "difference : s.diff(4)"


# ---------------------------------------------------------------------------
# 2 & 3. Diagnostics de la regression predictive
# ---------------------------------------------------------------------------

def regression_diagnostics(x: pd.Series, y: pd.Series, k: int,
                           groups: pd.Series | None = None,
                           grp: str | None = None) -> dict:
    """Diagnostics complets de  y_{t+k} = a + g*y_t + b*x_t + e.

    Renvoie stationnarite des entrees, autocorrelation et normalite des residus,
    RESET, White, et le VIF entre x et y_t.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_white, linear_reset

    idx = groups.index if groups is not None else y.index
    df = pd.DataFrame({"x": x.reindex(idx), "y0": y.reindex(idx),
                       "yk": y.reindex(idx).shift(-k)})
    if groups is not None and grp is not None:
        df = df[groups == grp]
    df = df.dropna()
    if len(df) < 30:
        return dict(erreur=f"seulement {len(df)} observations")

    xs = (df.x - df.x.mean()) / df.x.std(ddof=1)
    X = sm.add_constant(np.column_stack([df.y0.to_numpy(), xs.to_numpy()]))
    m_hac = max(k, int(np.floor(4 * (len(df) / 100) ** (2 / 9))))
    fit = sm.OLS(df.yk.to_numpy(), X).fit(cov_type="HAC",
                                          cov_kwds={"maxlags": m_hac})
    u = fit.resid

    out = dict(n=len(df), horizon=k, m_hac=m_hac,
               beta=round(float(fit.params[2]), 3),
               t_hac=round(float(fit.tvalues[2]), 2),
               R2=round(float(fit.rsquared), 3))

    out["stationnarite_x"] = stationarity(df.x, "x")["verdict"]
    out["stationnarite_y"] = stationarity(df.yk, "y")["verdict"]

    # Autocorrelation : ATTENDUE jusqu'a k-1 (chevauchement). On teste au-dela.
    try:
        lb = acorr_ljungbox(u, lags=[max(k, 4), max(2 * k, 8)], return_df=True)
        out["ljungbox_p"] = [round(float(p), 4) for p in lb["lb_pvalue"]]
        out["autocorr_residuelle"] = (
            "OK (couverte par HAC)" if float(lb["lb_pvalue"].iloc[-1]) > 0.05
            else f"presente au-dela de k -> augmenter m (actuel {m_hac})")
    except Exception:
        out["autocorr_residuelle"] = "indetermine"

    # Racine unitaire DANS LES RESIDUS : le test decisif de regression fallacieuse.
    # Si les residus sont eux-memes integres, la relation n'est pas cointegree
    # et le coefficient n'a aucun sens.
    try:
        from statsmodels.tsa.stattools import adfuller
        p = float(adfuller(u, autolag="AIC")[1])
        out["adf_residus_p"] = round(p, 4)
        out["residus"] = ("stationnaires -> relation exploitable" if p < 0.05
                          else "NON STATIONNAIRES -> REGRESSION FALLACIEUSE PROBABLE")
    except Exception:
        out["residus"] = "indetermine"

    try:
        out["reset_p"] = round(float(linear_reset(
            sm.OLS(df.yk.to_numpy(), X).fit(), power=2, use_f=True).pvalue), 4)
        out["linearite"] = ("OK" if out["reset_p"] > 0.05
                            else "rejetee -> essayer une forme non lineaire")
    except Exception:
        out["linearite"] = "indetermine"

    try:
        out["white_p"] = round(float(het_white(u, X)[1]), 4)
        out["heteroscedasticite"] = ("absente" if out["white_p"] > 0.05
                                     else "presente (HAC deja robuste)")
    except Exception:
        out["heteroscedasticite"] = "indetermine"

    try:
        r = np.corrcoef(df.y0, xs)[0, 1]
        out["corr_x_y0"] = round(float(r), 3)
        out["VIF"] = round(float(1 / max(1 - r ** 2, 1e-9)), 2)
        out["colinearite"] = ("OK" if out["VIF"] < 5
                              else "forte -> beta mal identifie")
    except Exception:
        out["colinearite"] = "indetermine"
    return out


def rapport(x: pd.Series, y: pd.Series, k: int, groups=None, grp=None,
            name: str = "x") -> None:
    d = regression_diagnostics(x, y, k, groups, grp)
    if "erreur" in d:
        print(f"{name} : {d['erreur']}")
        return
    print(f"=== {name} -> cible a +{k} trimestres"
          + (f" | groupe {grp}" if grp else "") + " ===")
    print(f"  n={d['n']}  beta={d['beta']:+.3f}  t_HAC={d['t_hac']:+.2f}  R2={d['R2']:.3f}")
    print(f"  stationnarite x   : {d['stationnarite_x']}")
    print(f"  residus (ADF)     : {d.get('residus')}")
    print(f"  autocorrelation   : {d.get('autocorr_residuelle')}")
    print(f"  linearite (RESET) : {d.get('linearite')}")
    print(f"  heteroscedastic.  : {d.get('heteroscedasticite')}")
    print(f"  colinearite x/y0  : VIF={d.get('VIF')} -> {d.get('colinearite')}")
