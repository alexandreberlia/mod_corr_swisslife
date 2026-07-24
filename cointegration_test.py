
from __future__ import annotations
 
import warnings
 
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def _benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Controle du FDR. Retourne un masque booleen des rejets."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    reject = np.zeros_like(p, dtype=bool)
    if ok.sum() == 0:
        return reject
    idx = np.argsort(p[ok])
    m = ok.sum()
    sorted_p = p[ok][idx]
    thresh = alpha * np.arange(1, m + 1) / m
    passed = sorted_p <= thresh
    if passed.any():
        k = np.max(np.where(passed)[0])
        keep = np.zeros(m, dtype=bool)
        keep[idx[: k + 1]] = True
        reject[np.where(ok)[0]] = keep
    return reject
 
 
def cointegration_block(
    block: pd.DataFrame,
    integration_orders: dict[str, str],
    block_name: str = "Unnamed Block",
    alpha: float = 0.05,
    min_observations: int = MIN_OBS_DEFAULT,
    trend: str = "c",
    fdr_correction: bool = True,
) -> pd.DataFrame:
    """
    Engle-Granger par paires, restreint aux paires I(1)/I(1).
 
    integration_orders : dict {nom_variable: "I(0)" | "I(1)" | ...}
        OBLIGATOIRE. Une paire ou au moins une variable n'est pas I(1)
        est marquee 'non eligible' et non testee.
 
        Justification : si y est I(0) et x est I(1), le residu de la
        regression de y sur x est asymptotiquement y lui-meme, donc
        stationnaire. Le test rejette H0 quasi systematiquement, meme
        sur des series parfaitement independantes. Ce n'est pas de la
        cointegration, c'est un artefact mecanique.
 
    Le test est effectue dans les DEUX sens : coint(y, x) != coint(x, y).
    """
    if not isinstance(block, pd.DataFrame):
        raise TypeError("block doit etre un pandas.DataFrame")
    if block.shape[1] < 2:
        raise ValueError("le bloc doit contenir au moins deux variables")
    if not 0 < alpha < 1:
        raise ValueError("alpha doit etre dans ]0, 1[")
 
    cols = list(block.columns)
    rows = []
 
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v1, v2 = cols[i], cols[j]
            o1 = integration_orders.get(v1, "inconnu")
            o2 = integration_orders.get(v2, "inconnu")
 
            base = {"Block": block_name, "Variable 1": v1, "Variable 2": v2,
                    "Ordre 1": o1, "Ordre 2": o2}
 
            if o1 != "I(1)" or o2 != "I(1)":
                rows.append({**base, "Observations": np.nan,
                             "p-value (1~2)": np.nan, "p-value (2~1)": np.nan,
                             "Cointegre": False,
                             "Statut": "non eligible : ordres non I(1)/I(1)"})
                continue
 
            merged = pd.concat(
                [pd.to_numeric(block[v1], errors="coerce").rename(v1),
                 pd.to_numeric(block[v2], errors="coerce").rename(v2)],
                axis=1,
            ).dropna()
            n = len(merged)
 
            if n < min_observations:
                rows.append({**base, "Observations": n,
                             "p-value (1~2)": np.nan, "p-value (2~1)": np.nan,
                             "Cointegre": False,
                             "Statut": "observations insuffisantes"})
                continue
 
            try:
                s12, p12, cv = coint(merged[v1], merged[v2], trend=trend, autolag="aic")
                s21, p21, _ = coint(merged[v2], merged[v1], trend=trend, autolag="aic")
                rows.append({
                    **base, "Observations": n,
                    "Stat (1~2)": s12, "p-value (1~2)": p12,
                    "Stat (2~1)": s21, "p-value (2~1)": p21,
                    "CV 5%": cv[1],
                    "p-value": min(p12, p21),
                    "Accord des deux sens": bool((p12 < alpha) == (p21 < alpha)),
                    "Cointegre": bool(p12 < alpha and p21 < alpha),
                    "Statut": "teste",
                })
            except Exception as err:
                rows.append({**base, "Observations": n,
                             "p-value (1~2)": np.nan, "p-value (2~1)": np.nan,
                             "Cointegre": False, "Statut": f"echec : {err}"})
 
    df = pd.DataFrame(rows)
    if df.empty:
        return df
 
    if fdr_correction and "p-value" in df.columns:
        tested = df["Statut"] == "teste"
        if tested.any():
            df.loc[tested, "Rejet FDR 5%"] = _benjamini_hochberg(
                df.loc[tested, "p-value"].to_numpy(), alpha
            )
            df["Rejet FDR 5%"] = df["Rejet FDR 5%"].fillna(False)
 
    sort_col = "p-value" if "p-value" in df.columns else "Statut"
    return df.sort_values(by=sort_col, na_position="last").reset_index(drop=True)
 
 
def johansen_report(block: pd.DataFrame, integration_orders: dict[str, str],
                    det_order: int = 0, k_ar_diff: int = 2) -> dict:
    """
    Test de Johansen sur le systeme complet, restreint aux variables I(1).
 
    Le test par paires ne peut PAS detecter une relation entre 3 variables
    ou plus. Si l'objectif est un VECM, c'est Johansen qu'il faut, pas
    Engle-Granger par paires.
 
    det_order : -1 (aucun terme deterministe), 0 (constante dans la
    cointegration), 1 (constante + tendance lineaire).
    """
    i1 = [c for c in block.columns if integration_orders.get(c) == "I(1)"]
    excluded = [c for c in block.columns if c not in i1]
 
    if len(i1) < 2:
        return {"error": "moins de deux variables I(1)", "variables": i1,
                "exclues": excluded}
 
    data = block[i1].apply(pd.to_numeric, errors="coerce").dropna()
    res = coint_johansen(data, det_order, k_ar_diff)
 
    trace = pd.DataFrame(
        {"r <=": range(len(i1)),
         "Trace stat": res.lr1,
         "CV 90%": res.cvt[:, 0], "CV 95%": res.cvt[:, 1], "CV 99%": res.cvt[:, 2],
         "Rejet 5%": res.lr1 > res.cvt[:, 1]}
    )
    eigen = pd.DataFrame(
        {"r =": range(len(i1)),
         "Max-eigen stat": res.lr2,
         "CV 90%": res.cvm[:, 0], "CV 95%": res.cvm[:, 1], "CV 99%": res.cvm[:, 2],
         "Rejet 5%": res.lr2 > res.cvm[:, 1]}
    )
 
    rank = int(trace["Rejet 5%"].to_numpy().argmin()) if not trace["Rejet 5%"].all() else len(i1)
 
    beta = pd.DataFrame(res.evec[:, :max(rank, 1)], index=i1,
                        columns=[f"beta_{k+1}" for k in range(max(rank, 1))])
    if rank >= 1:
        beta = beta / beta.iloc[np.abs(beta.iloc[:, 0]).to_numpy().argmax(), 0]
 
    warn = []
    if excluded:
        warn.append(
            f"variables exclues (non I(1)) : {excluded}. Les inclure gonflerait "
            "le rang d'une unite par variable stationnaire (vecteur trivial)."
        )
    if rank >= 1:
        top = beta.iloc[:, 0].abs().sort_values(ascending=False)
        if top.iloc[0] > 5 * top.iloc[1]:
            warn.append(
                f"le vecteur charge massivement sur '{top.index[0]}' : "
                "verifier qu'il ne s'agit pas d'un vecteur trivial."
            )
 
    return {"variables": i1, "exclues": excluded, "n_obs": len(data),
            "trace": trace, "max_eigen": eigen, "rang_retenu": rank,
            "beta_normalise": beta, "alertes": warn, "raw": res}
