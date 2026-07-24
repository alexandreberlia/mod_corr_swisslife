from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

__all__ = [
    "prepare_series",
    "unit_root_tests",
    "integration_order",
    "stationarity_block",
    "cointegration_block",
    "johansen_report",
]

MIN_OBS_DEFAULT = 50


def prepare_series(series: pd.Series, name: str | None = None) -> tuple[pd.Series, list[str]]:
    notes: list[str] = []
    name = name or series.name or "series"
    numeric = pd.to_numeric(series, errors="coerce")
    n_coerced = int(numeric.isna().sum() - series.isna().sum())
    if n_coerced > 0:
        notes.append(f"{n_coerced} valeurs non numeriques converties en NaN")
    trimmed = numeric.loc[numeric.first_valid_index():numeric.last_valid_index()] \
        if numeric.notna().any() else numeric
    n_internal = int(trimmed.isna().sum())
    if n_internal > 0:
        notes.append(
            f"ATTENTION : {n_internal} trou(s) interne(s) supprime(s) -- "
            "l'espacement temporel n'est plus regulier, ADF/KPSS biaises")
    clean = trimmed.dropna()
    if isinstance(clean.index, pd.DatetimeIndex) and len(clean) > 2:
        if pd.infer_freq(clean.index) is None:
            notes.append("frequence non inferable (index irregulier)")
    return clean, notes


def unit_root_tests(series: pd.Series, regression: str = "c", alpha: float = 0.05) -> dict:
    out: dict = {"regression": regression}
    try:
        adf_stat, adf_p, adf_lags, adf_nobs, *_ = adfuller(series, regression=regression, autolag="AIC")
        out.update(adf_stat=adf_stat, adf_p=adf_p, adf_lags=adf_lags,
                   adf_reject=bool(adf_p < alpha), adf_error=None)
    except Exception as err:
        out.update(adf_stat=np.nan, adf_p=np.nan, adf_lags=np.nan, adf_reject=np.nan, adf_error=str(err))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kpss_stat, kpss_p, kpss_lags, _ = kpss(series, regression=regression, nlags="auto")
            clipped = any("look-up table" in str(w.message) for w in caught)
        out.update(kpss_stat=kpss_stat, kpss_p=kpss_p, kpss_lags=kpss_lags,
                   kpss_reject=bool(kpss_p < alpha), kpss_p_clipped=clipped, kpss_error=None)
    except Exception as err:
        out.update(kpss_stat=np.nan, kpss_p=np.nan, kpss_lags=np.nan, kpss_reject=np.nan,
                   kpss_p_clipped=False, kpss_error=str(err))
    adf_r, kpss_r = out.get("adf_reject"), out.get("kpss_reject")
    if pd.isna(adf_r) or pd.isna(kpss_r):
        verdict = "indetermine (test en echec)"
    elif adf_r and not kpss_r:
        verdict = "I(0) - accord"
    elif not adf_r and kpss_r:
        verdict = "I(1)+ - accord"
    elif adf_r and kpss_r:
        verdict = "conflit : rupture structurelle ou memoire longue"
    else:
        verdict = "conflit : echantillon peu informatif"
    out["verdict"] = verdict
    return out


def integration_order(series, alpha=0.05, max_order=2, regression="auto") -> dict:
    result: dict = {"order": None, "detail": {}, "notes": []}
    if len(series) < 30:
        result["order"] = "observations insuffisantes"
        return result
    for d in range(max_order, -1, -1):
        s = series.diff(1).dropna() if d == 1 else \
            series.diff(1).diff(1).dropna() if d == 2 else series
        if d > 2:
            s = series.diff().dropna()
            for _ in range(d - 1):
                s = s.diff().dropna()
        if len(s) < 30:
            continue
        specs = ["c", "ct"] if (regression == "auto" and d == 0) else \
                (["c"] if regression == "auto" else [regression])
        tests = {spec: unit_root_tests(s, regression=spec, alpha=alpha) for spec in specs}
        result["detail"][f"d={d}"] = tests
        stationary = any(t.get("adf_reject") is True and t.get("kpss_reject") is False for t in tests.values())
        adf_only = any(t.get("adf_reject") is True for t in tests.values())
        if stationary or (adf_only and d == max_order):
            result["order"] = f"I({d})"
            if d == 0 and stationary:
                winning = [k for k, t in tests.items()
                           if t.get("adf_reject") is True and t.get("kpss_reject") is False]
                if winning == ["ct"]:
                    result["notes"].append(
                        "stationnaire UNIQUEMENT avec tendance deterministe : "
                        "trend-stationnaire, pas I(0) au sens usuel. "
                        "Ne pas melanger avec des I(1) sans precaution.")
            if adf_only and not stationary:
                result["notes"].append("ADF rejette mais KPSS aussi : verdict fragile")
            continue
        if result["order"] is not None:
            if adf_only:                                    # <-- AJOUT
                result["notes"].append(
                    f"descente interrompue a d={d} sur un conflit ADF/KPSS : "
                    f"ordre retenu {result['order']} probablement sur-differencie"
                )
            break
    if result["order"] is None:
        result["order"] = f"> I({max_order})"
        result["notes"].append("aucun ordre teste ne donne la stationnarite")
    return result


def stationarity_block(block: pd.DataFrame, block_name: str = "Unnamed Block", alpha: float = 0.05) -> pd.DataFrame:
    if not isinstance(block, pd.DataFrame):
        raise TypeError("block doit etre un pandas.DataFrame")
    rows = []
    for col in block.columns:
        clean, notes = prepare_series(block[col], name=col)
        if len(clean) < 30:
            rows.append({"Block": block_name, "Variable": col, "N": len(clean),
                         "Ordre": "observations insuffisantes", "Notes": "; ".join(notes)})
            continue
        lvl_c = unit_root_tests(clean, "c", alpha)
        lvl_ct = unit_root_tests(clean, "ct", alpha)
        order = integration_order(clean, alpha=alpha)
        rows.append({
            "Block": block_name, "Variable": col, "N": len(clean),
            "ADF stat (c)": lvl_c["adf_stat"], "ADF p (c)": lvl_c["adf_p"],
            "ADF stat (ct)": lvl_ct["adf_stat"], "ADF p (ct)": lvl_ct["adf_p"],
            "KPSS stat (c)": lvl_c["kpss_stat"], "KPSS p (c)": lvl_c["kpss_p"],
            "KPSS p borne": lvl_c["kpss_p_clipped"],
            "KPSS stat (ct)": lvl_ct["kpss_stat"], "KPSS p (ct)": lvl_ct["kpss_p"],
            "Verdict niveau (c)": lvl_c["verdict"], "Verdict niveau (ct)": lvl_ct["verdict"],
            "Ordre": order["order"], "Notes": "; ".join(notes + order["notes"]),
        })
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# Cointegration                                                                #
# --------------------------------------------------------------------------- #

def _benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
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
