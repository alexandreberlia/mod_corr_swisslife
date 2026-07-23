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
 
 
# --------------------------------------------------------------------------- #
# Preparation                                                                  #
# --------------------------------------------------------------------------- #
 
def prepare_series(series: pd.Series, name: str | None = None) -> tuple[pd.Series, list[str]]:
    """
    Nettoie une serie et signale ce qui a ete perdu.
 
    Contrairement a un dropna() aveugle, on distingue les NaN de bord
    (inoffensifs) des trous internes (qui detruisent l'espacement temporel
    et rendent l'ADF non interpretable).
    """
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
            "l'espacement temporel n'est plus regulier, ADF/KPSS biaises"
        )
 
    clean = trimmed.dropna()
 
    if isinstance(clean.index, pd.DatetimeIndex) and len(clean) > 2:
        freq = pd.infer_freq(clean.index)
        if freq is None:
            notes.append("frequence non inferable (index irregulier)")
 
    return clean, notes
 
 
# --------------------------------------------------------------------------- #
# Tests de racine unitaire                                                     #
# --------------------------------------------------------------------------- #
 
def unit_root_tests(series: pd.Series, regression: str = "c", alpha: float = 0.05) -> dict:
    """
    ADF et KPSS sur une meme serie, avec la meme specification deterministe.
 
    Hypotheses nulles inversees :
        ADF  H0 : racine unitaire        -> rejet = stationnaire
        KPSS H0 : stationnarite          -> rejet = racine unitaire
 
    Les p-values KPSS renvoyees par statsmodels sont interpolees dans une table
    et bornees a [0.01, 0.10]. Une p-value de 0.10 signifie donc "au moins 0.10",
    pas "exactement 0.10". Le flag kpss_p_clipped le signale.
    """
    out: dict = {"regression": regression}
 
    try:
        adf_stat, adf_p, adf_lags, adf_nobs, *_ = adfuller(
            series, regression=regression, autolag="AIC"
        )
        out.update(adf_stat=adf_stat, adf_p=adf_p, adf_lags=adf_lags,
                   adf_reject=bool(adf_p < alpha), adf_error=None)
    except Exception as err:
        out.update(adf_stat=np.nan, adf_p=np.nan, adf_lags=np.nan,
                   adf_reject=np.nan, adf_error=str(err))
 
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kpss_stat, kpss_p, kpss_lags, _ = kpss(
                series, regression=regression, nlags="auto"
            )
            clipped = any("look-up table" in str(w.message) for w in caught)
        out.update(kpss_stat=kpss_stat, kpss_p=kpss_p, kpss_lags=kpss_lags,
                   kpss_reject=bool(kpss_p < alpha), kpss_p_clipped=clipped,
                   kpss_error=None)
    except Exception as err:
        out.update(kpss_stat=np.nan, kpss_p=np.nan, kpss_lags=np.nan,
                   kpss_reject=np.nan, kpss_p_clipped=False, kpss_error=str(err))
 
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
 
 
def integration_order(
    series: pd.Series,
    alpha: float = 0.05,
    max_order: int = 2,
    regression: str = "auto",
) -> dict:
    """
    Determine l'ordre d'integration par la sequence de Pantula.
 
    On teste du PLUS differencie vers le MOINS differencie. Tester dans
    l'autre sens (niveau -> d1 -> d2, comme dans la version initiale) peut
    conclure I(0) sur une serie I(2) : le niveau d'une serie I(2) peut
    accidentellement rejeter, et on s'arrete avant d'avoir vu le probleme.
 
    regression='auto' teste 'c' et 'ct' sur le niveau et retient la
    specification la plus favorable a la stationnarite, en le signalant.
    Une serie trend-stationnaire testee uniquement avec 'c' ressort I(1)
    dans la quasi-totalite des cas.
    """
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
 
        stationary = any(
            t.get("adf_reject") is True and t.get("kpss_reject") is False
            for t in tests.values()
        )
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
                        "Ne pas melanger avec des I(1) sans precaution."
                    )
            if adf_only and not stationary:
                result["notes"].append("ADF rejette mais KPSS aussi : verdict fragile")
            # on continue la descente : si un ordre inferieur est aussi
            # stationnaire, c'est lui qui prime (principe de Pantula)
            continue
 
        if result["order"] is not None:
            break
 
    if result["order"] is None:
        result["order"] = f"> I({max_order})"
        result["notes"].append("aucun ordre teste ne donne la stationnarite")
 
    return result
 
 
def stationarity_block(block: pd.DataFrame, block_name: str = "Unnamed Block",
                       alpha: float = 0.05) -> pd.DataFrame:
    """Rapport de stationnarite pour toutes les colonnes d'un bloc."""
    if not isinstance(block, pd.DataFrame):
        raise TypeError("block doit etre un pandas.DataFrame")
 
    rows = []
    for col in block.columns:
        clean, notes = prepare_series(block[col], name=col)
 
        if len(clean) < 30:
            rows.append({"Block": block_name, "Variable": col, "N": len(clean),
                         "Ordre": "observations insuffisantes",
                         "Notes": "; ".join(notes)})
            continue
 
        lvl_c = unit_root_tests(clean, "c", alpha)
        lvl_ct = unit_root_tests(clean, "ct", alpha)
        order = integration_order(clean, alpha=alpha)
 
        rows.append({
            "Block": block_name,
            "Variable": col,
            "N": len(clean),
            "ADF stat (c)": lvl_c["adf_stat"],
            "ADF p (c)": lvl_c["adf_p"],
            "ADF stat (ct)": lvl_ct["adf_stat"],
            "ADF p (ct)": lvl_ct["adf_p"],
            "KPSS stat (c)": lvl_c["kpss_stat"],
            "KPSS p (c)": lvl_c["kpss_p"],
            "KPSS p borne": lvl_c["kpss_p_clipped"],
            "KPSS stat (ct)": lvl_ct["kpss_stat"],
            "KPSS p (ct)": lvl_ct["kpss_p"],
            "Verdict niveau (c)": lvl_c["verdict"],
            "Verdict niveau (ct)": lvl_ct["verdict"],
            "Ordre": order["order"],
            "Notes": "; ".join(notes + order["notes"]),
        })
 
    return pd.DataFrame(rows)
 
