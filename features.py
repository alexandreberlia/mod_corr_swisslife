"""
features.py — catalogue de features orientées + calibration empirique des poids.

RÔLE : laboratoire de recherche. Tourne une fois par trimestre, pas tous les jours.
Utilise le futur (rendements réalisés) pour MESURER — c'est légitime ici, ça ne
l'est jamais dans portefeuille.py.

PRINCIPE — MONOTONIE
Un indicateur n'entre dans un CLASSEMENT que si sa relation au rendement futur est
monotone : "plus c'est haut, plus c'est haussier", toujours dans le même sens.
    ADX = 60 n'est PAS meilleur que ADX = 30           -> non classable
    momentum élevé EST meilleur que momentum faible    -> classable
Les indicateurs non monotones (ADX, ER, ATR, RVOL) mesurent un CONTEXTE. Pour les
rendre classables on leur adjoint un signe : ER x sign(momentum).

CATALOGUE RÉDUIT
8 features au lieu de 14. Les retirées (mom_3m, pente_ema50, ext_ema200, rev_21,
regularite, vol_confirme) étaient soit corrélées à plus de 0.85 avec une conservée,
soit non significatives à l'IC. Multiplier les candidats corrélés augmente le risque
de retenir par hasard celle qui a le mieux marché in-sample.

CHAÎNE D'USAGE
    features_orientees   -> QUI PEUT concourir      (8 candidats)
    ic_rapport           -> QUI MÉRITE (mesure + intervalle de confiance)
    selection_decorrelee -> on écarte les redondants
    poids_depuis_ic      -> QUI ENTRE, et à quel poids
    -> ParamsPF.poids
"""

import numpy as np
import pandas as pd


# ============================================================================
# 1. Catalogue de features ORIENTÉES
# ============================================================================

def features_orientees(ind, p_atr: int = 14) -> dict:
    """Indicateurs transformés en signaux orientés et normalisés.
    Convention respectée par toutes : 'plus haut = plus haussier'."""
    px = ind.price
    atr = ind.atr(p_atr)
    lr = ind.logret(px, 1)
    signe = np.sign(lr.rolling(20).sum())      # orientation du mouvement récent

    return {
        # --- momentum (monotone par construction) ---
        "mom_12_1":   lr.rolling(252).sum() - lr.rolling(21).sum(),
        "rev_5":     -lr.rolling(5).sum(),          # survendu = signal long
        "mom_sharpe": lr.rolling(126).mean() / lr.rolling(126).std().replace(0, np.nan),

        # --- position vs moyennes (distance signée, normalisée ATR) ---
        "ext_ema50":  (px - ind.ema(50)) / atr,
        "ext_kama":   (px - ind.kama(10, 2, 30)) / atr,

        # --- qualité de tendance ORIENTÉE (ER et ADX seuls n'ont pas de direction) ---
        "er_signe":   ind.er(10) * signe,
        "adx_signe":  (ind.adx(14) / 100) * signe,

        # --- anomalie low-vol : signe NÉGATIF (moins volatil = mieux) ---
        "low_vol":   -(atr / px),
    }


def modulateurs(ind) -> dict:
    """Indicateurs NON orientés : ils filtrent ou modulent, ils ne se classent pas."""
    return {
        "er":      ind.er(10),
        "er_rk":   ind.rank_pct(ind.er(10), 252),
        "adx":     ind.adx(14),
        "atr_pct": ind.atr(14) / ind.price,
        "rvol":    ind.rvol(50),
    }


# ============================================================================
# 2. Information Coefficient
# ============================================================================
# ATTENTION AU SIGLE : ici IC = Information Coefficient (Grinold & Kahn), une
# corrélation de rang — PAS un intervalle de confiance. L'intervalle de confiance
# est fourni séparément par les colonnes ic_bas / ic_haut.

def ic_serie(panel: pd.DataFrame, close: pd.DataFrame,
             horizon: int = 20, min_titres: int = 10) -> pd.Series:
    """IC = corrélation de rang, à chaque date, entre la feature et le rendement futur.
    Positif => la feature classe correctement les titres."""
    fwd = close.pct_change(horizon).shift(-horizon)
    f, r = panel.align(fwd, join="inner")
    valides = (f.notna() & r.notna()).sum(axis=1)
    return f.corrwith(r, axis=1, method="spearman").where(valides >= min_titres)


def newey_west_se(x: pd.Series, lag: int) -> float:
    """Erreur-type corrigée de l'autocorrélation (noyau de Bartlett).

    INDISPENSABLE : avec un horizon de h jours et des observations quotidiennes, les
    fenêtres de rendement futur SE CHEVAUCHENT — deux IC consécutifs partagent h-1
    jours sur h. L'écart-type naïf suppose l'indépendance et sous-estime l'incertitude
    d'un facteur ~sqrt(h). Vérifié : x1.0 sur bruit i.i.d., x3.7 sur série autocorrélée.
    """
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = (e @ e) / n
    for j in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - j / (lag + 1.0)) * (e[j:] @ e[:-j]) / n
    return np.sqrt(max(s, 0.0) / n)


def ic_rapport(panels: dict, close: pd.DataFrame, features: list,
               horizons=(5, 20, 60), min_titres: int = 10,
               alpha: float = 0.05) -> pd.DataFrame:
    """Tableau IC par feature et par horizon, avec intervalle de confiance.

    IC_moy      : pouvoir prédictif moyen. En equity RÉELLE, 0.02-0.05 est déjà bon.
    IR          : IC_moy / IC_std -> la STABILITÉ, ce qui compte le plus.
    ic_bas/haut : intervalle de confiance. S'il contient 0, la feature n'est PAS
                  significative, quel que soit son IC moyen.
    t_NW        : t corrigé du chevauchement. |t| > 2 -> significatif.
    t_naif      : t non corrigé, pour mesurer l'ampleur de l'illusion.
    n_eff       : taille d'échantillon effective (n_dates / horizon).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    lignes = []

    for f in features:
        if f not in panels:
            continue
        for h in horizons:
            ic = ic_serie(panels[f], close, h, min_titres).dropna()
            if len(ic) < 30:
                continue
            m, s, n = ic.mean(), ic.std(), len(ic)
            se = newey_west_se(ic, lag=h)
            lignes.append({
                "feature": f, "horizon": h, "IC_moy": m, "IC_std": s,
                "IR": m / s if s > 0 else np.nan,
                "se_NW": se, "ic_bas": m - z * se, "ic_haut": m + z * se,
                "t_NW": m / se if se and se > 0 else np.nan,
                "t_naif": m / s * np.sqrt(n) if s > 0 else np.nan,
                "significatif": bool(se and se > 0 and abs(m) > z * se),
                "n_dates": n, "n_eff": int(n / h),
            })

    df = pd.DataFrame(lignes)
    return df.sort_values("IR", key=abs, ascending=False).reset_index(drop=True) if len(df) else df


def poids_depuis_ic(rapport: pd.DataFrame, horizon: int = 20,
                    ir_min: float = 0.05, n_max: int = 5,
                    exiger_significatif: bool = True) -> dict:
    """Poids proportionnels à |IR|, signe repris de l'IC.
    Les poids peuvent être NÉGATIFS : Portefeuille normalise par la somme des
    valeurs absolues et centre les rangs, ce qui gère correctement ce cas."""
    r = rapport[rapport.horizon == horizon]
    if exiger_significatif and "significatif" in r.columns:
        r = r[r.significatif]
    r = r[r.IR.abs() > ir_min]
    r = r.reindex(r.IR.abs().sort_values(ascending=False).index).head(n_max)
    if r.empty:
        return {}
    total = r.IR.abs().sum()
    return {row.feature: round(float(np.sign(row.IC_moy) * abs(row.IR) / total), 3)
            for row in r.itertuples()}


# ============================================================================
# 3. Décorrélation
# ============================================================================

def correlation_features(panels: dict, features: list, date_min=None) -> pd.DataFrame:
    """Corrélation de rang moyenne entre features, calculée cross-sectionnellement.
    Au-delà de |0.8|, deux features apportent la même information."""
    dispo = [f for f in features if f in panels]
    n = len(dispo)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = panels[dispo[i]], panels[dispo[j]]
            if date_min is not None:
                a, b = a.loc[date_min:], b.loc[date_min:]
            M[i, j] = M[j, i] = a.corrwith(b, axis=1, method="spearman").mean()
    return pd.DataFrame(M, index=dispo, columns=dispo)


def selection_decorrelee(rapport: pd.DataFrame, corr: pd.DataFrame,
                         horizon: int = 20, seuil: float = 0.8,
                         exiger_significatif: bool = True) -> list:
    """Garde les features par |IR| décroissant, en écartant toute candidate corrélée
    au-delà du seuil à une feature déjà retenue."""
    r = rapport[rapport.horizon == horizon]
    if exiger_significatif and "significatif" in r.columns:
        r = r[r.significatif]
    ordre = r.reindex(r.IR.abs().sort_values(ascending=False).index).feature.tolist()
    gardees = []
    for f in ordre:
        if f in corr.columns and all(abs(corr.loc[f, g]) < seuil for g in gardees):
            gardees.append(f)
    return gardees


AUX = ("close", "open", "high", "low", "atr", "er_rk", "adx", "dvol")


def calibrer(panels: dict, close: pd.DataFrame, features: list = None,
             horizon: int = 20, split=None, seuil_corr: float = 0.8,
             n_max: int = 5, min_titres: int = 10) -> tuple:
    """Chaîne complète de calibration. Renvoie (poids, rapport, correlations).

    split : date de fin d'échantillon IN-SAMPLE. OBLIGATOIRE en pratique — calibrer
            sur toute la période puis backtester dessus est du look-ahead pur.
    """
    if features is None:
        features = [k for k in panels if k not in AUX]
    pan = {k: (v.loc[:split] if split is not None else v) for k, v in panels.items()}
    cl = close.loc[:split] if split is not None else close

    rapport = ic_rapport(pan, cl, features, horizons=(horizon,), min_titres=min_titres)
    corr = correlation_features(pan, features)
    gardees = selection_decorrelee(rapport, corr, horizon, seuil_corr)
    poids = poids_depuis_ic(rapport[rapport.feature.isin(gardees)], horizon, n_max=n_max)
    return poids, rapport, corr
