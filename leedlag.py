"""
leadlag.py — Estimation du décalage (avance/retard) entre deux séries,
avec distribution de probabilite sur le decalage estime.

Convention de signe unique dans tout le module
----------------------------------------------
    lag k > 0  <=>  x est EN AVANCE de k periodes sur y
                    (on correle x[t-k] avec y[t])
    lag k < 0  <=>  x est en retard sur y

Trois problemes traites explicitement
-------------------------------------
1. CORRELATION FALLACIEUSE. Deux series autocorrelees exhibent des correlations
   croisees elevees meme independantes. La variance de la CCF empirique est
   gonflee par sum_k rho_x(k) rho_y(k) (formule de Bartlett). Correctif :
   pre-blanchiment de Box-Jenkins — on ajuste un AR(p) sur x, on filtre LES DEUX
   series par ce meme filtre, on calcule la CCF sur les residus. x devient un
   bruit blanc, l'erreur-type redevient 1/sqrt(n) et les pics sont interpretables.

2. TESTS MULTIPLES. Chercher le max de |rho| sur 25 decalages puis tester ce max
   comme s'il avait ete choisi a priori gonfle massivement le taux de faux
   positifs. Correctif : la p-value globale porte sur la statistique
   max_k |rho(k)|, dont la loi sous H0 est obtenue par bootstrap avec les deux
   series rendues independantes mais leur autocorrelation marginale preservee.

3. INCERTITUDE SUR argmax. Le decalage estime est lui-meme une variable
   aleatoire, souvent tres instable : deplacer le pic d'un trimestre coute
   parfois 0.01 de correlation. Correctif : bootstrap stationnaire (Politis-Romano)
   sur les PAIRES (x,y) — ce qui preserve la structure de dependance croisee —
   d'ou une distribution empirique de argmax, i.e. P(avance = k).

Dependances : numpy, pandas, statsmodels, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------

def _align(x, y):
    """Aligne deux series sur leur index commun et retire les manquants."""
    df = pd.concat([pd.Series(x).rename("x"), pd.Series(y).rename("y")], axis=1)
    df = df.dropna()
    if len(df) < 30:
        raise ValueError(f"Echantillon commun trop court : {len(df)} observations.")
    return df["x"].astype(float), df["y"].astype(float)


def _standardise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std(ddof=1)


def optimal_block_length(a: np.ndarray) -> float:
    """Longueur de bloc moyenne pour le bootstrap stationnaire.

    Regle pratique : b = n^(1/3) ajustee par la persistance AR(1). Une serie
    tres persistante exige des blocs longs pour que le reechantillonnage
    reproduise sa memoire.
    """
    n = len(a)
    a = _standardise(a)
    rho = np.corrcoef(a[:-1], a[1:])[0, 1]
    rho = np.clip(abs(rho), 0.0, 0.95)
    return max(2.0, min(n / 4.0, n ** (1 / 3) * (1 + 2 * rho) / (1 - rho + 1e-6) ** 0.5))


def stationary_bootstrap_index(n: int, block_len: float, rng) -> np.ndarray:
    """Indices d'un tirage de bootstrap stationnaire (Politis-Romano 1994).

    Blocs de longueur geometrique de moyenne block_len, enroules circulairement.
    Contrairement au bootstrap par blocs fixes, la serie reechantillonnee reste
    stationnaire, ce qui importe ici puisqu'on va lui appliquer des decalages.
    """
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    jumps = rng.integers(0, n, size=n)
    for t in range(1, n):
        idx[t] = jumps[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def _fit_ar_filter(a: np.ndarray, max_p: int = 8):
    """Ajuste un AR(p) par AIC et renvoie (coefficients, p).

    Ce sont ces coefficients qui serviront de filtre de pre-blanchiment.
    """
    from statsmodels.tsa.ar_model import AutoReg

    best_aic, best = np.inf, (np.array([]), 0)
    for p in range(1, max_p + 1):
        try:
            res = AutoReg(a, lags=p, old_names=False).fit()
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic, best = res.aic, (np.asarray(res.params)[1:], p)
        except Exception:
            continue
    return best


def _apply_filter(a: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Applique le filtre (1 - phi_1 L - ... - phi_p L^p) a la serie a."""
    p = len(phi)
    if p == 0:
        return a - a.mean()
    out = a[p:].copy()
    for j, c in enumerate(phi, start=1):
        out = out - c * a[p - j: -j]
    return out


def prewhiten(x: np.ndarray, y: np.ndarray, max_p: int = 8):
    """Pre-blanchiment de Box-Jenkins.

    Le filtre est estime SUR X SEULEMENT puis applique aux deux series. C'est le
    point crucial : filtrer y avec son propre modele detruirait la relation
    croisee qu'on cherche a mesurer.
    """
    phi, p = _fit_ar_filter(np.asarray(x, float), max_p)
    return _apply_filter(np.asarray(x, float), phi), _apply_filter(np.asarray(y, float), phi), p


def _ccf(x: np.ndarray, y: np.ndarray, lags) -> np.ndarray:
    """Correlations croisees. lag k > 0 : x[t-k] vs y[t], i.e. x en avance."""
    x, y = _standardise(x), _standardise(y)
    out = []
    for k in lags:
        if k > 0:
            a, b = x[:-k], y[k:]
        elif k < 0:
            a, b = x[-k:], y[:k]
        else:
            a, b = x, y
        out.append(np.nan if len(a) < 10 else np.corrcoef(a, b)[0, 1])
    return np.asarray(out)


# ----------------------------------------------------------------------------
# Resultat
# ----------------------------------------------------------------------------

@dataclass
class LeadLagResult:
    method: str
    lags: np.ndarray
    stat: np.ndarray                  # rho (ccf) ou pseudo-R2 (probit)
    se: np.ndarray | None
    best_lag: int
    best_stat: float
    prob_lag: pd.Series               # P(avance = k), somme a 1
    hdi: tuple                        # intervalle de plus haute densite a 90 %
    p_global: float                   # p-value corrigee des tests multiples
    n_obs: int
    detail: dict = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "lag": self.lags,
            "stat": self.stat,
            "P(avance=k)": self.prob_lag.reindex(self.lags).to_numpy(),
        })
        if self.se is not None:
            df["se"] = self.se
            df["t"] = self.stat / self.se
        return df.set_index("lag")

    def summary(self) -> str:
        sens = "en avance de" if self.best_lag > 0 else ("en retard de" if self.best_lag < 0 else "synchrone avec")
        L = [
            f"Methode           : {self.method}",
            f"Observations      : {self.n_obs}",
            f"Decalage estime   : x est {sens} {abs(self.best_lag)} periode(s) sur y",
            f"Statistique       : {self.best_stat:+.3f}",
            f"P(avance = {self.best_lag:+d})   : {self.prob_lag.get(self.best_lag, np.nan):.1%}",
            f"IPD 90 %          : [{self.hdi[0]:+d} ; {self.hdi[1]:+d}] periodes",
            f"p globale (max|.|): {self.p_global:.4f}"
            + ("   <- aucune relation decelable" if self.p_global > 0.10 else ""),
        ]
        if self.detail.get("ar_order") is not None:
            L.append(f"Ordre AR filtre   : {self.detail['ar_order']}")
        return "\n".join(L)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        a0, a1 = ax
        a0.axhline(0, lw=.8, color="k")
        a0.plot(self.lags, self.stat, marker="o", ms=3, lw=1.2, color="#185FA5")
        if self.se is not None:
            a0.fill_between(self.lags, -1.96 * self.se, 1.96 * self.se,
                            color="#888780", alpha=.18, label="IC 95 % ponctuel")
            a0.legend(fontsize=8, frameon=False)
        a0.axvline(self.best_lag, color="#D85A30", lw=1, ls="--")
        a0.set_ylabel("statistique")
        a0.set_title(f"{self.method} — pic a k={self.best_lag:+d}", fontsize=10)
        pr = self.prob_lag.reindex(self.lags).fillna(0)
        a1.bar(self.lags, pr.to_numpy(), color="#1D9E75", width=.7)
        a1.axvspan(self.hdi[0] - .5, self.hdi[1] + .5, color="#1D9E75", alpha=.12)
        a1.set_ylabel("P(avance = k)")
        a1.set_xlabel("decalage k (k>0 : x en avance sur y)")
        return ax


def _hdi_from_prob(pr: pd.Series, mass: float = 0.90) -> tuple:
    """Intervalle de plus haute densite : plus petit ensemble de lags contigus
    couvrant `mass` de la probabilite."""
    lags = pr.index.to_numpy()
    v = pr.to_numpy()
    best = (lags[0], lags[-1], len(lags) + 1)
    for i in range(len(lags)):
        s = 0.0
        for j in range(i, len(lags)):
            s += v[j]
            if s >= mass:
                if (j - i) < best[2]:
                    best = (int(lags[i]), int(lags[j]), j - i)
                break
    return best[0], best[1]


# ----------------------------------------------------------------------------
# 1. Corrélation croisée — cible continue
# ----------------------------------------------------------------------------

def leadlag_ccf(x, y, max_lag: int = 12, prewhiten_series: bool = True,
                n_boot: int = 2000, seed: int = 0, max_ar: int = 8) -> LeadLagResult:
    """Decalage entre deux series CONTINUES par correlation croisee.

    Parameters
    ----------
    x, y : array-like ou pd.Series
        Series a comparer. Elles doivent etre STATIONNAIRES : utilisez l'ecart
        au potentiel, une variation, ou une serie differenciee — jamais un niveau
        integre, sous peine de correlation fallacieuse insensible au
        pre-blanchiment.
    prewhiten_series : bool
        True (defaut) applique Box-Jenkins. Mettre False uniquement pour comparer
        et constater a quel point la CCF brute exagere.
    n_boot : int
        Reechantillonnages pour P(avance=k) et pour la p-value globale.

    Returns
    -------
    LeadLagResult
    """
    sx, sy = _align(x, y)
    xa, ya = sx.to_numpy(), sy.to_numpy()
    ar_order = None

    if prewhiten_series:
        xf, yf, ar_order = prewhiten(xa, ya, max_ar)
    else:
        xf, yf = xa, ya

    lags = np.arange(-max_lag, max_lag + 1)
    rho = _ccf(xf, yf, lags)
    n = len(xf)

    # Erreur-type. Apres pre-blanchiment x est un bruit blanc : 1/sqrt(n-|k|)
    # est valide. Sans pre-blanchiment on applique la formule de Bartlett, qui
    # gonfle l'erreur-type par la persistance conjointe des deux series.
    if prewhiten_series:
        se = 1.0 / np.sqrt(n - np.abs(lags))
    else:
        m = min(n // 4, 40)
        rx = np.array([np.corrcoef(xf[:-k], xf[k:])[0, 1] for k in range(1, m + 1)])
        ry = np.array([np.corrcoef(yf[:-k], yf[k:])[0, 1] for k in range(1, m + 1)])
        infl = np.sqrt(1 + 2 * np.sum(rx * ry))
        se = infl / np.sqrt(n - np.abs(lags))

    k_star = int(lags[np.nanargmax(np.abs(rho))])
    r_star = float(rho[np.nanargmax(np.abs(rho))])

    rng = np.random.default_rng(seed)
    b = optimal_block_length(xf)

    # (a) P(avance = k) : bootstrap sur les PAIRES, la dependance croisee est
    #     conservee, seule l'incertitude d'echantillonnage joue.
    counts = np.zeros(len(lags))
    for _ in range(n_boot):
        idx = stationary_bootstrap_index(n, b, rng)
        r = _ccf(xf[idx], yf[idx], lags)
        if np.all(np.isnan(r)):
            continue
        counts[np.nanargmax(np.abs(r))] += 1
    prob = pd.Series(counts / max(counts.sum(), 1), index=lags, name="P(avance=k)")

    # (b) p-value globale : sous H0 les deux series sont independantes mais
    #     gardent leur autocorrelation. On les reechantillonne separement et on
    #     compare le max|rho| observe a la loi du max|rho| simule. Ceci corrige
    #     la recherche sur 2*max_lag+1 decalages.
    by = optimal_block_length(yf)
    null_max = np.empty(n_boot)
    for i in range(n_boot):
        xi = stationary_bootstrap_index(n, b, rng)
        yi = stationary_bootstrap_index(n, by, rng)
        null_max[i] = np.nanmax(np.abs(_ccf(xf[xi], yf[yi], lags)))
    p_glob = float((np.sum(null_max >= abs(r_star)) + 1) / (n_boot + 1))

    return LeadLagResult(
        method="CCF pre-blanchie" if prewhiten_series else "CCF brute",
        lags=lags, stat=rho, se=se, best_lag=k_star, best_stat=r_star,
        prob_lag=prob, hdi=_hdi_from_prob(prob), p_global=p_glob, n_obs=n,
        detail={"ar_order": ar_order, "block_len": round(b, 2)},
    )


# ----------------------------------------------------------------------------
# 2. Probit décalé — cible binaire
# ----------------------------------------------------------------------------

def leadlag_probit(x, event, max_lag: int = 12, n_boot: int = 1000, seed: int = 0,
                   hac_lags: int | None = None, criterion: str = "pseudo_r2") -> LeadLagResult:
    """Decalage d'un predicteur continu sur un EVENEMENT BINAIRE.

    Specification d'Estrella-Mishkin (1998) : pour chaque decalage k on estime
        P(event_t = 1) = Phi(a + b * x_{t-k})
    et on retient le k qui maximise le critere d'ajustement.

    Parameters
    ----------
    x : array-like        predicteur continu (ex. pente 10 ans - 3 mois)
    event : array-like    indicatrice 0/1 (ex. 1 si en Decrochage au trimestre t)
    hac_lags : int | None Retards de Newey-West. Par defaut floor(4*(n/100)^(2/9)).
                          Indispensable : les evenements sont fortement
                          autocorreles (une recession dure plusieurs trimestres),
                          les erreurs-types MV seraient trop optimistes.
    criterion : {"pseudo_r2", "auc"}

    Notes
    -----
    Le pseudo-R2 de McFadden reste bas sur ce type de donnees ; 0.15-0.30 est un
    resultat solide pour une indicatrice de recession. Comparez les k entre eux,
    pas a 1.
    """
    import statsmodels.api as sm

    sx, sy = _align(x, event)
    xa, ya = sx.to_numpy(), sy.to_numpy().astype(int)
    if not np.isin(np.unique(ya), [0, 1]).all():
        raise ValueError("`event` doit etre binaire (0/1).")
    if ya.sum() < 8:
        warnings.warn(f"Seulement {ya.sum()} evenements : estimations tres instables.")

    if hac_lags is None:
        hac_lags = int(np.floor(4 * (len(xa) / 100) ** (2 / 9)))

    lags = np.arange(-max_lag, max_lag + 1)

    def _fit(xv, yv, k):
        if k > 0:
            a, b = xv[:-k], yv[k:]
        elif k < 0:
            a, b = xv[-k:], yv[:k]
        else:
            a, b = xv, yv
        if len(a) < 25 or b.sum() < 3 or b.sum() == len(b):
            return None
        X = sm.add_constant(a)
        try:
            m = sm.Probit(b, X).fit(disp=0, maxiter=200)
            m_hac = sm.Probit(b, X).fit(disp=0, maxiter=200, cov_type="HAC",
                                        cov_kwds={"maxlags": hac_lags})
        except Exception:
            return None
        if criterion == "auc":
            from sklearn.metrics import roc_auc_score
            crit = roc_auc_score(b, m.predict(X))
        else:
            crit = m.prsquared
        return dict(crit=crit, coef=m.params[1], se=m_hac.bse[1],
                    p=m_hac.pvalues[1], n=len(a), ev=int(b.sum()))

    fits = {int(k): _fit(xa, ya, int(k)) for k in lags}
    crit = np.array([fits[int(k)]["crit"] if fits[int(k)] else np.nan for k in lags])
    coef = np.array([fits[int(k)]["coef"] if fits[int(k)] else np.nan for k in lags])
    pval = np.array([fits[int(k)]["p"] if fits[int(k)] else np.nan for k in lags])
    ses = np.array([fits[int(k)]["se"] if fits[int(k)] else np.nan for k in lags])

    k_star = int(lags[np.nanargmax(crit)])
    c_star = float(np.nanmax(crit))

    rng = np.random.default_rng(seed)
    b_len = optimal_block_length(xa)

    # P(avance = k) : bootstrap stationnaire sur les paires (x, event).
    counts = np.zeros(len(lags))
    for _ in range(n_boot):
        idx = stationary_bootstrap_index(len(xa), b_len, rng)
        xb, yb = xa[idx], ya[idx]
        if yb.sum() < 5:
            continue
        c = np.array([(_fit(xb, yb, int(k)) or {"crit": np.nan})["crit"] for k in lags])
        if np.all(np.isnan(c)):
            continue
        counts[np.nanargmax(c)] += 1
    prob = pd.Series(counts / max(counts.sum(), 1), index=lags, name="P(avance=k)")

    # p globale sur max_k du critere, series rendues independantes.
    by = optimal_block_length(ya.astype(float))
    null_max = np.empty(n_boot)
    for i in range(n_boot):
        xi = stationary_bootstrap_index(len(xa), b_len, rng)
        yi = stationary_bootstrap_index(len(ya), by, rng)
        c = np.array([(_fit(xa[xi], ya[yi], int(k)) or {"crit": np.nan})["crit"] for k in lags])
        null_max[i] = np.nanmax(c) if not np.all(np.isnan(c)) else np.nan
    null_max = null_max[~np.isnan(null_max)]
    p_glob = float((np.sum(null_max >= c_star) + 1) / (len(null_max) + 1))

    return LeadLagResult(
        method=f"Probit decale ({criterion}, HAC {hac_lags})",
        lags=lags, stat=crit, se=None, best_lag=k_star, best_stat=c_star,
        prob_lag=prob, hdi=_hdi_from_prob(prob), p_global=p_glob, n_obs=len(xa),
        detail={"coef": coef, "p_coef": pval, "se_coef": ses, "ar_order": None,
                "n_events": int(ya.sum()), "hac_lags": hac_lags},
    )


# ----------------------------------------------------------------------------
# 3. Balayage sur plusieurs candidats
# ----------------------------------------------------------------------------

def scan(candidates: dict, target, binary: bool = False, max_lag: int = 12,
         n_boot: int = 1000, **kw) -> pd.DataFrame:
    """Applique l'estimateur a plusieurs predicteurs et classe les resultats.

    ATTENTION : balayer N variables rouvre le probleme des tests multiples a un
    second niveau. La colonne `p_bonferroni` applique la correction la plus
    conservatrice. Avec 6 candidats, une p globale de 0.04 devient 0.24.
    """
    rows = []
    for name, s in candidates.items():
        try:
            r = (leadlag_probit if binary else leadlag_ccf)(
                s, target, max_lag=max_lag, n_boot=n_boot, **kw)
            rows.append(dict(variable=name, avance=r.best_lag, stat=round(r.best_stat, 3),
                             prob_pic=round(float(r.prob_lag.get(r.best_lag, np.nan)), 3),
                             ipd_bas=r.hdi[0], ipd_haut=r.hdi[1],
                             p_globale=round(r.p_global, 4)))
        except Exception as e:
            rows.append(dict(variable=name, avance=np.nan, stat=np.nan,
                             prob_pic=np.nan, ipd_bas=np.nan, ipd_haut=np.nan,
                             p_globale=np.nan, erreur=str(e)[:60]))
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p_globale"] * len(candidates)).clip(upper=1.0)
    return out.sort_values("p_globale").reset_index(drop=True)
