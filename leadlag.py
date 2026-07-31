"""
leadlag.py — Estimation du décalage (avance/retard) entre deux séries,
avec distribution de probabilité sur le décalage estimé.

Convention de signe unique dans tout le module
---------------------------------------------
lag k > 0  <=>  x est EN AVANCE de k périodes sur y
                (on corrèle x[t-k] avec y[t])

lag k < 0  <=>  x est en retard sur y

Trois problèmes traités explicitement
-------------------------------------
1. CORRÉLATION FALLACIEUSE.
   Deux séries autocorrélées exhibent des corrélations croisées élevées même
   indépendantes. La variance de la CCF empirique est gonflée par :

       Σk rho_x(k) rho_y(k)

   (formule de Bartlett).

   Correctif : pré-blanchiment de Box-Jenkins.
   On ajuste un AR(p) sur x, on filtre LES DEUX séries par ce même filtre,
   puis on calcule la CCF sur les résidus.

2. TESTS MULTIPLES.
   Chercher le max de |rho| sur plusieurs décalages puis tester ce max comme
   s'il avait été choisi a priori gonfle massivement le taux de faux positifs.

   Correctif : bootstrap de la statistique

       max_k |rho(k)|

3. INCERTITUDE SUR argmax.
   Le décalage estimé est lui-même une variable aléatoire.

   Correctif : bootstrap stationnaire (Politis-Romano) sur les paires (x, y),
   ce qui permet d'obtenir une distribution empirique de :

       P(avance = k)

Dépendances : numpy, pandas, statsmodels, scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _to_period_index(s: pd.Series, freq: str | None):
    """
    Convertit un index temporel en PeriodIndex.

    Renvoie :
        (serie, freq)
    ou :
        (None, None)
    """
    idx = s.index

    if isinstance(idx, pd.PeriodIndex):
        return s, idx.freqstr

    if isinstance(idx, pd.DatetimeIndex):
        f = freq or pd.infer_freq(idx)

        if f is None:
            raise ValueError(
                "Index de dates à pas irrégulier : impossible d'inférer "
                "la fréquence. Passez explicitement freq='Q', 'M', 'A'..."
            )

        return s.to_period(f), s.to_period(f).index.freqstr

    return None, None


def _align(x, y, freq: str | None = None, strict: bool = True):
    """
    Aligne deux séries SUR LES DATES et sur une grille temporelle régulière.

    Les périodes manquantes sont conservées sous forme de NaN plutôt que
    supprimées.

    Returns
    -------
    sx, sy : pd.Series
    mode   : "date" ou "position"
    freq   : fréquence retenue
    """
    sx = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, float))
    sy = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, float))

    sx = sx.astype(float).copy()
    sy = sy.astype(float).copy()

    px, fx = _to_period_index(sx, freq)
    py, fy = _to_period_index(sy, freq)

    if px is None or py is None:
        msg = (
            "ALIGNEMENT POSITIONNEL : au moins une des deux séries n'a pas "
            "d'index temporel. Les décalages seront comptés en LIGNES."
        )

        if strict and (px is not None or py is not None):
            raise ValueError(
                msg +
                " (une seule des deux séries est datée : mélange interdit)"
            )

        warnings.warn(msg)

        n = min(len(sx), len(sy))

        if len(sx) != len(sy):
            raise ValueError(
                f"Longueurs différentes ({len(sx)} vs {len(sy)}) "
                "sans index temporel."
            )

        idx = pd.RangeIndex(n)

        return (
            pd.Series(sx.to_numpy()[:n], index=idx),
            pd.Series(sy.to_numpy()[:n], index=idx),
            "position",
            None
        )

    if fx != fy:
        raise ValueError(
            f"Fréquences incompatibles : x='{fx}', y='{fy}'. "
            "Agrégerez au préalable la série la plus fine."
        )

    for s, nm in ((px, "x"), (py, "y")):
        if s.index.has_duplicates:
            raise ValueError(
                f"Index de {nm} comporte des dates dupliquées."
            )

    lo = max(px.index.min(), py.index.min())
    hi = min(px.index.max(), py.index.max())

    if lo > hi:
        raise ValueError(
            "Les deux séries ne se recouvrent sur aucune période."
        )

    grid = pd.period_range(lo, hi, freq=fx)

    ax = px.reindex(grid)
    ay = py.reindex(grid)

    n_ok = int((ax.notna() & ay.notna()).sum())

    if n_ok < 30:
        raise ValueError(
            f"Seulement {n_ok} périodes communes renseignées."
        )

    n_gap = len(grid) - n_ok

    if n_gap:
        warnings.warn(
            f"{n_gap} période(s) sur {len(grid)} incomplètes entre "
            f"{lo} et {hi}."
        )

    return ax, ay, "date", fx


def _standardise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / np.nanstd(a, ddof=1)


def optimal_block_length(a: np.ndarray) -> float:
    """
    Longueur de bloc moyenne pour le bootstrap stationnaire.
    """
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]

    n = len(a)

    a = _standardise(a)

    rho = np.corrcoef(a[:-1], a[1:])[0, 1]
    rho = np.clip(abs(rho), 0.0, 0.95)

    return max(
        2.0,
        min(
            n / 4.0,
            n ** (1 / 3)
            * (1 + 2 * rho)
            / (1 - rho + 1e-6) ** 0.5
        )
    )

def stationary_bootstrap_index(
    n: int,
    block_len: float,
    rng
) -> np.ndarray:
    """
    Indices d'un tirage de bootstrap stationnaire
    (Politis-Romano 1994).

    Blocs de longueur géométrique de moyenne block_len,
    enroulés circulairement.
    """

    p = 1.0 / max(block_len, 1.0)

    idx = np.empty(n, dtype=int)

    idx[0] = rng.integers(n)

    new_block = rng.random(n) < p
    jumps = rng.integers(0, n, size=n)

    for t in range(1, n):

        idx[t] = (
            jumps[t]
            if new_block[t]
            else (idx[t - 1] + 1) % n
        )

    return idx


def _longest_finite_run(a: np.ndarray):
    """
    Plus long segment sans NaN.
    Sert à estimer le filtre AR.
    """

    fin = np.isfinite(a)

    best = (0, 0)

    s = None

    for i, f in enumerate(np.append(fin, False)):

        if f and s is None:
            s = i

        elif not f and s is not None:

            if i - s > best[1] - best = (s, i)

            s = None

    return best


def purge_contemporaneous(
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Retire de x sa projection sur y contemporain.

    Le résidu obtenu représente la composante anticipative
    de x après retrait de la partie expliquée par la
    conjoncture courante.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 20:
        return x.copy()

    Z = np.column_stack([
        np.ones(m.sum()),
        y[m]
    ])

    beta = np.linalg.lstsq(
        Z,
        x[m],
        rcond=None
    )[0]

    out = np.full(len(x), np.nan)

    out[m] = x[m] - Z @ beta

    return out


def _fit_ar_filter(
    a: np.ndarray,
    max_p: int = 8
):
    """
    Ajuste un AR(p) par AIC.

    Renvoie :
        (coefficients, p)
    """

    from statsmodels.tsa.ar_model import AutoReg

    a = np.asarray(a, float)

    i0, i1 = _longest_finite_run(a)

    if (i1 - i0) >= max(
        40,
        0.6 * np.isfinite(a).sum()
    ):

        a = a[i0:i1]

    else:

        s = pd.Series(a).interpolate(
            limit_area="inside"
        )

        a = s.to_numpy()[
            np.isfinite(s.to_numpy())
        ]

    if len(a) < 40:
        return np.array([]), 0

    best_aic = np.inf

    best = (
        np.array([]),
        0
    )

    for p in range(1, max_p + 1):

        try:

            res = AutoReg(
                a,
                lags=p,
                old_names=False
            ).fit()

            if (
                np.isfinite(res.aic)
                and res.aic < best_aic
            ):

                best_aic = res.aic

                best = (
                    np.asarray(res.params)[1:],
                    p
                )

        except Exception:
            continue

    return best


def _apply_filter(
    a: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    """
    Applique le filtre :

        (1 - phi1*L - ... - phip*L^p)
    """

    p = len(phi)

    if p == 0:
        return a - np.nanmean(a)

    out = a[p:].copy()

    for j, c in enumerate(phi, start=1):

        out = out - c * a[
            p - j:-j
        ]

    return out


def prewhiten(
    x: np.ndarray,
    y: np.ndarray,
    max_p: int = 8
):
    """
    Pré-blanchiment Box-Jenkins.

    Le filtre est estimé sur x seulement,
    puis appliqué aux deux séries.
    """

    phi, p = _fit_ar_filter(
        np.asarray(x, float),
        max_p
    )

    return (
        _apply_filter(
            np.asarray(x, float),
            phi
        ),
        _apply_filter(
            np.asarray(y, float),
            phi
        ),
        p
    )


def _ccf(
    x: np.ndarray,
    y: np.ndarray,
    lags
):
    """
    Corrélations croisées robustes aux NaN.

    lag k > 0 :
        x[t-k] vs y[t]
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    rho = []
    cnt = []

    for k in lags:

        if k > 0:

            a = x[:-k]
            b = y[k:]

        elif k < 0:

            a = x[-k:]
            b = y[:k]

        else:

            a = x
            b = y

        m = np.isfinite(a) & np.isfinite(b)

        if (
            m.sum() < 12
            or np.std(a[m]) == 0
            or np.std(b[m]) == 0
        ):

            rho.append(np.nan)
            cnt.append(int(m.sum()))

        else:

            rho.append(
                float(
                    np.corrcoef(
                        a[m],
                        b[m]
                    )[0, 1]
                )
            )

            cnt.append(int(m.sum()))

    return (
        np.asarray(rho),
        np.asarray(cnt)
    )

# ---------------------------------------------------------------------------
# Resultat
# ---------------------------------------------------------------------------

@dataclass
class LeadLagResult:
    method: str
    lags: np.ndarray
    stat: np.ndarray                  # rho (ccf) ou pseudo-R2 (probit)
    se: np.ndarray | None
    best_lag: int
    best_stat: float
    prob_lag: pd.Series               # P(avance = k), somme à 1
    hdi: tuple                        # intervalle de plus haute densité à 90 %
    p_global: float                   # p-value corrigée des tests multiples
    n_obs: int
    detail: dict = field(default_factory=dict)

    def table(self) -> pd.DataFrame:

        df = pd.DataFrame({
            "lag": self.lags,
            "stat": self.stat,
            "P(avance=k)": self.prob_lag.reindex(
                self.lags
            ).to_numpy(),
        })

        if self.se is not None:

            df["se"] = self.se
            df["t"] = self.stat / self.se

        return df.set_index("lag")

    def summary(self) -> str:

        sens = (
            "en avance de"
            if self.best_lag > 0
            else (
                "en retard de"
                if self.best_lag < 0
                else "synchrone avec"
            )
        )

        unit = {
            "Q": "trimestre",
            "M": "mois",
            "A": "annee",
            "Y": "annee",
        }.get(
            str(
                self.detail.get("freq") or ""
            )[:1],
            "periode",
        )

        am = self.detail.get("align_mode")

        L = [

            f"Methode           : {self.method}",

            (
                "Alignement        : "
                + (
                    f"SUR LES DATES ({self.detail.get('freq')}), "
                    f"{self.detail.get('periode')}"
                    if am == "date"
                    else
                    "PAR POSITION (aucun index temporel) — a verifier"
                )
            ),

            f"Observations      : {self.n_obs}",

            (
                f"Decalage estime   : x est "
                f"{sens} {abs(self.best_lag)} "
                f"{unit}(s) sur y"
            ),

            f"Statistique       : {self.best_stat:+.3f}",

            (
                f"P(avance = {self.best_lag:+d})   : "
                f"{self.prob_lag.get(self.best_lag, np.nan):.1%}"
            ),

            (
                f"IPD 90 %          : "
                f"[{self.hdi+d} ; {self.hdi+d}] "
                f"{unit}s"
            ),

            (
                f"p globale (max|.|): "
                f"{self.p_global:.4f}"
                + (
                    "   <- aucune relation decelable"
                    if self.p_global > 0.10
                    else ""
                )
            ),
        ]

        if self.detail.get("ar_order") is not None:
            L.append(
                f"Ordre AR filtre   : "
                f"{self.detail['ar_order']}"
            )

        return "\n".join(L)

    def plot(self, ax=None):

        import matplotlib.pyplot as plt

        if ax is None:

            _, ax = plt.subplots(
                2,
                1,
                figsize=(9, 6),
                sharex=True,
                gridspec_kw={
                    "height_ratios": [2, 1]
                }
            )

        a0, a1 = ax

        a0.axhline(
            0,
            lw=.8,
            color="k"
        )

        a0.plot(
            self.lags,
            self.stat,
            marker="o",
            ms=3,
            lw=1.2,
            color="#185FA5"
        )

        if self.se is not None:

            a0.fill_between(
                self.lags,
                -1.96 * self.se,
                1.96 * self.se,
                color="#888780",
                alpha=.18,
                label="IC 95 % ponctuel"
            )

            a0.legend(
                fontsize=8,
                frameon=False
            )

        a0.axvline(
            self.best_lag,
            color="#D85A30",
            lw=1,
            ls="--"
        )

        a0.set_ylabel("statistique")

        a0.set_title(
            f"{self.method} — pic a k={self.best_lag:+d}",
            fontsize=10
        )

        pr = (
            self.prob_lag
            .reindex(self.lags)
            .fillna(0)
        )

        a1.bar(
            self.lags,
            pr.to_numpy(),
            color="#1D9E75",
            width=.7
        )

        a1.axvspan(
            self.hdi[0] - .5,
            self.hdi[1] + .5,
            color="#1D9E75",
            alpha=.12
        )

        a1.set_ylabel("P(avance = k)")

        a1.set_xlabel(
            "decalage k (k>0 : x en avance sur y)"
        )

        return ax


def _hdi_from_prob(
    pr: pd.Series,
    mass: float = 0.90
) -> tuple:
    """
    Intervalle de plus haute densite :
    plus petit ensemble de lags contigus
    couvrant `mass` de la probabilite.
    """

    lags = pr.index.to_numpy()
    v = pr.to_numpy()

    best = (
        lags[0],
        lags[-1],
        len(lags) + 1
    )

    for i in range(len(lags)):

        s = 0.0

        for j in range(i, len(lags)):

            s += v[j]

            if s >= mass:

                if (j - i) < best = (
                        int(lags[i]),
                        int(lags[j]),
                        j - i
                    )

                break

    return best[0], best[1]


# ---------------------------------------------------------------------------
# 1. Corrélation croisée — cible continue
# ---------------------------------------------------------------------------

def leadlag_ccf(
    x,
    y,
    max_lag: int = 12,
    prewhiten_series: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    max_ar: int = 8,
    freq: str | None = None,
    min_lag: int | None = None,
    purge: bool = False,
) -> LeadLagResult:
    """
    Décalage entre deux séries continues par
    corrélation croisée.
    """

    sx, sy, mode, freq = _align(
        x,
        y,
        freq=freq
    )

    xa = sx.to_numpy(float)
    ya = sy.to_numpy(float)

    if purge:
        xa = purge_contemporaneous(xa, ya)

    ar_order = None

    if prewhiten_series:

        xf, yf, ar_order = prewhiten(
            xa,
            ya,
            max_ar
        )

    else:

        xf = xa
        yf = ya

    lo = (
        -max_lag
        if min_lag is None
        else int(min_lag)
    )

    if lo > max_lag:
        raise ValueError(
            "min_lag superieur a max_lag."
        )

    lags = np.arange(
        lo,
        max_lag + 1
    )

    rho, npair = _ccf(
        xf,
        yf,
        lags
    )

    n = len(xf)

    # Erreur-type.
    # Après pré-blanchiment x est un bruit blanc :
    # 1/sqrt(n-|k|) est valide.
    #
    # Sans pré-blanchiment on applique la formule
    # de Bartlett qui gonfle l'erreur-type par la
    # persistance conjointe des deux séries.

    neff = np.maximum(npair, 3)

    if prewhiten_series:

        se = 1.0 / np.sqrt(neff)

    else:

        m = min(n // 4, 40)

        rx, _ = _ccf(
            xf,
            xf,
            np.arange(1, m + 1)
        )

        ry, _ = _ccf(
            yf,
            yf,
            np.arange(1, m + 1)
        )

        infl = np.sqrt(
            1 + 2 * np.nansum(rx * ry)
        )

        se = infl / np.sqrt(neff)

    k_star = int(
        lags[
            np.nanargmax(
                np.abs(rho)
            )
        ]
    )

    r_star = float(
        rho[
            np.nanargmax(
                np.abs(rho)
            )
        ]
    )

    rng = np.random.default_rng(seed)

    b = optimal_block_length(xf)

    # (a) P(avance = k)
    # Bootstrap sur les PAIRES.
    # La dépendance croisée est conservée,
    # seule l'incertitude d'échantillonnage joue.

    counts = np.zeros(len(lags))

    for _ in range(n_boot):

        idx = stationary_bootstrap_index(
            n,
            b,
            rng
        )

        r, _ = _ccf(
            xf[idx],
            yf[idx],
            lags
        )

        if np.all(np.isnan(r)):
            continue

        counts[
            np.nanargmax(
                np.abs(r)
            )
        ] += 1

    prob = pd.Series(
        counts / max(counts.sum(), 1),
        index=lags,
        name="P(avance=k)"
    )

    # (b) p-value globale
    #
    # Sous H0 les deux séries sont indépendantes
    # mais conservent leur autocorrélation.
    #
    # On les rééchantillonne séparément puis on
    # compare le max|rho| observé à la distribution
    # bootstrap de max|rho|.

    by = optimal_block_length(yf)

    null_max = np.empty(n_boot)

    for i in range(n_boot):

        xi = stationary_bootstrap_index(
            n,
            b,
            rng
        )

        yi = stationary_bootstrap_index(
            n,
            by,
            rng
        )

        null_max[i] = np.nanmax(
            np.abs(
                _ccf(
                    xf[xi],
                    yf[yi],
                    lags
                )[0]
            )
        )

    p_glob = float(
        (
            np.sum(
                null_max >= abs(r_star)
            ) + 1
        ) / (n_boot + 1)
    )

    return LeadLagResult(

        method=(
            "CCF pre-blanchie"
            if prewhiten_series
            else "CCF brute"
        )
        + (
            " + purge"
            if purge
            else ""
        ),

        lags=lags,

        stat=rho,

        se=se,

        best_lag=k_star,

        best_stat=r_star,

        prob_lag=prob,

        hdi=_hdi_from_prob(prob),

        p_global=p_glob,

        n_obs=n,

        detail={
            "ar_order": ar_order,
            "block_len": round(b, 2),
            "n_pairs": npair,
            "align_mode": mode,
            "freq": freq,

            "periode":
                f"{sx.index.min()} -> "
                f"{sx.index.max()}"
                if mode == "date"
                else None,
        },
    )


# ---------------------------------------------------------------------------
# 2. Probit décalé — cible binaire
# ---------------------------------------------------------------------------

def leadlag_probit(
    x,
    event,
    max_lag: int = 12,
    n_boot: int = 1000,
    seed: int = 0,
    hac_lags: int | None = None,
    criterion: str = "pseudo_r2",
    freq: str | None = None,
    min_lag: int | None = None,
    purge: bool = False,
) -> LeadLagResult:
    """
    Décalage d'un prédicteur continu
    sur un événement binaire.

    Spécification d'Estrella-Mishkin :

        P(event_t = 1)
            = Phi(a + b * x_{t-k})
    """

    import statsmodels.api as sm

    sx, sy, mode, freq = _align(
        x,
        event,
        freq=freq
    )

    xa = sx.to_numpy(float)
    ya = sy.to_numpy(float)

    obs = ya[np.isfinite(ya)]

    if not np.isin(
        np.unique(obs),
        [0, 1]
    ).all():

        raise ValueError(
            "`event` doit etre binaire (0/1)."
        )

    if purge:
        xa = purge_contemporaneous(
            xa,
            ya
        )

    n_ev = int(np.nansum(ya))

    if n_ev < 8:

        warnings.warn(
            f"Seulement {n_ev} evenements : "
            "estimations tres instables."
        )

    if hac_lags is None:

        hac_lags = int(
            np.floor(
                4 * (len(xa) / 100) ** (2 / 9)
            )
        )

    ya_int = np.nan_to_num(
        ya,
        nan=0.0
    )

    lo = (
        -max_lag
        if min_lag is None
        else int(min_lag)
    )

    if lo > max_lag:
        raise ValueError(
            "min_lag superieur a max_lag."
        )

    lags = np.arange(
        lo,
        max_lag + 1
    )

    def _fit(xv, yv, k):

        if k > 0:

            a = xv[:-k]
            b = yv[k:]

        elif k < 0:

            a = xv[-k:]
            b = yv[:k]

        else:

            a = xv
            b = yv

        m = np.isfinite(a) & np.isfinite(b)

        a = a[m]
        b = b[m]

        if (
            len(a) < 25
            or b.sum() < 3
            or b.sum() == len(b)
        ):
            return None

        X = sm.add_constant(a)

        try:

            m = sm.Probit(
                b,
                X
            ).fit(
                disp=0,
                maxiter=200
            )

            m_hac = sm.Probit(
                b,
                X
            ).fit(
                disp=0,
                maxiter=200,
                cov_type="HAC",
                cov_kwds={
                    "maxlags": hac_lags
                }
            )

        except Exception:
            return None

        if criterion == "auc":

            from sklearn.metrics import roc_auc_score

            crit = roc_auc_score(
                b,
                m.predict(X)
            )

        else:

            crit = m.prsquared

        return dict(
            crit=crit,
            coef=m.params[1],
            se=m_hac.bse[1],
            p=m_hac.pvalues[1],
            n=len(a),
            ev=int(b.sum())
        )

    fits = {
        int(k): _fit(
            xa,
            ya,
            int(k)
        )
        for k in lags
    }

    crit = np.array([
        fits[int(k)]["crit"]
        if fits[int(k)]
        else np.nan
        for k in lags
    ])

    coef = np.array([
        fits[int(k)]["coef"]
        if fits[int(k)]
        else np.nan
        for k in lags
    ])

    pval = np.array([
        fits[int(k)]["p"]
        if fits[int(k)]
        else np.nan
        for k in lags
    ])

    ses = np.array([
        fits[int(k)]["se"]
        if fits[int(k)]
        else np.nan
        for k in lags
    ])

    k_star = int(
        lags[
            np.nanargmax(crit)
        ]
    )

    c_star = float(
        np.nanmax(crit)
    )

    rng = np.random.default_rng(seed)

    b_len = optimal_block_length(xa)

    # P(avance = k)
    # Bootstrap stationnaire sur les paires
    # (x, event)

    counts = np.zeros(
        len(lags)
    )

    for _ in range(n_boot):

        idx = (
            stationary_bootstrap_index(
                len(xa),
                b_len,
                rng
            )
        )

        xb = xa[idx]
        yb = ya[idx]

        if np.nansum(yb) < 5:
            continue

        c = np.array([
            (
                _fit(
                    xb,
                    yb,
                    int(k)
                )
                or {"crit": np.nan}
            )["crit"]
            for k in lags
        ])

        if np.all(np.isnan(c)):
            continue

        counts[
            np.nanargmax(c)
        ] += 1

    prob = pd.Series(
        counts / max(
            counts.sum(),
            1
        ),
        index=lags,
        name="P(avance=k)"
    )

    # p globale sur max_k du critère,
    # séries rendues indépendantes.

    by = optimal_block_length(
        ya_int
    )

    null_max = np.empty(
        n_boot
    )

    for i in range(n_boot):

        xi = (
            stationary_bootstrap_index(
                len(xa),
                b_len,
                rng
            )
        )

        yi = (
            stationary_bootstrap_index(
                len(ya),
                by,
                rng
            )
        )

        c = np.array([
            (
                _fit(
                    xa[xi],
                    ya[yi],
                    int(k)
                )
                or {"crit": np.nan}
            )["crit"]
            for k in lags
        ])

        null_max[i] = (
            np.nanmax(c)
            if not np.all(
                np.isnan(c)
            )
            else np.nan
        )

    null_max = null_max[
        ~np.isnan(null_max)
    ]

    p_glob = float(
        (
            np.sum(
                null_max >= c_star
            ) + 1
        )
        /
        (
            len(null_max) + 1
        )
    )

    return LeadLagResult(
        method=(
            f"Probit decale "
            f"({criterion}, HAC {hac_lags})"
            + (
                " + purge"
                if purge
                else ""
            )
        ),
        lags=lags,
        stat=crit,
        se=None,
        best_lag=k_star,
        best_stat=c_star,
        prob_lag=prob,
        hdi=_hdi_from_prob(prob),
        p_global=p_glob,
        n_obs=len(xa),
        detail={
            "coef": coef,
            "p_coef": pval,
            "se_coef": ses,
            "ar_order": None,
            "n_events": n_ev,
            "hac_lags": hac_lags,
            "align_mode": mode,
            "freq": freq,
            "periode":
                (
                    f"{sx.index.min()} -> "
                    f"{sx.index.max()}"
                )
                if mode == "date"
                else None,
        },
    )


# ---------------------------------------------------------------------------
# 3. Balayage sur plusieurs candidats
# ---------------------------------------------------------------------------
def scan(
    candidates: dict,
    target,
    binary: bool = False,
    max_lag: int = 12,
    n_boot: int = 1000,
    min_lag: int | None = 0,
    **kw
) -> pd.DataFrame:
    """
    Applique l'estimateur à plusieurs prédicteurs
    et classe les résultats.

    min_lag vaut 0 par DÉFAUT ici :
    un balayage de sélection de prédicteurs est une
    question prédictive, la grille doit exclure les
    décalages négatifs.

    Passez min_lag=None pour rouvrir la grille
    symétrique (analyse descriptive).

    ATTENTION :
    balayer N variables rouvre le problème des
    tests multiples à un second niveau.

    La colonne `p_bonferroni` applique la correction
    la plus conservatrice.
    """

    rows = []

    for name, s in candidates.items():

        try:

            r = (
                leadlag_probit
                if binary
                else leadlag_ccf
            )(
                s,
                target,
                max_lag=max_lag,
                n_boot=n_boot,
                min_lag=min_lag,
                **kw
            )

            rows.append(
                dict(
                    variable=name,
                    avance=r.best_lag,
                    stat=round(
                        r.best_stat,
                        3
                    ),
                    prob_pic=round(
                        float(
                            r.prob_lag.get(
                                r.best_lag,
                                np.nan
                            )
                        ),
                        3,
                    ),
                    ipd_bas=r.hdi[0],
                    ipd_haut=r.hdi[1],
                    p_globale=round(
                        r.p_global,
                        4,
                    ),
                )
            )

        except Exception as e:

            rows.append(
                dict(
                    variable=name,
                    avance=np.nan,
                    stat=np.nan,
                    prob_pic=np.nan,
                    ipd_bas=np.nan,
                    ipd_haut=np.nan,
                    p_globale=np.nan,
                    erreur=str(e)[:60],
                )
            )

    out = pd.DataFrame(rows)

    out["p_bonferroni"] = (
        out["p_globale"] * len(candidates)
    ).clip(upper=1.0)

    return (
        out
        .sort_values("p_globale")
        .reset_index(drop=True)
    )
