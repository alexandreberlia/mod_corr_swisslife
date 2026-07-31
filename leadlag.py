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

def _to_period_index(s: pd.Series, freq: str | None):
    """Convertit un index temporel en PeriodIndex. Renvoie (serie, freq) ou (None, None)."""
    idx = s.index
    if isinstance(idx, pd.PeriodIndex):
        return s, idx.freqstr
    if isinstance(idx, pd.DatetimeIndex):
        f = freq or pd.infer_freq(idx)
        if f is None:
            raise ValueError(
                "Index de dates a pas irregulier : impossible d'inferer la frequence. "
                "Passez explicitement freq='Q', 'M', 'A'..."
            )
        return s.to_period(f), s.to_period(f).index.freqstr
    return None, None


def _align(x, y, freq: str | None = None, strict: bool = True):
    """Aligne deux series SUR LES DATES et sur une grille temporelle REGULIERE.

    Point critique : les periodes manquantes sont conservees sous forme de NaN
    plutot que supprimees. Sans cela un seul trimestre absent decalerait toutes
    les observations suivantes d'un cran, et un « retard de 4 » vaudrait 4 lignes
    au lieu de 4 trimestres. Toute la suite du module repose sur l'equivalence
    « une ligne = une periode calendaire », qui n'est vraie que sur grille pleine.

    Returns
    -------
    sx, sy : pd.Series indexees par un PeriodIndex complet (avec NaN eventuels)
    mode   : "date" si l'alignement est calendaire, "position" sinon
    freq   : frequence retenue
    """
    sx = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, float))
    sy = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, float))
    sx, sy = sx.astype(float).copy(), sy.astype(float).copy()

    px, fx = _to_period_index(sx, freq)
    py, fy = _to_period_index(sy, freq)

    if px is None or py is None:
        msg = ("ALIGNEMENT POSITIONNEL : au moins une des deux series n'a pas "
               "d'index temporel. Les decalages seront comptes en LIGNES, pas en "
               "periodes calendaires. Fournissez des pd.Series indexees par "
               "DatetimeIndex ou PeriodIndex.")
        if strict and (px is not None or py is not None):
            raise ValueError(msg + " (une seule des deux series est datee : "
                                   "melange interdit, risque de decalage silencieux.)")
        warnings.warn(msg)
        n = min(len(sx), len(sy))
        if len(sx) != len(sy):
            raise ValueError(f"Longueurs differentes ({len(sx)} vs {len(sy)}) sans index "
                             "temporel : alignement impossible.")
        idx = pd.RangeIndex(n)
        return (pd.Series(sx.to_numpy()[:n], index=idx),
                pd.Series(sy.to_numpy()[:n], index=idx), "position", None)

    if fx != fy:
        raise ValueError(
            f"Frequences incompatibles : x est en '{fx}', y en '{fy}'. "
            "Agregez la serie la plus fine avant appel (ex. moyenne trimestrielle "
            "d'une serie mensuelle) — l'alignement automatique masquerait un choix "
            "de convention qui doit rester explicite."
        )
    for s, nm in ((px, "x"), (py, "y")):
        if s.index.has_duplicates:
            raise ValueError(f"Index de {nm} comporte des dates dupliquees.")

    # Grille reguliere couvrant l'intersection des deux supports.
    lo = max(px.index.min(), py.index.min())
    hi = min(px.index.max(), py.index.max())
    if lo > hi:
        raise ValueError("Les deux series ne se recouvrent sur aucune periode.")
    grid = pd.period_range(lo, hi, freq=fx)

    ax, ay = px.reindex(grid), py.reindex(grid)
    n_ok = int((ax.notna() & ay.notna()).sum())
    if n_ok < 30:
        raise ValueError(f"Seulement {n_ok} periodes communes renseignees.")
    n_gap = len(grid) - n_ok
    if n_gap:
        warnings.warn(f"{n_gap} periode(s) sur {len(grid)} incompletes entre {lo} et {hi} : "
                      "conservees en NaN, traitees par paires completes a chaque decalage.")
    return ax, ay, "date", fx


def _standardise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / np.nanstd(a, ddof=1)


def optimal_block_length(a: np.ndarray) -> float:
    """Longueur de bloc moyenne pour le bootstrap stationnaire.

    Regle pratique : b = n^(1/3) ajustee par la persistance AR(1). Une serie
    tres persistante exige des blocs longs pour que le reechantillonnage
    reproduise sa memoire.
    """
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
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


def _longest_finite_run(a: np.ndarray):
    """Plus long segment sans NaN — sert a estimer le filtre AR."""
    fin = np.isfinite(a); best = (0, 0); s = None
    for i, f in enumerate(np.append(fin, False)):
        if f and s is None:
            s = i
        elif not f and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    return best



def purge_contemporaneous(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Retire de x sa projection sur y CONTEMPORAIN.

    Motivation : la plupart des variables macro-financieres contiennent deux
    canaux vis-a-vis du cycle — une part qui l'anticipe, une part qui y REAGIT
    de facon contemporaine. La seconde a presque toujours la variance la plus
    grande, et c'est elle que le pic de la CCF designe. Restreindre la grille aux
    decalages positifs ne suffit pas : le pic se loge alors a +0, qui reste dans
    la grille. Il faut retirer la composante reactive, pas l'exclure du balayage.

    Ce residu est la part de x inexpliquee par la conjoncture courante,
    c'est-a-dire sa composante anticipative.

    N'ajoutez PAS de retards de y a la projection : en simulation, purger aussi
    y[t-1] sur-corrige et deplace le pic vers l'aval (avance vraie +6 estimee +8).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return x.copy()
    Z = np.column_stack([np.ones(m.sum()), y[m]])
    beta = np.linalg.lstsq(Z, x[m], rcond=None)[0]
    out = np.full(len(x), np.nan)
    out[m] = x[m] - Z @ beta
    return out


def _fit_ar_filter(a: np.ndarray, max_p: int = 8):
    """Ajuste un AR(p) par AIC et renvoie (coefficients, p).

    Ce sont ces coefficients qui serviront de filtre de pre-blanchiment.
    """
    from statsmodels.tsa.ar_model import AutoReg

    a = np.asarray(a, float)
    i0, i1 = _longest_finite_run(a)
    if (i1 - i0) >= max(40, 0.6 * np.isfinite(a).sum()):
        a = a[i0:i1]
    else:
        # Trous eparpilles : le plus long segment contigu est trop court pour
        # identifier un AR. On interpole lineairement les trous INTERNES pour la
        # seule ESTIMATION des coefficients. Le filtre est ensuite applique a la
        # serie d'origine, trous compris : aucune valeur fabriquee n'entre dans
        # la correlation croisee, elle sert uniquement a calibrer le filtre.
        s = pd.Series(a).interpolate(limit_area="inside")
        a = s.to_numpy()[np.isfinite(s.to_numpy())]
    if len(a) < 40:
        return np.array([]), 0
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
        return a - np.nanmean(a)
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


def _ccf(x: np.ndarray, y: np.ndarray, lags):
    """Correlations croisees, robustes aux NaN. lag k > 0 : x[t-k] vs y[t].

    A chaque decalage on ne retient que les PAIRES completes : les trous ne
    deplacent donc jamais les observations, ils reduisent seulement l'effectif.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, cnt = [], []
    for k in lags:
        if k > 0:
            a, b = x[:-k], y[k:]
        elif k < 0:
            a, b = x[-k:], y[:k]
        else:
            a, b = x, y
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 12 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            rho.append(np.nan); cnt.append(int(m.sum()))
        else:
            rho.append(float(np.corrcoef(a[m], b[m])[0, 1])); cnt.append(int(m.sum()))
    return np.asarray(rho), np.asarray(cnt)


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
        _u = None
        unit = {"Q": "trimestre", "M": "mois", "A": "annee", "Y": "annee"}.get(
            str(self.detail.get("freq") or "")[:1], "periode")
        am = self.detail.get("align_mode")
        L = [
            f"Methode           : {self.method}",
            f"Alignement        : "
            + (f"SUR LES DATES ({self.detail.get('freq')}), {self.detail.get('periode')}"
               if am == "date" else "PAR POSITION (aucun index temporel) — a verifier"),
            f"Observations      : {self.n_obs}",
            f"Decalage estime   : x est {sens} {abs(self.best_lag)} {unit}(s) sur y",
            f"Statistique       : {self.best_stat:+.3f}",
            f"P(avance = {self.best_lag:+d})   : {self.prob_lag.get(self.best_lag, np.nan):.1%}",
            f"IPD 90 %          : [{self.hdi[0]:+d} ; {self.hdi[1]:+d}] {unit}s",
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



def _probit_pseudo_r2(a: np.ndarray, b: np.ndarray, iters: int = 30) -> float:
    """Pseudo-R2 de McFadden d'un probit univarie, par Newton-Raphson direct.

    Sert UNIQUEMENT dans les boucles de bootstrap, ou seul le critere
    d'ajustement est utilise. Les statistiques rapportees (coefficient,
    erreur-type HAC, p-value) restent estimees par statsmodels sur l'echantillon
    d'origine. Environ 30x plus rapide que d'instancier un sm.Probit par tirage.
    """
    from scipy.stats import norm
    n = len(b)
    X = np.column_stack([np.ones(n), a])
    beta = np.zeros(2)
    pbar = b.mean()
    if pbar <= 0 or pbar >= 1:
        return np.nan
    beta[0] = norm.ppf(pbar)
    for _ in range(iters):
        eta = np.clip(X @ beta, -8, 8)
        Phi = np.clip(norm.cdf(eta), 1e-10, 1 - 1e-10)
        phi = norm.pdf(eta)
        lam = phi * (b - Phi) / (Phi * (1 - Phi))
        w = phi ** 2 / (Phi * (1 - Phi))
        g = X.T @ lam
        H = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(2), g)
        except np.linalg.LinAlgError:
            return np.nan
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    eta = np.clip(X @ beta, -8, 8)
    Phi = np.clip(norm.cdf(eta), 1e-10, 1 - 1e-10)
    ll = float(np.sum(b * np.log(Phi) + (1 - b) * np.log(1 - Phi)))
    ll0 = float(n * (pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar)))
    return 1.0 - ll / ll0 if ll0 != 0 else np.nan


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
                n_boot: int = 500, seed: int = 0, max_ar: int = 8,
                freq: str | None = None, min_lag: int | None = None,
                purge: bool = False) -> LeadLagResult:
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
    sx, sy, mode, freq = _align(x, y, freq=freq)
    xa, ya = sx.to_numpy(float), sy.to_numpy(float)
    if purge:
        xa = purge_contemporaneous(xa, ya)
    ar_order = None

    if prewhiten_series:
        xf, yf, ar_order = prewhiten(xa, ya, max_ar)
    else:
        xf, yf = xa, ya


    # Borne inferieure de la recherche. min_lag=0 (ou plus) restreint aux
    # decalages ou x PRECEDE y : c'est la seule grille coherente avec une
    # question predictive. Une grille symetrique demande « ou le co-mouvement
    # est-il maximal ? », question differente, que remporte presque toujours le
    # canal REACTIF quand il existe — la pente des taux se repentifie de +2,5 pt
    # pendant la recession contre une inversion de -0,8 pt avant, donc une
    # recherche libre la declare « en retard » sur le cycle.
    lo = -max_lag if min_lag is None else int(min_lag)
    if lo > max_lag:
        raise ValueError("min_lag superieur a max_lag.")
    lags = np.arange(lo, max_lag + 1)
    if purge and lo <= 0:
        warnings.warn(
            "purge=True retire la relation CONTEMPORAINE : le decalage 0 n'a plus "
            "de sens interpretable, et il capte les residus du filtrage. Sous "
            "independance il rafle le pic dans 33% des cas au lieu de 8%. "
            "min_lag releve a 1.")
        lo = 1
        lags = np.arange(lo, max_lag + 1)
    rho, npair = _ccf(xf, yf, lags)
    n = len(xf)

    # Erreur-type. Apres pre-blanchiment x est un bruit blanc : 1/sqrt(n-|k|)
    # est valide. Sans pre-blanchiment on applique la formule de Bartlett, qui
    # gonfle l'erreur-type par la persistance conjointe des deux series.
    neff = np.maximum(npair, 3)
    if prewhiten_series:
        se = 1.0 / np.sqrt(neff)
    else:
        m = min(n // 4, 40)
        rx, _ = _ccf(xf, xf, np.arange(1, m + 1))
        ry, _ = _ccf(yf, yf, np.arange(1, m + 1))
        infl = np.sqrt(1 + 2 * np.nansum(rx * ry))
        se = infl / np.sqrt(neff)

    k_star = int(lags[np.nanargmax(np.abs(rho))])
    r_star = float(rho[np.nanargmax(np.abs(rho))])

    rng = np.random.default_rng(seed)
    b = optimal_block_length(xf)

    # (a) P(avance = k) : bootstrap sur les PAIRES, la dependance croisee est
    #     conservee, seule l'incertitude d'echantillonnage joue.
    counts = np.zeros(len(lags))
    for _ in range(n_boot):
        idx = stationary_bootstrap_index(n, b, rng)
        r, _ = _ccf(xf[idx], yf[idx], lags)
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
        null_max[i] = np.nanmax(np.abs(_ccf(xf[xi], yf[yi], lags)[0]))
    p_glob = float((np.sum(null_max >= abs(r_star)) + 1) / (n_boot + 1))

    return LeadLagResult(
        method=("CCF pre-blanchie" if prewhiten_series else "CCF brute")
               + (" + purge" if purge else ""),
        lags=lags, stat=rho, se=se, best_lag=k_star, best_stat=r_star,
        prob_lag=prob, hdi=_hdi_from_prob(prob), p_global=p_glob, n_obs=n,
        detail={"ar_order": ar_order, "block_len": round(b, 2), "n_pairs": npair,
                "align_mode": mode, "freq": freq,
                "periode": f"{sx.index.min()} -> {sx.index.max()}" if mode == "date" else None},
    )


# ----------------------------------------------------------------------------
# 2. Probit décalé — cible binaire
# ----------------------------------------------------------------------------

def leadlag_probit(x, event, max_lag: int = 12, n_boot: int = 300, seed: int = 0,
                   hac_lags: int | None = None, criterion: str = "pseudo_r2",
                   freq: str | None = None, min_lag: int | None = None,
                purge: bool = False) -> LeadLagResult:
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

    sx, sy, mode, freq = _align(x, event, freq=freq)
    xa, ya = sx.to_numpy(float), sy.to_numpy(float)
    obs = ya[np.isfinite(ya)]
    if not np.isin(np.unique(obs), [0, 1]).all():
        raise ValueError("`event` doit etre binaire (0/1).")
    if purge:
        xa = purge_contemporaneous(xa, ya)
    n_ev = int(np.nansum(ya))
    if n_ev < 8:
        warnings.warn(f"Seulement {n_ev} evenements : estimations tres instables.")

    if hac_lags is None:
        hac_lags = int(np.floor(4 * (len(xa) / 100) ** (2 / 9)))
    ya_int = np.nan_to_num(ya, nan=0.0)


    # Borne inferieure de la recherche. min_lag=0 (ou plus) restreint aux
    # decalages ou x PRECEDE y : c'est la seule grille coherente avec une
    # question predictive. Une grille symetrique demande « ou le co-mouvement
    # est-il maximal ? », question differente, que remporte presque toujours le
    # canal REACTIF quand il existe — la pente des taux se repentifie de +2,5 pt
    # pendant la recession contre une inversion de -0,8 pt avant, donc une
    # recherche libre la declare « en retard » sur le cycle.
    lo = -max_lag if min_lag is None else int(min_lag)
    if lo > max_lag:
        raise ValueError("min_lag superieur a max_lag.")
    lags = np.arange(lo, max_lag + 1)
    if purge and lo <= 0:
        warnings.warn(
            "purge=True retire la relation CONTEMPORAINE : le decalage 0 n'a plus "
            "de sens interpretable, et il capte les residus du filtrage. Sous "
            "independance il rafle le pic dans 33% des cas au lieu de 8%. "
            "min_lag releve a 1.")
        lo = 1
        lags = np.arange(lo, max_lag + 1)

    def _fit(xv, yv, k, hac: bool = True):
        if k > 0:
            a, b = xv[:-k], yv[k:]
        elif k < 0:
            a, b = xv[-k:], yv[:k]
        else:
            a, b = xv, yv
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if len(a) < 25 or b.sum() < 3 or b.sum() == len(b):
            return None
        if not hac:
            # Chemin rapide : seul le critere est utilise dans le bootstrap.
            if criterion == "auc":
                from sklearn.metrics import roc_auc_score
                try:
                    from scipy.stats import norm
                    return dict(crit=roc_auc_score(b, a))
                except Exception:
                    return None
            c = _probit_pseudo_r2(a, b.astype(float))
            return None if not np.isfinite(c) else dict(crit=c)
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
        if np.nansum(yb) < 5:
            continue
        c = np.array([(_fit(xb, yb, int(k), hac=False) or {"crit": np.nan})["crit"] for k in lags])
        if np.all(np.isnan(c)):
            continue
        counts[np.nanargmax(c)] += 1
    prob = pd.Series(counts / max(counts.sum(), 1), index=lags, name="P(avance=k)")

    # p globale sur max_k du critere, series rendues independantes.
    by = optimal_block_length(ya_int)
    null_max = np.empty(n_boot)
    for i in range(n_boot):
        xi = stationary_bootstrap_index(len(xa), b_len, rng)
        yi = stationary_bootstrap_index(len(ya), by, rng)
        c = np.array([(_fit(xa[xi], ya[yi], int(k), hac=False) or {"crit": np.nan})["crit"] for k in lags])
        null_max[i] = np.nanmax(c) if not np.all(np.isnan(c)) else np.nan
    null_max = null_max[~np.isnan(null_max)]
    p_glob = float((np.sum(null_max >= c_star) + 1) / (len(null_max) + 1))

    return LeadLagResult(
        method=f"Probit decale ({criterion}, HAC {hac_lags})" + (" + purge" if purge else ""),
        lags=lags, stat=crit, se=None, best_lag=k_star, best_stat=c_star,
        prob_lag=prob, hdi=_hdi_from_prob(prob), p_global=p_glob, n_obs=len(xa),
        detail={"coef": coef, "p_coef": pval, "se_coef": ses, "ar_order": None,
                "n_events": n_ev, "hac_lags": hac_lags,
                "align_mode": mode, "freq": freq,
                "periode": f"{sx.index.min()} -> {sx.index.max()}" if mode == "date" else None},
    )


# ----------------------------------------------------------------------------
# 3. Balayage sur plusieurs candidats
# ----------------------------------------------------------------------------

def scan(candidates: dict, target, binary: bool = False, max_lag: int = 12,
         n_boot: int = 1000, min_lag: int | None = 0, **kw) -> pd.DataFrame:
    """Applique l'estimateur a plusieurs predicteurs et classe les resultats.

    min_lag vaut 0 par DEFAUT ici : un balayage de selection de predicteurs est
    une question predictive, la grille doit exclure les decalages negatifs.
    Passez min_lag=None pour rouvrir la grille symetrique (analyse descriptive).

    ATTENTION : balayer N variables rouvre le probleme des tests multiples a un
    second niveau. La colonne `p_bonferroni` applique la correction la plus
    conservatrice. Avec 6 candidats, une p globale de 0.04 devient 0.24.
    """
    rows = []
    for name, s in candidates.items():
        try:
            r = (leadlag_probit if binary else leadlag_ccf)(
                s, target, max_lag=max_lag, n_boot=n_boot, min_lag=min_lag, **kw)
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
