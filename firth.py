"""firth.py — Vraisemblance penalisee de Firth pour le cloglog.

Le probleme
-----------
Avec 2 a 6 evenements par branche, le maximum de vraisemblance echoue souvent :
il existe un hyperplan qui separe parfaitement les sorties des non-sorties. La
vraisemblance croit alors indefiniment le long d'une direction, aucun maximum
interieur n'existe, et l'algorithme renvoie des coefficients qui divergent
(+13.66 sur la branche Ralentissement -> Explosion) avec des erreurs-types
enormes. Ce ne sont pas des estimations, ce sont des artefacts numeriques.

La solution de Firth (1993)
---------------------------
Penaliser la vraisemblance par le prior de Jeffreys :

    l*(beta) = l(beta) + (1/2) log |I(beta)|

ou I(beta) = X' W X est l'information de Fisher. Trois consequences.

1. La penalite s'annule quand |I| tend vers 0, ce qui arrive precisement quand
   beta diverge. Elle empeche donc mecaniquement la divergence : l'estimateur
   penalise est TOUJOURS FINI, meme sous separation complete.

2. Elle retire le biais d'ordre O(1/n) du maximum de vraisemblance. C'etait
   l'objectif initial de Firth ; le traitement de la separation en a ete un
   sous-produit (Heinze-Schemper 2002).

3. Elle agit comme un retrait vers zero (« shrinkage ») d'autant plus fort que
   l'echantillon est petit. Sur grand echantillon, la penalite devient
   negligeable devant l(beta) et l'on retrouve le maximum de vraisemblance.

Implementation
--------------
On maximise l* directement par optimisation numerique plutot que de deriver le
score modifie. Avec quelques centaines d'observations le cout est negligeable,
et cela evite les erreurs d'algebre propres a la forme analytique du score
ajuste pour un lien non canonique comme le cloglog.

Pour le cloglog :
    eta = X beta
    mu  = 1 - exp(-exp(eta))
    dmu/deta = exp(eta) (1 - mu)
    w   = (dmu/deta)^2 / [mu (1-mu)] = exp(2 eta) (1-mu) / mu
    l(beta) = SUM [ y log mu - (1-y) exp(eta) ]     car log(1-mu) = -exp(eta)

Inference
---------
Les erreurs-types de Wald sont peu fiables sous separation, meme apres
penalisation : la log-vraisemblance n'est pas quadratique dans ces conditions.
On rapporte donc AUSSI un test du rapport de vraisemblance penalise, qui reste
valide et constitue la reference (Heinze-Schemper 2002).

Dependances : numpy, scipy
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import optimize, stats

warnings.filterwarnings("ignore")

_EPS = 1e-10


def _cloglog_bits(eta: np.ndarray):
    """mu, w et log(1-mu), avec bornage pour la stabilite numerique."""
    eta = np.clip(eta, -30.0, 5.0)
    e = np.exp(eta)
    mu = 1.0 - np.exp(-e)
    mu = np.clip(mu, _EPS, 1.0 - _EPS)
    dmu = e * (1.0 - mu)
    w = np.maximum(dmu ** 2 / (mu * (1.0 - mu)), _EPS)
    return mu, w, -e


def loglik(beta, X, y):
    eta = X @ beta
    mu, _, log1mmu = _cloglog_bits(eta)
    return float(np.sum(y * np.log(mu) + (1.0 - y) * log1mmu))


def penalised_loglik(beta, X, y):
    """l(beta) + (1/2) log |X' W X|."""
    eta = X @ beta
    mu, w, log1mmu = _cloglog_bits(eta)
    ll = float(np.sum(y * np.log(mu) + (1.0 - y) * log1mmu))
    I = X.T @ (X * w[:, None])
    sign, logdet = np.linalg.slogdet(I + 1e-12 * np.eye(X.shape[1]))
    if sign <= 0 or not np.isfinite(logdet):
        return -1e12
    return ll + 0.5 * logdet


def fit_firth(X: np.ndarray, y: np.ndarray, maxiter: int = 500) -> dict:
    """Estime un cloglog par vraisemblance penalisee de Firth.

    Returns
    -------
    dict : params, se, pvalues (Wald), pvalues_lr (rapport de vraisemblance),
           converge, separation, ic (profil du rapport de vraisemblance).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    if y.sum() < 1:
        raise ValueError("Aucun evenement dans l'echantillon.")

    # Detection de separation : le MV non penalise diverge-t-il ?
    sep = False
    try:
        import statsmodels.api as sm
        m0 = sm.GLM(y, X, family=sm.families.Binomial(
            sm.families.links.CLogLog())).fit(maxiter=100)
        sep = bool(np.max(np.abs(np.asarray(m0.params))) > 10
                   or np.max(np.asarray(m0.bse)) > 50
                   or not np.all(np.isfinite(np.asarray(m0.bse))))
    except Exception:
        sep = True

    b0 = np.zeros(k)
    b0[0] = np.log(-np.log(1.0 - np.clip(y.mean(), 1e-3, 1 - 1e-3)))
    res = optimize.minimize(lambda b: -penalised_loglik(b, X, y), b0,
                            method="BFGS",
                            options=dict(maxiter=maxiter, gtol=1e-8))
    beta = res.x

    # Erreurs-types depuis l'information observee au point penalise.
    eta = X @ beta
    _, w, _ = _cloglog_bits(eta)
    I = X.T @ (X * w[:, None])
    try:
        V = np.linalg.pinv(I)
        se = np.sqrt(np.maximum(np.diag(V), 0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    # Test du rapport de vraisemblance PENALISE, coefficient par coefficient.
    # Plus fiable que Wald sous separation, ou la log-vraisemblance n'est pas
    # quadratique et ou l'erreur-type perd son sens.
    ll_full = penalised_loglik(beta, X, y)
    p_lr = np.full(k, np.nan)
    for j in range(k):
        idx = [c for c in range(k) if c != j]
        if not idx:
            continue
        Xr = X[:, idx]
        try:
            r = optimize.minimize(lambda b: -penalised_loglik(b, Xr, y),
                                  np.zeros(len(idx)), method="BFGS",
                                  options=dict(maxiter=maxiter))
            lr = 2.0 * (ll_full - penalised_loglik(r.x, Xr, y))
            p_lr[j] = float(1 - stats.chi2.cdf(max(lr, 0.0), 1))
        except Exception:
            pass

    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
        p_wald = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))

    return dict(params=beta, se=se, pvalues=p_wald, pvalues_lr=p_lr,
                loglik=loglik(beta, X, y), loglik_pen=ll_full,
                converge=bool(res.success), separation=sep, n=n,
                evenements=int(y.sum()))


def ic_profil(X, y, j: int, niveau: float = 0.95, borne: float = 25.0) -> tuple:
    """Intervalle de confiance par profil de vraisemblance penalisee.

    On cherche les valeurs de beta_j telles que 2[l*(beta_hat) - l*_profil(b)]
    egale le quantile du chi2(1). Cet intervalle reste valide sous separation,
    contrairement a beta_hat +/- 1.96 se, qui suppose une log-vraisemblance
    quadratique — hypothese fausse en petit echantillon.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    k = X.shape[1]
    fit = fit_firth(X, y)
    ll_max = fit["loglik_pen"]
    seuil = stats.chi2.ppf(niveau, 1) / 2.0

    def profil(val):
        idx = [c for c in range(k) if c != j]
        off = X[:, j] * val

        def obj(b):
            bb = np.zeros(k)
            bb[idx] = b
            bb[j] = val
            return -penalised_loglik(bb, X, y)
        r = optimize.minimize(obj, np.zeros(len(idx)), method="BFGS",
                              options=dict(maxiter=300))
        return -r.fun

    def ecart(val):
        return (ll_max - profil(val)) - seuil

    bhat = fit["params"][j]
    bornes = []
    for sens in (-1, 1):
        lo, hi = bhat, bhat + sens * borne
        try:
            if ecart(hi) < 0:
                bornes.append(np.nan)      # non borne de ce cote
                continue
            bornes.append(float(optimize.brentq(ecart, lo, hi, xtol=1e-4)))
        except Exception:
            bornes.append(np.nan)
    return tuple(sorted([b for b in bornes if np.isfinite(b)]) or (np.nan, np.nan))
