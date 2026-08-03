"""phase_predict.py — Predictibilite conditionnelle a la phase du cycle.

Question posee
--------------
« Sachant que l'economie est en phase p aujourd'hui, la variable x apporte-t-elle
de l'information sur le PIB dans k trimestres, au-dela de ce que le PIB courant
dit deja ? Et cette information depend-elle de la phase ? »

Specification
-------------
    y_{t+k} = SUM_p  D_{p,t} * ( alpha_p + gamma_p * y_t + beta_p * x_t )  + e_{t+k}

    D_{p,t} : indicatrice de la phase a la date du PREDICTEUR (pas de la cible)
    gamma_p : persistance propre a la phase — le banc d'essai autoregressif
    beta_p  : APPORT PREDICTIF de x, en sus du PIB courant, dans la phase p

Quatre choix de conception, et leurs raisons
--------------------------------------------
1. CONDITIONNEMENT SUR LA DATE DU PREDICTEUR. On veut pouvoir dire « nous sommes
   en Ralentissement, que regarder ? ». La cible y_{t+k} a donc le droit d'etre
   dans une autre phase : c'est meme le cas interessant, puisqu'on cherche a
   anticiper un changement de phase. Exiger que les deux dates soient dans le
   meme episode viderait l'echantillon — a k=6, il ne resterait que 7 paires
   pour Decrochage.

2. UNE SEULE EQUATION A INTERACTIONS, pas quatre regressions separees. On garde
   les 306 observations, la variance residuelle est estimee sur tout
   l'echantillon, et surtout on dispose d'un TEST DE WALD de l'egalite des
   beta_p — c'est-a-dire un test formel de « la predictibilite depend de la
   phase », que des regressions separees ne fournissent jamais.

3. y_t EN CONTROLE PLUTOT QUE LA PURGE. Par le theoreme de Frisch-Waugh,
   inclure y_t dans la regression est ALGEBRIQUEMENT EQUIVALENT a purger x de sa
   projection sur y_t puis regresser. La version en controle est preferable ici :
   elle est standard, elle laisse gamma_p s'ajuster par phase, et elle evite
   l'artefact de decalage 0 constate avec la purge explicite.

4. x STANDARDISE (z-score plein echantillon). beta_p se lit alors « effet sur
   y_{t+k}, en unites de y, d'un ecart-type de x ». Sans cela, les coefficients
   ne seraient pas comparables entre variables et le tableau final serait
   illisible.

Inference
---------
- Erreurs-types de Newey-West. Indispensable : des horizons k>1 sur donnees
  trimestrielles creent des observations CHEVAUCHANTES (y_{t+k} et y_{t+1+k}
  partagent k-1 trimestres), donc des residus fortement autocorreles par
  construction. Fenetre m = max(k, regle usuelle).
- Bootstrap par EPISODE (cluster). On reechantillonne des episodes entiers avec
  remise. Aucune jonction artificielle n'est creee, contrairement a une
  concatenation des observations d'une meme phase, ou 59 a 81 % des paires
  (x_{t-k}, y_t) associeraient des observations distantes de decennies.

Dependances : numpy, pandas
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Briques d'estimation
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray):
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    return b, u, XtX_inv


def _newey_west(X: np.ndarray, u: np.ndarray, XtX_inv: np.ndarray, m: int):
    """Matrice de covariance HAC (noyau de Bartlett).

    Omega = G_0 + SUM_j (1 - j/(m+1)) (G_j + G_j')   avec G_j = SUM_t u_t u_{t-j} X_t X_{t-j}'
    V     = (X'X)^-1 Omega (X'X)^-1

    Le poids 1 - j/(m+1) decroit lineairement et garantit qu'Omega reste
    semi-definie positive.
    """
    n = len(u)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for j in range(1, m + 1):
        if j >= n:
            break
        w = 1.0 - j / (m + 1.0)
        A = (X[j:] * u[j:, None]).T @ (X[:-j] * u[:-j, None])
        S = S + w * (A + A.T)
    return XtX_inv @ S @ XtX_inv


def _wald(b: np.ndarray, V: np.ndarray, idx: list, n_clusters: int | None = None,
          n: int | None = None, K: int | None = None) -> tuple:
    """Test de Wald de l'egalite des coefficients d'indices `idx`.

    H0 : b[idx[0]] = b[idx[1]] = ... . On construit les contrastes deux a deux
    avec le premier, soit q = len(idx)-1 restrictions, et
        W = (Rb)' (R V R')^-1 (Rb)  ~  chi2(q)  sous H0.
    """
    from scipy import stats
    q = len(idx) - 1
    if q < 1:
        return np.nan, np.nan, 0
    R = np.zeros((q, len(b)))
    for i in range(q):
        R[i, idx[0]] = 1.0
        R[i, idx[i + 1]] = -1.0
    Rb = R @ b
    M = R @ V @ R.T
    try:
        W = float(Rb @ np.linalg.solve(M, Rb))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, q
    if n_clusters and n_clusters > q + 1:
        # Correction de cluster (Cameron-Miller) : avec G groupes seulement,
        # la loi asymptotique chi2 sur-rejette lourdement. On applique le
        # facteur d'echelle fini et on lit W/q dans une F(q, G-1) au lieu
        # d'une chi2(q). Ici G = nombre d'EPISODES, pas d'observations : c'est
        # lui qui borne l'information disponible.
        c = (n_clusters / (n_clusters - 1.0))
        if n and K and n > K:
            c *= (n - 1.0) / (n - K)
        W = W / c
        return W, float(1 - stats.f.cdf(W / q, q, n_clusters - 1)), q
    return W, float(1 - stats.chi2.cdf(W, q)), q


# ---------------------------------------------------------------------------
# Construction du jeu de donnees
# ---------------------------------------------------------------------------

def build_design(phases: pd.DataFrame, x: pd.Series, y: pd.Series, k: int,
                 phase_col: str = "phase", control_y: bool = True,
                 min_phase_obs: int = 15):
    """Assemble la matrice de regression pour l'horizon k.

    Chaque ligne est un couple (date du predicteur t, cible t+k). L'appariement
    se fait SUR LES DATES via l'index de periodes, jamais par position.
    """
    idx = phases.index
    ph = phases[phase_col]
    df = pd.DataFrame(index=idx)
    df["phase"] = ph
    df["episode"] = (ph != ph.shift()).cumsum()
    df["x"] = x.reindex(idx)
    df["y0"] = y.reindex(idx)
    df["ycible"] = y.reindex(idx).shift(-k)          # y_{t+k}, aligne sur les dates
    df = df.dropna(subset=["x", "y0", "ycible", "phase"])

    keep = df.phase.value_counts()
    keep = sorted(keep[keep >= min_phase_obs].index.tolist())
    df = df[df.phase.isin(keep)]
    if len(df) < 40 or len(keep) < 2:
        return None

    # x standardise : beta se lit « par ecart-type de x »
    xs = (df.x - df.x.mean()) / df.x.std(ddof=1)

    blocks, names = [], []
    for p in keep:
        D = (df.phase == p).astype(float).to_numpy()
        blocks.append(D)
        names.append(f"const[{p}]")
        if control_y:
            blocks.append(D * df.y0.to_numpy())
            names.append(f"y0[{p}]")
        blocks.append(D * xs.to_numpy())
        names.append(f"beta[{p}]")
    X = np.column_stack(blocks)
    return dict(X=X, y=df.ycible.to_numpy(), names=names, phases=keep,
                episode=df.episode.to_numpy(), n=len(df),
                beta_idx=[i for i, nm in enumerate(names) if nm.startswith("beta[")],
                n_by_phase={p: int((df.phase == p).sum()) for p in keep})


# ---------------------------------------------------------------------------
# Estimation pour une variable
# ---------------------------------------------------------------------------

def fit_horizon(design: dict, k: int, n_boot: int = 0, seed: int = 0) -> dict:
    X, y = design["X"], design["y"]
    b, u, XtX_inv = _ols(X, y)
    n = len(y)
    m = max(k, int(np.floor(4 * (n / 100) ** (2 / 9))))
    V = _newey_west(X, u, XtX_inv, m)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    G = len(np.unique(design["episode"]))
    W, p_wald, q = _wald(b, V, design["beta_idx"], n_clusters=G, n=n, K=X.shape[1])

    out = dict(n=n, m_hac=m, n_episodes=G, W=W, p_wald=p_wald, q=q,
               beta={}, se={}, t={}, p={})
    from scipy import stats
    for i, p_ in zip(design["beta_idx"], design["phases"]):
        out["beta"][p_] = float(b[i])
        out["se"][p_] = float(se[i])
        tt = b[i] / se[i] if se[i] > 0 else np.nan
        out["t"][p_] = float(tt)
        out["p"][p_] = float(2 * (1 - stats.norm.cdf(abs(tt)))) if np.isfinite(tt) else np.nan

    if n_boot > 0:
        rng = np.random.default_rng(seed)
        eps = np.unique(design["episode"])
        Wb = []
        bb = {p_: [] for p_ in design["phases"]}
        for _ in range(n_boot):
            pick = rng.choice(eps, len(eps), replace=True)
            rows = np.concatenate([np.where(design["episode"] == e)[0] for e in pick])
            if len(rows) < X.shape[1] + 10:
                continue
            try:
                bs, us, inv_s = _ols(X[rows], y[rows])
                Vs = _newey_west(X[rows], us, inv_s, m)
                # Statistique PIVOTALE : on recentre sur b observe. Sans cela la
                # loi bootstrap serait centree sur W_obs et la p-value tendrait
                # mecaniquement vers 0.5, quelle que soit la verite.
                Ws, _, _ = _wald(bs - b, Vs, design["beta_idx"],
                                 n_clusters=G, n=len(rows), K=X.shape[1])
                if np.isfinite(Ws):
                    Wb.append(Ws)
                for i, p_ in zip(design["beta_idx"], design["phases"]):
                    bb[p_].append(bs[i])
            except Exception:
                continue
        out["p_wald_boot"] = (float(np.mean(np.array(Wb) >= W)) if Wb and np.isfinite(W)
                              else np.nan)
        out["ic_beta"] = {p_: (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
                          for p_, v in bb.items() if len(v) > 20}
    return out


def analyse_variable(phases: pd.DataFrame, x: pd.Series, y: pd.Series,
                     horizons=range(1, 9), n_boot: int = 0, **kw) -> pd.DataFrame:
    rows = []
    for k in horizons:
        d = build_design(phases, x, y, k, **kw)
        if d is None:
            continue
        r = fit_horizon(d, k, n_boot=n_boot)
        for p_ in d["phases"]:
            rows.append(dict(horizon=k, phase=p_, beta=r["beta"][p_], se=r["se"][p_],
                             t=r["t"][p_], p=r["p"][p_], n=d["n_by_phase"][p_],
                             p_wald=r["p_wald"], W=r["W"],
                             p_wald_boot=r.get("p_wald_boot", np.nan)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Balayage multi-variables
# ---------------------------------------------------------------------------

def benjamini_hochberg(p: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Controle du taux de fausses decouvertes. Moins conservateur que
    Bonferroni, adapte a une phase exploratoire : on accepte qu'une part alpha
    des decouvertes soit fausse, plutot que d'exiger aucune fausse decouverte."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    mtot = len(idx)
    seuil = 0
    for i, j in enumerate(idx, start=1):
        if p[j] <= alpha * i / mtot:
            seuil = i
    out[idx[:seuil]] = True
    return out


def scan_phases(phases: pd.DataFrame, panel: pd.DataFrame, y: pd.Series,
                horizons=range(1, 9), n_boot: int = 0, alpha: float = 0.10,
                min_obs: int = 120, **kw) -> dict:
    """Applique l'analyse a toutes les colonnes du panel.

    Renvoie {'detail', 'resume', 'tableau'} :
      detail  : une ligne par (variable, horizon, phase)
      resume  : une ligne par variable — meilleur horizon, p de Wald, verdict FDR
      tableau : croisement variable x phase des beta au meilleur horizon
    """
    det = []
    for nm in panel.columns:
        s = panel[nm]
        if s.notna().sum() < min_obs:
            continue
        try:
            r = analyse_variable(phases, s, y, horizons=horizons, n_boot=n_boot, **kw)
        except Exception:
            continue
        if len(r) == 0:
            continue
        r.insert(0, "variable", nm)
        det.append(r)
    if not det:
        return dict(detail=pd.DataFrame(), resume=pd.DataFrame(), tableau=pd.DataFrame())
    detail = pd.concat(det, ignore_index=True)

    # Un horizon par variable : celui qui maximise |t| moyen entre phases.
    force = (detail.assign(at=detail.t.abs())
                   .groupby(["variable", "horizon"]).at.mean().reset_index())
    best = force.loc[force.groupby("variable").at.idxmax()][["variable", "horizon"]]
    best = best.rename(columns={"horizon": "h_opt"})

    res = (detail.merge(best, on="variable")
                 .query("horizon == h_opt")
                 .groupby("variable")
                 .agg(h_opt=("h_opt", "first"), p_wald=("p_wald", "first"),
                      p_wald_boot=("p_wald_boot", "first"),
                      beta_max=("beta", lambda v: v.abs().max()),
                      t_max=("t", lambda v: v.abs().max()))
                 .reset_index())
    res["FDR_wald"] = benjamini_hochberg(res.p_wald.to_numpy(), alpha)
    res = res.sort_values("p_wald").reset_index(drop=True)

    tab = (detail.merge(best, on="variable").query("horizon == h_opt")
                 .pivot(index="variable", columns="phase", values="beta").round(3))
    return dict(detail=detail, resume=res, tableau=tab)
