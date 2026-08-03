"""lead_rank.py — Classement des variables par AVANCE predictive sur le PIB.

Question posee
--------------
Pour chaque variable x prise SEPAREMENT : a quel horizon k anticipe-t-elle le
mieux la cible y ? Avec quelle probabilite cet horizon est-il le bon ? Et quelles
variables anticipent le mieux ?

Modele, pour chaque horizon k
-----------------------------
    y_{t+k} = alpha + gamma * y_t + beta_k * x_t + e_{t+k}

y_t en controle : beta_k mesure l'APPORT de x EN SUS de ce que la cible courante
dit deja d'elle-meme. Sans ce controle, toute variable procyclique semblerait
predire, alors qu'elle ne ferait que reproduire la persistance de y.

x est standardise (z-score). beta se lit « effet en unites de y d'un ecart-type
de x », ce qui rend les variables comparables entre elles.

Choix de l'horizon
------------------
    k* = argmax_k |t_k|

Le critere est la statistique de Student, pas |beta| : elle penalise
automatiquement les horizons ou l'effectif s'effondre ou la variance explose.

Trois quantites rapportees
--------------------------
1. k*                   l'avance estimee
2. P(avance = k)        frequence bootstrap de k* — mesure de STABILITE, pas de
                        croyance a posteriori. Somme a 1 par construction, donc
                        elle place de la masse quelque part meme sans signal :
                        a lire APRES p_max, jamais avant.
3. p_max                p-value du max_k |t_k|, corrigee de la recherche sur la
                        grille d'horizons. Sans elle, retenir le meilleur des 8
                        horizons puis le tester comme s'il avait ete choisi a
                        priori serait du p-hacking.

Le bootstrap est fait par EPISODE (cluster), jamais par observation : les
303 trimestres sont regroupes en ~41 episodes, et c'est le nombre d'episodes qui
borne l'information disponible.

Dependances : numpy, pandas, scipy
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Regroupements de phases
# ---------------------------------------------------------------------------

GROUPINGS = {
    "phase": None,           # les 4 phases d'origine
    "niveau": {"Decrochage": "Sous_potentiel", "Reprise": "Sous_potentiel",
               "Ralentissement": "Sur_potentiel", "Explosion": "Sur_potentiel"},
    "momentum": {"Decrochage": "Deceleration", "Ralentissement": "Deceleration",
                 "Reprise": "Acceleration", "Explosion": "Acceleration"},
    "global": {"Decrochage": "Tous", "Reprise": "Tous",
               "Ralentissement": "Tous", "Explosion": "Tous"},
}


def apply_grouping(phases: pd.DataFrame, how: str = "global",
                   phase_col: str = "phase") -> pd.Series:
    """Renvoie la serie de groupes.

    Pourquoi regrouper : avec 4 phases, chaque cellule ne compte que 8 a 12
    episodes. En les fusionnant deux a deux on double l'information par cellule.
    La classification reposant deja sur deux binaires (niveau, momentum), il est
    naturel de les tester SEPAREMENT avant de tester leur croisement.
    """
    if how not in GROUPINGS:
        raise ValueError(f"how doit etre dans {list(GROUPINGS)}")
    m = GROUPINGS[how]
    return phases[phase_col] if m is None else phases[phase_col].map(m)


# ---------------------------------------------------------------------------
# Briques
# ---------------------------------------------------------------------------

def _ols_t(X: np.ndarray, y: np.ndarray, m: int, j: int = -1):
    """OLS + t de Student HAC (Newey-West) sur le coefficient d'indice j."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    n = len(u)
    for lag in range(1, m + 1):
        if lag >= n:
            break
        w = 1.0 - lag / (m + 1.0)
        A = Xu[lag:].T @ Xu[:-lag]
        S = S + w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(max(V[j, j], 1e-12))
    return float(b[j]), float(se), float(b[j] / se)


def _panel_for(x: pd.Series, y: pd.Series, groups: pd.Series, k: int,
               episode: pd.Series):
    """Assemble (x_t, y_t, y_{t+k}, groupe_t, episode_t) aligne SUR LES DATES."""
    idx = groups.index
    df = pd.DataFrame({
        "g": groups, "ep": episode,
        "x": x.reindex(idx), "y0": y.reindex(idx),
        "yk": y.reindex(idx).shift(-k),
    }).dropna()
    return df


def _fit_k(df: pd.DataFrame, k: int, min_n: int = 30):
    """Regression pour un horizon et un sous-echantillon donnes."""
    if len(df) < min_n:
        return None
    xs = df["x"].to_numpy()
    sd = xs.std(ddof=1)
    if sd <= 0:
        return None
    xs = (xs - xs.mean()) / sd
    X = np.column_stack([np.ones(len(df)), df["y0"].to_numpy(), xs])
    m = max(k, int(np.floor(4 * (len(df) / 100) ** (2 / 9))))
    b, se, t = _ols_t(X, df["yk"].to_numpy(), m, j=2)
    return dict(beta=b, se=se, t=t, n=len(df))


# ---------------------------------------------------------------------------
# Coeur : une variable, un groupe
# ---------------------------------------------------------------------------

def lead_of(x: pd.Series, y: pd.Series, groups: pd.Series, episode: pd.Series,
            grp: str, horizons=range(1, 9), n_boot: int = 500,
            seed: int = 0, min_n: int = 30) -> dict | None:
    """Avance estimee, sa probabilite, et la p-value corrigee."""
    ks = list(horizons)
    fits, dfs = {}, {}
    for k in ks:
        d = _panel_for(x, y, groups, k, episode)
        d = d[d.g == grp] if grp is not None else d
        r = _fit_k(d, k, min_n)
        if r is not None:
            fits[k], dfs[k] = r, d
    if len(fits) < 2:
        return None

    kk = sorted(fits)
    t_obs = np.array([fits[k]["t"] for k in kk])
    i_star = int(np.argmax(np.abs(t_obs)))
    k_star = kk[i_star]

    rng = np.random.default_rng(seed)
    eps = np.unique(np.concatenate([dfs[k]["ep"].to_numpy() for k in kk]))
    rows = {k: {e: np.where(dfs[k]["ep"].to_numpy() == e)[0] for e in eps} for k in kk}

    cnt = {k: 0 for k in kk}
    maxdev, ok = [], 0
    for _ in range(n_boot):
        pick = rng.choice(eps, len(eps), replace=True)
        tb = np.full(len(kk), np.nan)
        for i, k in enumerate(kk):
            sel = np.concatenate([rows[k][e] for e in pick if len(rows[k][e])])
            if len(sel) < min_n:
                continue
            r = _fit_k(dfs[k].iloc[sel], k, min_n)
            if r is not None:
                tb[i] = r["t"]
        if np.all(np.isnan(tb)):
            continue
        cnt[kk[int(np.nanargmax(np.abs(tb)))]] += 1
        # Statistique PIVOTALE : on recentre sur t observe. Sans recentrage la
        # loi bootstrap serait centree sur le max observe et la p-value
        # tendrait mecaniquement vers 0.5.
        dev = np.abs(tb - t_obs)
        if np.any(np.isfinite(dev)):
            maxdev.append(np.nanmax(dev))
            ok += 1
    tot = max(sum(cnt.values()), 1)
    prob = {k: cnt[k] / tot for k in kk}
    p_max = (float(np.sum(np.array(maxdev) >= np.abs(t_obs[i_star]))) + 1) / (ok + 1) \
        if ok else np.nan

    # intervalle de plus haute densite sur les horizons
    v = np.array([prob[k] for k in kk])
    best = (kk[0], kk[-1], len(kk) + 1)
    for a in range(len(kk)):
        s = 0.0
        for b_ in range(a, len(kk)):
            s += v[b_]
            if s >= 0.90:
                if (b_ - a) < best[2]:
                    best = (kk[a], kk[b_], b_ - a)
                break

    return dict(avance=k_star, beta=fits[k_star]["beta"], t=t_obs[i_star],
                prob_avance=prob[k_star], ipd_bas=best[0], ipd_haut=best[1],
                p_max=p_max, n=fits[k_star]["n"], prob=prob)


# ---------------------------------------------------------------------------
# Classement
# ---------------------------------------------------------------------------

def benjamini_hochberg(p, alpha=0.10):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    cut = 0
    for i, j in enumerate(order, start=1):
        if p[j] <= alpha * i / len(order):
            cut = i
    out[order[:cut]] = True
    return out


def rank_leads(phases: pd.DataFrame, panel: pd.DataFrame, y: pd.Series,
               grouping: str = "global", horizons=range(1, 9),
               n_boot: int = 400, min_obs: int = 100, alpha: float = 0.10,
               phase_col: str = "phase", seed: int = 0) -> pd.DataFrame:
    """Classe les variables du panel par avance predictive.

    grouping : 'global' (tout l'echantillon), 'niveau' (sous/sur potentiel),
               'momentum' (acceleration/deceleration), 'phase' (les 4 phases).

    Colonnes de sortie, dans l'ordre de lecture :
        groupe, variable, avance, beta, t, prob_avance, ipd_bas, ipd_haut,
        p_max, p_adj (Bonferroni sur les variables), FDR (Benjamini-Hochberg), n
    """
    g = apply_grouping(phases, grouping, phase_col)
    episode = (phases[phase_col] != phases[phase_col].shift()).cumsum()
    groupes = sorted(g.dropna().unique())

    rows = []
    for grp in groupes:
        for nm in panel.columns:
            s = panel[nm]
            if s.notna().sum() < min_obs:
                continue
            try:
                r = lead_of(s, y, g, episode, grp, horizons, n_boot, seed)
            except Exception:
                r = None
            if r is None:
                continue
            r.pop("prob", None)
            rows.append(dict(groupe=grp, variable=nm, **r))
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    nv = out.variable.nunique()
    out["p_adj"] = (out.p_max * nv).clip(upper=1.0)
    out["FDR"] = False
    for grp, sub in out.groupby("groupe"):
        out.loc[sub.index, "FDR"] = benjamini_hochberg(sub.p_max.to_numpy(), alpha)
    cols = ["groupe", "variable", "avance", "beta", "t", "prob_avance",
            "ipd_bas", "ipd_haut", "p_max", "p_adj", "FDR", "n"]
    return (out[cols].sort_values(["groupe", "p_max"])
                     .round({"beta": 3, "t": 2, "prob_avance": 3,
                             "p_max": 4, "p_adj": 3})
                     .reset_index(drop=True))
