"""rank_sortie.py — Quels indicateurs surveiller, selon la phase ou l'on est.

    « Je suis en Ralentissement. Quels indicateurs regarder en priorite pour
      anticiper la sortie, et avec quelle avance ? »

Trois questions distinctes, trois modules
------------------------------------------
  rank_leads   cible CONTINUE (PIB, chomage) : que predit le niveau futur ?
  rank_phase   cible ETAT : dans quelle phase serons-nous dans k trimestres ?
  rank_sortie  cible SORTIE, conditionnelle a la phase courante : allons-nous
               QUITTER cette phase, et qu'est-ce qui l'annonce ?          <- ici

La difference avec rank_phase est decisive. Les phases sont persistantes : y
etre aujourd'hui predit tres bien qu'on y sera demain, si bien qu'un modele qui
recopie l'etat courant obtient deja un bon score. Predire la SORTIE est une
question plus dure et plus utile.

Specification
-------------
Pour chaque phase P, l'echantillon se limite aux trimestres passes en P. Pour
chaque horizon k :

    P(sortie dans les k prochains trimestres | encore en P en t) = Phi(a + b x_t)

C'est un risque cumule a temps discret. Chaque trimestre de la phase est une
observation, pas seulement les trimestres de sortie — ce qui donne un
echantillon exploitable la ou compter les seules sorties n'en laisserait que
9 a 14.

Le signe de b se lit : NEGATIF = un indicateur eleve eloigne la sortie ;
POSITIF = il la rapproche.

Distinguer la destination
-------------------------
L'argument `direction` restreint l'evenement a une destination precise. C'est
souvent indispensable : un meme indicateur peut agir en sens OPPOSES selon la
direction de sortie. Mesure sur l'ISM en Ralentissement — rapport de risque
0,93 vers le Decrochage, 2,26 vers l'Explosion. Le risque agrege les moyenne a
zero, et l'indicateur parait sans effet alors qu'il informe fortement.

Precautions
-----------
BOOTSTRAP PAR EPISODE. Les trimestres d'un meme episode partagent le meme
destin : ils ne comptent pas pour des observations independantes. On
reechantillonne des episodes entiers.

STATISTIQUE DU MAXIMUM. On retient le meilleur des k horizons ; la p-value
porte donc sur max_k |t(k)|, dont la loi est simulee.

EFFECTIFS. Avec 9 a 14 sorties par phase, la puissance est faible. La colonne
`n_sorties` doit etre lue AVANT toute p-value : sous 8 evenements, aucune
conclusion n'est defendable.

Dependances : numpy, pandas, scipy, statsmodels
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def _episodes(ph: pd.Series) -> pd.Series:
    return (ph != ph.shift()).cumsum()


def construire_sortie(phases: pd.Series, phase_cible: str, horizon: int,
                      direction: str | None = None) -> pd.DataFrame:
    """Panel des trimestres passes en `phase_cible`, avec l'evenement de sortie.

    sortie = 1 si l'on a quitte la phase dans les `horizon` trimestres suivants.
    Si `direction` est precise, seules les sorties vers cette destination
    comptent ; les autres sont ECARTEES (et non codees 0), pour ne pas melanger
    « pas de sortie » et « sortie ailleurs ».
    """
    ep = _episodes(phases)
    dernier = ep.max()
    lignes = []
    for e in sorted(ep.unique()):
        idx = phases.index[ep == e]
        if phases[idx[0]] != phase_cible:
            continue
        censure = e == dernier
        dest = None if censure else phases[phases.index[ep == e + 1][0]]
        fin = idx[-1]
        for t in idx:
            reste = (fin - t).n            # trimestres avant la fin de l'episode
            if censure and reste < horizon:
                continue                   # on ignore : issue inconnue
            sort = int(reste < horizon)
            if direction is not None and sort == 1 and dest != direction:
                continue                   # sortie vers ailleurs : hors champ
            lignes.append(dict(date=t, episode=int(e), sortie=sort,
                               anciennete=(t - idx[0]).n + 1))
    return pd.DataFrame(lignes).set_index("date")


def _fit(x: np.ndarray, y: np.ndarray):
    import statsmodels.api as sm
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 25 or y.sum() < 4 or y.sum() == len(y) or np.std(x) == 0:
        return None
    X = sm.add_constant(x)
    try:
        r = sm.Probit(y, X).fit(disp=0, maxiter=200)
        return dict(coef=float(r.params[1]), t=float(r.tvalues[1]),
                    r2=float(r.prsquared), n=len(x), ev=int(y.sum()))
    except Exception:
        return None


def avance_sortie(x: pd.Series, phases: pd.Series, phase_cible: str,
                  horizons=range(1, 9), direction: str | None = None,
                  n_boot: int = 300, seed: int = 0) -> dict | None:
    """Avance d'une variable sur la sortie d'une phase donnee."""
    ks = list(horizons)
    fits, panels = {}, {}
    for k in ks:
        d = construire_sortie(phases, phase_cible, k, direction)
        if len(d) == 0:
            continue
        d["x"] = x.reindex(d.index)
        d = d.dropna(subset=["x"])
        r = _fit(d["x"].to_numpy(float), d["sortie"].to_numpy(float))
        if r:
            fits[k], panels[k] = r, d
    if len(fits) < 2:
        return None

    kk = sorted(fits)
    tobs = np.array([fits[k]["t"] for k in kk])
    i_star = int(np.argmax(np.abs(tobs)))
    k_star = kk[i_star]

    rng = np.random.default_rng(seed)
    eps = np.unique(np.concatenate([panels[k]["episode"].to_numpy() for k in kk]))
    rows = {k: {e: np.where(panels[k]["episode"].to_numpy() == e)[0] for e in eps}
            for k in kk}

    cnt = {k: 0 for k in kk}
    maxdev, ok = [], 0
    for _ in range(n_boot):
        pick = rng.choice(eps, len(eps), replace=True)
        tb = np.full(len(kk), np.nan)
        for i, k in enumerate(kk):
            sel = np.concatenate([rows[k][e] for e in pick if len(rows[k][e])])
            if len(sel) < 25:
                continue
            d = panels[k].iloc[sel]
            r = _fit(d["x"].to_numpy(float), d["sortie"].to_numpy(float))
            if r:
                tb[i] = r["t"]
        if np.all(np.isnan(tb)):
            continue
        cnt[kk[int(np.nanargmax(np.abs(tb)))]] += 1
        dev = np.abs(tb - tobs)          # statistique pivotale, recentree
        if np.any(np.isfinite(dev)):
            maxdev.append(np.nanmax(dev))
            ok += 1
    tot = max(sum(cnt.values()), 1)
    prob = {k: cnt[k] / tot for k in kk}
    p_max = ((np.sum(np.array(maxdev) >= abs(tobs[i_star])) + 1) / (ok + 1)
             if ok else np.nan)

    v = np.array([prob[k] for k in kk])
    best = (kk[0], kk[-1], len(kk) + 1)
    for a in range(len(kk)):
        s = 0.0
        for b in range(a, len(kk)):
            s += v[b]
            if s >= 0.90:
                if (b - a) < best[2]:
                    best = (kk[a], kk[b], b - a)
                break

    f = fits[k_star]
    return dict(avance=k_star, coef=round(f["coef"], 4), t=round(f["t"], 2),
                sens=("rapproche la sortie" if f["coef"] > 0
                      else "eloigne la sortie"),
                pseudo_R2=round(f["r2"], 3), prob_avance=round(prob[k_star], 3),
                ipd_bas=best[0], ipd_haut=best[1],
                p_max=round(float(p_max), 4) if np.isfinite(p_max) else np.nan,
                n=f["n"], n_sorties=f["ev"])


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


def rank_sortie(phases: pd.Series, panel: pd.DataFrame, phase_cible: str,
                horizons=range(1, 9), direction: str | None = None,
                n_boot: int = 300, min_obs: int = 80,
                alpha: float = 0.10) -> pd.DataFrame:
    """Classe les indicateurs annoncant la sortie d'une phase donnee.

    Parameters
    ----------
    phases : Series des phases, indexee par periode
    phase_cible : "Ralentissement", "Explosion", "Reprise", "Decrochage"
    direction : destination a isoler, ou None pour toute sortie

    Lecture, dans cet ordre : n_sorties d'abord (sous 8, rien n'est
    defendable), puis p_max, puis le SENS, puis l'intervalle ipd.
    """
    lignes = []
    for nom in panel.columns:
        s = panel[nom]
        if s.notna().sum() < min_obs:
            continue
        try:
            r = avance_sortie(s, phases, phase_cible, horizons, direction, n_boot)
        except Exception:
            r = None
        if r:
            lignes.append(dict(variable=nom, **r))
    if not lignes:
        return pd.DataFrame()
    out = pd.DataFrame(lignes)
    out["p_adj"] = (out.p_max * out.variable.nunique()).clip(upper=1.0)
    out["FDR"] = benjamini_hochberg(out.p_max.to_numpy(), alpha)
    return out.sort_values("p_max").reset_index(drop=True)


def tableau_par_phase(phases: pd.Series, panel: pd.DataFrame,
                      horizons=range(1, 9), n_boot: int = 200,
                      top: int = 4) -> pd.DataFrame:
    """Un tableau recapitulatif : que surveiller dans chaque phase."""
    out = []
    for ph in sorted(phases.dropna().unique()):
        R = rank_sortie(phases, panel, ph, horizons, None, n_boot)
        if len(R) == 0:
            continue
        R.insert(0, "phase_courante", ph)
        out.append(R.head(top))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
