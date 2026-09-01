"""local_projection.py — Reponses dynamiques a un choc identifie (Jorda 2005).

Ce que ce module fait, et ce qu'il exige
-----------------------------------------
Il estime la reponse d'une variable y a un choc s, horizon par horizon :

    y_{t+h} - y_{t-1} = alpha_h + beta_h * s_t + gamma_h' Z_t + e_{t+h}

Le coefficient beta_h EST la reponse a l'horizon h. On empile les beta_h pour
obtenir la fonction de reponse. Une regression par horizon, rien de plus.

MAIS le resultat n'est causal que si s_t est un VRAI choc : une variation de la
politique monetaire qui n'est explicable ni par l'etat de l'economie, ni par ce
que la banque centrale anticipait. Sans cela, beta_h melange l'effet du choc et
la reaction de la banque centrale a la conjoncture — ce qui est le probleme
qu'on cherche precisement a eviter.

Illustration du danger : sur un VAR a deux variables PIB/chomage, changer le
seul ordre suppose des variables fait passer la reponse du PIB a l'impact de
0,00 a -1,15. Memes donnees, meme modele, resultat oppose. L'identification
n'est pas un detail technique, c'est tout le probleme.

SERIES DE CHOCS UTILISABLES (a telecharger)
--------------------------------------------
  Romer & Romer (2004)      chocs narratifs construits sur les minutes du FOMC,
                            1969-1996, prolonges a 2007 par Wieland-Yang.
  Bauer & Swanson (2023)    surprises haute frequence dans la fenetre
                            d'annonce du FOMC, 1988-2023.
  Jarocinski & Karadi (2020) separe le choc de politique du choc
                            d'information — utile car une hausse de taux
                            « parce que l'economie va bien » n'est pas un
                            resserrement restrictif.
  Nakamura & Steinsson (2018), Gertler & Karadi (2015)

Le module ne verifie pas que votre serie est un choc valide : il applique la
methode a ce que vous lui donnez. La responsabilite de l'identification vous
revient.

POURQUOI PROJECTIONS LOCALES PLUTOT QUE VAR
--------------------------------------------
Le VAR impose une dynamique commune a tous les horizons ; une erreur de
specification a un pas se propage et s'amplifie a mesure que l'horizon
s'allonge. La projection locale estime chaque horizon separement : elle est
robuste a une mauvaise specification, au prix d'une perte d'efficacite si le
VAR est correct (Jorda 2005, Ramey 2016). Sur des echantillons macro courts,
l'arbitrage penche pour la projection locale.

Elle se prete en outre naturellement aux reponses DEPENDANTES DE L'ETAT
(Ramey-Zubairy 2018) : il suffit d'interagir le choc avec une indicatrice de
regime. C'est le lien direct avec le decoupage en phases.

INFERENCE
---------
Erreurs-types de Newey-West avec fenetre m >= h. A l'horizon h les observations
se chevauchent par construction — y_{t+h} et y_{t+1+h} partagent h-1 periodes —
donc les residus suivent un MA(h-1). Sans cette correction les erreurs-types
sont trop optimistes d'un facteur 2 a 3.

Dependances : numpy, pandas, scipy
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def _hac(X: np.ndarray, y: np.ndarray, m: int):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for j in range(1, m + 1):
        if j >= len(u):
            break
        w = 1.0 - j / (m + 1.0)
        A = Xu[j:].T @ Xu[:-j]
        S = S + w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.maximum(np.diag(V), 0.0)), u


class LocalProjection:
    """Reponse de y a un choc s, par projections locales.

    Parameters
    ----------
    p_lags : int
        Retards de controle de y et du choc. Ils absorbent la dynamique propre
        de la variable et l'autocorrelation eventuelle du choc.
    cumul : bool
        True  : reponse du NIVEAU cumule, y_{t+h} - y_{t-1}. A utiliser quand y
                est une variation (croissance, inflation) et qu'on veut l'effet
                cumule.
        False : reponse de y_{t+h} directement. Pour y deja en niveau.
    """

    def __init__(self, p_lags: int = 4, cumul: bool = True):
        self.p_lags = p_lags
        self.cumul = cumul
        self.resultats_: pd.DataFrame | None = None

    # -- construction ------------------------------------------------------

    def _design(self, y, s, Z, h, etat=None):
        idx = y.index
        d = pd.DataFrame({"y0": y, "s": s.reindex(idx)})
        d["cible"] = y.shift(-h) - y.shift(1) if self.cumul else y.shift(-h)
        for j in range(1, self.p_lags + 1):
            d[f"y_l{j}"] = y.shift(j)
            d[f"s_l{j}"] = s.reindex(idx).shift(j)
        if Z is not None:
            for c in Z.columns:
                d[c] = Z[c].reindex(idx)
        if etat is not None:
            d["etat"] = etat.reindex(idx).astype(float)
        d = d.dropna()
        if len(d) < 30:
            return None

        cols = [c for c in d.columns if c not in ("cible", "s", "etat")]
        blocs = [np.ones(len(d))] + [d[c].to_numpy() for c in cols]
        noms = ["const"] + cols
        if etat is None:
            blocs.append(d["s"].to_numpy())
            noms.append("choc")
        else:
            # Reponse dependante de l'etat : un coefficient par regime, sans
            # terme de choc commun — sinon les deux ne sont pas identifiables.
            blocs += [d["etat"].to_numpy() * d["s"].to_numpy(),
                      (1 - d["etat"].to_numpy()) * d["s"].to_numpy(),
                      d["etat"].to_numpy()]
            noms += ["choc_etat1", "choc_etat0", "etat"]
        return dict(X=np.column_stack(blocs), y=d["cible"].to_numpy(),
                    noms=noms, n=len(d))

    # -- estimation --------------------------------------------------------

    def fit(self, y: pd.Series, choc: pd.Series, controls: pd.DataFrame | None = None,
            horizons=range(0, 17), etat: pd.Series | None = None,
            niveau: float = 0.90):
        """Estime la reponse a chaque horizon.

        etat : indicatrice 0/1 optionnelle. Si fournie, deux reponses sont
            estimees — une par regime — ce qui repond a « l'effet d'un
            resserrement est-il le meme en expansion et en ralentissement ? ».
        """
        z = stats.norm.ppf(0.5 + niveau / 2)
        lignes = []
        for h in horizons:
            d = self._design(y, choc, controls, h, etat)
            if d is None:
                continue
            m = max(h, int(np.floor(4 * (d["n"] / 100) ** (2 / 9))))
            b, se, u = _hac(d["X"], d["y"], m)
            r = dict(horizon=h, n=d["n"], m_hac=m)
            for nom in (["choc"] if etat is None else ["choc_etat1", "choc_etat0"]):
                j = d["noms"].index(nom)
                r[nom] = float(b[j])
                r[nom + "_se"] = float(se[j])
                r[nom + "_bas"] = float(b[j] - z * se[j])
                r[nom + "_haut"] = float(b[j] + z * se[j])
                r[nom + "_p"] = float(2 * (1 - stats.norm.cdf(abs(b[j] / se[j]))))
            if etat is not None:
                j1, j0 = d["noms"].index("choc_etat1"), d["noms"].index("choc_etat0")
                diff = b[j1] - b[j0]
                sed = np.sqrt(se[j1] ** 2 + se[j0] ** 2)   # borne haute : ignore la covariance
                r["ecart"] = float(diff)
                r["ecart_p"] = float(2 * (1 - stats.norm.cdf(abs(diff / sed))))
            lignes.append(r)
        self.resultats_ = pd.DataFrame(lignes)
        self.etat_ = etat is not None
        self.niveau_ = niveau
        return self

    # -- restitution -------------------------------------------------------

    def summary(self) -> str:
        if self.resultats_ is None:
            raise RuntimeError("Modele non ajuste.")
        r = self.resultats_
        pc = int(100 * self.niveau_)
        if not self.etat_:
            L = [f"{'h':>3}{'reponse':>10}{'se':>8}{'IC' + str(pc) + ' bas':>11}"
                 f"{'haut':>9}{'p':>8}{'n':>6}"]
            for _, q in r.iterrows():
                et = " *" if q.choc_p < 0.10 else ""
                L.append(f"{int(q.horizon):>3}{q.choc:>+10.3f}{q.choc_se:>8.3f}"
                         f"{q.choc_bas:>+11.3f}{q.choc_haut:>+9.3f}"
                         f"{q.choc_p:>8.3f}{int(q.n):>6}{et}")
        else:
            L = [f"{'h':>3}{'etat=1':>10}{'etat=0':>10}{'ecart':>10}{'p(ecart)':>10}{'n':>6}"]
            for _, q in r.iterrows():
                et = " *" if q.ecart_p < 0.10 else ""
                L.append(f"{int(q.horizon):>3}{q.choc_etat1:>+10.3f}"
                         f"{q.choc_etat0:>+10.3f}{q.ecart:>+10.3f}"
                         f"{q.ecart_p:>10.3f}{int(q.n):>6}{et}")
        pic = r.loc[r[("choc" if not self.etat_ else "choc_etat1")].abs().idxmax()]
        L += ["", f"pic de reponse a h={int(pic.horizon)}",
              "* = significatif a 10 %", "",
              "Rappel : causal SEULEMENT si le choc est correctement identifie."]
        return "\n".join(L)


def taylor_residu(taux: pd.Series, inflation: pd.Series, activite: pd.Series,
                  p: int = 4) -> pd.Series:
    """Residu d'une regle de Taylor — PLACEHOLDER, PAS un choc identifie.

    Sert uniquement a tester la mecanique du module en attendant une vraie
    serie. Ce residu contient encore tout ce que la banque centrale observe et
    que ces trois variables ne captent pas : anticipations, conditions
    financieres, jugement. Ne publiez aucun resultat fonde dessus.
    """
    d = pd.concat([taux.rename("i"), inflation.rename("pi"),
                   activite.rename("x")], axis=1)
    for j in range(1, p + 1):
        d[f"i_l{j}"] = taux.shift(j)
    d = d.dropna()
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy()
                                             for c in d.columns if c != "i"])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ d["i"].to_numpy())
    warnings.warn("taylor_residu n'est PAS un choc identifie : test uniquement.")
    return pd.Series(d["i"].to_numpy() - X @ b, index=d.index, name="residu_taylor")


def charger_chocs(chemin: str, colonne: str | None = None,
                  col_date: str = None, freq_cible: str = "Q") -> pd.Series:
    """Charge une serie de chocs (CSV ou Excel) et l'agrege a la frequence voulue.

    AGREGATION PAR SOMME, jamais par moyenne ni derniere valeur. Un choc est une
    innovation : trois resserrements de 0,1 point dans un trimestre font 0,3, pas
    0,1. Prendre la moyenne diviserait l'amplitude par le nombre de reunions, et
    prendre la derniere valeur jetterait les autres.

    Les periodes sans reunion sont a zero, pas manquantes : une absence de choc
    est une information, pas un trou. La reindexation le fait explicitement.

    Formats acceptes : mensuel (Romer-Romer, Jarocinski-Karadi) ou par evenement
    (Bauer-Swanson, une ligne par reunion du FOMC).
    """
    d = (pd.read_excel(chemin) if str(chemin).endswith((".xlsx", ".xls"))
         else pd.read_csv(chemin))
    col_date = col_date or d.columns[0]
    if colonne is None:
        num = [c for c in d.columns if c != col_date
               and pd.api.types.is_numeric_dtype(d[c])]
        if len(num) != 1:
            raise ValueError(f"Preciser `colonne`. Candidates : {num}")
        colonne = num[0]
    s = pd.Series(pd.to_numeric(d[colonne], errors="coerce").to_numpy(),
                  index=pd.to_datetime(d[col_date])).dropna()
    per = s.groupby(pd.PeriodIndex(s.index, freq=freq_cible)).sum()
    grille = pd.period_range(per.index.min(), per.index.max(), freq=freq_cible)
    out = per.reindex(grille, fill_value=0.0)
    out.name = colonne
    n0 = int((out == 0).sum())
    if n0 > 0.5 * len(out):
        warnings.warn(f"{n0} periodes sur {len(out)} a zero : verifiez que la "
                      "serie couvre bien toute la plage.")
    return out
