"""sensitivity.py — Effet predictif PARTIEL d'une variable, controles inclus.

Question posee
--------------
« Quelle information x_t apporte-t-il sur y_{t+h}, une fois retiree la
persistance de y et l'information deja contenue dans les autres predicteurs ? »

Modele
------
    y_{t+h} = alpha + gamma y_t + beta x_t + delta' Z_t + e_{t+h}

    gamma y_t   banc d'essai autoregressif : ce que la cible dit deja d'elle-meme
    Z_t         controles — les autres variables predictives
    beta        CE QU'ON CHERCHE : l'apport SINGULIER de x

AVERTISSEMENT CENTRAL — ceci n'est PAS un effet causal
-------------------------------------------------------
beta repond a « quelle information x ajoute-t-il ? », pas a « que se passe-t-il
si x bouge de 1 point ? ». Les deux coincident seulement sous des hypotheses
d'identification (exogeneite conditionnelle) qui ne tiennent pas ici : x et y
sont co-determines par le meme cycle, et les chocs qui font bouger x font aussi
bouger y par d'autres canaux.

Formulation correcte :
    « un ecart-type de plus sur x est associe a beta unites de y, h periodes
      plus tard, au-dela de ce que la persistance et les controles expliquent »
Formulation incorrecte :
    « si x baisse de 1 point, y reagira de beta »

Pour une lecture causale il faudrait un choc identifie (VAR structurel,
instrument, experience naturelle). Ce module ne fournit rien de tel.

LE PIEGE PRINCIPAL : la colinearite
------------------------------------
Les variables macro se ressemblent. Si les controles expliquent deja 90 % de la
variance de x, seuls 10 % restent pour identifier beta — et ces 10 % sont
souvent du bruit de mesure plutot que du signal economique. Le coefficient
devient alors instable, change de signe entre sous-echantillons, et son
erreur-type explose.

Le module rapporte donc SYSTEMATIQUEMENT :
    R2_x_sur_Z   part de x expliquee par les controles
    VIF          facteur d'inflation de la variance = 1/(1 - R2_x_sur_Z)
    attenuation  ecart entre coefficient brut et coefficient controle

Regle de lecture : au-dela de VIF = 10 (soit R2_x_sur_Z > 0.90), beta n'est plus
interpretable, quelle que soit sa p-value.

CAS PARTICULIER A CONNAITRE — les relations definitionnelles
-------------------------------------------------------------
Chomage et PIB sont lies par la loi d'Okun, qui est une regularite comptable
autant qu'economique. Y ajouter en controle la production industrielle et
l'emploi revient a retirer exactement la variance qui porte la relation. Le
beta resultant mesure alors le residu, c'est-a-dire pas grand-chose.
Choisissez des controles qui ne sont pas des mesures alternatives de x.

Inference
---------
Erreurs-types de Newey-West avec fenetre m >= h : a l'horizon h les
observations SE CHEVAUCHENT (y_{t+h} et y_{t+1+h} partagent h-1 periodes), donc
les residus suivent mecaniquement un MA(h-1). Sans cette correction les
erreurs-types sont trop optimistes d'un facteur 2 a 3.

Dependances : numpy, pandas, scipy, statsmodels
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Briques
# ---------------------------------------------------------------------------

def _hac(X: np.ndarray, y: np.ndarray, m: int):
    """OLS + covariance de Newey-West (noyau de Bartlett)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    n = len(u)
    for j in range(1, m + 1):
        if j >= n:
            break
        w = 1.0 - j / (m + 1.0)
        A = Xu[j:].T @ Xu[:-j]
        S = S + w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    r2 = 1.0 - u.var() / y.var() if y.var() > 0 else np.nan
    return b, se, u, r2


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


# ---------------------------------------------------------------------------
# Modele
# ---------------------------------------------------------------------------

class SensitivityModel:
    """Effet predictif partiel de x sur y, a plusieurs horizons.

    Parameters
    ----------
    standardiser : bool
        Standardise x et les controles. beta se lit alors « effet en unites de
        y d'un ecart-type de x », ce qui rend les variables comparables.
    controler_y : bool
        Inclut y_t. Sans lui, beta capte la simple procyclicite de x.
    vif_max : float
        Au-dela, le coefficient est marque comme non interpretable.
    """

    def __init__(self, standardiser: bool = True, controler_y: bool = True,
                 vif_max: float = 10.0):
        self.standardiser = standardiser
        self.controler_y = controler_y
        self.vif_max = vif_max
        self.resultats_: pd.DataFrame | None = None
        self.detail_: dict = {}

    # -- construction ------------------------------------------------------

    def _design(self, y: pd.Series, x: pd.Series, Z: pd.DataFrame | None, h: int):
        idx = y.index
        d = pd.DataFrame({"y0": y, "yh": y.shift(-h), "x": x.reindex(idx)})
        noms_z = []
        if Z is not None:
            for c in Z.columns:
                if c == x.name:
                    continue
                d[c] = Z[c].reindex(idx)
                noms_z.append(c)
        d = d.dropna()
        if len(d) < 30 + len(noms_z):
            return None
        xs = _z(d["x"]) if self.standardiser else d["x"]
        cols, noms = [np.ones(len(d))], ["const"]
        if self.controler_y:
            cols.append(d["y0"].to_numpy())
            noms.append("y_t")
        cols.append(xs.to_numpy())
        noms.append("x")
        for c in noms_z:
            v = _z(d[c]) if self.standardiser else d[c]
            cols.append(v.to_numpy())
            noms.append(c)
        return dict(X=np.column_stack(cols), y=d["yh"].to_numpy(),
                    noms=noms, n=len(d), noms_z=noms_z, index=d.index,
                    x_brut=xs.to_numpy())

    # -- estimation --------------------------------------------------------

    def fit(self, y: pd.Series, x: pd.Series, controls: pd.DataFrame | None = None,
            horizons=range(1, 9)):
        """Estime l'effet partiel de x sur y a chaque horizon."""
        if x.name is None:
            x = x.rename("x")
        lignes = []
        for h in horizons:
            d = self._design(y, x, controls, h)
            if d is None:
                continue
            m = max(h, int(np.floor(4 * (d["n"] / 100) ** (2 / 9))))
            b, se, u, r2 = _hac(d["X"], d["y"], m)
            j = d["noms"].index("x")

            # Modele SANS controles, pour mesurer l'attenuation.
            d0 = self._design(y, x, None, h)
            b0, se0, _, r20 = _hac(d0["X"], d0["y"], m)
            j0 = d0["noms"].index("x")

            # Colinearite : part de x expliquee par les autres regresseurs.
            autres = [k for k in range(d["X"].shape[1]) if k != j]
            if len(autres) > 1:
                bb = np.linalg.pinv(d["X"][:, autres]) @ d["x_brut"]
                res = d["x_brut"] - d["X"][:, autres] @ bb
                r2_xz = 1.0 - res.var() / d["x_brut"].var()
            else:
                r2_xz = 0.0
            vif = 1.0 / max(1.0 - r2_xz, 1e-6)

            t = b[j] / se[j] if se[j] > 0 else np.nan
            lignes.append(dict(
                horizon=h, n=d["n"],
                beta=float(b[j]), se=float(se[j]), t=float(t),
                p=float(2 * (1 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                beta_brut=float(b0[j0]), t_brut=float(b0[j0] / se0[j0]),
                attenuation=float(1 - b[j] / b0[j0]) if b0[j0] != 0 else np.nan,
                R2=float(r2), R2_sans_x=np.nan,
                R2_x_sur_Z=float(r2_xz), VIF=float(vif),
                interpretable=bool(vif <= self.vif_max),
                m_hac=m))
            self.detail_[h] = dict(coefs=dict(zip(d["noms"], np.round(b, 4))),
                                   se=dict(zip(d["noms"], np.round(se, 4))),
                                   noms_z=d["noms_z"])

        # Apport incremental : R2 du modele complet vs modele sans x.
        for i, r in enumerate(lignes):
            h = r["horizon"]
            d = self._design(y, x, controls, h)
            j = d["noms"].index("x")
            X_sans = np.delete(d["X"], j, axis=1)
            _, _, _, r2s = _hac(X_sans, d["y"], r["m_hac"])
            lignes[i]["R2_sans_x"] = float(r2s)
            lignes[i]["gain_R2"] = float(r["R2"] - r2s)

        self.resultats_ = pd.DataFrame(lignes)
        self.x_nom_ = x.name
        self.y_nom_ = y.name or "y"
        return self

    # -- restitution -------------------------------------------------------

    def summary(self, seuil_p: float = 0.10) -> str:
        if self.resultats_ is None:
            raise RuntimeError("Modele non ajuste.")
        r = self.resultats_
        L = ["=" * 74,
             f"SENSIBILITE : {self.x_nom_}  ->  {self.y_nom_}",
             "=" * 74,
             "beta = effet sur y d'un ecart-type de x, h periodes plus tard,",
             "       NET de la persistance de y et des controles.",
             "",
             f"{'h':>3}{'n':>6}{'beta':>9}{'se':>8}{'t':>7}{'p':>8}"
             f"{'gain R2':>9}{'VIF':>7}  statut"]
        for _, q in r.iterrows():
            st = "" if q.interpretable else "  COLINEAIRE"
            if q.interpretable and q.p < seuil_p:
                st = "  *"
            L.append(f"{int(q.horizon):>3}{int(q.n):>6}{q.beta:>+9.3f}{q.se:>8.3f}"
                     f"{q.t:>+7.2f}{q.p:>8.3f}{q.gain_R2:>9.3f}{q.VIF:>7.1f}{st}")
        best = r[r.interpretable]
        if len(best):
            b = best.loc[best.t.abs().idxmax()]
            L += ["", f"Horizon le plus net : h={int(b.horizon)}  "
                      f"beta={b.beta:+.3f} (p={b.p:.3f})",
                  f"  coefficient sans controles : {b.beta_brut:+.3f}  "
                  f"-> attenuation {100*b.attenuation:.0f} %",
                  f"  part de x expliquee par les controles : "
                  f"{100*b.R2_x_sur_Z:.0f} %"]
        if (~r.interpretable).any():
            L += ["", "ATTENTION : VIF > seuil a certains horizons. Les controles "
                      "expliquent",
                  "l'essentiel de x ; beta y mesure un residu, pas un effet."]
        L += ["", "Rappel : effet PREDICTIF PARTIEL, pas causal. Voir l'en-tete "
                  "du module."]
        return "\n".join(L)

    def contributions(self, h: int) -> pd.DataFrame:
        """Coefficients de tous les regresseurs a un horizon donne."""
        if h not in self.detail_:
            raise KeyError(f"Horizon {h} non estime. Disponibles : {list(self.detail_)}")
        d = self.detail_[h]
        t = pd.DataFrame({"coef": pd.Series(d["coefs"]), "se": pd.Series(d["se"])})
        t["t"] = (t.coef / t.se.replace(0, np.nan)).round(2)
        return t


# ---------------------------------------------------------------------------
# Balayage
# ---------------------------------------------------------------------------

def scan_sensibilite(y: pd.Series, candidats: pd.DataFrame,
                     controls: pd.DataFrame | None = None,
                     horizons=range(1, 9), vif_max: float = 10.0,
                     controls_communs: bool = True) -> pd.DataFrame:
    """Applique le modele a plusieurs variables, une par une.

    controls_communs : si True, le meme jeu de controles sert pour toutes les
        variables testees (la variable testee est automatiquement retiree des
        controles). Si False, aucun controle — utile pour comparer l'effet brut.

    ATTENTION aux tests multiples : tester N variables sur H horizons cree N*H
    combinaisons. La colonne p_bonferroni multiplie par N.
    """
    lignes = []
    for c in candidats.columns:
        Z = None
        if controls is not None and controls_communs:
            Z = controls.drop(columns=[c], errors="ignore")
        try:
            m = SensitivityModel(vif_max=vif_max).fit(
                y, candidats[c].rename(c), Z, horizons)
        except Exception:
            continue
        r = m.resultats_
        r = r[r.interpretable]
        if len(r) == 0:
            lignes.append(dict(variable=c, horizon=np.nan, beta=np.nan,
                               statut="colineaire a tous les horizons"))
            continue
        b = r.loc[r.t.abs().idxmax()]
        lignes.append(dict(variable=c, horizon=int(b.horizon), beta=round(b.beta, 3),
                           t=round(b.t, 2), p=round(b.p, 4),
                           gain_R2=round(b.gain_R2, 3),
                           beta_brut=round(b.beta_brut, 3),
                           attenuation=round(b.attenuation, 2),
                           VIF=round(b.VIF, 1), n=int(b.n), statut="ok"))
    out = pd.DataFrame(lignes)
    if "p" in out.columns:
        nv = out.variable.nunique()
        out["p_bonferroni"] = (out["p"] * nv).clip(upper=1.0)
        out = out.sort_values("p", na_position="last")
    return out.reset_index(drop=True)
