"""cycle_model.py — Modele de cycle economique, interface orientee objet.

Consolide prevision.py et transitions.py en une seule classe. Le passage a
l'objet n'est pas cosmetique : l'etat estime (coefficients de hasard, matrice
de transition, table de recalibration) doit rester solidaire du modele et etre
reutilise a chaque prevision. Le faire circuler par dictionnaires exposait a
melanger le hasard d'un ajustement avec la matrice d'un autre.

Usage
-----
    from cycle_model import CycleModel

    m = CycleModel().fit(phases["phase"])
    print(m.summary())
    print(m.predict(H=6))
    print(m.explain(H=4))

    # avec covariable
    m = CycleModel().fit(phases["phase"], exog=pd.DataFrame({"dgno": serie}))
    print(m.predict(H=4, x={"dgno": 1.2}))

    # recalibration sur backtest
    bt = m.backtest(phases["phase"], debut="1990Q1", H=4)
    m.calibrate(bt)
    print(m.predict(H=4))          # p_change devient p_change_cal

Ce que le modele fait, et ce qu'il ne fait pas
-----------------------------------------------
IL FAIT     estimer P(changer de phase dans H trimestres), et la repartition
            entre destinations si changement.
IL NE FAIT  PAS predire de facon fiable QUELLE phase on observera. Sur backtest,
            l'etiquette predite ne bat pas un predicteur trivial « rien ne
            change ». L'information exploitable est dans la PROBABILITE
            continue, pas dans le label — d'ou l'accent mis sur `p_change` et
            sur la recalibration.

Dependances : numpy, pandas, scipy, statsmodels
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fonctions utilitaires (publiques : utiles hors du modele)
# ---------------------------------------------------------------------------

def episodes(phases: pd.Series) -> pd.Series:
    """Numerote les blocs consecutifs de meme phase."""
    return (phases != phases.shift()).cumsum()


def anciennete(phases: pd.Series) -> pd.Series:
    """Trimestres deja passes dans la phase courante (1 = premier)."""
    ep = episodes(phases)
    return ep.groupby(ep).cumcount() + 1



def _assurer_period_index(s, nom="phases"):
    """Convertit l'index en PeriodIndex, ou explique precisement quoi faire.

    Un read_csv laisse un Index de chaines, sans attribut freqstr ; un
    DatetimeIndex en a un, mais il vaut None si la frequence n'est pas posee.
    Les deux cassent toute comparaison de dates en aval, d'ou cette conversion
    systematique a l'entree plutot qu'un plantage sur un attribut manquant.
    """
    s = s.copy()
    idx = s.index
    if isinstance(idx, pd.PeriodIndex):
        return s
    if isinstance(idx, pd.DatetimeIndex):
        f = idx.freqstr or pd.infer_freq(idx)
        # infer_freq renvoie des alias de DateOffset ('MS', 'QS-OCT', 'YE-DEC')
        # que to_period refuse : on ne garde que la lettre de base.
        if f is not None:
            f = {"M": "M", "Q": "Q", "A": "A", "Y": "A", "W": "W", "D": "D"}.get(
                f.split("-")[0].rstrip("SE") or f[0], f.split("-")[0])
        if f is None:
            raise ValueError(
                f"L'index de `{nom}` est un DatetimeIndex sans frequence "
                "identifiable. Convertissez-le explicitement :\n"
                f"    {nom}.index = pd.PeriodIndex({nom}.index, freq='M')  # ou 'Q'")
        return s.set_axis(idx.to_period(f))
    try:
        conv = pd.PeriodIndex(idx)
        return s.set_axis(conv)
    except Exception:
        pass
    # Chaines de dates ('1949-10', '2024Q2') : on passe par to_datetime puis on
    # infere la frequence sur l'ecart median entre observations. Plus robuste
    # que pd.infer_freq, qui echoue des qu'une date manque.
    try:
        dt = pd.to_datetime(pd.Index(idx).astype(str), errors="raise")
        jours = pd.Series(dt).diff().dt.days.median()
        f = "M" if jours < 45 else ("Q" if jours < 120 else "A")
        return s.set_axis(dt.to_period(f))
    except Exception:
        raise TypeError(
            f"L'index de `{nom}` est un {type(idx).__name__}, pas un "
            "PeriodIndex. Apres un read_csv, l'index reste une simple chaine ; "
            "convertissez-le :\n"
            f"    {nom}.index = pd.PeriodIndex({nom}.index, freq='M')  # ou 'Q'")
    return s.set_axis(conv)


class CycleModel:
    """Modele de duree + matrice de transition pour un decoupage en phases.

    Parameters
    ----------
    exclure : tuple
        Phases ecartees de l'estimation (episodes trop courts pour estimer).
    min_evenements : int
        En dessous, l'estimation d'une phase est signalee comme instable.
    lissage : bool
        Posterieur de Laplace sur la matrice de transition. Une destination
        jamais observee recoit alors une probabilite faible mais NON NULLE :
        l'absence sur onze episodes ne prouve pas l'impossibilite.
    firth : bool
        Vraisemblance penalisee de Firth pour les hasards par destination.
        Indispensable avec 2 a 6 evenements par branche : le maximum de
        vraisemblance ordinaire y diverge par separation quasi complete. La
        penalite de Jeffreys garantit un estimateur fini et retire le biais
        d'ordre 1/n. Sur grand echantillon, elle devient negligeable et l'on
        retrouve le MV ordinaire.
    risques_concurrents : bool
        Estime un hasard DISTINCT par destination, au lieu d'un hasard unique
        « sortir ». Necessaire des lors qu'une covariable agit en sens opposes
        selon la destination : un ISM eleve eloigne le Decrochage et rapproche
        l'Explosion, si bien qu'un hasard agrege les moyenne a zero. Le cout est
        le comptage — 5 et 3 evenements par branche en Ralentissement — d'ou
        l'affichage systematique des effectifs a cote des coefficients.
    """

    def __init__(self, exclure: tuple = ("Choc Covid",),
                 min_evenements: int = 3, lissage: bool = True,
                 risques_concurrents: bool = False, firth: bool = True):
        self.exclure = exclure
        self.min_evenements = min_evenements
        self.lissage = lissage
        self.risques_concurrents = risques_concurrents
        self.firth = firth
        self.hazard_: dict = {}
        self.hazard_dest_: dict = {}
        self.transitions_: pd.DataFrame | None = None
        self.comptes_: pd.DataFrame | None = None
        self.intervalles_: dict = {}
        self.exog_noms_: list = []
        self.phase_courante_: str | None = None
        self.anciennete_: int | None = None
        self.calibration_: pd.DataFrame | None = None
        self.unite_ = "periodes"
        self._ajuste = False

    # -- estimation ---------------------------------------------------------

    def fit(self, phases: pd.Series, exog: pd.DataFrame | None = None):
        """Estime le hasard par phase et la matrice de transition."""
        phases = _assurer_period_index(phases, "phases")
        if exog is not None:
            exog = _assurer_period_index(exog, "exog")
        p = phases[~phases.isin(self.exclure)].dropna()
        if len(p) < 40:
            raise ValueError(f"Serie trop courte : {len(p)} trimestres.")
        self.exog_noms_ = list(exog.columns) if exog is not None else []
        self._fit_transitions(p)
        self._fit_hazard(p, exog)
        if self.risques_concurrents:
            self._fit_hazard_dest(p, exog)
        f = str(getattr(phases.index, "freqstr", "Q"))[:1]
        self.unite_ = {"M": "mois", "Q": "trimestres", "A": "annees"}.get(f, "periodes")
        self.phase_courante_ = phases.dropna().iloc[-1]
        self.anciennete_ = int(anciennete(phases.dropna()).iloc[-1])
        self._ajuste = True
        return self

    def _fit_transitions(self, p: pd.Series):
        ep = episodes(p)
        suite = [(p[ep == e].iloc[0], p[ep == e + 1].iloc[0])
                 for e in sorted(ep.unique())[:-1] if (ep == e + 1).any()]
        if not suite:
            raise ValueError("Aucune transition observee.")
        dep, arr = sorted({a for a, _ in suite}), sorted({b for _, b in suite})
        C = pd.DataFrame(0, index=dep, columns=arr, dtype=int)
        for a, b in suite:
            C.loc[a, b] += 1
        P = C.astype(float).copy()
        for d in dep:
            k = C.loc[d].to_numpy(float)
            n, K = k.sum(), len(arr)
            P.loc[d] = (k + 1.0) / (n + K) if self.lissage else k / max(n, 1)
            for j, dest in enumerate(arr):
                lo, hi = stats.beta.ppf([0.05, 0.95], k[j] + 1, n - k[j] + K - 1)
                self.intervalles_[(d, dest)] = (round(float(lo), 3), round(float(hi), 3))
        self.comptes_, self.transitions_ = C, P.round(3)

    def _fit_hazard(self, p: pd.Series, exog: pd.DataFrame | None):
        import statsmodels.api as sm
        d = self._panel_survie(p, exog)
        for g in sorted(d.phase.unique()):
            sub = d[d.phase == g]
            sub = sub.dropna(subset=self.exog_noms_) if self.exog_noms_ else sub
            ev = int(sub.sortie.sum())
            if ev < self.min_evenements or len(sub) < 20:
                warnings.warn(f"Phase '{g}' : {ev} sorties, phase ignoree.")
                continue
            X = np.column_stack([np.ones(len(sub)), np.log(sub.t.to_numpy())]
                                + [sub[c].to_numpy() for c in self.exog_noms_])
            try:
                mod = sm.GLM(sub.sortie.to_numpy(), X,
                             family=sm.families.Binomial(
                                 sm.families.links.CLogLog())).fit()
            except Exception as e:
                warnings.warn(f"Phase '{g}' : estimation impossible ({e}).")
                continue
            self.hazard_[g] = dict(
                params=np.asarray(mod.params), se=np.asarray(mod.bse),
                pvalues=np.asarray(mod.pvalues),
                noms=["const", "log_t"] + self.exog_noms_,
                n=len(sub), evenements=ev,
                duree_med=float(sub.groupby("episode").t.max().median()))

    def _fit_hazard_dest(self, p: pd.Series, exog: pd.DataFrame | None):
        """Un hasard par couple (phase de depart, destination)."""
        import statsmodels.api as sm
        d = self._panel_survie(p, exog, avec_dest=True)
        for g in sorted(d.phase.unique()):
            sub = d[d.phase == g]
            sub = sub.dropna(subset=self.exog_noms_) if self.exog_noms_ else sub
            if len(sub) < 20:
                continue
            X = np.column_stack([np.ones(len(sub)), np.log(sub.t.to_numpy())]
                                + [sub[c].to_numpy() for c in self.exog_noms_])
            for dest in sorted(sub.dest.dropna().unique()):
                y = ((sub.sortie == 1) & (sub.dest == dest)).astype(int).to_numpy()
                ev = int(y.sum())
                if ev < 2:
                    continue
                if self.firth:
                    from firth import fit_firth
                    try:
                        f = fit_firth(X, y.astype(float))
                    except Exception:
                        continue
                    par, bse = f["params"], f["se"]
                    pv, pv_lr, sep = f["pvalues"], f["pvalues_lr"], f["separation"]
                else:
                    try:
                        mod = sm.GLM(y, X, family=sm.families.Binomial(
                            sm.families.links.CLogLog())).fit(maxiter=200)
                    except Exception:
                        continue
                    par, bse = np.asarray(mod.params), np.asarray(mod.bse)
                    pv = np.asarray(mod.pvalues)
                    pv_lr, sep = np.full(len(par), np.nan), False
                # Separation quasi complete : avec 2-3 evenements, la
                # vraisemblance n'a pas de maximum interieur et les
                # coefficients divergent. On les marque plutot que de les
                # publier comme des estimations.
                degen = bool(np.max(np.abs(par)) > 10 or np.max(bse) > 50
                             or not np.all(np.isfinite(bse)))
                self.hazard_dest_[(g, dest)] = dict(
                    params=par, se=bse, pvalues=pv, pvalues_lr=pv_lr,
                    separation=sep, methode="Firth" if self.firth else "MV",
                    noms=["const", "log_t"] + self.exog_noms_,
                    n=len(sub), evenements=ev,
                    fiable=(ev >= 8 and not degen), degenere=degen)

    def _panel_survie(self, p: pd.Series, exog: pd.DataFrame | None,
                      avec_dest: bool = False) -> pd.DataFrame:
        """Panel trimestre-risque. Le dernier episode est CENSURE a droite :
        on sait qu'il a dure au moins t trimestres, pas quand il finira.
        Le compter comme une sortie biaiserait les durees vers le bas."""
        ep, anc = episodes(p), anciennete(p)
        dernier = ep.max()
        dest_de = {}
        if avec_dest:
            for e in sorted(ep.unique())[:-1]:
                if (ep == e + 1).any():
                    dest_de[e] = p[ep == e + 1].iloc[0]
        lignes = []
        for dt, e, a, ph in zip(p.index, ep, anc, p):
            fin = ((ep == e).sum() == a) and e != dernier
            r = dict(date=dt, episode=int(e), phase=ph, t=int(a), sortie=int(fin))
            if avec_dest:
                r["dest"] = dest_de.get(int(e)) if fin else None
            lignes.append(r)
        d = pd.DataFrame(lignes).set_index("date")
        if exog is not None:
            for c in exog.columns:
                d[c] = exog[c].reindex(d.index)
        return d

    # -- prevision ----------------------------------------------------------

    def _check(self):
        if not self._ajuste:
            raise RuntimeError("Modele non ajuste : appelez .fit() d'abord.")

    def hazard(self, phase: str, t: int, x: dict | None = None) -> float:
        """Hasard discret h(t) = 1 - exp(-exp(eta)) : probabilite de sortir au
        trimestre t sachant qu'on y est encore."""
        self._check()
        if phase not in self.hazard_:
            raise KeyError(f"Phase '{phase}' non estimee. "
                           f"Disponibles : {list(self.hazard_)}")
        m = self.hazard_[phase]
        eta = m["params"][0] + m["params"][1] * np.log(max(t, 1))
        for k, nom in enumerate(m["noms"][2:], start=2):
            if x is None or nom not in x:
                raise ValueError(f"Valeur manquante pour la covariable '{nom}'.")
            eta += m["params"][k] * x[nom]
        return float(1.0 - np.exp(-np.exp(np.clip(eta, -20, 20))))

    def predict(self, H: int = 6, phase: str | None = None,
                t: int | None = None, x: dict | None = None) -> pd.DataFrame:
        """Prevision a H trimestres.

        Returns
        -------
        DataFrame : une ligne par horizon.
            hasard        taux de sortie du trimestre
            p_change      P(avoir quitte la phase) — CUMULEE
            p_change_cal  version recalibree, si .calibrate() a ete appele
            <phase>       P(etre dans cette phase), par destination
            lecture       traduction en clair du niveau de risque

        Le modele est ABSORBANT : on ne chaine pas les transitions ulterieures.
        Chainer supposerait un processus markovien, ce que la dependance de
        duree contredit — le hasard depend de l'anciennete, pas de la seule phase.
        """
        self._check()
        phase = phase or self.phase_courante_
        t = t if t is not None else self.anciennete_
        if phase not in self.transitions_.index:
            raise KeyError(f"Phase '{phase}' absente de la matrice de transition.")
        dests = list(self.transitions_.columns)

        lignes, surv, cumul = [], 1.0, {d: 0.0 for d in dests}
        for j in range(1, H + 1):
            hz = self.hazard(phase, t + j - 1, x)
            sortie = surv * hz
            surv *= (1.0 - hz)
            for d in dests:
                cumul[d] += sortie * float(self.transitions_.loc[phase, d])
            pc = 1.0 - surv
            ligne = {"horizon": j, "anciennete": t + j - 1,
                     "hasard": round(hz, 3), "p_change": round(pc, 3)}
            if self.calibration_ is not None:
                ligne["p_change_cal"] = round(self._calibrer(pc), 3)
            ligne[f"reste_{phase}"] = round(surv, 3)
            ligne.update({d: round(cumul[d], 3) for d in dests})
            ligne["lecture"] = self.lecture(
                ligne.get("p_change_cal", pc))
            lignes.append(ligne)
        return pd.DataFrame(lignes)

    @staticmethod
    def lecture(p: float) -> str:
        """Traduction du risque. Seuils issus de la calibration observee sur
        backtest : la zone 0.30-0.45 correspondait a 68 % de changements reels,
        au-dela de 0.45 a 94 %. Le seuil d'alerte est donc 0.30, pas 0.50."""
        if p < 0.15:
            return "stable"
        if p < 0.30:
            return "surveillance"
        if p < 0.45:
            return "changement probable"
        return "changement quasi certain"

    def explain(self, H: int = 4, phase: str | None = None,
                t: int | None = None, x: dict | None = None) -> str:
        """Formule la prevision en clair, avec l'intervalle sur la destination."""
        self._check()
        phase = phase or self.phase_courante_
        t = t if t is not None else self.anciennete_
        pv = self.predict(H, phase, t, x)
        r = pv.iloc[-1]
        pc = float(r.get("p_change_cal", r["p_change"]))
        dests = [c for c in self.transitions_.columns if c != phase]
        best = max(dests, key=lambda d: float(r[d]))
        cond = float(self.transitions_.loc[phase, best])
        lo, hi = self.intervalles_.get((phase, best), (np.nan, np.nan))
        return (f"{phase} depuis {t} {self.unite_}. A {H} {self.unite_[:-1] if self.unite_.endswith('s') else self.unite_}(s) : "
                f"{100*pc:.0f} % de probabilite d'en etre sorti ({self.lecture(pc)}). "
                f"Destination la plus probable : {best} — {100*cond:.0f} % "
                f"des sorties historiques, IC90 [{100*lo:.0f} ; {100*hi:.0f}] %.")

    # -- validation et recalibration ---------------------------------------

    def backtest(self, phases: pd.Series, debut: str, H: int = 4,
                 exog: pd.DataFrame | None = None) -> pd.DataFrame:
        """Backtest RECURSIF : a chaque date, le modele est reestime sur le
        passe seul. Sans cette reestimation, le test serait sans valeur.

        Reserve : les etiquettes de phase proviennent d'une datation etablie en
        plein echantillon. Un test veritablement en temps reel exigerait de
        redater recursivement, ce qui degraderait les resultats.
        """
        phases = _assurer_period_index(phases, "phases")
        if exog is not None:
            exog = _assurer_period_index(exog, "exog")
        p = phases.dropna()
        d0 = debut if isinstance(debut, pd.Period) else pd.Period(debut, freq=p.index.freqstr)
        lignes = []
        for i, t in enumerate(p.index):
            if t < d0 or i < 40:
                continue
            passe = p.iloc[:i + 1]
            try:
                m = CycleModel(self.exclure, self.min_evenements, self.lissage)
                m.fit(passe, exog.reindex(passe.index) if exog is not None else None)
            except Exception:
                continue
            cur, anc = m.phase_courante_, m.anciennete_
            if cur not in m.hazard_ or cur not in m.transitions_.index:
                continue
            xx = ({k: float(exog[k].reindex([t]).iloc[0]) for k in self.exog_noms_}
                  if exog is not None else None)
            if xx is not None and any(pd.isna(v) for v in xx.values()):
                continue
            try:
                pv = m.predict(H, cur, anc, xx)
            except Exception:
                continue
            for h in range(1, H + 1):
                cible = t + h
                if cible not in p.index:
                    continue
                lignes.append(dict(
                    date=str(t), phase_t=cur, anciennete=anc, h=h, cible=str(cible),
                    p_change=float(pv.iloc[h - 1]["p_change"]),
                    reelle=p.loc[cible], change_reel=bool(p.loc[cible] != cur)))
        return pd.DataFrame(lignes)

    def calibrate(self, backtest: pd.DataFrame, n_bins: int = 4):
        """Corrige l'ecrasement d'echelle constate sur backtest.

        Le modele ne connait que la duree : il ne distingue pas un episode qui
        va degenerer d'un autre qui va se resorber, donc il moyenne et
        sous-estime systematiquement le risque (de ~19 points sur le backtest
        realise). On apprend la correspondance annonce -> observe, par
        regression isotone (monotone, donc l'ordre est preserve).
        """
        b = backtest.dropna(subset=["p_change", "change_reel"])
        if len(b) < 30:
            raise ValueError(f"Backtest trop court : {len(b)} lignes.")
        b = b.sort_values("p_change")
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(b.p_change.to_numpy(), b.change_reel.astype(float).to_numpy())
            grille = np.linspace(0, 1, 101)
            self.calibration_ = pd.DataFrame(
                {"annonce": grille, "corrige": iso.predict(grille)})
        except ImportError:
            # repli : table par tranches, sans dependance a scikit-learn
            b["bin"] = pd.qcut(b.p_change, min(n_bins, b.p_change.nunique()),
                               duplicates="drop")
            t = b.groupby("bin").agg(annonce=("p_change", "mean"),
                                     corrige=("change_reel", "mean")).reset_index(drop=True)
            self.calibration_ = t.sort_values("annonce")
        return self

    def _calibrer(self, p: float) -> float:
        c = self.calibration_
        return float(np.interp(p, c["annonce"], c["corrige"]))

    # -- restitution --------------------------------------------------------

    def summary(self) -> str:
        self._check()
        L = ["=" * 68, "MODELE DE CYCLE", "=" * 68,
             f"Etat courant : {self.phase_courante_} depuis {self.anciennete_} {self.unite_}",
             f"Covariables  : {self.exog_noms_ or 'aucune'}",
             f"Calibration  : {'appliquee' if self.calibration_ is not None else 'non appliquee'}",
             "", f"-- Hasard de sortie (cloglog), duree en {self.unite_} --",
             f"{'phase':<16}{'n':>5}{'sorties':>9}{'duree med':>11}   {'log_t (dependance de duree)'}"]
        for g, m in self.hazard_.items():
            L.append(f"{g:<16}{m['n']:>5}{m['evenements']:>9}{m['duree_med']:>11.1f}   "
                     f"{m['params'][1]:+.3f} (p={m['pvalues'][1]:.3f})")
        if self.hazard_dest_:
            meth = next(iter(self.hazard_dest_.values())).get("methode", "MV")
            L += ["", f"-- Hasards par destination ({meth}) --",
                  f"{'depart -> destination':<34}{'evts':>5}{'log_t':>9}"
                  + (f"{self.exog_noms_[0]:>11}{'p(RV)':>8}" if self.exog_noms_ else "")
                  + "   sep."]
            for (g, dst), m in sorted(self.hazard_dest_.items()):
                if m.get("degenere") and not self.firth:
                    L.append(f"{g + ' -> ' + dst:<34}{m['evenements']:>5}"
                             f"   DIVERGENT — activez firth=True")
                    continue
                ligne = f"{g + ' -> ' + dst:<34}{m['evenements']:>5}{m['params'][1]:>+9.3f}"
                if self.exog_noms_:
                    p = m.get("pvalues_lr", [np.nan] * 3)[2]
                    ligne += f"{m['params'][2]:>+11.4f}"
                    ligne += f"{p:>8.3f}" if np.isfinite(p) else f"{'-':>8}"
                ligne += "    oui" if m.get("separation") else "    non"
                L.append(ligne)
        L += ["", "-- Matrice de transition (probabilites) --",
              self.transitions_.to_string()]
        return "\n".join(L)

    def __repr__(self):
        if not self._ajuste:
            return "<CycleModel non ajuste>"
        return (f"<CycleModel {self.phase_courante_} t={self.anciennete_} "
                f"| {len(self.hazard_)} phases | "
                f"exog={self.exog_noms_ or 'aucune'}>")
