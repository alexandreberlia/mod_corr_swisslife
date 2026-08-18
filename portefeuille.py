"""
Moteur cross-sectionnel — classement de l'univers, allocation du capital, backtest.

Chaîne de traitement :
    prix (dict ticker -> OHLCV)
        -> panels de features        (dates x tickers)
        -> winsorisation             (écrête les aberrants, à chaque date)
        -> neutralisation secteur    (retire l'effet sectoriel)
        -> rang percentile           (0-1, à chaque date, sur l'univers)
        -> score composite           (moyenne pondérée des rangs)
        -> filtre d'éligibilité      (régime, liquidité)
        -> sélection + allocation    (pondération inverse-volatilité)

Ordre d'exécution à chaque rééquilibrage : SORTIES d'abord, ENTRÉES ensuite,
pour que la liquidité libérée soit réinvestissable le jour même.

Convention : signaux évalués sur la clôture de J, exécution à l'ouverture de J+1.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================================
# Paramètres
# ============================================================================

@dataclass
class ParamsPF:
    # --- score composite : feature -> poids ---
    poids: dict = field(default_factory=lambda: {
        "mom_12_1": 1.00,    # momentum 12-1 : l'anomalie la mieux documentée
        "rev_5":    0.40,    # réversion court terme (signe déjà inversé)
        "ext50":    0.25,    # position vs EMA50, normalisée ATR
    })
    winsor: tuple = (0.02, 0.98)
    neutraliser_secteur: bool = True
    min_titres: int = 15          # sous ce seuil, pas de classement à cette date

    # --- sélection ---
    n_long: int = 10
    n_short: int = 0              # 0 = long-only
    rang_entree: float = 0.85     # percentile mini pour ouvrir
    rang_sortie: float = 0.50     # on ferme si le rang retombe sous ce seuil
    rang_entree_short: float = 0.15
    rang_sortie_short: float = 0.50

    # --- éligibilité ---
    er_rank_min: float = 0.45
    adx_min: float = 15.0
    dollar_vol_min: float = 5e6
    prix_min: float = 5.0

    # --- allocation ---
    exposition: float = 0.95      # fraction du capital investie
    poids_max: float = 0.20       # plafond par ligne
    poids_min: float = 0.02       # sous ce poids, on n'ouvre pas
    ponderation: str = "inv_vol"  # "inv_vol" | "equal" | "score"

    # --- risque et sorties ---
    atr_n: int = 14
    stop_atr: float = 2.5         # stop initial, en ATR
    trail_atr: float = 3.5        # chandelier : plus haut depuis l'entrée - k*ATR
    max_hold: int = 250
    sortie_regime: float = 0.20   # ER rank sous ce seuil -> sortie

    # --- exécution ---
    freq_rebal: int = 5           # rééquilibrage tous les N jours ouvrés
    cost_bps: float = 10.0        # aller simple


# ============================================================================
# Construction des panels
# ============================================================================

def construire_panels(prix: dict, Indicateurs, p: ParamsPF) -> dict:
    """prix : {ticker: DataFrame OHLCV}. Renvoie un dict de panels (dates x tickers)."""
    feats, aux = {}, {}

    for t, df in prix.items():
        ind = Indicateurs(df, burnin=False)
        lr = ind.logret(ind.price, 1)

        feats[t] = pd.DataFrame({
            "mom_12_1": lr.rolling(252).sum() - lr.rolling(21).sum(),
            "rev_5":   -lr.rolling(5).sum(),          # survendu = signal long
            "ext50":    ind.ext(50, p.atr_n),
        })
        aux[t] = pd.DataFrame({
            "close":  ind.close,
            "open":   df["Open"].astype(float) if "Open" in df.columns else ind.close,
            "high":   ind.high,
            "low":    ind.low,
            "atr":    ind.atr(p.atr_n),
            "er_rk":  ind.rank_pct(ind.er(10), 252),
            "adx":    ind.adx(p.atr_n),
            "dvol":  (ind.close * ind.volume).rolling(20).median(),
        })

    def panel(source, col):
        return pd.DataFrame({t: d[col] for t, d in source.items()}).sort_index()

    out = {c: panel(feats, c) for c in next(iter(feats.values())).columns}
    out |= {c: panel(aux, c) for c in next(iter(aux.values())).columns}
    return out


# ============================================================================
# Transformations cross-sectionnelles
# ============================================================================

def winsorize_cs(df, lo=0.02, hi=0.98):
    """Écrête les extrêmes ligne par ligne (à chaque date, sur l'univers)."""
    return df.clip(lower=df.quantile(lo, axis=1), upper=df.quantile(hi, axis=1), axis=0)


def neutralize_cs(df, secteurs: pd.Series, min_grp=3):
    """Retranche la moyenne du secteur : évite de parier sur un secteur sans le vouloir."""
    if secteurs is None:
        return df
    g = secteurs.reindex(df.columns)
    out = df.copy()
    for _, cols in g.groupby(g).groups.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) >= min_grp:
            out[cols] = df[cols].sub(df[cols].mean(axis=1), axis=0)
    return out


def rank_cs(df, min_titres=15):
    """Rang percentile 0-1 à chaque date. axis=1 : à travers les titres, pas le temps."""
    valide = df.notna().sum(axis=1) >= min_titres
    return df.rank(axis=1, pct=True).where(valide, np.nan)


# ============================================================================
# Moteur
# ============================================================================

class Portefeuille:

    def __init__(self, panels: dict, p: ParamsPF | None = None, secteurs=None):
        self.pn = panels
        self.p = p or ParamsPF()
        self.secteurs = secteurs
        self._score()

    # ---------- score composite et éligibilité ----------

    def _score(self):
        p = self.p
        total = 0.0
        acc = None
        for nom, w in p.poids.items():
            if nom not in self.pn or w == 0:
                continue
            x = winsorize_cs(self.pn[nom], *p.winsor)
            if p.neutraliser_secteur:
                x = neutralize_cs(x, self.secteurs)
            r = rank_cs(x, p.min_titres)
            acc = r * w if acc is None else acc.add(r * w, fill_value=np.nan)
            total += w

        self.score = acc / total                       # moyenne pondérée des rangs
        self.rang = rank_cs(self.score, p.min_titres)  # re-classement du composite

        self.eligible = (
            (self.pn["er_rk"] > p.er_rank_min)
            & (self.pn["adx"] > p.adx_min)
            & (self.pn["dvol"] > p.dollar_vol_min)
            & (self.pn["close"] > p.prix_min)
            & self.rang.notna()
            & self.pn["atr"].gt(0)
        ).fillna(False)

    # ---------- allocation ----------

    def _poids_cibles(self, tickers, date, equity) -> pd.Series:
        """Pondération des lignes retenues. inv_vol => risque € identique par ligne."""
        p = self.p
        if not len(tickers):
            return pd.Series(dtype=float)

        if p.ponderation == "inv_vol":
            atrp = (self.pn["atr"].loc[date, tickers] /
                    self.pn["close"].loc[date, tickers]).replace(0, np.nan)
            raw = 1.0 / atrp
        elif p.ponderation == "score":
            raw = self.score.loc[date, tickers].clip(lower=0.01)
        else:
            raw = pd.Series(1.0, index=tickers)

        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            return pd.Series(dtype=float)

        w = raw / raw.sum() * p.exposition

        # plafonnement itératif : l'excédent est redistribué sur les lignes non plafonnées.
        # Si TOUTES sont plafonnées, l'exposition totale reste < p.exposition : c'est voulu
        # (moins de candidats que n_long => on n'investit pas tout, on ne concentre pas).
        for _ in range(10):
            trop = w > p.poids_max
            if not trop.any():
                break
            excedent = (w[trop] - p.poids_max).sum()
            w[trop] = p.poids_max
            libre = ~trop
            if not libre.any() or w[libre].sum() <= 0:
                break
            w[libre] += excedent * w[libre] / w[libre].sum()

        return w[w >= p.poids_min]

    # ---------- carnet du jour (usage live) ----------

    def book(self, date=None, equity: float = 10_000.0) -> pd.DataFrame:
        """Classement + allocation à une date. C'est l'ordre à passer."""
        p = self.p
        date = self.rang.index[-1] if date is None else pd.Timestamp(date)

        rg = self.rang.loc[date]
        el = self.eligible.loc[date]
        px = self.pn["close"].loc[date]
        atr = self.pn["atr"].loc[date]

        longs = rg[(rg >= p.rang_entree) & el].nlargest(p.n_long).index
        shorts = rg[(rg <= p.rang_entree_short) & el].nsmallest(p.n_short).index if p.n_short else []

        lignes = []
        for cote, sel in (("LONG", longs), ("SHORT", shorts)):
            if not len(sel):
                continue
            w = self._poids_cibles(list(sel), date, equity)
            for t, wi in w.items():
                capital = wi * equity
                qty = capital / px[t]
                dist = p.stop_atr * atr[t]
                stop = px[t] - dist if cote == "LONG" else px[t] + dist
                lignes.append({
                    "ticker": t, "sens": cote,
                    "rang": round(rg[t], 3), "score": round(self.score.loc[date, t], 3),
                    "prix": round(px[t], 2), "poids_%": round(wi * 100, 2),
                    "capital": round(capital, 2), "qty": round(qty, 4),
                    "stop": round(stop, 2),
                    "risque_€": round(qty * dist, 2),
                    "risque_%_capital": round(qty * dist / equity * 100, 2),
                    "atr_%": round(atr[t] / px[t] * 100, 2),
                })

        cols = ["ticker", "sens", "rang", "score", "prix", "poids_%", "capital",
                "qty", "stop", "risque_€", "risque_%_capital", "atr_%"]
        df = pd.DataFrame(lignes, columns=cols)
        return df.sort_values(["sens", "rang"], ascending=[True, False]).reset_index(drop=True)

    # ---------- backtest ----------

    def backtest(self, capital: float = 10_000.0) -> dict:
        p = self.p
        idx = self.rang.index
        cost = p.cost_bps / 10_000.0

        op, hi, lo, cl = (self.pn["open"], self.pn["high"], self.pn["low"], self.pn["close"])
        atr = self.pn["atr"]

        cash = capital
        pos = {}                       # ticker -> dict(qty, px_in, date_in, stop, plus_haut, held)
        trades, courbe = [], np.full(len(idx), np.nan)
        ordres = {"sorties": [], "entrees": {}}

        for i, d in enumerate(idx):
            if i == 0:
                courbe[i] = cash
                continue
            dv = idx[i - 1]            # barre de décision (la veille)

            # ---- 1. exécution des ordres décidés hier, à l'ouverture ----
            for t in ordres["sorties"]:
                if t in pos and not np.isnan(op.loc[d, t]):
                    px = op.loc[d, t]
                    cash += pos[t]["qty"] * px * (1 - cost)
                    trades.append(_trade(t, pos.pop(t), d, px, "signal"))

            for t, cap in ordres["entrees"].items():
                px, a = op.loc[d, t], atr.loc[dv, t]
                if t in pos or np.isnan(px) or np.isnan(a) or a <= 0:
                    continue
                cap = min(cap, cash / (1 + cost))
                qty = cap / px
                if qty * px < 1:
                    continue
                cash -= qty * px * (1 + cost)
                pos[t] = {"qty": qty, "px_in": px, "date_in": d,
                          "stop": px - p.stop_atr * a, "plus_haut": px, "held": 0}
            ordres = {"sorties": [], "entrees": {}}

            # ---- 2. stops touchés en séance (avant toute autre décision) ----
            for t in list(pos):
                s = pos[t]
                s["held"] += 1
                low, high = lo.loc[d, t], hi.loc[d, t]
                if np.isnan(low):
                    continue
                if low <= s["stop"]:
                    px = min(op.loc[d, t], s["stop"])      # gap : on subit l'ouverture
                    cash += s["qty"] * px * (1 - cost)
                    trades.append(_trade(t, pos.pop(t), d, px, "stop"))
                    continue
                # chandelier : trailing depuis le plus haut atteint, jamais en arrière
                s["plus_haut"] = max(s["plus_haut"], high)
                a = atr.loc[d, t]
                if not np.isnan(a):
                    s["stop"] = max(s["stop"], s["plus_haut"] - p.trail_atr * a)

            # ---- 3. rééquilibrage : SORTIES d'abord, ENTRÉES ensuite ----
            if i % p.freq_rebal == 0:
                rg, el = self.rang.loc[d], self.eligible.loc[d]
                er = self.pn["er_rk"].loc[d]

                # 3a. sorties par signal -> libèrent de la liquidité pour 3b
                for t, s in pos.items():
                    if (pd.isna(rg.get(t)) or rg.get(t, 0) < p.rang_sortie
                            or er.get(t, 1) < p.sortie_regime
                            or s["held"] >= p.max_hold):
                        ordres["sorties"].append(t)

                # 3b. entrées, en tenant compte du cash libéré par 3a
                restants = [t for t in pos if t not in ordres["sorties"]]
                libere = sum(pos[t]["qty"] * cl.loc[d, t] for t in ordres["sorties"]
                             if not np.isnan(cl.loc[d, t]))
                equity = cash + sum(pos[t]["qty"] * cl.loc[d, t] for t in pos
                                    if not np.isnan(cl.loc[d, t]))
                dispo = cash + libere

                n_libres = p.n_long - len(restants)
                if n_libres > 0 and dispo > equity * p.poids_min:
                    cand = rg[(rg >= p.rang_entree) & el].drop(labels=restants, errors="ignore")
                    cand = cand.nlargest(n_libres).index
                    if len(cand):
                        w = self._poids_cibles(list(cand), d, equity)
                        besoin = (w * equity).sum()
                        k = min(1.0, dispo / besoin) if besoin > 0 else 0.0
                        for t, wi in w.items():
                            ordres["entrees"][t] = wi * equity * k

            courbe[i] = cash + sum(pos[t]["qty"] * cl.loc[d, t] for t in pos
                                   if not np.isnan(cl.loc[d, t]))

        eq = pd.Series(courbe, index=idx).ffill().fillna(capital)
        return {"equity": eq, "trades": pd.DataFrame(trades), "params": p}


def _trade(t, s, d_out, px_out, motif):
    return {"ticker": t, "entree": s["date_in"], "sortie": d_out,
            "px_entree": s["px_in"], "px_sortie": px_out, "qty": s["qty"],
            "pnl": (px_out - s["px_in"]) * s["qty"],
            "ret_%": (px_out / s["px_in"] - 1) * 100,
            "jours": s["held"], "motif": motif}


# ============================================================================
# Métriques
# ============================================================================

def stats(res: dict, freq: int = 252) -> pd.Series:
    eq, tr = res["equity"], res["trades"]
    r = eq.pct_change().dropna()
    ans = len(eq) / freq
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / ans) - 1 if ans > 0 else np.nan
    vol = r.std() * np.sqrt(freq)
    dd = (eq / eq.cummax() - 1).min()

    out = {"perf_%": (eq.iloc[-1] / eq.iloc[0] - 1) * 100, "cagr_%": cagr * 100,
           "vol_%": vol * 100, "sharpe": cagr / vol if vol > 0 else np.nan,
           "max_dd_%": dd * 100, "calmar": cagr / abs(dd) if dd < 0 else np.nan,
           "nb_trades": len(tr)}
    if len(tr):
        g, pr = tr[tr.pnl > 0], tr[tr.pnl <= 0]
        out |= {"win_%": len(g) / len(tr) * 100,
                "gain_moy_%": g["ret_%"].mean() if len(g) else np.nan,
                "perte_moy_%": pr["ret_%"].mean() if len(pr) else np.nan,
                "profit_factor": g.pnl.sum() / abs(pr.pnl.sum()) if pr.pnl.sum() else np.inf,
                "duree_moy_j": tr.jours.mean()}
    return pd.Series(out)
