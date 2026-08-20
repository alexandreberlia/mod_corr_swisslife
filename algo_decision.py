"""
portefeuille.py — moteur cross-sectionnel : classement, allocation, backtest.

RÔLE : machine de production. Ne regarde QUE le passé. Tourne tous les jours.

CHAÎNE
    prix (dict ticker -> OHLCV)
      -> panels de features           (dates x tickers)
      -> winsorisation                (écrête les aberrants, à chaque date)
      -> neutralisation secteur       (retire l'effet sectoriel)
      -> rang percentile CENTRÉ       (-0.5 à +0.5, à chaque date, sur l'univers)
      -> score composite              (moyenne pondérée, normalisée par somme des |w|)
      -> filtre d'éligibilité         (régime, liquidité)
      -> sélection + allocation       (pondération inverse-volatilité)

ORDRE À CHAQUE RÉÉQUILIBRAGE : SORTIES d'abord, ENTRÉES ensuite, pour que la
liquidité libérée soit réinvestissable le jour même.

CONVENTION : signal évalué sur la clôture de J, exécution à l'OUVERTURE de J+1.

LONG ONLY. La vente à découvert n'est pas implémentée — c'était une incohérence de
la version précédente (book() générait des SHORT que backtest() ignorait). Mieux vaut
une absence explicite qu'un mensonge silencieux.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================================
# Paramètres
# ============================================================================

@dataclass
class ParamsPF:
    # --- score composite : feature -> poids (peut être négatif) ---
    poids: dict = field(default_factory=lambda: {
        "mom_12_1": 1.00, "rev_5": 0.40, "ext_ema50": 0.25,
    })
    winsor: tuple = (0.02, 0.98)
    neutraliser_secteur: bool = True
    min_titres: int = 15          # sous ce seuil, pas de classement à cette date

    # --- sélection (long only) ---
    n_long: int = 10
    rang_entree: float = 0.85     # percentile mini pour ouvrir
    rang_sortie: float = 0.50     # hystérésis : on ferme sous ce seuil seulement

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
    stop_atr: float = 2.5         # stop initial
    trail_atr: float = 5.0        # chandelier. 3.5 donnait 58 % de sorties sur stop :
                                  # trop serré, la position n'a pas la place de respirer.
    max_hold: int = 250
    sortie_regime: float = 0.20   # ER rank sous ce seuil -> sortie

    # --- exécution ---
    freq_rebal: int = 5           # rééquilibrage tous les N jours ouvrés
    cost_bps: float = 10.0        # aller simple


# ============================================================================
# Construction des panels
# ============================================================================

def construire_panels(prix: dict, Indicateurs, p: ParamsPF, generateur=None) -> dict:
    """prix : {ticker: DataFrame OHLCV}. Renvoie un dict de panels (dates x tickers).

    generateur : fonction ind -> dict[str, Series].
                 None => jeu minimal. Passer `features_orientees` (features.py) pour
                 le catalogue complet — OBLIGATOIRE si les poids viennent de l'IC.
    """
    feats, aux = {}, {}

    for t, df in prix.items():
        ind = Indicateurs(df, burnin=False)

        if generateur is not None:
            feats[t] = pd.DataFrame(generateur(ind))
        else:
            lr = ind.logret(ind.price, 1)
            feats[t] = pd.DataFrame({
                "mom_12_1":  lr.rolling(252).sum() - lr.rolling(21).sum(),
                "rev_5":    -lr.rolling(5).sum(),
                "ext_ema50": ind.ext(50, p.atr_n),
            })

        # colonnes techniques : toujours nécessaires (exécution, stops, éligibilité)
        aux[t] = pd.DataFrame({
            "close": ind.close,
            "open":  df["Open"].astype(float) if "Open" in df.columns else ind.close,
            "high":  ind.high,
            "low":   ind.low,
            "atr":   ind.atr(p.atr_n),
            "er_rk": ind.rank_pct(ind.er(10), 252),
            "adx":   ind.adx(p.atr_n),
            "dvol": (ind.close * ind.volume).rolling(20).median(),
        })

    def panel(src, col):
        return pd.DataFrame({t: d[col] for t, d in src.items()}).sort_index()

    out = {c: panel(feats, c) for c in next(iter(feats.values())).columns}
    out |= {c: panel(aux, c) for c in next(iter(aux.values())).columns}
    return out


# ============================================================================
# Transformations cross-sectionnelles
# ============================================================================

def winsorize_cs(df, lo=0.02, hi=0.98):
    """Écrête les extrêmes ligne par ligne (à chaque date, sur l'univers).
    À faire AVANT la neutralisation : sinon un outlier pollue la moyenne sectorielle."""
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

        # validation : échouer bruyamment plutôt que silencieusement
        demandees = {k for k, v in p.poids.items() if v != 0}
        manquantes = demandees - set(self.pn)
        if manquantes:
            raise KeyError(
                f"Features absentes des panels : {sorted(manquantes)}.\n"
                f"Disponibles : {sorted(self.pn)}.\n"
                f"-> construire_panels(..., generateur=features_orientees)"
            )
        if not demandees:
            raise ValueError("Aucune feature avec un poids non nul dans ParamsPF.poids")

        # Rangs CENTRÉS (-0.5 à +0.5) et normalisation par la somme des VALEURS
        # ABSOLUES : les poids issus de poids_depuis_ic peuvent être négatifs. Diviser
        # par la somme signée ferait exploser le score quand les poids se compensent
        # (ex. {0.5, -0.45} -> diviseur 0.05 -> score x20).
        acc, total = None, 0.0
        for nom, w in p.poids.items():
            if w == 0:
                continue
            x = winsorize_cs(self.pn[nom], *p.winsor)
            if p.neutraliser_secteur:
                x = neutralize_cs(x, self.secteurs)
            r = rank_cs(x, p.min_titres) - 0.5
            acc = r * w if acc is None else acc + r * w
            total += abs(w)

        self.score = acc / total                       # borné [-0.5, +0.5]
        self.rang = rank_cs(self.score, p.min_titres)

        self.eligible = (
            (self.pn["er_rk"] > p.er_rank_min)
            & (self.pn["adx"] > p.adx_min)
            & (self.pn["dvol"] > p.dollar_vol_min)
            & (self.pn["close"] > p.prix_min)
            & self.rang.notna()
            & self.pn["atr"].gt(0)
        ).fillna(False)

    # ---------- allocation ----------

    def _poids_cibles(self, tickers, date) -> pd.Series:
        """Pondération des lignes retenues.

        inv_vol => risque en euros IDENTIQUE par ligne :
            risque_i = capital_i x stop_atr x atr_pct_i
                     ∝ (1/atr_pct_i) x atr_pct_i = constante
        C'est ce qui réconcilie "allouer par poids" et "dimensionner par le risque".
        """
        p = self.p
        if not len(tickers):
            return pd.Series(dtype=float)

        if p.ponderation == "inv_vol":
            atrp = (self.pn["atr"].loc[date, tickers] /
                    self.pn["close"].loc[date, tickers]).replace(0, np.nan)
            raw = 1.0 / atrp
        elif p.ponderation == "score":
            raw = (self.score.loc[date, tickers] + 0.5).clip(lower=0.01)
        else:
            raw = pd.Series(1.0, index=tickers)

        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            return pd.Series(dtype=float)

        w = raw / raw.sum() * p.exposition

        # plafonnement itératif : l'excédent est redistribué sur les non plafonnées.
        # Si toutes sont plafonnées, l'exposition reste < p.exposition : c'est voulu
        # (moins de candidats que n_long => on n'investit pas tout, on ne concentre pas).
        for _ in range(10):
            trop = w > p.poids_max
            if not trop.any():
                break
            exc = (w[trop] - p.poids_max).sum()
            w[trop] = p.poids_max
            libre = ~trop
            if not libre.any() or w[libre].sum() <= 0:
                break
            w[libre] += exc * w[libre] / w[libre].sum()

        return w[w >= p.poids_min]

    # ---------- carnet du jour (usage live) ----------

    def book(self, date=None, equity: float = 10_000.0) -> pd.DataFrame:
        """Classement + allocation à une date. C'est l'ordre à passer. LONG ONLY."""
        p = self.p
        date = self.rang.index[-1] if date is None else pd.Timestamp(date)

        rg, el = self.rang.loc[date], self.eligible.loc[date]
        px, atr = self.pn["close"].loc[date], self.pn["atr"].loc[date]

        sel = rg[(rg >= p.rang_entree) & el].nlargest(p.n_long).index
        w = self._poids_cibles(list(sel), date)

        lignes = []
        for t, wi in w.items():
            capital = wi * equity
            qty = capital / px[t]
            dist = p.stop_atr * atr[t]
            lignes.append({
                "ticker": t, "sens": "LONG",
                "rang": round(rg[t], 3), "score": round(self.score.loc[date, t], 3),
                "prix": round(px[t], 2), "poids_%": round(wi * 100, 2),
                "capital": round(capital, 2), "qty": round(qty, 4),
                "stop": round(px[t] - dist, 2),
                "risque_€": round(qty * dist, 2),
                "risque_%": round(qty * dist / equity * 100, 2),
                "atr_%": round(atr[t] / px[t] * 100, 2),
            })

        cols = ["ticker", "sens", "rang", "score", "prix", "poids_%", "capital",
                "qty", "stop", "risque_€", "risque_%", "atr_%"]
        return (pd.DataFrame(lignes, columns=cols)
                .sort_values("rang", ascending=False).reset_index(drop=True))

    # ---------- backtest ----------

    def backtest(self, capital: float = 10_000.0) -> dict:
        p = self.p
        idx = self.rang.index
        cost = p.cost_bps / 10_000.0

        op, hi, lo, cl = (self.pn["open"], self.pn["high"],
                          self.pn["low"], self.pn["close"])
        atr = self.pn["atr"]

        cash = capital
        pos = {}                       # ticker -> dict d'état
        trades, courbe, expo = [], np.full(len(idx), np.nan), np.zeros(len(idx))
        ordres = {"sorties": [], "entrees": {}}

        for i, d in enumerate(idx):
            if i == 0:
                courbe[i] = cash
                continue
            dv = idx[i - 1]

            # ---- 1. exécution des ordres décidés hier, à l'ouverture ----
            for t in ordres["sorties"]:
                if t in pos and not np.isnan(op.loc[d, t]):
                    px = op.loc[d, t]
                    cash += pos[t]["qty"] * px * (1 - cost)
                    trades.append(_trade(t, pos.pop(t), d, px, "signal", cost))

            # servir dans l'ordre du score : pas de biais alphabétique si le cash manque
            for t, cap in sorted(ordres["entrees"].items(),
                                 key=lambda kv: -kv[1]):
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
                    px = min(op.loc[d, t], s["stop"])   # gap : on subit l'ouverture
                    cash += s["qty"] * px * (1 - cost)
                    trades.append(_trade(t, pos.pop(t), d, px, "stop", cost))
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
                    cand = (rg[(rg >= p.rang_entree) & el]
                            .drop(labels=restants, errors="ignore")
                            .nlargest(n_libres).index)
                    if len(cand):
                        w = self._poids_cibles(list(cand), d)
                        besoin = (w * equity).sum()
                        k = min(1.0, dispo / besoin) if besoin > 0 else 0.0
                        for t, wi in w.items():
                            ordres["entrees"][t] = wi * equity * k

            val_pos = sum(pos[t]["qty"] * cl.loc[d, t] for t in pos
                          if not np.isnan(cl.loc[d, t]))
            courbe[i] = cash + val_pos
            expo[i] = val_pos / courbe[i] if courbe[i] > 0 else 0.0

        eq = pd.Series(courbe, index=idx).ffill().fillna(capital)
        return {"equity": eq, "trades": pd.DataFrame(trades),
                "exposition": pd.Series(expo, index=idx), "params": p}


def _trade(t, s, d_out, px_out, motif, cost):
    """PnL NET des coûts (aller + retour). La version précédente le calculait brut,
    ce qui surestimait profit_factor et gain_moyen."""
    frais = s["qty"] * (s["px_in"] + px_out) * cost
    pnl = (px_out - s["px_in"]) * s["qty"] - frais
    return {"ticker": t, "entree": s["date_in"], "sortie": d_out,
            "px_entree": s["px_in"], "px_sortie": px_out, "qty": s["qty"],
            "frais": frais, "pnl": pnl,
            "ret_%": pnl / (s["px_in"] * s["qty"]) * 100,
            "jours": s["held"], "motif": motif}


# ============================================================================
# Métriques
# ============================================================================

def stats(res: dict, close: pd.DataFrame | None = None, freq: int = 252) -> pd.Series:
    """close : panel de clôtures. Si fourni, ajoute le benchmark buy & hold.

    POURQUOI LE BENCHMARK EST INDISPENSABLE
    Sur une marche aléatoire en log, le PRIX a une dérive positive (inégalité de
    Jensen). Une stratégie long-only capture cette dérive SANS aucun edge. Comparer
    à zéro ne prouve donc rien : le bon contrôle négatif est le buy & hold
    équipondéré, ajusté de l'exposition moyenne.
    """
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

    if "exposition" in res:
        out["expo_moy_%"] = res["exposition"].mean() * 100

    if len(tr):
        g, pr = tr[tr.pnl > 0], tr[tr.pnl <= 0]
        out |= {"win_%": len(g) / len(tr) * 100,
                "gain_moy_%": g["ret_%"].mean() if len(g) else np.nan,
                "perte_moy_%": pr["ret_%"].mean() if len(pr) else np.nan,
                "profit_factor": g.pnl.sum() / abs(pr.pnl.sum()) if len(pr) and pr.pnl.sum() else np.inf,
                "duree_moy_j": tr.jours.mean(),
                "frais_tot": tr.frais.sum(),
                "frais_%_an": tr.frais.sum() / eq.iloc[0] / ans * 100}

    if close is not None:
        bh = close.pct_change().mean(axis=1).reindex(eq.index).fillna(0).add(1).cumprod()
        bh_cagr = bh.iloc[-1] ** (1 / ans) - 1 if ans > 0 else np.nan
        expo = res.get("exposition", pd.Series(1.0, index=eq.index)).mean()
        out |= {"bh_cagr_%": bh_cagr * 100,
                "alpha_vs_bh_%": (cagr - bh_cagr * expo) * 100}

    return pd.Series(out)
