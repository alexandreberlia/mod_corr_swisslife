"""
simulation.py — simulation walk-forward à partir d'une date donnée.

QUESTION À LAQUELLE CE MODULE RÉPOND
"Si j'avais lancé cet algo le 2025-01-01 et l'avais fait tourner chaque jour,
où en serais-je aujourd'hui ?"

CAUSALITÉ — le point critique
À chaque date t, la décision n'utilise QUE l'information disponible à t :

  1. Indicateurs   : EMA, ATR, ADX, ER, KAMA sont récursifs ou glissants -> causaux
                     par construction. La valeur à t ne dépend que de <= t.
  2. Rangs         : winsorize / neutralize / rank sont cross-sectionnels, calculés
                     date par date sur l'univers -> causaux.
  3. Exécution     : signal sur la clôture de t, ordre exécuté à l'OUVERTURE de t+1.
  4. POIDS         : c'est ici que le look-ahead se glisse. calibrer() utilise
                     shift(-horizon), donc le futur. Si on calibre une fois sur toute
                     la période, la simulation triche.
                     -> RECALIBRATION WALK-FORWARD : à chaque date de recalibrage R,
                        les poids sont estimés uniquement sur les données < R, puis
                        appliqués sur [R, R_suivant[. Jamais l'inverse.

DÉCOMPOSITION DU P&L
  réalisé  : positions fermées (net de frais aller + retour)
  latent   : positions encore ouvertes, valorisées au dernier cours connu
  total    = equity_finale - capital_initial = réalisé + latent
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd



# ============================================================================
# Résultat
# ============================================================================

@dataclass
class Simulation:
    equity: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    cash: float
    capital_initial: float
    date_debut: pd.Timestamp
    date_fin: pd.Timestamp
    poids_utilises: dict = field(default_factory=dict)
    benchmark: pd.Series | None = None

    # ---------- décomposition du P&L ----------

    @property
    def pnl(self) -> pd.Series:
        realise = self.trades.pnl.sum() if len(self.trades) else 0.0
        latent = self.positions.pnl_latent.sum() if len(self.positions) else 0.0
        frais_c = self.trades.frais.sum() if len(self.trades) else 0.0
        frais_o = self.positions.frais.sum() if len(self.positions) else 0.0
        total = self.equity.iloc[-1] - self.capital_initial
        return pd.Series({
            "capital_initial": self.capital_initial,
            "valeur_finale": self.equity.iloc[-1],
            "pnl_total": total,
            "pnl_total_%": total / self.capital_initial * 100,
            "pnl_realise": realise,
            "pnl_latent": latent,
            "frais_payes": frais_c + frais_o,
            "cash_disponible": self.cash,
            "valeur_positions": self.positions.valeur.sum() if len(self.positions) else 0.0,
            "nb_positions_ouvertes": len(self.positions),
            "nb_trades_clotures": len(self.trades),
        })

    # ---------- rapport lisible ----------

    def rapport(self) -> str:
        p = self.pnl
        jours = (self.date_fin - self.date_debut).days
        L = []
        L.append("=" * 72)
        L.append(f"SIMULATION  {self.date_debut.date()}  ->  {self.date_fin.date()}"
                 f"   ({jours} jours, {len(self.equity)} séances)")
        L.append("=" * 72)

        L.append(f"\nCapital initial      {p.capital_initial:>14,.2f}")
        L.append(f"Valeur aujourd'hui   {p.valeur_finale:>14,.2f}")
        L.append(f"{'':21}{'-' * 14}")
        signe = "+" if p.pnl_total >= 0 else ""
        L.append(f"P&L TOTAL            {signe}{p.pnl_total:>13,.2f}"
                 f"   ({signe}{p['pnl_total_%']:.2f} %)")

        L.append(f"\n  dont réalisé (positions fermées)   {p.pnl_realise:>12,.2f}")
        L.append(f"  dont latent  (positions ouvertes)  {p.pnl_latent:>12,.2f}")
        L.append(f"  frais payés                        {-p.frais_payes:>12,.2f}")

        L.append(f"\nComposition actuelle")
        L.append(f"  cash                               {p.cash_disponible:>12,.2f}")
        L.append(f"  titres ({int(p.nb_positions_ouvertes)} lignes)"
                 f"{'':>21}{p.valeur_positions:>12,.2f}")

        if self.benchmark is not None:
            bh = (self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1) * 100
            L.append(f"\nBuy & hold équipondéré             {bh:>+12.2f} %")
            L.append(f"Écart                              "
                     f"{p['pnl_total_%'] - bh:>+12.2f} %")

        if len(self.positions):
            L.append("\n" + "-" * 72)
            L.append("POSITIONS ENCORE OUVERTES")
            L.append("-" * 72)
            cols = ["ticker", "entree", "px_entree", "px_actuel", "qty",
                    "valeur", "pnl_latent", "ret_%", "stop", "jours"]
            pos = self.positions[cols].copy()
            pos["entree"] = pos["entree"].dt.date
            L.append(pos.round(2).to_string(index=False))

        if len(self.trades):
            L.append("\n" + "-" * 72)
            L.append(f"TRADES CLÔTURÉS ({len(self.trades)})")
            L.append("-" * 72)
            t = self.trades
            g, pr = t[t.pnl > 0], t[t.pnl <= 0]
            L.append(f"  gagnants {len(g):>3} | moy {g['ret_%'].mean() if len(g) else 0:>+6.2f} %"
                     f" | total {g.pnl.sum():>+10,.2f}")
            L.append(f"  perdants {len(pr):>3} | moy {pr['ret_%'].mean() if len(pr) else 0:>+6.2f} %"
                     f" | total {pr.pnl.sum():>+10,.2f}")
            L.append(f"  durée moyenne {t.jours.mean():.0f} jours")
            L.append(f"  motifs : {dict(t.motif.value_counts())}")
            L.append("\n  10 derniers :")
            der = t.tail(10)[["ticker", "entree", "sortie", "px_entree",
                              "px_sortie", "pnl", "ret_%", "motif"]].copy()
            der["entree"] = der["entree"].dt.date
            der["sortie"] = der["sortie"].dt.date
            L.append(der.round(2).to_string(index=False))

        return "\n".join(L)


# ============================================================================
# Score walk-forward
# ============================================================================

def score_walk_forward(panels: dict, close: pd.DataFrame, p: ParamsPF,
                       secteurs, date_debut, freq_recalib: int = 63,
                       horizon: int = 20, n_max: int = 5,
                       poids_fixes: dict | None = None,
                       verbose: bool = True) -> tuple:
    """Construit le panel de rangs en recalibrant les poids périodiquement.

    À chaque date de recalibrage R, les poids sont estimés sur panels.loc[:R]
    UNIQUEMENT, puis appliqués jusqu'au recalibrage suivant. Aucun poids n'est
    jamais utilisé sur les données qui ont servi à l'estimer... vers le futur.

    poids_fixes : court-circuite la calibration (utile pour comparer).
    """
    idx = close.index
    d0 = pd.Timestamp(date_debut)
    if d0 < idx[0]:
        d0 = idx[0]
    dates_sim = idx[idx >= d0]
    if len(dates_sim) == 0:
        raise ValueError(f"Aucune donnée après {d0.date()}")

    # dates de recalibrage : la 1re est le début de simulation
    bornes = list(range(0, len(dates_sim), freq_recalib))
    recalib = [dates_sim[i] for i in bornes] + [idx[-1] + pd.Timedelta(days=1)]

    rangs, journal = [], []

    for k in range(len(recalib) - 1):
        R, R_fin = recalib[k], recalib[k + 1]

        if poids_fixes is not None:
            poids = dict(poids_fixes)
        else:
            # calibration sur le PASSÉ STRICT (< R). ic_serie utilise shift(-horizon),
            # donc les dernières dates avant R ont un rendement futur NaN et sont
            # automatiquement écartées : pas de fuite.
            avant = idx[idx < R]
            if len(avant) < 252 + horizon:
                poids = dict(p.poids)                   # pas assez d'historique
            else:
                poids, _, _ = calibrer(panels, close, split=avant[-1],
                                       horizon=horizon, n_max=n_max)
                if not poids:
                    poids = dict(p.poids)

        # score sur toute la période (causal), mais on ne garde que [R, R_fin[
        p_seg = ParamsPF(**{**p.__dict__, "poids": poids})
        pf_seg = Portefeuille(panels, p_seg, secteurs)
        seg = pf_seg.rang.loc[(pf_seg.rang.index >= R) & (pf_seg.rang.index < R_fin)]
        rangs.append(seg)
        journal.append({"debut": R.date(), "poids": poids})

        if verbose:
            desc = ", ".join(f"{a}{b:+.2f}" for a, b in poids.items())
            print(f"  {R.date()} -> {desc}")

    return pd.concat(rangs), journal


# ============================================================================
# Simulation
# ============================================================================

def simuler(prix: dict, date_debut, Indicateurs, p: ParamsPF | None = None,
            secteurs=None, capital: float = 10_000.0,
            freq_recalib: int = 63, horizon: int = 20,
            poids_fixes: dict | None = None, verbose: bool = True) -> Simulation:
    """Rejoue la stratégie jour par jour depuis `date_debut` jusqu'à la fin des données.

    prix        : {ticker: DataFrame OHLCV} — historique COMPLET (l'antériorité sert
                  à amorcer les indicateurs, elle n'est jamais tradée).
    date_debut  : première date où une position peut être ouverte.
    poids_fixes : si fourni, pas de recalibration (comparaison / debug).
    """
    p = p or ParamsPF()

    if verbose:
        print("Construction des panels…")
    panels = construire_panels(prix, Indicateurs, p, generateur=features_orientees)
    close = panels["close"]

    if verbose:
        print(f"\nRecalibrage walk-forward (tous les {freq_recalib} jours) :")
    rang_wf, journal = score_walk_forward(panels, close, p, secteurs, date_debut,
                                          freq_recalib, horizon,
                                          poids_fixes=poids_fixes, verbose=verbose)

    # on restreint les panels à la période simulée, puis on injecte le rang walk-forward
    dates = rang_wf.index
    pan_sim = {k: v.loc[dates] for k, v in panels.items()}

    pf = Portefeuille(pan_sim, p, secteurs)
    pf.rang = rang_wf                                  # rangs recalibrés périodiquement
    pf.eligible = pf.eligible & rang_wf.notna()

    if verbose:
        print(f"\nBacktest sur {len(dates)} séances…")
    res = pf.backtest(capital)

    cl = pan_sim["close"]
    bench = cl.pct_change().fillna(0).mean(axis=1).add(1).cumprod()

    return Simulation(
        equity=res["equity"], trades=res["trades"], positions=res["positions"],
        cash=res["cash"], capital_initial=capital,
        date_debut=dates[0], date_fin=dates[-1],
        poids_utilises={j["debut"]: j["poids"] for j in journal},
        benchmark=bench,
    )


# ============================================================================
# Contrôle de causalité
# ============================================================================

def test_causalite(prix: dict, Indicateurs, p: ParamsPF, secteurs,
                   date_test, poids: dict) -> bool:
    """Vérifie qu'à la date T, le carnet est IDENTIQUE selon qu'on dispose de
    l'historique complet ou seulement des données jusqu'à T.

    C'est LE test de non-anticipation : si les deux carnets diffèrent, une
    information postérieure à T a fui dans la décision.
    """
    T = pd.Timestamp(date_test)
    p2 = ParamsPF(**{**p.__dict__, "poids": poids})

    pan_complet = construire_panels(prix, Indicateurs, p2, generateur=features_orientees)
    book_complet = Portefeuille(pan_complet, p2, secteurs).book(date=T)

    prix_tronque = {t: df.loc[:T] for t, df in prix.items()}
    pan_tronque = construire_panels(prix_tronque, Indicateurs, p2,
                                    generateur=features_orientees)
    book_tronque = Portefeuille(pan_tronque, p2, secteurs).book(date=T)

    if len(book_complet) != len(book_tronque):
        return False
    if len(book_complet) == 0:
        return True
    cols = ["ticker", "poids_%", "stop"]
    a = book_complet[cols].sort_values("ticker").reset_index(drop=True)
    b = book_tronque[cols].sort_values("ticker").reset_index(drop=True)
    return a.ticker.equals(b.ticker) and np.allclose(
        a[["poids_%", "stop"]], b[["poids_%", "stop"]], atol=1e-6)
