"""
bootstrap.py — bootstrap par fenêtres disjointes.

PRINCIPE
Pour un couple (panier d'entrée, panier de sortie) et un horizon h :
  - on découpe la période d'entraînement en fenêtres DISJOINTES de h séances
  - dans chaque fenêtre on simule jour par jour, capital neuf de 10 000 €
  - à t+h on relève le P&L, positions ouvertes valorisées à la clôture SANS
    ordre de vente (donc sans frais de sortie)
  - on moyenne sur toutes les fenêtres

CAUSALITÉ
Les panels sont calculés UNE FOIS sur l'historique complet, mais tous les
indicateurs sont causaux (EMA/ATR/ADX récursifs, rolling glissants) : la valeur
au jour t n'utilise jamais t+1. La contrainte "comme si on n'avait pas les
données futures" est donc satisfaite sans recalcul par fenêtre. Le décalage
signal(t) -> exécution à l'ouverture(t+1) est appliqué dans la boucle.

DEUX CURSEURS
  seuil_entree  score d'entrée minimal (en plus des barrières dures)
  seuil_sortie  score de sortie déclenchant la vente
                0 = neutre | +0.3 = sortie rare | -0.3 = sortie nerveuse

SORTIE ASYMÉTRIQUE (désactivée par défaut)
  sensibilite > 0 abaisse le seuil de sortie quand la position est en gain :
  on verrouille plus vite. C'est un effet de disposition ASSUMÉ — la littérature
  le désigne comme coûteux en suivi de tendance (il coupe les rares gros gains).
  À comparer sur les mêmes fenêtres, pas à trancher par principe.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================================
# Paramètres
# ============================================================================

@dataclass
class ParamsBS:
    capital: float = 10_000.0
    n_long: int = 5
    poids_max: float = 0.25
    poids_min: float = 0.02
    exposition: float = 0.95

    stop_atr: float = 2.5             # stop-loss, en ATR
    trail_atr: float | None = None    # None = pas de trailing (stop fixe)

    seuil_entree: float = 0.0
    seuil_sortie: float = 0.0
    sensibilite: float = 0.0          # sortie asymétrique. 0 = désactivée
    plancher_sortie: float = -0.5     # borne basse du seuil asymétrique

    freq_decision: int = 1            # décisions tous les N jours
    cost_bps: float = 10.0            # aller simple
    min_titres: int = 20
    couverture_min: float = 5.0       # sous ce seuil, panier non exploitable


HORIZONS = {"court": 15, "moyen": 63, "long": 126}


# ============================================================================
# Découpage en fenêtres disjointes
# ============================================================================

def fenetres_disjointes(index: pd.DatetimeIndex, h: int, debut=None, fin=None,
                        marge_init: int = 300, n_max: int = None,
                        graine: int = 0) -> list:
    """Fenêtres consécutives de h séances, SANS RECOUVREMENT.

    marge_init : séances réservées à l'amorçage des indicateurs (mom_12_1 exige
                 252 séances). Aucune fenêtre ne commence avant.
    """
    idx = index
    if debut is not None:
        idx = idx[idx >= pd.Timestamp(debut)]
    if fin is not None:
        idx = idx[idx <= pd.Timestamp(fin)]

    depart = index.get_loc(idx[0]) if len(idx) else 0
    if depart < marge_init:
        pos0 = marge_init
    else:
        pos0 = depart
    pos_fin = index.get_loc(idx[-1]) if len(idx) else len(index) - 1

    fen = []
    p = pos0
    while p + h <= pos_fin:
        fen.append((index[p], index[p + h]))
        p += h

    if n_max and len(fen) > n_max:
        rng = np.random.default_rng(graine)
        keep = sorted(rng.choice(len(fen), n_max, replace=False))
        fen = [fen[i] for i in keep]
    return fen


# ============================================================================
# Simulation d'une fenêtre
# ============================================================================

def simuler_fenetre(sc_in: pd.DataFrame, sc_out: pd.DataFrame, panels: dict,
                    d0, d1, p: ParamsBS) -> dict:
    """Une fenêtre, jour par jour. Renvoie le P&L en % du capital.

    Ordre à chaque barre :
      1. exécution des ordres décidés la veille, à l'OUVERTURE
      2. stop-loss touché en séance (prioritaire sur tout signal)
      3. décision de sortie sur la clôture -> exécutée demain
      4. décision d'entrée sur la clôture -> exécutée demain
    """
    op, hi, lo, cl = panels["open"], panels["high"], panels["low"], panels["close"]
    atr = panels["atr"]
    idx = cl.loc[d0:d1].index
    if len(idx) < 3:
        return None

    cost = p.cost_bps / 10_000.0
    cash = p.capital
    pos = {}                      # ticker -> dict d'état
    frais_tot = 0.0
    n_trades = 0
    ordres = {"out": [], "in": {}}
    couv = []

    for i, d in enumerate(idx):
        if i == 0:
            continue
        dv = idx[i - 1]

        # ---- 1. exécution à l'ouverture ----
        for t in ordres["out"]:
            if t in pos and not np.isnan(op.loc[d, t]):
                px = op.loc[d, t]
                f = pos[t]["qty"] * px * cost
                cash += pos[t]["qty"] * px - f
                frais_tot += f
                pos.pop(t)
                n_trades += 1

        for t, cap in sorted(ordres["in"].items(), key=lambda kv: -kv[1]):
            px, a = op.loc[d, t], atr.loc[dv, t]
            if t in pos or np.isnan(px) or np.isnan(a) or a <= 0 or px <= 0:
                continue
            cap = min(cap, cash / (1 + cost))
            qty = cap / px
            if qty * px < 1:
                continue
            f = qty * px * cost
            cash -= qty * px + f
            frais_tot += f
            pos[t] = {"qty": qty, "px_in": px, "stop": px - p.stop_atr * a,
                      "plus_haut": px, "atr_in": a}
        ordres = {"out": [], "in": {}}

        # ---- 2. stop-loss en séance (prioritaire) ----
        for t in list(pos):
            s = pos[t]
            low = lo.loc[d, t]
            if np.isnan(low):
                continue
            if low <= s["stop"]:
                px = min(op.loc[d, t], s["stop"])     # gap : on subit l'ouverture
                f = s["qty"] * px * cost
                cash += s["qty"] * px - f
                frais_tot += f
                pos.pop(t)
                n_trades += 1
                continue
            if p.trail_atr:
                s["plus_haut"] = max(s["plus_haut"], hi.loc[d, t])
                a = atr.loc[d, t]
                if not np.isnan(a):
                    s["stop"] = max(s["stop"], s["plus_haut"] - p.trail_atr * a)

        # ---- pas de décision le dernier jour : on valorise et on s'arrête ----
        if i == len(idx) - 1 or (i % p.freq_decision):
            continue

        # ---- 3. sorties (score de sortie sur la clôture du jour) ----
        so = sc_out.loc[d] if d in sc_out.index else None
        if so is not None:
            for t, s in pos.items():
                v = so.get(t, np.nan)
                if np.isnan(v):
                    continue
                seuil = p.seuil_sortie
                if p.sensibilite > 0 and s["atr_in"] > 0:
                    gain_atr = (cl.loc[d, t] - s["px_in"]) / s["atr_in"]
                    seuil = max(p.plancher_sortie,
                                seuil - p.sensibilite * max(0.0, gain_atr))
                if v > seuil:
                    ordres["out"].append(t)

        # ---- 4. entrées ----
        si = sc_in.loc[d] if d in sc_in.index else None
        if si is None:
            continue
        elig = si.dropna()
        couv.append(len(elig))
        restants = [t for t in pos if t not in ordres["out"]]
        libres = p.n_long - len(restants)
        if libres <= 0:
            continue

        cand = elig[elig > p.seuil_entree].drop(labels=restants, errors="ignore")
        if cand.empty:
            continue
        cand = cand.nlargest(libres)

        libere = sum(pos[t]["qty"] * cl.loc[d, t] for t in ordres["out"]
                     if not np.isnan(cl.loc[d, t]))
        equity = cash + sum(pos[t]["qty"] * cl.loc[d, t] for t in pos
                            if not np.isnan(cl.loc[d, t]))
        dispo = cash + libere
        if dispo < equity * p.poids_min:
            continue

        w = _poids(list(cand.index), d, panels, p)
        if w.empty:
            continue
        besoin = (w * equity).sum()
        k = min(1.0, dispo / besoin) if besoin > 0 else 0.0
        for t, wi in w.items():
            ordres["in"][t] = wi * equity * k

    # ---- valorisation finale : PAS d'ordre de vente ----
    d_fin = idx[-1]
    val_pos = sum(s["qty"] * cl.loc[d_fin, t] for t, s in pos.items()
                  if not np.isnan(cl.loc[d_fin, t]))
    equity = cash + val_pos

    return {
        "debut": idx[0], "fin": d_fin,
        "pnl_pct": (equity / p.capital - 1) * 100,
        "equity": equity,
        "n_trades": n_trades,
        "n_ouvertes": len(pos),
        "frais": frais_tot,
        "frais_pct": frais_tot / p.capital * 100,
        "couverture_moy": float(np.mean(couv)) if couv else 0.0,
        "expo_finale": val_pos / equity if equity > 0 else 0.0,
    }


def _poids(tickers, date, panels, p: ParamsBS) -> pd.Series:
    """Inverse-volatilité => risque en euros identique par ligne, puisque
    risque_i = capital_i x stop_atr x atr_pct_i ∝ (1/atr_pct) x atr_pct."""
    atrp = (panels["atr"].loc[date, tickers] /
            panels["close"].loc[date, tickers]).replace(0, np.nan)
    raw = (1.0 / atrp).replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return pd.Series(dtype=float)
    w = raw / raw.sum() * p.exposition
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


# ============================================================================
# Bootstrap sur un couple de paniers
# ============================================================================

def bootstrap_couple(p_in, p_out, panels: dict, h: int, fenetres: list,
                     p: ParamsBS) -> dict:
    """Un couple (entrée, sortie), un horizon, toutes les fenêtres."""
    sc_in = p_in.calculer(panels, p.min_titres, appliquer_masque=True)
    sc_out = p_out.calculer(panels, p.min_titres, appliquer_masque=False)

    res = [r for d0, d1 in fenetres
           if (r := simuler_fenetre(sc_in, sc_out, panels, d0, d1, p)) is not None]
    if not res:
        return None

    df = pd.DataFrame(res)
    pnl = df.pnl_pct
    couv = df.couverture_moy.mean()
    return {
        "entree": p_in.nom, "sortie": p_out.nom, "horizon": h,
        "pnl_moyen_%": pnl.mean(),
        "pnl_median_%": pnl.median(),
        "pnl_std_%": pnl.std(),
        "pnl_min_%": pnl.min(), "pnl_max_%": pnl.max(),
        "%_fenetres_positives": (pnl > 0).mean() * 100,
        # t de Student : les fenêtres sont disjointes, donc indépendantes
        "t_stat": pnl.mean() / (pnl.std() / np.sqrt(len(pnl))) if pnl.std() > 0 else np.nan,
        "n_fenetres": len(pnl),
        "trades_moy": df.n_trades.mean(),
        "frais_moy_%": df.frais_pct.mean(),
        "couverture_moy": couv,
        "exploitable": couv >= p.couverture_min,
        "detail": df,
    }


def bootstrap_tous(paniers_in: list, paniers_out: list, panels: dict,
                   horizons: dict = None, debut=None, fin=None,
                   p: ParamsBS = None, marge_init: int = 300,
                   verbose: bool = True) -> dict:
    """Produit cartésien entrée x sortie x horizon."""
    p = p or ParamsBS()
    horizons = horizons or HORIZONS
    idx = panels["close"].index

    lignes, detail = [], {}
    for nom_h, h in horizons.items():
        fen = fenetres_disjointes(idx, h, debut, fin, marge_init)
        if verbose:
            print(f"  {nom_h} (h={h}) : {len(fen)} fenêtres disjointes")
        if not fen:
            continue
        for pi in paniers_in:
            for po in paniers_out:
                r = bootstrap_couple(pi, po, panels, h, fen, p)
                if r is None:
                    continue
                cle = (nom_h, pi.nom, po.nom)
                detail[cle] = r.pop("detail")
                r["groupe"] = nom_h
                lignes.append(r)

    tab = pd.DataFrame(lignes)
    if tab.empty:
        return {"tableau": tab, "meilleurs": {}, "detail": detail}

    meilleurs = {}
    for g in tab.groupe.unique():
        sous = tab[(tab.groupe == g) & tab.exploitable]
        if sous.empty:
            sous = tab[tab.groupe == g]
        best = sous.sort_values("pnl_moyen_%", ascending=False).iloc[0]
        meilleurs[g] = {"entree": best.entree, "sortie": best.sortie,
                        "pnl": best["pnl_moyen_%"], "t": best.t_stat,
                        "n_fen": best.n_fenetres, "couverture": best.couverture_moy}

    return {"tableau": tab.sort_values(["groupe", "pnl_moyen_%"],
                                       ascending=[True, False]),
            "meilleurs": meilleurs, "detail": detail}


def resume_bootstrap(res: dict, top: int = 5) -> str:
    tab = res["tableau"]
    if tab.empty:
        return "Aucun couple exploitable."
    L = []
    cols = ["entree", "sortie", "pnl_moyen_%", "pnl_std_%", "t_stat",
            "%_fenetres_positives", "n_fenetres", "trades_moy",
            "couverture_moy", "exploitable"]
    for g in tab.groupe.unique():
        m = res["meilleurs"][g]
        L.append("=" * 100)
        L.append(f"HORIZON {g.upper()}  —  retenu : {m['entree']}  /  {m['sortie']}"
                 f"   (P&L {m['pnl']:+.2f} %, t={m['t']:.2f}, "
                 f"{int(m['n_fen'])} fenêtres)")
        L.append("=" * 100)
        L.append(tab[tab.groupe == g][cols].head(top).round(3).to_string(index=False))
        L.append("")
    L.append("LECTURE")
    L.append("  pnl_moyen_%  P&L moyen sur la fenêtre, en % du capital de départ.")
    L.append("  t_stat       fenêtres disjointes = observations indépendantes.")
    L.append("               |t| > 2 -> P&L moyen non nul. Mais attention au")
    L.append("               nombre d'essais : on retient le max de 9 couples.")
    L.append("  couverture   titres passant les barrières dures. Sous 5, le panier")
    L.append("               est marqué non exploitable.")
    return "\n".join(L)
