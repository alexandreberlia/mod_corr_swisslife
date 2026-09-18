"""
carnet.py — le carnet d'ordres à passer à la prochaine ouverture.

QUESTION
"Sur la base de la clôture d'hier, qu'est-ce que j'achète et je vends ce matin ?"

MÉCANIQUE
Pour savoir quoi faire aujourd'hui, il faut connaître les positions qu'on
détiendrait — donc rejouer l'historique depuis une date de départ. On ne peut pas
répondre "que dois-je acheter" sans savoir ce qu'on a déjà : le nombre de lignes
libres, le cash disponible et les stops en cours en dépendent.

Le script rejoue donc la stratégie jusqu'à la dernière clôture connue, puis prend
la décision de ce jour-là — celle qui s'exécute à l'ouverture suivante.

IMPORTANT
Les quantités sont calculées au dernier cours de CLÔTURE. Le prix d'ouverture
sera différent. Passe des ordres en MONTANT (colonne `montant`) plutôt qu'en
quantité, ou recalcule la quantité sur le prix d'ouverture réel.
"""

import numpy as np
import pandas as pd

from bootstrap import ParamsBS, simuler_fenetre


def carnet_du_jour(prix: dict, Indicateurs, panier_entree, panier_sortie,
                   date_depart, p: ParamsBS = None, capital: float = 10_000.0,
                   generateur=None, verbose: bool = True) -> dict:
    """Rejoue la stratégie depuis `date_depart` et renvoie les ordres du jour.

    date_depart : quand le portefeuille a été (ou aurait été) ouvert. L'historique
                  antérieur sert à amorcer les indicateurs, il n'est pas tradé.
    """
    from portefeuille import ParamsPF, construire_panels
    from features import features_completes

    p = p or ParamsBS(capital=capital)
    p.capital = capital
    gen = generateur or features_completes

    if verbose:
        print("Construction des panels…")
    panels = construire_panels(prix, Indicateurs,
                               ParamsPF(min_titres=p.min_titres), generateur=gen)
    cl, op_ = panels["close"], panels["open"]
    idx = cl.index

    d0 = pd.Timestamp(date_depart)
    d0 = idx[idx >= d0][0] if (idx >= d0).any() else idx[0]
    d_fin = idx[-1]

    sc_in = panier_entree.calculer(panels, p.min_titres, appliquer_masque=True)
    sc_out = panier_sortie.calculer(panels, p.min_titres, appliquer_masque=False)

    if verbose:
        print(f"Rejeu {d0.date()} → {d_fin.date()} ({len(cl.loc[d0:d_fin])} séances)…")

    r = simuler_fenetre(sc_in, sc_out, panels, d0, d_fin, p,
                        journal=True, decision_finale=True)
    if r is None:
        raise ValueError("période trop courte")

    pos = r["positions_vives"]
    att = r["ordres_en_attente"]
    equity = r["equity"]

    # ---------- lignes de vente ----------
    ventes = []
    for t in att.get("ventes", []):
        s = pos.get(t)
        if s is None:
            continue
        px = cl.loc[d_fin, t]
        frais_in = s.get("frais_in", 0.0)
        ventes.append({
            "ticker": t, "action": "VENDRE", "qty": round(s["qty"], 4),
            "px_cloture": round(px, 2),
            "montant_estime": round(s["qty"] * px, 2),
            "px_entree": round(s["px_in"], 2),
            "pnl_latent": round((px - s["px_in"]) * s["qty"] - frais_in, 2),
            "ret_%": round(((px - s["px_in"]) * s["qty"] - frais_in)
                           / (s["px_in"] * s["qty"]) * 100, 2),
            "jours": s.get("held", 0),
            "score_sortie": round(float(sc_out.loc[d_fin, t]), 3),
        })

    # ---------- lignes d'achat ----------
    achats = []
    for t, montant in sorted(att.get("achats", {}).items(), key=lambda kv: -kv[1]):
        px, a = cl.loc[d_fin, t], panels["atr"].loc[d_fin, t]
        if np.isnan(px) or np.isnan(a) or px <= 0:
            continue
        qty = montant / px
        dist = p.stop_atr * a
        achats.append({
            "ticker": t, "action": "ACHETER",
            "montant": round(montant, 2),
            "px_cloture": round(px, 2),
            "qty_indicative": round(qty, 4),
            "stop_indicatif": round(px - dist, 2),
            "risque_€": round(qty * dist, 2),
            "risque_%": round(qty * dist / equity * 100, 2),
            "atr_%": round(a / px * 100, 2),
            "score_entree": round(float(sc_in.loc[d_fin, t]), 3),
        })

    # ---------- positions conservées ----------
    gardees = []
    vendus = set(att.get("ventes", []))
    for t, s in pos.items():
        if t in vendus:
            continue
        px = cl.loc[d_fin, t]
        frais_in = s.get("frais_in", 0.0)
        gardees.append({
            "ticker": t, "qty": round(s["qty"], 4),
            "px_entree": round(s["px_in"], 2), "px_cloture": round(px, 2),
            "valeur": round(s["qty"] * px, 2),
            "stop_en_cours": round(s["stop"], 2),
            "marge_au_stop_%": round((px / s["stop"] - 1) * 100, 2),
            "pnl_latent": round((px - s["px_in"]) * s["qty"] - frais_in, 2),
            "jours": s.get("held", 0),
            "score_sortie": round(float(sc_out.loc[d_fin, t]), 3)
                            if not np.isnan(sc_out.loc[d_fin, t]) else np.nan,
        })

    return {
        "date_cloture": d_fin, "date_execution": "prochaine ouverture",
        "equity": equity, "cash": r["cash"],
        "ventes": pd.DataFrame(ventes), "achats": pd.DataFrame(achats),
        "positions": pd.DataFrame(gardees),
        "couverture": int(sc_in.loc[d_fin].notna().sum()),
        "equity_curve": r["equity_curve"], "trades": r["trades"],
        "capital_initial": capital,
    }


def rapport_carnet(c: dict) -> str:
    L = []
    L.append("=" * 88)
    L.append(f"CARNET D'ORDRES — décision sur la clôture du {c['date_cloture'].date()}")
    L.append(f"                  exécution à la PROCHAINE OUVERTURE")
    L.append("=" * 88)

    pnl = c["equity"] - c["capital_initial"]
    L.append(f"\nPortefeuille   {c['equity']:>12,.2f}   "
             f"(P&L {pnl:+,.2f} · {pnl / c['capital_initial'] * 100:+.2f} %)")
    L.append(f"  cash         {c['cash']:>12,.2f}")
    L.append(f"  titres       {c['equity'] - c['cash']:>12,.2f}   "
             f"({len(c['positions'])} lignes)")
    L.append(f"\nTitres éligibles à l'entrée ce jour : {c['couverture']}")

    if len(c["ventes"]):
        L.append("\n" + "─" * 88)
        L.append(f"VENDRE ({len(c['ventes'])})")
        L.append("─" * 88)
        L.append(c["ventes"].to_string(index=False))
    else:
        L.append("\nVENDRE : aucune")

    if len(c["achats"]):
        L.append("\n" + "─" * 88)
        L.append(f"ACHETER ({len(c['achats'])})")
        L.append("─" * 88)
        L.append(c["achats"].to_string(index=False))
        L.append("\n  Les quantités sont indicatives : calculées sur la clôture, pas")
        L.append("  sur l'ouverture d'exécution. Passer des ordres en MONTANT, ou")
        L.append("  recalculer qty = montant / prix d'ouverture réel.")
        L.append("  Le stop est à recalculer de même sur le prix d'exécution :")
        L.append("     stop = px_execution − stop_atr × ATR")
    else:
        L.append("\nACHETER : aucune")

    if len(c["positions"]):
        L.append("\n" + "─" * 88)
        L.append(f"CONSERVER ({len(c['positions'])})")
        L.append("─" * 88)
        L.append(c["positions"].to_string(index=False))
        proches = c["positions"][c["positions"]["marge_au_stop_%"] < 3]
        if len(proches):
            L.append(f"\n  ATTENTION : {len(proches)} ligne(s) à moins de 3 % du stop "
                     f"({', '.join(proches.ticker)})")

    L.append("\n" + "=" * 88)
    L.append("À VÉRIFIER AVANT DE PASSER LES ORDRES")
    L.append("=" * 88)
    L.append("""
  - Les données vont-elles bien jusqu'à la dernière clôture ? (yfinance a du
    retard en cours de séance et le jour même après la clôture)
  - Les stops en cours doivent être replacés chez le courtier : le backtest les
    applique en intrajournalier, pas ton broker s'ils ne sont pas passés.
  - Le montant total des achats ne doit pas dépasser le cash après ventes.
    Le calcul en tient compte, mais les prix d'ouverture peuvent décaler.
""")
    return "\n".join(L)
