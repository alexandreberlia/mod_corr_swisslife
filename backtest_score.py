"""backtest_seuils.py — Strategie d'entree/sortie sur un score d'activite.

    from backtest_seuils import charger, simuler, rapport

    score, prix = charger("scores.xlsx", "spx.csv")
    r = simuler(score, prix, n_entree=30, n_sortie=70)
    print(rapport(r))

REGLE SIMULEE
-------------
On parcourt le score dans l'ordre chronologique :

    hors position, score FRANCHIT n_entree vers le BAS   -> achat
    en position,   score FRANCHIT n_sortie vers le HAUT  -> vente, P&L enregistre

Le declenchement se fait au FRANCHISSEMENT, pas au niveau. Sans cela, un score
qui reste sous le seuil dix trimestres declencherait dix achats successifs.
On compare donc la valeur courante a la precedente : il faut etre passe de
au-dessus a en-dessous.

TROIS POINTS QUI CHANGENT LE RESULTAT
--------------------------------------
1. DECALAGE D'EXECUTION. Le score d'un trimestre n'est pas connu a la fin de ce
   trimestre : les composantes sont publiees avec retard et revisees. Executer
   au prix du trimestre du signal revient a acheter avec une information
   indisponible. `decalage=1` (defaut) execute au trimestre SUIVANT. Mettre 0
   donne une performance flatteuse et fausse.

2. POSITION UNIQUE. Une seule position a la fois : pas de rachat tant qu'on n'a
   pas vendu. C'est la regle la plus simple, et elle evite d'avoir a gerer un
   dimensionnement.

3. POSITION OUVERTE EN FIN D'ECHANTILLON. Si le dernier signal est un achat
   sans vente, le trade reste OUVERT. Il figure dans `position_ouverte` et n'est
   PAS compte dans le P&L — le compter au dernier prix connu reviendrait a
   supposer une sortie qui n'a pas eu lieu.

CE QUE CE BACKTEST NE DIT PAS
------------------------------
Il ne prouve pas qu'une strategie fonctionne. Avec deux seuils libres, on trouve
toujours un couple qui a bien marche sur l'historique : c'est du surajustement,
pas un resultat. `balayer` produit la surface complete pour le montrer — si le
P&L n'est eleve que sur un ilot etroit de la grille, le reglage ne survivra pas.

Le P&L est calcule en points d'indice et en rendement, sans frais, sans
dividendes, sans cout de portage.

Dependances : numpy, pandas
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Entrees
# ---------------------------------------------------------------------------

def _index_periode(idx, freq="Q") -> pd.PeriodIndex:
    """Ramene un index quelconque en PeriodIndex."""
    if isinstance(idx, pd.PeriodIndex):
        return idx
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_period(freq)
    try:
        return pd.PeriodIndex(idx, freq=freq)
    except Exception:
        return pd.PeriodIndex(pd.to_datetime(pd.Index(idx).astype(str)), freq=freq)


def charger(fichier_score: str, fichier_prix: str, colonne_score: str | None = None,
            colonne_prix: str | None = None, feuille=0,
            freq: str = "Q") -> tuple[pd.Series, pd.Series]:
    """Lit le score (Excel) et le prix (CSV), les ramene a la meme frequence.

    La colonne est detectee automatiquement s'il n'y en a qu'une de numerique ;
    sinon il faut la nommer. La premiere colonne est prise pour les dates.
    """
    s = pd.read_excel(fichier_score, sheet_name=feuille)
    p = pd.read_csv(fichier_prix)

    def extraire(d, col, quoi):
        dates = d.columns[0]
        num = [c for c in d.columns[1:] if pd.api.types.is_numeric_dtype(d[c])]
        if col is None:
            if len(num) != 1:
                raise ValueError(f"Preciser la colonne {quoi}. Candidates : {num}")
            col = num[0]
        out = pd.Series(pd.to_numeric(d[col], errors="coerce").to_numpy(),
                        index=_index_periode(d[dates], freq), name=col).dropna()
        return out[~out.index.duplicated(keep="last")].sort_index()

    score = extraire(s, colonne_score, "de score")
    prix = extraire(p, colonne_prix, "de prix")
    # si le prix est plus fin que le score, on garde la derniere valeur de chaque
    # periode : c'est le prix auquel on pourrait effectivement traiter
    if len(prix) > len(score) * 1.5:
        prix = prix.groupby(prix.index).last()
    return score, prix


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simuler(score: pd.Series, prix: pd.Series, n_entree: float, n_sortie: float,
            decalage: int = 1) -> dict:
    """Parcourt le score et enregistre les allers-retours.

    Returns
    -------
    dict avec
        achats, ventes   listes des prix d'execution
        pnl              liste des differences vente - achat
        trades           DataFrame detaille (dates, prix, P&L, rendement)
        position_ouverte dict ou None si le dernier achat n'a pas ete solde
    """
    if n_entree >= n_sortie:
        raise ValueError(f"n_entree ({n_entree}) doit etre < n_sortie ({n_sortie}).")

    d = pd.concat([score.rename("s"), prix.rename("p")], axis=1)
    d["p_exec"] = d["p"].shift(-decalage)      # execution differee
    d = d.dropna(subset=["s"])
    if len(d) < 10:
        raise ValueError(f"Echantillon trop court apres alignement : {len(d)}.")

    achats, ventes, pnl, trades = [], [], [], []
    en_position = False
    prix_achat, date_achat = None, None
    prec = None

    for date, row in d.iterrows():
        s, pe = row["s"], row["p_exec"]
        if prec is not None and np.isfinite(pe):
            franchit_bas = prec >= n_entree and s < n_entree
            franchit_haut = prec <= n_sortie and s > n_sortie
            if not en_position and franchit_bas:
                achats.append(float(pe))
                prix_achat, date_achat = float(pe), date
                en_position = True
            elif en_position and franchit_haut:
                ventes.append(float(pe))
                g = float(pe) - prix_achat
                pnl.append(g)
                trades.append(dict(entree=str(date_achat), sortie=str(date),
                                   prix_achat=round(prix_achat, 2),
                                   prix_vente=round(float(pe), 2),
                                   pnl=round(g, 2),
                                   rendement_pct=round(100 * g / prix_achat, 2),
                                   duree=(date - date_achat).n))
                en_position = False
                prix_achat, date_achat = None, None
        prec = s

    ouverte = (dict(entree=str(date_achat), prix_achat=round(prix_achat, 2))
               if en_position else None)
    return dict(achats=achats, ventes=ventes, pnl=pnl,
                trades=pd.DataFrame(trades), position_ouverte=ouverte,
                n_entree=n_entree, n_sortie=n_sortie, decalage=decalage,
                periode=(str(d.index.min()), str(d.index.max())),
                prix_debut=float(d["p"].dropna().iloc[0]),
                prix_fin=float(d["p"].dropna().iloc[-1]))


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------

def rapport(r: dict) -> str:
    t = r["trades"]
    L = ["=" * 62,
         f"SEUILS  entree < {r['n_entree']}   sortie > {r['n_sortie']}"
         f"   (execution a t+{r['decalage']})",
         f"PERIODE {r['periode'][0]} -> {r['periode'][1]}", "=" * 62]
    if len(t) == 0:
        L.append("Aucun aller-retour complet sur la periode.")
    else:
        gains = t.pnl
        L += [f"  trades            : {len(t)}",
              f"  gagnants          : {(gains > 0).sum()} "
              f"({100 * (gains > 0).mean():.0f} %)",
              f"  P&L total         : {gains.sum():+.1f} points",
              f"  P&L moyen         : {gains.mean():+.1f} points "
              f"({t.rendement_pct.mean():+.1f} %)",
              f"  meilleur / pire   : {gains.max():+.1f} / {gains.min():+.1f}",
              f"  duree moyenne     : {t.duree.mean():.1f} periodes"]
        # reference : acheter au debut et conserver
        bh = r["prix_fin"] - r["prix_debut"]
        L += ["", f"  reference achat-conservation : {bh:+.1f} points "
                  f"({100 * bh / r['prix_debut']:+.0f} %)",
              "  (le P&L en points n'est PAS comparable directement : la "
              "strategie",
              "   n'est investie qu'une partie du temps, et sur des niveaux "
              "d'indice",
              "   differents. Comparer les rendements par trade.)"]
    if r["position_ouverte"]:
        o = r["position_ouverte"]
        L += ["", f"  POSITION OUVERTE depuis {o['entree']} a {o['prix_achat']} "
                  "— non comptee dans le P&L."]
    return "\n".join(L)


def balayer(score: pd.Series, prix: pd.Series,
            entrees=range(10, 51, 5), sorties=range(50, 96, 5),
            decalage: int = 1) -> pd.DataFrame:
    """P&L pour chaque couple de seuils.

    A lire comme une SURFACE, pas comme un classement. Si le P&L n'est eleve que
    sur un ilot etroit, le reglage est un artefact : un bon parametrage doit
    former un plateau, c'est-a-dire rester correct quand on bouge un peu les
    seuils.
    """
    out = []
    for e in entrees:
        for s in sorties:
            if e >= s:
                continue
            try:
                r = simuler(score, prix, e, s, decalage)
            except Exception:
                continue
            t = r["trades"]
            out.append(dict(n_entree=e, n_sortie=s, trades=len(t),
                            pnl_total=round(sum(r["pnl"]), 1) if r["pnl"] else 0.0,
                            rdt_moyen=round(t.rendement_pct.mean(), 2) if len(t) else np.nan,
                            taux_gagnants=round(100 * (t.pnl > 0).mean()) if len(t) else np.nan))
    return pd.DataFrame(out).sort_values("pnl_total", ascending=False)
