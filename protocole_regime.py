"""
protocole_regimes.py — un sous-algo par régime d'activité macro.

PRINCIPE
Au lieu d'un algo unique, trois : chacun entraîné sur les seules dates de son
régime (high / medium / low). En production, on lit le score macro du jour et on
applique le panier correspondant.

DÉCOUPAGE
Le train/test se fait PAR ÉPISODES, régime par régime : au sein de chaque régime,
les épisodes sont triés chronologiquement et les plus récents (30 %) vont au test.
Cela garantit que les trois régimes disposent d'une période de test — un découpage
chronologique global ne le garantit pas (si les 30 % récents sont entièrement
"high", medium et low ne seraient jamais testés).

LE TEST QUI DÉCIDE SI LA SUBDIVISION PAIE
Subdiviser triple le nombre de sélections : trois gagnants retenus au lieu d'un,
sur des échantillons plus petits. Le biais de sélection augmente mécaniquement.
La seule façon de savoir si l'idée apporte quelque chose est de comparer, SUR LES
MÊMES FENÊTRES DE TEST, le panier spécialisé au panier global entraîné sur tout
l'échantillon. Si le spécialisé ne bat pas le global, la subdivision n'a fait
qu'ajouter des degrés de liberté.
"""

import numpy as np
import pandas as pd

from bootstrap import ParamsBS, HORIZONS, bootstrap_tous, bootstrap_couple, resume_bootstrap
from fusion import fusionner_couples
from regimes import (REGIMES, episodes, split_episodes, fenetres_regime,
                     resume_regimes)


# ============================================================================
# Un régime
# ============================================================================

def lancer_regime(paniers_in: list, paniers_out: list, panels: dict,
                  serie_regime: pd.Series, regime: str,
                  horizons: dict = None, p: ParamsBS = None,
                  part_train: float = 0.70, marge_init: int = 300,
                  recouvrement_long: bool = True, seuil_corr: float = 0.8,
                  max_dures: int = 4, verbose: bool = True) -> dict:
    """Protocole complet sur un seul régime."""
    p = p or ParamsBS()
    horizons = horizons or HORIZONS
    idx = panels["close"].index

    eps = episodes(serie_regime, regime)
    if not eps:
        return {"ERREUR": f"aucun épisode pour le régime '{regime}'"}
    eps_tr, eps_te = split_episodes(eps, part_train)

    if verbose:
        print(f"\n{'=' * 100}")
        print(f"RÉGIME {regime.upper()}  —  {len(eps)} épisodes "
              f"({len(eps_tr)} train / {len(eps_te)} test)")
        print("=" * 100)
        print(f"  train : {eps_tr[0][0].date()} → {eps_tr[-1][1].date()}, "
              f"{sum(e[2] for e in eps_tr)} séances")
        if eps_te:
            print(f"  test  : {eps_te[0][0].date()} → {eps_te[-1][1].date()}, "
                  f"{sum(e[2] for e in eps_te)} séances")

    # --- fenêtres d'entraînement, contenues dans les épisodes ---
    fen_tr = {}
    for nom_h, h in horizons.items():
        rec = recouvrement_long and h >= 100
        f, ids = fenetres_regime(idx, h, eps_tr, marge_init, recouvrement=rec)
        if len(f) >= 3:
            fen_tr[nom_h] = (f, ids)
        elif verbose:
            print(f"  {nom_h} (h={h}) : {len(f)} fenêtres -> horizon abandonné")
    if not fen_tr:
        return {"ERREUR": f"régime '{regime}' : aucun horizon exploitable "
                          f"(épisodes trop courts pour les h demandés)"}

    train = bootstrap_tous(paniers_in, paniers_out, panels, horizons, p=p,
                           fenetres_par_horizon=fen_tr, verbose=verbose)
    if train["tableau"].empty:
        return {"ERREUR": f"régime '{regime}' : aucun couple exploitable"}
    if verbose:
        print()
        print(resume_bootstrap(train, top=3))

    fus = fusionner_couples(train["meilleurs"], paniers_in, paniers_out, panels,
                            seuil_corr, max_dures, verbose=verbose)

    # --- fenêtres de test ---
    fen_te = {}
    for nom_h, h in horizons.items():
        if nom_h not in fen_tr or not eps_te:
            continue
        rec = h >= 60                       # test plus court : on assouplit
        f, ids = fenetres_regime(idx, h, eps_te, 0, recouvrement=rec)
        if f:
            fen_te[nom_h] = (f, ids)

    return {"regime": regime, "train": train, "fusion": fus,
            "episodes": eps, "eps_train": eps_tr, "eps_test": eps_te,
            "fenetres_train": fen_tr, "fenetres_test": fen_te,
            "panier_entree": fus["panier_entree"],
            "panier_sortie": fus["panier_sortie"]}


# ============================================================================
# Les trois régimes + le global
# ============================================================================

def lancer_tous_regimes(paniers_in: list, paniers_out: list, panels: dict,
                        serie_regime: pd.Series, horizons: dict = None,
                        p: ParamsBS = None, part_train: float = 0.70,
                        marge_init: int = 300, comparer_global: bool = True,
                        verbose: bool = True) -> dict:
    """Trois sous-algos, puis comparaison au panier global sur chaque régime."""
    p = p or ParamsBS()
    horizons = horizons or HORIZONS

    if verbose:
        print("=" * 100)
        print("RÉGIMES D'ACTIVITÉ MACRO")
        print("=" * 100)
        print(resume_regimes(serie_regime).to_string(index=False))
        print("\n  Rappel : l'unité d'indépendance est l'ÉPISODE, pas la fenêtre.")
        print("  Sous 6 épisodes, le classement entre paniers reste fragile quel")
        print("  que soit le nombre de séances.")

    res = {}
    for r in REGIMES:
        res[r] = lancer_regime(paniers_in, paniers_out, panels, serie_regime, r,
                               horizons, p, part_train, marge_init,
                               verbose=verbose)
        if "ERREUR" in res[r] and verbose:
            print(f"  ** {res[r]['ERREUR']}")

    # --- panier global : entraîné sur TOUT, tous régimes confondus ---
    glob = None
    if comparer_global:
        if verbose:
            print(f"\n{'=' * 100}")
            print("PANIER GLOBAL (entraîné sur tous régimes confondus) — référence")
            print("=" * 100)
        eps_all = [(serie_regime.index[0], serie_regime.index[-1],
                    len(serie_regime))]
        fen_all = {}
        for nom_h, h in horizons.items():
            f, ids = fenetres_regime(panels["close"].index, h, eps_all, marge_init)
            n_tr = max(int(len(f) * part_train), 1)
            if n_tr >= 3:
                fen_all[nom_h] = (f[:n_tr], ids[:n_tr])
        if fen_all:
            tg = bootstrap_tous(paniers_in, paniers_out, panels, horizons, p=p,
                                fenetres_par_horizon=fen_all, verbose=False)
            if not tg["tableau"].empty:
                fg = fusionner_couples(tg["meilleurs"], paniers_in, paniers_out,
                                       panels, verbose=False)
                glob = {"train": tg, "fusion": fg,
                        "panier_entree": fg["panier_entree"],
                        "panier_sortie": fg["panier_sortie"]}
                if verbose:
                    print(f"  couples retenus : "
                          f"{[(g, m['entree'], m['sortie']) for g, m in tg['meilleurs'].items()]}")

    # --- validation : spécialisé vs global, sur les mêmes fenêtres de test ---
    lignes = []
    for r in REGIMES:
        rr = res.get(r)
        if not rr or "ERREUR" in rr or not rr["fenetres_test"]:
            continue
        objets = {"specialise": (rr["panier_entree"], rr["panier_sortie"])}
        if glob:
            objets["global"] = (glob["panier_entree"], glob["panier_sortie"])

        for nom_h, (fen, ids) in rr["fenetres_test"].items():
            h = horizons[nom_h]
            for lib, (pi, po) in objets.items():
                out = bootstrap_couple(pi, po, panels, h, fen, p, ids)
                if out is None:
                    continue
                out.pop("detail", None)
                out.update({"regime": r, "objet": lib, "groupe": nom_h})
                lignes.append(out)

    return {"regimes": res, "global": glob, "test": pd.DataFrame(lignes)}


# ============================================================================
# Rapport
# ============================================================================

def resume_regimes_final(res: dict) -> str:
    te = res["test"]
    L = ["=" * 100, "VALIDATION — SPÉCIALISÉ vs GLOBAL, sur les fenêtres de TEST",
         "=" * 100]
    if te.empty:
        L.append("\nAucune fenêtre de test exploitable : les épisodes réservés au")
        L.append("test sont plus courts que les horizons demandés. Réduire les")
        L.append("horizons ou abaisser part_train.")
        return "\n".join(L)

    cols = [c for c in ["objet", "pnl_moyen_%", "pnl_std_%", "t_stat", "t_groupe",
                        "%_fenetres_positives", "n_fenetres", "n_episodes",
                        "couverture_moy"] if c in te.columns]

    verdicts = []
    for r in sorted(te.regime.unique()):
        L.append(f"\n{'─' * 100}\nRÉGIME {r.upper()}\n{'─' * 100}")
        for g in te[te.regime == r].groupe.unique():
            sous = te[(te.regime == r) & (te.groupe == g)]
            L.append(f"\n  horizon {g}")
            L.append(sous[cols].round(3).to_string(index=False))
            sp = sous[sous.objet == "specialise"]
            gl = sous[sous.objet == "global"]
            if not sp.empty and not gl.empty:
                d = float(sp.iloc[0]["pnl_moyen_%"]) - float(gl.iloc[0]["pnl_moyen_%"])
                verdicts.append((r, g, d))
                L.append(f"  -> écart spécialisé - global : {d:+.2f} %"
                         + ("   la subdivision paie" if d > 0
                            else "   le global fait aussi bien ou mieux"))

    if verdicts:
        gagne = sum(1 for _, _, d in verdicts if d > 0)
        L.append(f"\n{'=' * 100}")
        L.append(f"BILAN : la spécialisation l'emporte dans {gagne}/{len(verdicts)} "
                 f"cas (régime x horizon).")
        moy = np.mean([d for _, _, d in verdicts])
        L.append(f"        écart moyen : {moy:+.2f} %")
        if gagne <= len(verdicts) / 2:
            L.append("        -> la subdivision n'apporte pas : elle a triplé les")
            L.append("           degrés de liberté sans gain mesurable.")

    L.append(f"\n{'=' * 100}")
    L.append("LECTURE")
    L.append("=" * 100)
    L.append("""
  t_groupe   groupé par épisode. C'est le seul honnête : trois fenêtres du même
             épisode partagent le même contexte macro. Toujours < t_stat.
  n_episodes nombre d'épisodes distincts. C'est la vraie taille d'échantillon.
             Sous 6, ne rien conclure.

  Le score macro repose sur des séries RÉVISÉES (FRED). Le classement en régimes
  est donc plus net qu'il n'était possible en temps réel : les résultats sont
  optimistes de ce fait, indépendamment du décalage de publication appliqué.
""")
    return "\n".join(L)


def paniers_finaux(res: dict) -> dict:
    """{regime: (panier_entree, panier_sortie)} — ce qu'on utilise en production."""
    out = {}
    for r, rr in res["regimes"].items():
        if rr and "ERREUR" not in rr:
            out[r] = (rr["panier_entree"], rr["panier_sortie"])
    return out
