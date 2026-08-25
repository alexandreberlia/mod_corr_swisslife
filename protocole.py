"""
protocole.py — enchaînement des 7 étapes.

  1. paniers d'entrée / sortie          -> paniers.py
  2. découpage train/test + horizons
  3. bootstrap par fenêtres disjointes  -> bootstrap.py
  4. P&L moyen par couple et horizon
  5. sélection du meilleur couple par horizon
  6. fusion décorrélée avec héritage    -> fusion.py
  7. validation sur la période de TEST, sur 4 objets :
     le panier fusionné + les 3 couples retenus

POURQUOI 4 OBJETS
Le panier fusionné est un objet NEUF, jamais mesuré aux étapes 3-5. La fusion
peut diluer ce qui marchait. On le compare donc aux couples d'origine sur les
mêmes fenêtres de test : s'il ne les bat pas, la fusion a détruit de la valeur.
"""

import numpy as np
import pandas as pd

from bootstrap import (ParamsBS, HORIZONS, bootstrap_tous, bootstrap_couple,
                       fenetres_disjointes, resume_bootstrap)
from fusion import fusionner_couples


def valider_test(objets: dict, panels: dict, horizons: dict, debut, fin,
                 p: ParamsBS, marge_init: int = 300,
                 verbose: bool = True) -> pd.DataFrame:
    """Étape 7 : chaque objet est rejoué sur les fenêtres de TEST.

    objets : {libelle: (panier_entree, panier_sortie)}
    """
    idx = panels["close"].index
    lignes = []
    for nom_h, h in horizons.items():
        fen = fenetres_disjointes(idx, h, debut, fin, marge_init)
        if verbose:
            print(f"  {nom_h} (h={h}) : {len(fen)} fenêtres de test")
        if not fen:
            continue
        for lib, (pi, po) in objets.items():
            r = bootstrap_couple(pi, po, panels, h, fen, p)
            if r is None:
                continue
            r.pop("detail", None)
            r["objet"] = lib
            r["groupe"] = nom_h
            lignes.append(r)
    return pd.DataFrame(lignes)


def lancer(paniers_in: list, paniers_out: list, panels: dict,
           split, horizons: dict = None, p: ParamsBS = None,
           seuil_corr: float = 0.8, max_dures: int = 4,
           marge_init: int = 300, verbose: bool = True) -> dict:
    """Protocole complet. `split` sépare entraînement et test."""
    p = p or ParamsBS()
    horizons = horizons or HORIZONS
    split = pd.Timestamp(split)
    idx = panels["close"].index

    # ---------------- étapes 2-3-4 : bootstrap sur l'entraînement ----------
    if verbose:
        print("=" * 100)
        print(f"ÉTAPES 2-4 — BOOTSTRAP SUR L'ENTRAÎNEMENT (jusqu'au {split.date()})")
        print("=" * 100)
        print("  Fenêtres DISJOINTES : chaque tirage est indépendant, l'écart-type")
        print("  du P&L moyen est donc honnête (pas de recouvrement).")
    train = bootstrap_tous(paniers_in, paniers_out, panels, horizons,
                           fin=split, p=p, marge_init=marge_init, verbose=verbose)
    if train["tableau"].empty:
        return {"ERREUR": "aucun couple exploitable sur l'entraînement"}
    if verbose:
        print()
        print(resume_bootstrap(train, top=4))

    # ---------------- étape 6 : fusion ------------------------------------
    fus = fusionner_couples(train["meilleurs"], paniers_in, paniers_out,
                            panels, seuil_corr, max_dures,
                            fin=split, verbose=verbose)

    # ---------------- étape 7 : validation sur le test --------------------
    idx_in = {q.nom: q for q in paniers_in}
    idx_out = {q.nom: q for q in paniers_out}

    objets = {"FUSION": (fus["panier_entree"], fus["panier_sortie"])}
    for g, m in train["meilleurs"].items():
        pi, po = idx_in.get(m["entree"]), idx_out.get(m["sortie"])
        if pi is not None and po is not None:
            objets[f"retenu {g}"] = (pi, po)

    if verbose:
        print("\n" + "=" * 100)
        print("ÉTAPE 7 — VALIDATION SUR LA PÉRIODE DE TEST")
        print("=" * 100)
        print(f"  {len(objets)} objets : {list(objets)}")
    test = valider_test(objets, panels, horizons, split, idx[-1], p,
                        marge_init=0, verbose=verbose)

    return {"train": train, "fusion": fus, "test": test, "objets": objets,
            "split": split}


def resume_final(res: dict) -> str:
    """Comparaison entraînement / test, objet par objet."""
    if "ERREUR" in res:
        return res["ERREUR"]
    te = res["test"]
    if te.empty:
        return "Aucune fenêtre de test exploitable — période trop courte."

    L = ["=" * 100, "RÉSULTATS SUR LA PÉRIODE DE TEST", "=" * 100]
    cols = ["objet", "pnl_moyen_%", "pnl_median_%", "pnl_std_%", "t_stat",
            "%_fenetres_positives", "n_fenetres", "trades_moy", "couverture_moy"]

    tr = res["train"]["tableau"]
    for g in te.groupe.unique():
        L.append(f"\nHORIZON {g.upper()}")
        L.append("-" * 100)
        sous = te[te.groupe == g].sort_values("pnl_moyen_%", ascending=False)
        L.append(sous[cols].round(3).to_string(index=False))

        m = res["train"]["meilleurs"].get(g)
        f = sous[sous.objet == "FUSION"]
        if m is not None and not f.empty:
            pnl_te = float(f.iloc[0]["pnl_moyen_%"])
            L.append(f"\n  entraînement (meilleur couple) : {m['pnl']:+.2f} %")
            L.append(f"  test (fusion)                  : {pnl_te:+.2f} %")
            ecart = m["pnl"] - pnl_te
            L.append(f"  écart train - test             : {ecart:+.2f} %"
                     + ("   -> surajustement probable" if ecart > abs(pnl_te) else ""))

    L.append("\n" + "=" * 100)
    L.append("LECTURE")
    L.append("=" * 100)
    L.append("""
  - Le P&L de test n'a JAMAIS servi au choix : c'est ce qui le rend lisible.
  - Si FUSION ne bat pas les couples retenus, la fusion a dilué le signal :
    préférer le couple d'origine.
  - t_stat : les fenêtres étant disjointes, chacune est une observation
    indépendante. Mais on retient le max de N couples à l'étape 5, donc le
    seuil réel est supérieur à |t| = 2. Calibrer par permutation si le
    résultat est limite.
  - À l'horizon long, le nombre de fenêtres est faible (12-14 sur 10 ans) :
    l'écart-type est large et le classement peu fiable. Le court terme est
    de loin le plus solide statistiquement.
  - couverture_moy : titres passant les barrières dures. Sous 5, le panier
    est marqué non exploitable — durcir les conditions au-delà de 4 fait
    tomber la couverture à zéro (mesuré).
""")
    return "\n".join(L)
