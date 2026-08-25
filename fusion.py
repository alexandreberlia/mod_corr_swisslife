"""
fusion.py — étape 6 : fusionner les paniers retenus en un panier final.

PROCÉDURE
  1. rassembler tous les indicateurs des paniers retenus (un par horizon)
  2. compter la FRÉQUENCE d'apparition de chacun
  3. matrice de corrélation cross-sectionnelle. Au-delà de |0.8|, un seul survit :
     le plus fréquent, départagé par le P&L du panier d'origine.
     Le survivant HÉRITE de la fréquence des évincés — il n'est pas lésé d'avoir
     eu des doublons.
  4. poids = rang percentile de la fréquence, rapporté au plus fréquent
  5. deux fusions PARALLÈLES : une pour l'entrée, une pour la sortie

CONDITIONS DURES
Une condition dure n'est conservée que si elle apparaît dans la MAJORITÉ des
paniers retenus (elle fait consensus), et le plafond de 4 est respecté en gardant
les plus fréquentes. Cumuler les barrières de tous les paniers ferait s'effondrer
la couverture — mesuré : 0 titre médian au-delà de 8 conditions.

AVERTISSEMENT
Le panier fusionné est un objet NEUF, jamais testé au bootstrap. Rien ne garantit
qu'il batte le meilleur des paniers d'origine : la fusion peut diluer ce qui
marchait. D'où l'étape 7 lancée sur 4 objets (fusionné + les 3 retenus).
"""

import numpy as np
import pandas as pd

from paniers import Panier


# ============================================================================
# Corrélation entre indicateurs
# ============================================================================

def matrice_correlation(indicateurs: list, panels: dict,
                        debut=None, fin=None) -> pd.DataFrame:
    """Corrélation de rang MOYENNE entre indicateurs, cross-sectionnelle.

    On corrèle date par date à travers les titres, puis on moyenne : c'est la
    corrélation qui compte pour un système de classement, pas la corrélation
    temporelle titre par titre.
    """
    dispo = [f for f in indicateurs if f in panels]
    n = len(dispo)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = panels[dispo[i]], panels[dispo[j]]
            if debut is not None:
                a, b = a.loc[debut:], b.loc[debut:]
            if fin is not None:
                a, b = a.loc[:fin], b.loc[:fin]
            c = a.corrwith(b, axis=1, method="spearman").mean()
            M[i, j] = M[j, i] = 0.0 if np.isnan(c) else c
    return pd.DataFrame(M, index=dispo, columns=dispo)


# ============================================================================
# Fusion
# ============================================================================

def fusionner(paniers_retenus: list, pnl_par_panier: dict, panels: dict,
              seuil_corr: float = 0.8, max_dures: int = 4,
              sens: str = "entree", nom: str = None,
              debut=None, fin=None, verbose: bool = True) -> dict:
    """Fusionne des paniers de même sens en un panier unique.

    paniers_retenus : liste d'objets Panier (un par horizon)
    pnl_par_panier  : {nom_panier: pnl} — sert à départager à fréquence égale
    """
    if not paniers_retenus:
        raise ValueError("aucun panier à fusionner")

    # ---- 1-2. fréquences et spécifications ----
    freq, specs, freq_dures, specs_dures = {}, {}, {}, {}
    for pk in paniers_retenus:
        for f, (op, s, s2, w) in pk.specs_score().items():
            freq[f] = freq.get(f, 0) + 1
            specs.setdefault(f, []).append((op, s, s2, w, pk.nom))
        for f, spec in pk.dures.items():
            freq_dures[f] = freq_dures.get(f, 0) + 1
            specs_dures.setdefault(f, []).append((spec, pk.nom))

    if not freq:
        raise ValueError("aucun indicateur de score dans les paniers retenus")

    # ---- 3. décorrélation avec héritage ----
    corr = matrice_correlation(list(freq), panels, debut, fin)

    def cle(f):
        return (freq[f], pnl_par_panier.get(specs[f][0][4], 0.0))

    ordre = sorted(freq, key=cle, reverse=True)
    gardes, evinces, herite = [], {}, dict(freq)

    for f in ordre:
        if f not in corr.columns:
            gardes.append(f)
            continue
        proche = next((g for g in gardes
                       if abs(corr.loc[f, g]) >= seuil_corr), None)
        if proche is None:
            gardes.append(f)
        else:
            evinces[f] = (proche, float(corr.loc[f, proche]))
            herite[proche] += freq[f]          # héritage de fréquence
            herite.pop(f, None)

    # ---- 4. poids = rang percentile de la fréquence héritée ----
    s_freq = pd.Series({f: herite[f] for f in gardes}, dtype=float)
    if len(s_freq) == 1:
        poids = {s_freq.index[0]: 1.0}
    else:
        rangs = s_freq.rank(pct=True)          # 0-1, le plus fréquent = 1.0
        poids = (rangs / rangs.max()).round(3).to_dict()

    # ---- spécification retenue : celle du panier au meilleur P&L ----
    score_final = {}
    for f in gardes:
        cands = sorted(specs[f], key=lambda x: pnl_par_panier.get(x[4], 0.0),
                       reverse=True)
        op, s, s2, _, _ = cands[0]
        score_final[f] = (op, s, s2, poids[f]) if s2 is not None else (op, s, poids[f])

    # ---- conditions dures : consensus majoritaire, plafonné ----
    n_pan = len(paniers_retenus)
    dures_final = {}
    if sens == "entree" and freq_dures:
        majoritaires = {f: c for f, c in freq_dures.items() if c > n_pan / 2}
        if not majoritaires:                    # aucun consensus -> les plus fréquentes
            majoritaires = freq_dures
        top = sorted(majoritaires, key=lambda f: -freq_dures[f])[:max_dures]
        for f in top:
            cands = sorted(specs_dures[f],
                           key=lambda x: pnl_par_panier.get(x[1], 0.0), reverse=True)
            dures_final[f] = cands[0][0]

    panier = Panier(nom=nom or f"FUSION {sens}", dures=dures_final,
                    score=score_final, sens=sens)

    if verbose:
        print(f"\n  {panier.nom}")
        print(f"    indicateurs candidats : {len(freq)}")
        if evinces:
            for f, (g, c) in evinces.items():
                print(f"    évincé  {f:<18} corr {c:+.2f} avec {g}  "
                      f"(+{freq[f]} de fréquence hérités)")
        print(f"    retenus : {len(gardes)}")
        for f in sorted(gardes, key=lambda x: -poids[x]):
            print(f"      {f:<18} freq {freq[f]} -> {herite[f]}  poids {poids[f]:.3f}")
        if dures_final:
            print(f"    conditions dures : {list(dures_final)}")

    return {"panier": panier, "correlation": corr, "frequences": freq,
            "frequences_heritees": herite, "evinces": evinces, "poids": poids}


def fusionner_couples(meilleurs: dict, tous_in: list, tous_out: list,
                      panels: dict, seuil_corr: float = 0.8,
                      max_dures: int = 4, debut=None, fin=None,
                      verbose: bool = True) -> dict:
    """Deux fusions parallèles à partir des couples retenus par horizon.

    meilleurs : {groupe: {"entree": nom, "sortie": nom, "pnl": float, ...}}
    """
    idx_in = {p.nom: p for p in tous_in}
    idx_out = {p.nom: p for p in tous_out}

    ret_in, ret_out, pnl_in, pnl_out = [], [], {}, {}
    for g, m in meilleurs.items():
        pi, po = idx_in.get(m["entree"]), idx_out.get(m["sortie"])
        if pi is not None and pi not in ret_in:
            ret_in.append(pi)
        if po is not None and po not in ret_out:
            ret_out.append(po)
        pnl_in[m["entree"]] = max(pnl_in.get(m["entree"], -1e9), m["pnl"])
        pnl_out[m["sortie"]] = max(pnl_out.get(m["sortie"], -1e9), m["pnl"])

    if verbose:
        print("=" * 90)
        print("ÉTAPE 6 — FUSION")
        print("=" * 90)
        print(f"  paniers d'entrée retenus : {[p.nom for p in ret_in]}")
        print(f"  paniers de sortie retenus : {[p.nom for p in ret_out]}")

    f_in = fusionner(ret_in, pnl_in, panels, seuil_corr, max_dures,
                     "entree", "FUSION entree", debut, fin, verbose)
    f_out = fusionner(ret_out, pnl_out, panels, seuil_corr, max_dures,
                      "sortie", "FUSION sortie", debut, fin, verbose)

    return {"entree": f_in, "sortie": f_out,
            "panier_entree": f_in["panier"], "panier_sortie": f_out["panier"],
            "retenus_entree": ret_in, "retenus_sortie": ret_out}
