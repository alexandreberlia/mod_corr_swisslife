"""
validation.py — protocole d'évaluation hors échantillon.

CE QUE FAISAIT LE CODE JUSQU'ICI (et pourquoi c'est insuffisant)
    pipeline_complet : UN seul découpage 60/40 -> UNE simulation hors échantillon
    simulation       : recalibrage des poids tous les 63 j, mais UN seul chemin
    classer_combinaisons : évalue ~200 combinaisons sur TOUT l'échantillon,
                           sans découpage du tout

Trois problèmes, par gravité croissante :

  1. UN SEUL DÉCOUPAGE = UN SEUL TIRAGE
     Le résultat dépend entièrement de la période qui est tombée en test. Un
     découpage 60/40 sur 2015-2025 met le COVID d'un côté ou de l'autre selon
     la date choisie. Ce n'est pas une mesure, c'est une anecdote.

  2. FUITE À LA FRONTIÈRE
     Avec un horizon de 20 jours, les 20 dernières observations d'entraînement
     ont un rendement futur qui déborde sur la période de test. Il faut PURGER
     ces observations, puis imposer un EMBARGO après la frontière (López de
     Prado, "Advances in Financial Machine Learning", ch. 7).

  3. SÉLECTION MULTIPLE — le plus grave
     Tester 150 combinaisons et garder la meilleure produit un résultat flatteur
     même sur du bruit pur. Mesuré : MCC maximum de 0.028 sur un univers SANS
     AUCUN SIGNAL. Le seuil à battre n'est donc pas 0, c'est le maximum attendu
     sous l'hypothèse nulle, qui croît avec le nombre d'essais.

CE MODULE FOURNIT
    decoupages_wf         plusieurs folds walk-forward, purgés et avec embargo
    evaluer_hors_echant   une combinaison évaluée sur TOUS les folds
    seuil_selection       la barre à franchir compte tenu du nombre d'essais
    selectionner          sélection honnête : choix in-sample, mesure out-of-sample
    bootstrap_bloc        intervalle de confiance par rééchantillonnage par blocs
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from features import evaluer_binaire, _score_combi


# ============================================================================
# 1. Découpages
# ============================================================================

@dataclass
class Fold:
    i: int
    train_debut: pd.Timestamp
    train_fin: pd.Timestamp
    test_debut: pd.Timestamp
    test_fin: pd.Timestamp

    def __repr__(self):
        return (f"Fold{self.i} train[{self.train_debut.date()}→{self.train_fin.date()}] "
                f"test[{self.test_debut.date()}→{self.test_fin.date()}]")


def decoupages_wf(index: pd.DatetimeIndex, n_folds: int = 5, horizon: int = 20,
                  embargo: int = 10, ancre: bool = True,
                  min_train: int = 504) -> list:
    """Folds walk-forward purgés.

    ancre=True  : la fenêtre d'entraînement s'allonge (2015-16, 2015-17, 2015-18…)
                  -> reproduit la situation réelle : on accumule de l'historique
    ancre=False : fenêtre glissante de taille constante
                  -> teste si le modèle vieillit

    PURGE   : on retire les `horizon` dernières observations d'entraînement, dont
              le rendement futur déborde sur la période de test.
    EMBARGO : on saute `embargo` observations supplémentaires après la frontière,
              pour couper l'autocorrélation résiduelle des features.
    """
    n = len(index)
    dispo = n - min_train
    if dispo <= 0:
        raise ValueError(f"Historique trop court : {n} barres, min_train={min_train}")

    taille = dispo // n_folds
    folds = []
    for k in range(n_folds):
        test_i0 = min_train + k * taille
        test_i1 = min_train + (k + 1) * taille if k < n_folds - 1 else n - 1
        train_i1 = test_i0 - horizon - embargo          # purge + embargo
        train_i0 = 0 if ancre else max(0, train_i1 - min_train)
        if train_i1 - train_i0 < min_train // 2:
            continue
        folds.append(Fold(k + 1, index[train_i0], index[train_i1],
                          index[test_i0], index[min(test_i1, n - 1)]))
    return folds


def decoupages_cpcv(index: pd.DatetimeIndex, n_groupes: int = 6, n_test: int = 2,
                    horizon: int = 20, embargo: int = 10) -> list:
    """Combinatorial Purged Cross-Validation (López de Prado, ch. 12).

    Découpe la période en n_groupes blocs, et teste TOUTES les façons d'en
    choisir n_test comme échantillon de test. Avec 6 groupes et 2 en test :
    C(6,2) = 15 chemins hors échantillon au lieu d'un seul.

    Bien plus informatif qu'un walk-forward : on obtient une DISTRIBUTION de
    performances, donc un intervalle de confiance, au lieu d'un point unique.
    Limite : certains folds entraînent sur du futur par rapport au test, ce qui
    est acceptable pour MESURER la robustesse d'un signal, pas pour simuler
    une exploitation réelle.
    """
    from itertools import combinations
    n = len(index)
    bornes = [int(i * n / n_groupes) for i in range(n_groupes + 1)]
    blocs = [(bornes[i], bornes[i + 1]) for i in range(n_groupes)]

    folds = []
    for k, choix in enumerate(combinations(range(n_groupes), n_test), 1):
        test_idx = np.zeros(n, bool)
        for g in choix:
            test_idx[blocs[g][0]:blocs[g][1]] = True
        train_idx = ~test_idx
        # purge + embargo autour de chaque frontière
        for g in choix:
            a, b = blocs[g]
            train_idx[max(0, a - horizon - embargo):a] = False
            train_idx[b:min(n, b + embargo)] = False
        folds.append({"i": k, "groupes_test": choix,
                      "train": index[train_idx], "test": index[test_idx]})
    return folds


# ============================================================================
# 2. Évaluation hors échantillon
# ============================================================================

def evaluer_hors_echant(combi: dict, panels: dict, close: pd.DataFrame,
                        folds: list, horizon: int = 20, q: float = 0.10,
                        reference: str = "mediane", min_titres: int = 20,
                        critere: str = "MCC") -> pd.DataFrame:
    """Évalue une combinaison sur CHAQUE fold, séparément train et test.

    L'écart train - test est l'information la plus utile du tableau : un écart
    important signale un surajustement, même si le test reste positif.
    """
    lignes = []
    for f in folds:
        if isinstance(f, dict):
            idx_tr, idx_te, num = f["train"], f["test"], f["i"]
        else:
            idx_tr = close.loc[f.train_debut:f.train_fin].index
            idx_te = close.loc[f.test_debut:f.test_fin].index
            num = f.i

        for nom, idx in (("train", idx_tr), ("test", idx_te)):
            pan = {k: v.loc[v.index.isin(idx)] for k, v in panels.items()}
            cl = close.loc[close.index.isin(idx)]
            try:
                ev = evaluer_binaire(combi, pan, cl, (horizon,), q, reference, min_titres)
            except (KeyError, ValueError):
                continue
            if critere not in ev.columns or ev[critere].isna().all():
                continue
            r = ev.iloc[0]
            lignes.append({"fold": num, "echantillon": nom,
                           "debut": idx[0].date(), "fin": idx[-1].date(),
                           critere: r[critere], "edge_pt": r.get("edge_pt", np.nan),
                           "lift_hausse": r.get("lift_hausse", np.nan),
                           "n_dates": r.get("n_dates", 0)})
    return pd.DataFrame(lignes)


def resume_oos(detail: pd.DataFrame, critere: str = "MCC") -> pd.Series:
    """Agrège les folds. La STABILITÉ prime sur la moyenne."""
    tr = detail[detail.echantillon == "train"][critere].dropna()
    te = detail[detail.echantillon == "test"][critere].dropna()
    if te.empty:
        return pd.Series({"ERREUR": "aucun fold de test exploitable"})
    return pd.Series({
        f"{critere}_train": tr.mean(),
        f"{critere}_test": te.mean(),
        "ecart_train_test": tr.mean() - te.mean(),   # > 0 => surajustement
        f"{critere}_test_min": te.min(),
        f"{critere}_test_std": te.std(),
        "folds_positifs": int((te > 0).sum()),
        "n_folds": len(te),
        "%_folds_positifs": (te > 0).mean() * 100,
        # t de Student sur les folds : chaque fold est une observation indépendante
        "t_folds": te.mean() / (te.std() / np.sqrt(len(te))) if te.std() > 0 else np.nan,
    })


# ============================================================================
# 3. Sélection multiple
# ============================================================================

def seuil_selection(n_essais: int, n_obs: int, alpha: float = 0.05,
                    sigma: float = None) -> dict:
    """Barre THÉORIQUE quand on retient le meilleur de n_essais candidats.

    ATTENTION — cette formule suppose des tirages INDÉPENDANTS. Des combinaisons
    qui partagent des features sont fortement corrélées : le nombre d'essais
    effectifs est bien inférieur à n_essais, et ce seuil est donc trop sévère.
    Mesuré : la formule annonce 0.32 là où la permutation donne 0.03.

    À n'utiliser que comme borne supérieure grossière. Préférer seuil_empirique().
    """
    from scipy.stats import norm
    s = sigma if sigma is not None else 1.0 / np.sqrt(max(n_obs, 2))
    e_max = s * np.sqrt(2 * np.log(max(n_essais, 2)))
    z_bonf = norm.ppf(1 - alpha / (2 * max(n_essais, 1)))
    return {
        "n_essais": n_essais, "n_obs": n_obs, "sigma_estime": s,
        "max_attendu_sous_H0": e_max,
        "seuil_bonferroni": z_bonf * s,
        "z_naif": norm.ppf(1 - alpha / 2) * s,
        "avertissement": "suppose l'independance des essais -> trop severe",
    }


def seuil_empirique(combis: dict, panels: dict, close: pd.DataFrame,
                    horizon: int = 20, q: float = 0.10,
                    reference: str = "mediane", min_titres: int = 20,
                    critere: str = "MCC", n_permutations: int = 20,
                    graine: int = 0, verbose: bool = True) -> dict:
    """Distribution du MEILLEUR score sous l'hypothèse nulle, par permutation.

    Méthode : à chaque date, on permute les rendements futurs ENTRE LES TITRES.
    Cela détruit le lien feature -> rendement tout en préservant exactement la
    distribution cross-sectionnelle des rendements, l'autocorrélation temporelle
    et la corrélation entre combinaisons. On relance ensuite la sélection
    complète et on note le meilleur score obtenu.

    Répété n_permutations fois, on obtient la distribution du "meilleur de N"
    sous H0. C'est ÇA le seuil à battre — pas zéro, pas la formule théorique.

    Le p_valeur renvoyé se lit : proportion de permutations où le hasard fait
    aussi bien que le résultat réel.
    """
    rng = np.random.default_rng(graine)
    maxima = []

    for perm in range(n_permutations):
        if verbose and (perm + 1) % 5 == 0:
            print(f"    permutation {perm + 1}/{n_permutations}…")

        # permutation des colonnes ligne par ligne : le rendement du titre A à la
        # date d est attribué à un autre titre, tiré au sort à cette date.
        vals = close.to_numpy().copy()
        idx_perm = np.argsort(rng.random(vals.shape), axis=1)
        cl_perm = pd.DataFrame(np.take_along_axis(vals, idx_perm, axis=1),
                               index=close.index, columns=close.columns)

        scores = []
        for combi in combis.values():
            try:
                ev = evaluer_binaire(combi, panels, cl_perm, (horizon,), q,
                                     reference, min_titres)
            except (KeyError, ValueError):
                continue
            if critere in ev.columns and not ev[critere].isna().all():
                scores.append(float(ev[critere].iloc[0]))
        if scores:
            maxima.append(max(scores))

    if not maxima:
        return {"ERREUR": "aucune permutation exploitable"}

    m = np.array(maxima)
    return {
        "n_essais": len(combis),
        "n_permutations": len(m),
        "max_median_H0": float(np.median(m)),
        "max_p90_H0": float(np.quantile(m, 0.90)),
        "max_p95_H0": float(np.quantile(m, 0.95)),
        "max_observe_H0": float(m.max()),
        "distribution": m,
    }


def p_valeur_permutation(score_reel: float, seuils: dict) -> float:
    """Proportion de permutations où le hasard atteint au moins score_reel."""
    if "distribution" not in seuils:
        return np.nan
    m = seuils["distribution"]
    return float((m >= score_reel).mean())


def selectionner(combis: dict, panels: dict, close: pd.DataFrame,
                 folds: list, horizon: int = 20, q: float = 0.10,
                 reference: str = "mediane", min_titres: int = 20,
                 critere: str = "MCC", seuil_h0: float = None,
                 verbose: bool = True) -> dict:
    """Sélection honnête, en deux temps strictement séparés.

      ÉTAPE 1  on classe les combinaisons sur les folds d'ENTRAÎNEMENT seulement
      ÉTAPE 2  on mesure la retenue sur les folds de TEST, jamais vus

    La performance de test n'a jamais servi au choix : c'est ce qui la rend
    interprétable. Le tableau renvoie aussi le seuil de sélection multiple, à
    comparer au résultat obtenu.
    """
    detail = {}
    for i, (nom, combi) in enumerate(combis.items(), 1):
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(combis)}…")
        d = evaluer_hors_echant(combi, panels, close, folds, horizon, q,
                                reference, min_titres, critere)
        if not d.empty:
            detail[nom] = d

    if not detail:
        return {"ERREUR": "aucune combinaison exploitable"}

    lignes = []
    for nom, d in detail.items():
        r = resume_oos(d, critere)
        if "ERREUR" in r:
            continue
        r["combinaison"] = nom
        lignes.append(r)
    tab = pd.DataFrame(lignes).set_index("combinaison")

    # ÉTAPE 1 : le choix se fait sur le TRAIN uniquement
    classement_train = tab.sort_values(f"{critere}_train", ascending=False)
    retenue = classement_train.index[0]

    return {
        "retenue": retenue,
        "tableau": tab.sort_values(f"{critere}_test", ascending=False),
        "classement_train": classement_train,
        "detail_retenue": detail[retenue],
        "n_essais": len(detail),
        "verdict": _verdict(tab.loc[retenue], critere, seuil_h0),
        "seuil_h0": seuil_h0,
    }


def _verdict(r: pd.Series, critere: str, seuil_h0: float = None) -> str:
    """Verdict. `seuil_h0` doit venir de seuil_empirique() : c'est le 95e
    percentile du meilleur score obtenu par permutation."""
    te = r[f"{critere}_test"]
    msgs = []
    if te <= 0:
        msgs.append("REJET : performance de test negative.")
    elif seuil_h0 is not None:
        if te < seuil_h0:
            msgs.append(f"REJET : {critere}_test = {te:.4f} sous le seuil de "
                        f"permutation ({seuil_h0:.4f}).")
        else:
            msgs.append(f"RETENU : {critere}_test = {te:.4f} > seuil de "
                        f"permutation ({seuil_h0:.4f}).")
    else:
        msgs.append(f"{critere}_test = {te:.4f}. Seuil H0 non calibre : lancer "
                    f"seuil_empirique() avant de conclure.")
    if te > 0 and r["ecart_train_test"] > 2 * abs(te):
        msgs.append("ATTENTION : ecart train-test important, surajustement probable.")
    if r["%_folds_positifs"] < 60:
        msgs.append(f"ATTENTION : seulement {r['%_folds_positifs']:.0f} % de folds "
                    f"positifs — instable.")
    return " ".join(msgs)


# ============================================================================
# 4. Intervalle de confiance par bootstrap par blocs
# ============================================================================

def bootstrap_bloc(serie: pd.Series, n_tirages: int = 1000, taille_bloc: int = 21,
                   alpha: float = 0.05, graine: int = 0) -> dict:
    """Intervalle de confiance qui préserve l'autocorrélation.

    Un bootstrap classique tire les observations indépendamment et détruit la
    structure temporelle, ce qui produit des intervalles beaucoup trop étroits
    sur des séries financières. Le tirage par BLOCS de `taille_bloc` jours la
    conserve.
    """
    x = serie.dropna().to_numpy()
    n = len(x)
    if n < taille_bloc * 3:
        return {"ERREUR": f"série trop courte ({n})"}
    rng = np.random.default_rng(graine)
    n_blocs = int(np.ceil(n / taille_bloc))
    moyennes = np.empty(n_tirages)
    for i in range(n_tirages):
        deb = rng.integers(0, n - taille_bloc, n_blocs)
        ech = np.concatenate([x[d:d + taille_bloc] for d in deb])[:n]
        moyennes[i] = ech.mean()
    return {
        "moyenne": float(x.mean()),
        "ic_bas": float(np.quantile(moyennes, alpha / 2)),
        "ic_haut": float(np.quantile(moyennes, 1 - alpha / 2)),
        "p_valeur_unilat": float((moyennes <= 0).mean()),
        "n_tirages": n_tirages, "taille_bloc": taille_bloc,
    }
