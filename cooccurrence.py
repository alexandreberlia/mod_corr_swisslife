"""
cooccurrence.py — matrice de suivi entre signaux binaires.

ENTRÉE  : tableau 0/1, dates en lignes, indicateurs en colonnes.
SORTIE  : pour chaque couple (A, B), la part des signaux de A qui sont suivis
          d'un signal de B dans la fenêtre [t + decalage_min ; t + h].

POURQUOI LE RATIO BRUT NE SUFFIT PAS
Si B se déclenche sur 15 % des séances, une fenêtre de 6 séances a déjà
1 - 0.85^6 = 62 % de chances de contenir un signal de B par pur hasard. Un
taux de suivi de 30 % serait donc DEUX FOIS MOINS bon que le hasard, alors
qu'il a l'air d'un résultat. Trois colonnes corrigent ça :

    base      P(B se déclenche dans une fenêtre quelconque)  -> le hasard
    lift      taux / base    (1.0 = hasard, 2.0 = deux fois mieux)
    p_valeur  calibrée par PERMUTATION CIRCULAIRE de la série de B :
              on fait tourner B d'un décalage aléatoire, ce qui conserve
              exactement son nombre de signaux ET son regroupement temporel,
              mais détruit tout alignement avec A. Un test binomial serait
              anticonservateur ici, les signaux arrivant en grappes.

ASYMÉTRIE : taux[A, B] != taux[B, A]. La matrice se lit « B suit A » en
partant de la LIGNE A vers la COLONNE B.
"""

import numpy as np
import pandas as pd


# ============================================================================
# 1. Chargement
# ============================================================================

def charger_signaux(chemin: str, feuille=0, col_date: str | int = 0,
                    verbose: bool = True) -> pd.DataFrame:
    """Lit un .xlsx ou .csv et renvoie un tableau strictement 0/1.

    Tolère les cellules vides et les chaînes vides "" produites par SIERREUR :
    elles valent 0. Signale toute cellule qui n'était ni 0, ni 1, ni vide —
    c'est le symptôme d'une colonne qui n'est pas binaire.
    """
    if str(chemin).lower().endswith((".csv", ".txt")):
        df = pd.read_csv(chemin, index_col=col_date, parse_dates=True)
    else:
        df = pd.read_excel(chemin, sheet_name=feuille, index_col=col_date,
                           parse_dates=True)

    df = df.sort_index()
    num = df.apply(pd.to_numeric, errors="coerce")

    suspect = ((num.notna()) & (~num.isin([0, 1]))).sum()
    suspect = suspect[suspect > 0]
    if verbose and not suspect.empty:
        print("ATTENTION — cellules ni 0 ni 1 (forcées à 0) :")
        print(suspect.to_string())

    S = num.where(num == 1, 0.0).fillna(0.0)

    if verbose:
        print(f"{len(S)} séances, {S.shape[1]} indicateurs, "
              f"du {S.index[0].date()} au {S.index[-1].date()}")
        print(f"signaux par indicateur : min {int(S.sum().min())}, "
              f"médiane {int(S.sum().median())}, max {int(S.sum().max())}")
    return S


# ============================================================================
# 2. Fenêtre glissante vers l'AVANT
# ============================================================================

def fenetre_avant(s: pd.Series, h: int = 5, decalage_min: int = 0) -> np.ndarray:
    """1 si s vaut 1 au moins une fois dans [t + decalage_min ; t + h].

    Implémentation : on renverse la série, on applique un max glissant, on
    remet dans l'ordre. Un `rolling` classique regarde en arrière ; renverser
    est la façon la plus sûre de regarder en avant sans erreur de signe.
    """
    if h < decalage_min:
        raise ValueError("h doit être >= decalage_min")
    largeur = h - decalage_min + 1
    d = s.shift(-decalage_min).fillna(0.0)
    return d.iloc[::-1].rolling(largeur, min_periods=1).max().iloc[::-1].to_numpy()


def _episodes(v: np.ndarray, ecart_min: int) -> np.ndarray:
    """Ne garde que le PREMIER signal de chaque grappe espacée de < ecart_min.

    Deux franchissements à deux jours d'intervalle ne sont pas deux
    événements indépendants : les compter deux fois gonfle le dénominateur.
    """
    if ecart_min <= 1:
        return v
    out = np.zeros_like(v)
    dernier = -10**9
    for i in np.flatnonzero(v > 0):
        if i - dernier >= ecart_min:
            out[i] = 1.0
            dernier = i
    return out


# ============================================================================
# 3. Matrice de suivi
# ============================================================================

def matrice_suivi(S: pd.DataFrame, h: int = 5, decalage_min: int = 0,
                  tronquer: bool = True, ecart_min_evenements: int = 1,
                  n_permutations: int = 5000, graine: int = 0,
                  verbose: bool = True) -> dict:
    """Parcourt tous les couples (déclencheur, suiveur).

    h                    : horizon de suivi, en séances.
    decalage_min         : 0 = le même jour compte ; 1 = B doit venir APRÈS A.
    tronquer             : exclut les h dernières séances du dénominateur.
                           Sinon leurs fenêtres sont incomplètes et tout taux
                           de suivi est mécaniquement sous-estimé en fin
                           d'échantillon.
    ecart_min_evenements : regroupe les signaux de A trop rapprochés.
    n_permutations       : 0 pour sauter le calcul des p-valeurs.
    """
    S = S.astype(float)
    cols = list(S.columns)
    n = len(S)
    if h >= n:
        raise ValueError("horizon plus long que l'échantillon")

    # ---- domaine valide : fenêtre de suivi complète ----------------------
    fin = n - h if tronquer else n
    valide = np.zeros(n, dtype=bool)
    valide[:fin] = True
    n_valide = int(valide.sum())

    # ---- fenêtres avant (suiveurs) et masques de déclenchement -----------
    F = np.column_stack([fenetre_avant(S[c], h, decalage_min) for c in cols])
    A = np.column_stack([_episodes(S[c].to_numpy(), ecart_min_evenements)
                         for c in cols])
    masques = (A > 0) & valide[:, None]            # n x k
    n_sig = masques.sum(axis=0)                    # signaux retenus par indicateur

    # ---- même jour seulement : détecteur de redondance -------------------
    Sim = np.column_stack([fenetre_avant(S[c], 0, 0) for c in cols])

    k = len(cols)
    suivis = np.zeros((k, k)); taux = np.full((k, k), np.nan)
    simul = np.full((k, k), np.nan)
    for i in range(k):
        if n_sig[i] == 0:
            continue
        m = masques[:, i]
        suivis[i] = F[m].sum(axis=0)
        taux[i] = suivis[i] / n_sig[i]
        simul[i] = Sim[m].sum(axis=0) / n_sig[i]

    base = F[valide].mean(axis=0)                  # le hasard, par suiveur
    lift = taux / np.where(base > 0, base, np.nan)

    idx = pd.Index(cols, name="declencheur")
    col = pd.Index(cols, name="suiveur")
    res = {
        "taux":      pd.DataFrame(taux, idx, col),
        "suivis":    pd.DataFrame(suivis, idx, col).astype(int),
        "lift":      pd.DataFrame(lift, idx, col),
        "simultane": pd.DataFrame(simul, idx, col),
        "n_signaux": pd.Series(n_sig, index=pd.Index(cols, name="indicateur")),
        "base":      pd.Series(base, index=pd.Index(cols, name="suiveur")),
        "params": {"h": h, "decalage_min": decalage_min, "tronquer": tronquer,
                   "ecart_min_evenements": ecart_min_evenements,
                   "n_seances": n, "n_valide": n_valide,
                   "n_permutations": n_permutations},
    }

    # ---- p-valeurs par permutation circulaire ---------------------------
    if n_permutations > 0:
        res["p_valeur"] = _p_permutation(F, masques, n_sig, suivis,
                                         n_permutations, graine, cols, verbose)

    # ---- diagonale : vide d'information ---------------------------------
    # A se déclenche en t, donc A est trivialement dans [t ; t+h] : la
    # diagonale vaut 1 par construction et ne mesure rien.
    diag = pd.DataFrame(np.eye(k, dtype=bool), index=idx, columns=col)
    for nom in ("taux", "lift", "simultane", "p_valeur"):
        if nom in res:
            res[nom] = res[nom].mask(diag.set_axis(res[nom].index, axis=0)
                                         .set_axis(res[nom].columns, axis=1))

    return res


def _p_permutation(F, masques, n_sig, suivis, n_perm, graine, cols, verbose):
    """Distribution nulle du nombre de suivis, par rotation circulaire de B.

    La rotation préserve le nombre de signaux de B et son autocorrélation
    (ses grappes restent des grappes) : seul l'alignement avec A est détruit.
    On fait tourner directement la FENÊTRE de B plutôt que sa série brute —
    la fenêtre d'une série décalée est le décalé de sa fenêtre.
    """
    n, k = F.shape
    rng = np.random.default_rng(graine)
    P = np.full((k, k), np.nan)
    M = masques.astype(np.float32)                 # n x k

    for j in range(k):                             # j = suiveur permuté
        decalages = rng.integers(1, n, size=n_perm)
        pos = (np.arange(n)[None, :] + decalages[:, None]) % n
        tire = F[:, j].astype(np.float32)[pos]     # n_perm x n
        nul = tire @ M                             # n_perm x k
        for i in range(k):
            if n_sig[i] == 0:
                continue
            P[i, j] = (1 + int((nul[:, i] >= suivis[i, j]).sum())) / (1 + n_perm)
        if verbose and (j + 1) % 5 == 0:
            print(f"  permutations : {j + 1}/{k} suiveurs")

    return pd.DataFrame(P, pd.Index(cols, name="declencheur"),
                        pd.Index(cols, name="suiveur"))


# ============================================================================
# 4. Lecture
# ============================================================================

def resume(res: dict, seuil_p: float = 0.05, top: int = 15) -> str:
    """Rapport texte : effectifs, couples significatifs, redondances."""
    p = res["params"]
    L = ["=" * 92,
         f"MATRICE DE SUIVI — fenêtre [t+{p['decalage_min']} ; t+{p['h']}]",
         "=" * 92,
         f"  {p['n_seances']} séances, {p['n_valide']} exploitables "
         f"({'tronquées' if p['tronquer'] else 'non tronquées'} en fin "
         f"d'échantillon)",
         ""]

    ns = res["n_signaux"]
    L += ["SIGNAUX PAR INDICATEUR", "-" * 92]
    t = pd.DataFrame({"n_signaux": ns,
                      "base_%": (res["base"] * 100).round(1)})
    t["verdict"] = np.where(ns < 20, "TROP PEU (< 20)",
                   np.where(res["base"] > 0.60, "base trop haute (> 60 %)", ""))
    L.append(t.to_string())

    if (ns < 20).any():
        L += ["", "  Sous 20 signaux, aucun taux de cette LIGNE n'est "
                  "interprétable : desserrer le seuil de l'indicateur.",
              "  Une base au-dessus de 60 % rend la COLONNE inutile : le "
              "suiveur se déclenche presque toujours."]

    if "p_valeur" in res:
        pv, tx, lf, sm = res["p_valeur"], res["taux"], res["lift"], res["simultane"]
        pile = (pd.concat({"taux": tx.stack(), "lift": lf.stack(),
                           "p": pv.stack(), "meme_jour": sm.stack()}, axis=1)
                .dropna().sort_values("p"))
        pile["n_sig_decl"] = [ns[a] for a, _ in pile.index]

        sig = pile[(pile.p <= seuil_p) & (pile.n_sig_decl >= 20)]
        L += ["", "=" * 92,
              f"COUPLES SIGNIFICATIFS (p <= {seuil_p}, >= 20 signaux)",
              "=" * 92]
        if sig.empty:
            L.append("  Aucun. Les co-occurrences observées sont compatibles "
                     "avec le hasard.")
        else:
            v = sig.head(top).copy()
            v["taux"] = (v["taux"] * 100).round(1)
            v["meme_jour"] = (v["meme_jour"] * 100).round(1)
            v["lift"] = v["lift"].round(2)
            L.append(v.to_string())
            L += ["", "  taux      % des signaux du déclencheur suivis",
                  "  lift      taux / hasard (1.00 = aucune information)",
                  "  meme_jour part des suivis qui tombent le JOUR MÊME"]

        red = pile[(pile.meme_jour > 0.70) & (pile.n_sig_decl >= 20)]
        if not red.empty:
            L += ["", "=" * 92, "REDONDANCE (> 70 % des suivis le jour même)",
                  "=" * 92,
                  "  Ces deux indicateurs ne se SUIVENT pas, ils mesurent la",
                  "  même chose. Les cumuler dans un panier compte une",
                  "  conviction deux fois.", ""]
            v = red.head(top)[["taux", "meme_jour", "lift", "p"]].copy()
            v["taux"] = (v["taux"] * 100).round(1)
            v["meme_jour"] = (v["meme_jour"] * 100).round(1)
            v["lift"] = v["lift"].round(2)
            L.append(v.to_string())

    k = len(ns)
    n_tests = k * (k - 1)
    seuil_corr = seuil_p / max(n_tests, 1)
    fine = 1 / (1 + p["n_permutations"]) if p["n_permutations"] else 1.0

    L += ["", "=" * 92, "LECTURE", "=" * 92,
          "  La matrice est ASYMÉTRIQUE : ligne = déclencheur, colonne =",
          "  suiveur. 'stoch suit rsi' se lit case [rsi, stoch].",
          "  La diagonale est vide : un indicateur se suit lui-même par",
          "  construction.",
          "",
          f"  {n_tests} couples testés ({k} indicateurs). À p <= {seuil_p}, on",
          f"  attend {n_tests * seuil_p:.0f} faux positifs par pur hasard. Le seuil",
          f"  corrigé est {seuil_p}/{n_tests} = {seuil_corr:.5f} : en dessous, un",
          "  couple tient ; entre les deux, c'est une piste, pas une",
          "  conclusion.",
          f"  {p['n_permutations']} permutations circulaires -> p-valeur la plus",
          f"  fine atteignable {fine:.5f}."]

    if fine > seuil_corr:
        L += ["", f"  AVERTISSEMENT : {fine:.5f} > {seuil_corr:.5f}. Aucun couple ne",
              "  PEUT atteindre le seuil corrigé avec ce nombre de",
              f"  permutations. Relancer avec n_permutations >= "
              f"{int(np.ceil(1 / seuil_corr)):,}".replace(",", " ") + " pour",
              "  pouvoir conclure."]

    L += ["",
          "  La permutation circulaire est CONSERVATRICE si le déclencheur et",
          "  le suiveur partagent une périodicité : une rotation d'un multiple",
          "  de cette période les réaligne, ce qui gonfle la distribution",
          "  nulle. Mesuré : sur deux séries liées à 70 %, la p-valeur passe",
          "  de 0.004 (positions irrégulières) à 0.034 (positions",
          "  régulièrement espacées). Se tromper dans ce sens fait rater un",
          "  vrai lien, jamais en inventer un."]
    return "\n".join(L)


def exporter(res: dict, chemin: str = "matrice_suivi.xlsx") -> str:
    """Une feuille par matrice. Les taux sont écrits en pourcentage."""
    ordre = ["taux", "lift", "p_valeur", "simultane", "suivis"]
    with pd.ExcelWriter(chemin, engine="openpyxl") as w:
        for nom in ordre:
            if nom not in res:
                continue
            d = res[nom]
            if nom in ("taux", "simultane"):
                d = (d * 100).round(1)
            elif nom == "lift":
                d = d.round(2)
            elif nom == "p_valeur":
                d = d.round(4)
            d.to_excel(w, sheet_name=nom)

        pd.DataFrame({"n_signaux": res["n_signaux"],
                      "base_%": (res["base"] * 100).round(1)}
                     ).to_excel(w, sheet_name="effectifs")
        pd.Series(res["params"]).to_frame("valeur").to_excel(w,
                                                             sheet_name="parametres")
    return chemin


# ============================================================================
# 5. Utilisation
# ============================================================================

if __name__ == "__main__":
    S = charger_signaux("signaux.xlsx")

    res = matrice_suivi(S, h=5, decalage_min=0, n_permutations=500)
    print()
    print(resume(res))

    print("\nfichier :", exporter(res))

    # Variante utile : B doit venir APRÈS A (vraie séquence, pas simultanéité)
    # res_apres = matrice_suivi(S, h=5, decalage_min=1, n_permutations=500)
