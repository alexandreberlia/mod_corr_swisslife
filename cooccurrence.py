"""
cooccurrence.py — tableau de correspondance entre signaux binaires.

ENTRÉE  : tableau 0/1, dates en lignes, indicateurs en colonnes.
SORTIE  : pour chaque couple (A, B), la part des signaux de A pour lesquels B
          se déclenche dans la fenêtre [t + decalage_min ; t + h].

Tout est du comptage. Aucune modélisation, aucun test statistique.

    taux[A, B] = suivis[A, B] / n_signaux[A]

Deux filtres à l'affichage, et c'est tout :
    - un déclencheur avec moins de 20 signaux est écarté (ligne)
    - un suiveur présent dans plus de 60 % des fenêtres est écarté (colonne),
      il se déclenche trop souvent pour que « il suit A » veuille dire quelque
      chose

Lecture : LIGNE = déclencheur, COLONNE = suiveur. « stoch suit rsi » se lit
case [rsi, stoch]. Le tableau est asymétrique, taux[A,B] != taux[B,A].
"""

import numpy as np
import pandas as pd


# ============================================================================
# 1. Chargement
# ============================================================================

def charger_signaux(chemin: str, feuille=0, col_date: str | int = 0,
                    verbose: bool = True) -> pd.DataFrame:
    """Lit un .xlsx ou .csv et renvoie un tableau strictement 0/1.

    Les cellules vides et les chaînes vides "" produites par SIERREUR valent 0.
    Toute cellule numérique qui n'était ni 0 ni 1 est signalée : c'est le
    symptôme d'une colonne qui contient des valeurs d'indicateur au lieu de
    drapeaux.

    Le tri par date est indispensable : tout le module regarde VERS L'AVANT
    dans l'ordre des lignes. Un fichier trié du plus récent au plus ancien
    donnerait les 5 séances PRÉCÉDENTES sans aucune erreur visible.
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
    return S


# ============================================================================
# 2. Fenêtre glissante vers l'AVANT
# ============================================================================

def fenetre_avant(s: pd.Series, h: int = 5, decalage_min: int = 0) -> np.ndarray:
    """1 si s vaut 1 au moins une fois dans [t + decalage_min ; t + h].

    `rolling` de pandas regarde toujours en arrière. On renverse la série, on
    applique le max glissant, on remet dans l'ordre : c'est la façon la plus
    sûre de regarder en avant sans erreur de signe.

    Le max sur du 0/1 est un OU logique : « au moins un signal », jamais un
    comptage. Trois signaux dans la fenêtre comptent pour un.
    """
    if h < decalage_min:
        raise ValueError("h doit être >= decalage_min")
    largeur = h - decalage_min + 1
    d = s.shift(-decalage_min).fillna(0.0)
    return d.iloc[::-1].rolling(largeur, min_periods=1).max().iloc[::-1].to_numpy()


def _episodes(v: np.ndarray, ecart_min: int) -> np.ndarray:
    """Ne garde que le PREMIER signal de chaque grappe espacée de < ecart_min.

    Un franchissement lundi, un repli mardi, un refranchissement mercredi :
    trois lignes à 1 dans le tableau, un seul mouvement de marché. Les compter
    trois fois gonfle le dénominateur.
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
# 3. Comptage
# ============================================================================

def matrice_suivi(S: pd.DataFrame, h: int = 5, decalage_min: int = 0,
                  tronquer: bool = True,
                  ecart_min_evenements: int = 1) -> dict:
    """Parcourt tous les couples (déclencheur, suiveur) et compte.

    h                    : horizon de suivi, en séances.
    decalage_min         : 0 = le même jour compte ; 1 = le suiveur doit venir
                           APRÈS le déclencheur.
    tronquer             : exclut les h dernières séances. Leurs fenêtres sont
                           incomplètes, donc elles ne peuvent presque jamais
                           compter comme suivies et tirent tous les taux vers
                           le bas.
    ecart_min_evenements : regroupe les signaux trop rapprochés du déclencheur.
    """
    S = S.astype(float)
    cols = list(S.columns)
    n, k = len(S), len(S.columns)
    if h >= n:
        raise ValueError("horizon plus long que l'échantillon")

    # ---- domaine valide : fenêtre de suivi complète ----------------------
    fin = n - h if tronquer else n
    valide = np.zeros(n, dtype=bool)
    valide[:fin] = True

    # ---- fenêtres avant (suiveurs) et masques de déclenchement -----------
    F = np.column_stack([fenetre_avant(S[c], h, decalage_min) for c in cols])
    A = np.column_stack([_episodes(S[c].to_numpy(), ecart_min_evenements)
                         for c in cols])
    masques = (A > 0) & valide[:, None]            # n x k
    n_sig = masques.sum(axis=0)

    # ---- même jour seulement : séquence ou simple simultanéité ? ---------
    Sim = S.to_numpy()

    suivis = np.zeros((k, k))
    taux = np.full((k, k), np.nan)
    simul = np.full((k, k), np.nan)
    for i in range(k):
        if n_sig[i] == 0:
            continue
        m = masques[:, i]
        suivis[i] = F[m].sum(axis=0)
        taux[i] = suivis[i] / n_sig[i]
        simul[i] = Sim[m].sum(axis=0) / n_sig[i]

    # diagonale : A est dans [t ; t+h] par construction quand A se déclenche
    # en t. La case vaudrait 1 et ne mesurerait rien. On la vide ici, sur les
    # tableaux numpy, qui sont sûrement accessibles en écriture.
    np.fill_diagonal(taux, np.nan)
    np.fill_diagonal(simul, np.nan)

    # part des fenêtres de l'historique contenant un signal du suiveur :
    # c'est un second comptage, sur les mêmes données, sans conditionner
    # sur le déclencheur.
    base = F[valide].mean(axis=0)

    idx = pd.Index(cols, name="declencheur")
    col = pd.Index(cols, name="suiveur")
    res = {
        "taux":      pd.DataFrame(taux, idx, col),
        "suivis":    pd.DataFrame(suivis, idx, col).astype(int),
        "simultane": pd.DataFrame(simul, idx, col),
        "n_signaux": pd.Series(n_sig, index=pd.Index(cols, name="indicateur")),
        "base":      pd.Series(base, index=pd.Index(cols, name="suiveur")),
        "params": {"h": h, "decalage_min": decalage_min, "tronquer": tronquer,
                   "ecart_min_evenements": ecart_min_evenements,
                   "n_seances": n, "n_valide": int(valide.sum())},
    }

    return res


# ============================================================================
# 4. Filtrage et tableau
# ============================================================================

def filtrer(res: dict, min_signaux: int = 20,
            base_max: float = 0.60) -> tuple[list, list, pd.DataFrame]:
    """Retient les déclencheurs assez fréquents et les suiveurs pas trop bavards.

    Renvoie (declencheurs, suiveurs, journal) où journal dit, indicateur par
    indicateur, s'il est gardé en ligne, en colonne, et pourquoi.
    """
    ns, base = res["n_signaux"], res["base"]
    noms = list(ns.index)

    garde_ligne = ns >= min_signaux
    garde_col = base <= base_max

    journal = pd.DataFrame({
        "n_signaux": ns,
        "base_%": (base * 100).round(1),
        "ligne": np.where(garde_ligne.reindex(noms).to_numpy(), "gardee",
                          f"ECARTEE < {min_signaux} signaux"),
        "colonne": np.where(garde_col.reindex(noms).to_numpy(), "gardee",
                            f"ECARTEE base > {base_max:.0%}"),
    }, index=pd.Index(noms, name="indicateur"))

    return ([c for c in noms if garde_ligne[c]],
            [c for c in noms if garde_col[c]], journal)


def tableau(res: dict, quoi: str = "taux", min_signaux: int = 20,
            base_max: float = 0.60, pourcent: bool = True) -> pd.DataFrame:
    """La matrice filtrée, forme carrée déclencheurs x suiveurs.

    quoi : "taux" | "suivis" | "simultane"
    """
    lig, colo, _ = filtrer(res, min_signaux, base_max)
    d = res[quoi].loc[lig, colo]
    if pourcent and quoi != "suivis":
        d = (d * 100).round(1)
    return d


def _aplatir(d: pd.DataFrame, nom: str) -> pd.DataFrame:
    """Matrice carrée -> colonne longue indexée (declencheur, suiveur)."""
    return (d.rename_axis(index="declencheur", columns="suiveur")
             .reset_index()
             .melt(id_vars="declencheur", var_name="suiveur", value_name=nom))


def couples(res: dict, min_signaux: int = 20, base_max: float = 0.60,
            tri: str = "taux_%") -> pd.DataFrame:
    """TOUS les couples passant les filtres, une ligne par couple.

    Colonnes, toutes issues de comptages :
        n_signaux    signaux du déclencheur          (dénominateur)
        suivis       signaux suivis d'un suiveur     (numérateur)
        taux_%       suivis / n_signaux
        base_%       part des fenêtres de l'historique contenant le suiveur,
                     sans conditionner sur le déclencheur
        meme_jour_%  part des signaux du déclencheur où le suiveur sort LE
                     JOUR MÊME
    """
    lig, colo, _ = filtrer(res, min_signaux, base_max)
    if not lig or not colo:
        return pd.DataFrame()

    d = _aplatir((res["taux"].loc[lig, colo] * 100).round(1), "taux_%")
    for quoi, nom, mult in (("suivis", "suivis", 1),
                            ("simultane", "meme_jour_%", 100)):
        m = res[quoi].loc[lig, colo]
        m = (m * 100).round(1) if mult == 100 else m
        d = d.merge(_aplatir(m, nom), on=["declencheur", "suiveur"], how="left")

    d = d.dropna(subset=["taux_%"])                  # retire la diagonale
    d["n_signaux"] = d.declencheur.map(res["n_signaux"])
    d["base_%"] = d.suiveur.map((res["base"] * 100).round(1))

    return (d.set_index(["declencheur", "suiveur"])
             [["n_signaux", "suivis", "taux_%", "base_%", "meme_jour_%"]]
             .sort_values(tri, ascending=False))


# ============================================================================
# 5. Rapport
# ============================================================================

def resume(res: dict, min_signaux: int = 20, base_max: float = 0.60,
           top: int | None = None, tri: str = "taux_%") -> str:
    """Effectifs, puis TOUS les couples passant les filtres, un par ligne.

    top : None affiche tout. Un entier limite aux `top` premiers couples.
    """
    p = res["params"]
    lig, colo, journal = filtrer(res, min_signaux, base_max)

    L = ["=" * 92,
         f"TABLEAU DE CORRESPONDANCE — fenêtre [t+{p['decalage_min']} ; "
         f"t+{p['h']}]",
         "=" * 92,
         f"  {p['n_seances']} séances, {p['n_valide']} exploitables"
         f"{' (h dernières tronquées)' if p['tronquer'] else ''}",
         f"  filtres : >= {min_signaux} signaux en ligne, base <= "
         f"{base_max:.0%} en colonne",
         "",
         "EFFECTIFS", "-" * 92, journal.to_string(), ""]

    c = couples(res, min_signaux, base_max, tri)
    if c.empty:
        L.append("Aucun couple ne passe les filtres.")
        return "\n".join(L)

    montre = c if top is None else c.head(top)
    L += ["=" * 92,
          f"COUPLES — {len(c)} au total"
          + (f", {len(montre)} affichés" if top else "")
          + f", triés par {tri} décroissant",
          "=" * 92,
          montre.to_string(),
          "",
          "  n_signaux   signaux du declencheur                   denominateur",
          "  suivis      dont un signal du suiveur dans la fenetre numerateur",
          "  taux_%      suivis / n_signaux",
          "  base_%      part des fenetres de l'historique contenant le",
          "              suiveur, sans conditionner sur le declencheur",
          "  meme_jour_% part des signaux du declencheur ou le suiveur sort LE",
          "              JOUR MEME. Proche de taux_% => les deux sortent",
          "              ensemble plutot que l'un apres l'autre.",
          "",
          "  Lecture : la ligne (A, B) se lit « B suit A ». Le tableau est",
          "  asymetrique, (A, B) et (B, A) sont deux lignes distinctes.",
          "  La matrice carree reste disponible : tableau(res, \"taux\")."]
    return "\n".join(L)


def exporter(res: dict, chemin: str = "matrice_suivi.xlsx",
             min_signaux: int = 20, base_max: float = 0.60) -> str:
    """Les couples en liste, les matrices filtrées, puis les matrices complètes."""
    with pd.ExcelWriter(chemin, engine="openpyxl") as w:
        couples(res, min_signaux, base_max).to_excel(w, sheet_name="couples")

        for quoi in ("taux", "suivis", "simultane"):
            tableau(res, quoi, min_signaux, base_max).to_excel(
                w, sheet_name=f"{quoi}_filtre")

        (res["taux"] * 100).round(1).to_excel(w, sheet_name="taux_complet")
        res["suivis"].to_excel(w, sheet_name="suivis_complet")
        (res["simultane"] * 100).round(1).to_excel(w,
                                                   sheet_name="meme_jour_complet")

        _, _, journal = filtrer(res, min_signaux, base_max)
        journal.to_excel(w, sheet_name="effectifs")
        pd.Series(res["params"]).to_frame("valeur").to_excel(w,
                                                             sheet_name="parametres")
    return chemin


# ============================================================================
# 6. Utilisation
# ============================================================================

if __name__ == "__main__":
    S = charger_signaux("signaux.xlsx")

    res = matrice_suivi(S, h=5, decalage_min=0)
    print()
    print(resume(res))
    print("\nfichier :", exporter(res))

    # Le tableau seul, en DataFrame :
    #   t = tableau(res, "taux")
    #
    # Variante « vraie sequence » : le suiveur doit venir APRES
    #   res_apres = matrice_suivi(S, h=5, decalage_min=1)
