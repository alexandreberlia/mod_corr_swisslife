"""bbg_loader.py — Lecteur de l'export Bloomberg brut (colonnes date/valeur appairees).

    from bbg_loader import load_paired_csv, inventory, to_period_panel
    S = load_paired_csv("Data_in_value.csv")      # dict nom -> Series datee
    inventory(S)                                   # frequence native, couverture
    Q = to_period_panel(S, freq="Q", how="last", min_obs=40)

En ligne de commande, affiche l'inventaire sans rien ecrire :

    python bbg_loader.py Data_in_value.csv


LE PROBLEME QUE CE MODULE RESOUT
---------------------------------
L'export Bloomberg n'est pas un tableau rectangulaire. Chaque serie occupe DEUX
colonnes — sa propre colonne de dates, puis ses valeurs — et ces colonnes de
dates ne commencent pas au meme moment :

    GDP CQOQ Index   | 31/03/1970 | -0.6  | ADP CHNG Index | 28/02/2010 | -45
    ...

La ligne 1 du fichier contient donc le PIB du T1 1970 ET l'emploi ADP de
fevrier 2010. Lire le fichier comme un csv ordinaire et empiler les colonnes de
valeurs par POSITION reviendrait a dater l'ADP de 1970 : toutes les correlations
croisees seraient fausses, et d'autant plus fausses que les series commencent
loin l'une de l'autre (PIB 1970, ADP 2010, pente des taux 1982, NBER 1950).

Ce module reconstruit chaque serie depuis SA propre colonne de dates, puis
realigne tout sur une grille de periodes reguliere. Il absorbe aussi les deux
irregularites de l'export : les formats de date melanges (JJ/MM/AAAA et
AAAA-MM-JJ, parfois dans le meme fichier) et les frequences natives melangees
(quotidienne pour les marches, mensuelle pour les enquetes, trimestrielle pour
les comptes nationaux).


AGREGATION PAR DERNIERE OBSERVATION, PAS PAR MOYENNE
-----------------------------------------------------
`how="last"` par defaut, et ce n'est pas un detail. Moyenner les observations
d'un trimestre est un lissage CENTRE : il decale les correlations croisees d'a
peu pres une demi-periode vers zero. Il attenue donc precisement ce qu'on
cherche a mesurer — l'avance d'un indicateur sur le cycle. La derniere
observation du trimestre est aussi la seule disponible en temps reel a la
cloture du trimestre, ce qui rend les backtests honnetes.


CE QUE LE MODULE NE FAIT PAS
-----------------------------
Il n'interpole rien. Une serie trimestrielle native placee dans un panel
mensuel garde une observation sur trois, les deux autres sont vides. C'est
voulu : remplir ces trous inventerait de l'information et gonflerait les
tailles d'echantillon de toutes les regressions en aval.

Il ne nettoie pas les noms. Le nom de colonne est le nom Bloomberg tel quel,
y compris ses doubles espaces ("IP  YOY Index"), parce que les dictionnaires de
poids de cycle_score referencent ces noms a l'identique.
"""

import re
import sys

import numpy as np
import pandas as pd

# annotation d'axe ajoutee par le graphe Bloomberg, pas une partie du ticker :
# "S5MATR Index  (R1)" -> "S5MATR Index"
_AXE = re.compile(r"\s*\((?:[RL]\d+)\)\s*$")

# jours entre deux observations -> etiquette de frequence native
_FREQS = [(3.5, "D"), (10, "W"), (45, "M"), (135, "Q"), (400, "A")]


def _parse_dates(col: pd.Series) -> pd.Series:
    """Convertit une colonne de dates texte, en tolerant deux formats melanges.

    On tente JJ/MM/AAAA en premier parce que c'est le format dominant de
    l'export, puis on reprend les echecs en AAAA-MM-JJ. Passer directement par
    `to_datetime(dayfirst=True)` sans format explicite marche aussi mais
    reinterprete silencieusement 03/04/1986 selon la premiere ligne rencontree,
    ce qui peut decaler une serie entiere de deux mois.
    """
    s = col.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    reste = out.isna() & s.ne("") & s.ne("nan")
    if reste.any():
        out.loc[reste] = pd.to_datetime(s[reste], errors="coerce", format="mixed",
                                        dayfirst=True)
    return out


def _infer_freq(idx: pd.DatetimeIndex) -> str:
    """Frequence native, deduite de l'ecart MEDIAN entre observations.

    La mediane et non le mode : une serie quotidienne a des trous de trois jours
    tous les week-ends et de plus en plus de trous feries, une serie mensuelle a
    des ecarts de 28 a 31 jours. La mediane est insensible a ces irregularites,
    et aussi aux ruptures de frequence (une serie publiee mensuellement depuis
    2000 et trimestriellement avant).
    """
    if len(idx) < 3:
        return "?"
    j = float(np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int)))
    for seuil, nom in _FREQS:
        if j <= seuil:
            return nom
    return "A"


def load_paired_csv(path: str, sep: str = ";", encoding: str = "utf-8-sig") -> dict:
    """Lit l'export appaire et renvoie {nom de serie: Series indexee par date}.

    Les colonnes sont lues par paires (date, valeur) a partir de la position du
    nom dans la ligne d'en-tete. Une serie dont la colonne de valeurs est
    entierement illisible est ecartee, pas remplie de NaN.
    """
    brut = pd.read_csv(path, sep=sep, encoding=encoding, header=None, dtype=str,
                       keep_default_na=False, engine="python")
    entete = brut.iloc[0].tolist()
    corps = brut.iloc[1:].reset_index(drop=True)

    series, ignorees = {}, []
    for i in range(0, len(entete) - 1, 2):
        nom = _AXE.sub("", str(entete[i])).strip()
        # l'export se termine par des colonnes vides et un compteur de series
        if not nom or nom.isdigit():
            continue

        d = _parse_dates(corps.iloc[:, i])
        v = corps.iloc[:, i + 1].astype(str).str.strip()
        # un export en locale europeenne ecrit la virgule decimale ; avec ';'
        # comme separateur de colonnes, les deux conventions coexistent
        if v.str.contains(",", regex=False).any():
            v = v.str.replace(",", ".", regex=False)
        v = pd.to_numeric(v, errors="coerce")

        ok = d.notna() & v.notna()
        if not ok.any():
            ignorees.append(nom)
            continue

        s = pd.Series(v[ok].to_numpy(), index=pd.DatetimeIndex(d[ok]), name=nom)
        # un meme jour apparait deux fois dans quelques series (revision publiee
        # a la meme date) : on garde la derniere valeur du fichier
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if nom in series:
            nom = f"{nom} (2)"
            s.name = nom
        series[nom] = s

    if ignorees:
        print(f"bbg_loader : {len(ignorees)} colonne(s) sans valeur numerique "
              f"ignoree(s) : {', '.join(ignorees[:5])}"
              + (" ..." if len(ignorees) > 5 else ""))
    return series


def inventory(series: dict) -> pd.DataFrame:
    """Frequence native et couverture de chaque serie, triee par date de debut.

    C'est le tableau a regarder AVANT toute estimation : il dit quelle serie
    bride l'echantillon. Une covariable qui commence en 1982 ne peut pas
    expliquer des transitions de phase des annees 1970, et le modele ne le dira
    pas — il estimera sur ce qui reste.
    """
    lignes = []
    for nom, s in series.items():
        lignes.append({
            "serie": nom,
            "freq": _infer_freq(s.index),
            "n_obs": int(s.notna().sum()),
            "debut": s.index.min().date(),
            "fin": s.index.max().date(),
        })
    return (pd.DataFrame(lignes)
            .sort_values(["debut", "serie"])
            .reset_index(drop=True))


def to_period_panel(series: dict, freq: str = "Q", how: str = "last",
                    min_obs: int = 0) -> pd.DataFrame:
    """Realigne toutes les series sur une grille de periodes reguliere.

    freq     "Q" ou "M" (toute frequence pandas acceptee par PeriodIndex).
    how      "last" (defaut), "first", "mean", "sum", "median".
             Garder "last" sauf raison explicite : voir l'en-tete du module.
    min_obs  ecarte les series ayant moins de min_obs observations APRES
             agregation. Un seuil, pas un filtre de qualite : il protege les
             regressions en aval d'une colonne a 12 points qui passerait
             inapercue dans un panel de 57 series.

    Le resultat a un PeriodIndex CONTINU du debut de la serie la plus ancienne
    a la fin de la plus recente — trous compris. Le bord droit est en dents de
    scie, les series ne s'arretant pas toutes ensemble.
    """
    agg = {"last": "last", "first": "first", "mean": "mean", "sum": "sum",
           "median": "median"}
    if how not in agg:
        raise ValueError(f"how={how!r} inconnu, attendu un de {sorted(agg)}")

    cols, courtes = {}, []
    for nom, s in series.items():
        p = s.copy()
        p.index = p.index.to_period(freq)
        p = p.groupby(level=0).agg(agg[how])
        if p.notna().sum() < min_obs:
            courtes.append((nom, int(p.notna().sum())))
            continue
        cols[nom] = p

    if not cols:
        raise ValueError("aucune serie ne passe min_obs")

    debut = min(p.index.min() for p in cols.values())
    fin = max(p.index.max() for p in cols.values())
    grille = pd.period_range(debut, fin, freq=freq)

    P = pd.DataFrame({nom: p.reindex(grille) for nom, p in cols.items()},
                     index=grille)
    P.index.name = {"Q": "trimestre", "M": "mois"}.get(freq, "periode")

    if courtes:
        print(f"bbg_loader : {len(courtes)} serie(s) sous min_obs={min_obs} "
              f"ecartee(s) : "
              + ", ".join(f"{n} ({k})" for n, k in courtes[:5])
              + (" ..." if len(courtes) > 5 else ""))
    return P


def nber_indicators(series: dict, key: str = "NBER", freq: str = "Q") -> pd.DataFrame:
    """Indicatrice de recession et dates d'entree/sortie, depuis la colonne NBER.

    ATTENTION : la colonne NBER de cet export est fausse (136 trimestres de
    recession contre 48 reels, 54 % d'accord avec la chronologie officielle).
    Utiliser `usrec.py`, qui construit l'indicatrice depuis les dates publiees
    par le NBER. Cette fonction ne reste la que pour verifier une colonne NBER
    d'un autre export avant de s'y fier.
    """
    if key not in series:
        raise KeyError(f"serie {key!r} absente de l'export")
    r = series[key].copy()
    r.index = r.index.to_period(freq)
    r = (r.groupby(level=0).last() > 0).astype(float)
    d = r.diff()
    return pd.DataFrame({"recession": r,
                         "entree": (d > 0).astype(float),
                         "sortie": (d < 0).astype(float)})


def trailing_min(s: pd.Series, window: int) -> pd.Series:
    """Minimum glissant sur les `window` dernieres periodes, bord gauche exclu.

    Sert aux variables definies par rapport a un plancher recent — "le chomage
    est-il remonte de plus de x point au-dessus de son plus bas des 12 derniers
    mois", la regle de Sahm. `min_periods=window` : pas de valeur avant que la
    fenetre soit pleine, pour ne pas comparer a un minimum calcule sur 2 points.
    """
    return s.rolling(window, min_periods=window).min()


def _main(fichier="Data_in_value.csv"):
    S = load_paired_csv(fichier)
    inv = inventory(S)
    print(f"\n{len(S)} series lues depuis {fichier}\n")
    print("frequences natives :")
    print(inv.groupby("freq").serie.count().to_string())
    print(f"\nla plus ancienne : {inv.iloc[0].serie} ({inv.iloc[0].debut})")
    print(f"la plus recente  : {inv.iloc[-1].serie} ({inv.iloc[-1].debut})")
    print("\n10 series qui bornent l'echantillon (debut le plus tardif) :")
    print(inv.sort_values("debut").tail(10)[["serie", "freq", "debut", "fin"]]
          .to_string(index=False))
    return S, inv


if __name__ == "__main__":
    _main(sys.argv[1] if len(sys.argv) > 1 else "Data_in_value.csv")
