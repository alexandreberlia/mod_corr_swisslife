"""pipeline_dating.py — Banc d'essai pour cycle_dating.

    python pipeline_dating.py

Permet de modifier les parametres et de voir immediatement l'effet, sans
toucher au module. Trois usages :

    1. Datation courante + controle qualite
    2. Balayage d'un parametre, toutes choses egales par ailleurs
    3. Comparaison de deux jeux de parametres, trimestre par trimestre

POURQUOI PASSER PAR CE FICHIER
------------------------------
Les constantes de cycle_dating sont lues A L'APPEL, sauf CORRECTION_DERIVE qui
l'etait autrefois comme valeur par defaut d'argument — donc figee a l'import.
Ce piege a rendu un test parametrique silencieusement faux : cinq valeurs
differentes donnaient cinq resultats identiques. Le module est corrige, mais le
plus sur reste de recharger le module a chaque configuration, ce que fait
`dater_avec()` ci-dessous.

LES PARAMETRES ET LEUR EFFET
-----------------------------
  CORRECTION_DERIVE   0 = gap brut (ecart au potentiel, interpretable).
                      12 = gap moins sa moyenne mobile 3 ans : date mieux les
                      recessions NBER mais change le SENS du niveau — un PIB
                      au-dessus du potentiel peut etre classe « en dessous ».
  BANDE_NIVEAU        bande morte autour de zero, en ecarts-types du gap.
                      0 = bascule des que le gap change de signe.
  AMPLITUDE_MIN       points de croissance minimum entre deux retournements.
                      Bas = plus de phases, plus sensible au bruit.
                      Haut = phases longues, retournements tardifs ou manques.
  DUREE_MIN_RETOURNEMENT  trimestres minimum entre deux retournements.
  DUREE_MIN_PHASE     phases plus courtes absorbees par la voisine.
  SEUIL_CHOC          au-dela (en ecarts-types), une phase courte est protegee
                      de l'absorption. Sert au Covid, qui n'a dure que
                      2 trimestres.

CE QU'IL FAUT REGARDER
----------------------
Les metriques NBER (couverture, precision) sont utiles mais TROMPEUSES prises
seules : « Decrochage » est plus large que « recession » — sous le potentiel et
en deceleration, ce qui arrive hors recession. Une precision de 50 % n'est donc
pas un echec.

Les periodes temoins comptent davantage. Chacune teste un aspect precis :
  1980/81      deux recessions separees par 12 mois d'expansion : sont-elles
               distinguees, ou fusionnees en un seul Decrochage ?
  2001-2004    le creux de 2001Q4 est-il capte, ou noye ?
  2011-2013    sous le potentiel, momentum haussier apres le creux de 2011Q3 :
               doit donner Decrochage puis Reprise.
  Covid        2 trimestres : la phase survit-elle a la censure de duree ?
  2023-2026    au-dessus du potentiel en permanence : ni Reprise ni Decrochage
               ne devraient apparaitre.

Dependances : numpy, pandas, scipy, statsmodels
"""

import importlib
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

PIB = "GDPC1.xlsx"
POTENTIEL = "GDPPOT.xlsx"

TEMOINS = [
    ("1980Q2", "1982Q4", "1980/81 : deux recessions"),
    ("2001Q3", "2004Q2", "2001-2004 : creux de 2001Q4"),
    ("2010Q4", "2013Q4", "2011-2013 : sous potentiel, momentum haussier"),
    ("2020Q1", "2021Q2", "Covid : phase de 2 trimestres"),
    ("2023Q4", "2026Q2", "recent : au-dessus du potentiel"),
]


# ---------------------------------------------------------------------------
# Chargement d'une variante du module
# ---------------------------------------------------------------------------

def dater_avec(**params):
    """Date le cycle avec des parametres modifies, sans toucher au module.

    Une copie du source est ecrite dans un fichier temporaire, les constantes y
    sont substituees, puis le module est importe sous un nom unique. Cela evite
    tout effet de bord entre configurations — un simple `cd.PARAM = x` peut
    rester sans effet si le parametre est lu ailleurs qu'a l'appel.
    """
    src = Path("cycle_dating.py").read_text()
    for cle, val in params.items():
        motif = rf"^{cle} = [\d.]+"
        if not re.search(motif, src, flags=re.M):
            raise KeyError(f"Parametre inconnu : {cle}")
        src = re.sub(motif, f"{cle} = {val}", src, count=1, flags=re.M)
    d = Path(tempfile.mkdtemp())
    nom = "cd_" + "_".join(f"{k[:4]}{v}" for k, v in sorted(params.items())).replace(".", "")
    nom = (nom or "cd_base")[:60]
    (d / f"{nom}.py").write_text(src)
    for aux in ("firth.py", "usrec.py"):
        if Path(aux).exists():
            (d / aux).write_text(Path(aux).read_text())
    sys.path.insert(0, str(d))
    if nom in sys.modules:
        del sys.modules[nom]
    mod = importlib.import_module(nom)
    Y, P = mod.charger_fred(PIB), mod.charger_fred(POTENTIEL)
    return mod.dater(Y, P), mod


# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------

def metriques(D, mod, recession=None):
    ep = mod.episodes(D["phase"])
    out = dict(episodes=len(ep), transitions=len(ep) - 1,
               duree_med=ep.duree.median(), duree_max=int(ep.duree.max()),
               longues=int((ep.duree > 16).sum()))
    if recession is not None:
        y = recession.reindex(D.index)
        dec = D["phase"] == "Decrochage"
        ent = list(y.index[(y == 1) & (y.shift(1, fill_value=0) == 0)])
        hit = sum(1 for p in ent
                  if "Decrochage" in D["phase"].loc[max(D.index[0], p - 2):
                                                    min(D.index[-1], p + 3)].values)
        out.update(recess=f"{hit}/{len(ent)}",
                   couv=round(100 * (dec & (y == 1)).sum() / max((y == 1).sum(), 1)),
                   prec=round(100 * (dec & (y == 1)).sum() / max(dec.sum(), 1)))
    return out


def afficher_temoins(D, largeur=4):
    for a, b, lab in TEMOINS:
        try:
            seq = " ".join(x[:largeur] for x in D["phase"].loc[a:b])
        except Exception:
            seq = "hors periode"
        print(f"  {lab:<46} {seq}")


# ---------------------------------------------------------------------------
# 1. Datation courante
# ---------------------------------------------------------------------------

def courante(recession=None):
    D, mod = dater_avec()
    print("=" * 74)
    print("DATATION COURANTE")
    print("=" * 74)
    print(mod.controle_qualite(D, recession))
    print("\nPERIODES TEMOINS")
    afficher_temoins(D)
    return D, mod


# ---------------------------------------------------------------------------
# 2. Balayage d'un parametre
# ---------------------------------------------------------------------------

def balayer(parametre, valeurs, recession=None, temoin=None):
    """Fait varier UN parametre, tout le reste inchange.

    temoin : couple (debut, fin) dont la sequence est affichee a droite.
    """
    print("=" * 74)
    print(f"BALAYAGE : {parametre}")
    print("=" * 74)
    entete = f"{parametre:>10}{'ep':>5}{'med':>6}{'max':>5}{'>16':>5}"
    if recession is not None:
        entete += f"{'recess':>8}{'couv':>6}{'prec':>6}"
    if temoin:
        entete += f"   {temoin[0]}-{temoin[1]}"
    print(entete)
    lignes = []
    for v in valeurs:
        D, mod = dater_avec(**{parametre: v})
        m = metriques(D, mod, recession)
        s = (f"{v:>10}{m['episodes']:>5}{m['duree_med']:>6.1f}"
             f"{m['duree_max']:>5}{m['longues']:>5}")
        if recession is not None:
            s += f"{m['recess']:>8}{m['couv']:>5}%{m['prec']:>5}%"
        if temoin:
            s += "   " + " ".join(x[:4] for x in D["phase"].loc[temoin[0]:temoin[1]])
        print(s)
        lignes.append(dict(valeur=v, **m))
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 3. Comparaison de deux configurations
# ---------------------------------------------------------------------------

def comparer(config_a: dict, config_b: dict, recession=None, detail=True):
    """Compare deux jeux de parametres et liste les trimestres qui changent."""
    Da, ma = dater_avec(**config_a)
    Db, mb = dater_avec(**config_b)
    print("=" * 74)
    print(f"A = {config_a or 'defaut'}")
    print(f"B = {config_b or 'defaut'}")
    print("=" * 74)
    for lab, D, mod in (("A", Da, ma), ("B", Db, mb)):
        m = metriques(D, mod, recession)
        print(f"  {lab} : {m}")
    i = Da.index.intersection(Db.index)
    diff = Da["phase"][i] != Db["phase"][i]
    print(f"\n  {diff.sum()} trimestres changent ({100 * diff.mean():.0f} %)")
    if diff.sum() and detail:
        print("\n  QUI DEVIENT QUOI (lignes = A, colonnes = B)")
        print(pd.crosstab(Da["phase"][i][diff], Db["phase"][i][diff]).to_string())
        per = [str(p) for p in i[diff]]
        print(f"\n  periodes : {', '.join(per[:24])}"
              + (" ..." if len(per) > 24 else ""))
    print("\n  temoins A :")
    afficher_temoins(Da)
    print("  temoins B :")
    afficher_temoins(Db)
    return Da, Db


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from usrec import usrec
        rec = usrec("Q")
    except Exception:
        rec = None
        print("usrec.py absent : metriques NBER desactivees.\n")

    D, mod = courante(rec)

    print("\n")
    balayer("AMPLITUDE_MIN", [1.0, 1.25, 1.5, 2.0, 2.5], rec,
            temoin=("2023Q4", "2026Q2"))

    print("\n")
    balayer("CORRECTION_DERIVE", [0, 8, 12, 16], rec,
            temoin=("2011Q1", "2012Q2"))

    print("\n")
    comparer({"CORRECTION_DERIVE": 0}, {"CORRECTION_DERIVE": 12}, rec)

    # Autres balayages possibles :
    #   balayer("BANDE_NIVEAU", [0.0, 0.15, 0.35], rec)
    #   balayer("DUREE_MIN_PHASE", [1, 2, 3, 4], rec)
    #   balayer("SEUIL_CHOC", [1.5, 2.0, 2.5, 3.0], rec, temoin=("2020Q1","2021Q2"))
