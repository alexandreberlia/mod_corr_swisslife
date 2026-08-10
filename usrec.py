"""usrec.py — Indicatrice de recession NBER, saisie manuellement.

Pourquoi ce fichier
-------------------
La colonne 'NBER' du panel Bloomberg n'est PAS le NBER : 21 episodes au lieu de
11, avec 1961-1963 classe en recession et 2009Q2-2010Q4 aussi. Une validation
menee contre elle donne une AUC de 0.48 — pire que le hasard — non parce que le
score est mauvais, mais parce que la cible est fausse.

Sans acces reseau a FRED, les dates sont saisies depuis la chronologie
officielle du NBER (Business Cycle Dating Committee). Elles sont publiques,
stables, et verifiables sur nber.org/research/business-cycle-dating.

Interet : c'est la SEULE cible independante du filtre de Hamilton. Toute
validation menee contre la datation maison est partiellement circulaire — le
score et les phases derivent tous deux du PIB.
"""

import pandas as pd

# (pic, creux) — mois du pic = dernier mois d'expansion,
# mois du creux = dernier mois de recession.
RECESSIONS = [
    ("1948-11", "1949-10"), ("1953-07", "1954-05"), ("1957-08", "1958-04"),
    ("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"),
    ("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04"),
]


def usrec(freq: str = "Q", debut: str = "1947-01", fin: str = "2026-06") -> pd.Series:
    """Indicatrice de NIVEAU : 1 pendant toute la recession.

    En trimestriel, un trimestre est marque des qu'UN mois au moins est en
    recession — convention du NBER lui-meme pour ses dates trimestrielles.
    """
    m = pd.period_range(debut, fin, freq="M")
    s = pd.Series(0, index=m, name="usrec")
    for pic, creux in RECESSIONS:
        p, c = pd.Period(pic, "M"), pd.Period(creux, "M")
        s.loc[(s.index >= p) & (s.index <= c)] = 1
    if freq.upper().startswith("Q"):
        s = s.groupby(s.index.asfreq("Q")).max()
        s.index = pd.PeriodIndex(s.index, freq="Q")
    return s


def usrec_entree(freq: str = "Q", **kw) -> pd.Series:
    """Indicatrice de TRANSITION : 1 a la seule periode d'entree en recession.

    Distinction essentielle. Une indicatrice de niveau vaut 1 pendant toute la
    recession, si bien qu'un ajustement optimal place son pic au MILIEU de
    l'episode, pas a son entree. Pour mesurer une avance, c'est la transition
    qu'il faut viser.
    """
    s = usrec(freq, **kw)
    return ((s == 1) & (s.shift(1, fill_value=0) == 0)).astype(int).rename("usrec_entree")
