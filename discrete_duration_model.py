"""
================================================================================
TEMPS DE SEJOUR DANS LES PHASES CONJONCTURELLES
Modele de duree en temps discret a risques concurrents
================================================================================

POURQUOI PAS UN SEMI-MARKOVIEN A 4 ETATS
----------------------------------------
Sur 1949T4-2026T1 on dispose de 42 episodes : 8 decrochages, 11 reprises,
10 explosions, 12 ralentissements. Un semi-markovien a 4 etats demande une loi de
duree par etat plus une matrice de transition, soit une trentaine de parametres
pour 41 evenements de sortie. Ce n'est pas identifiable.

Surtout, ce serait redondant. Les quatre phases ne sont pas des etats elementaires :
ce sont les quatre cases du croisement de deux variables binaires,

    NIVEAU    L = 1 si l'activite est au-dessus de sa norme, 0 sinon
    MOMENTUM  M = 1 si le momentum est haussier, 0 sinon

    Explosion (1,1)   Ralentissement (1,0)   Reprise (0,1)   Decrochage (0,0)

Une phase se termine des que L ou M bascule. C'est donc un probleme de RISQUES
CONCURRENTS, et la duree de la phase est le minimum de deux durees residuelles.

Ce cadre explique la structure de transition observee sans avoir a l'estimer :

  - Depuis Decrochage (0,0), l'ecart est sous -1 pt et baisse : L ne peut pas
    basculer vers 1. Seul M le peut  ->  sortie forcee vers Reprise. Observe 8/8.
  - Depuis Explosion (1,1), l'ecart est au-dessus de +1 pt et monte : L ne peut
    pas basculer vers 0  ->  sortie forcee vers Ralentissement. Observe 10/10.
  - Depuis Reprise (0,1) et Ralentissement (1,0), les deux peuvent basculer.
    C'est la, et seulement la, qu'il y a une vraie incertitude de direction.

Deux horloges, pas une : quand M bascule, l'horloge de M repart a zero mais celle
de L continue. Au 1er trimestre 2026 la phase a 8 trimestres alors que l'etat de
niveau en a 14. Un modele qui ne suit que la duree de la phase perd cette
information.

CE QUE MESURE LE MODELE
-----------------------
    h_j(d | X) = P(l'etat j bascule au trimestre d | il a tenu jusqu'a d, X)

estime par un GLM binomial a lien complementaire log-log sur un panel
episode-trimestre. Le lien cloglog est l'analogue exact, en temps discret, d'un
modele a hasards proportionnels en temps continu. La censure a droite est prise en
charge naturellement : un episode en cours contribue ses trimestres avec y = 0.

    S(s) = prod_{k=1..s} (1 - h_L(d_L+k)) (1 - h_M(d_M+k))
    E[duree residuelle] = somme_{s>=1} S(s)

TROIS PIEGES
------------
1. Les regles de datation interdisent les phases de moins de 3 trimestres. Le
   hasard est donc nul par construction pour d < 3. Il faut conditionner sur
   d >= 3 (parametre dmin), sinon on prend un artefact d'algorithme pour de la
   dependance a la duree. Sur ces donnees, beta passe de +1,08 a +0,90 pour le
   momentum quand on corrige, et de +0,83 a +0,65 pour le niveau.

2. La datation est ex post : elle mobilise l'information jusqu'a t+9. Les durees
   estimees ici ne sont pas des durees temps reel. La validation de la section 5
   ne corrige que la partie estimation, pas la datation elle-meme.

3. |ecart au potentiel| est le meilleur predicteur de toutes les covariables
   testees. C'est circulaire : la bascule de niveau EST le franchissement du seuil
   sur l'ecart. Cette variable est deliberement exclue.

Donnees : phases_et_covariables.csv (306 trimestres, 1949T4-2026T1)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

NIVEAU   = {"Explosion": 1, "Ralentissement": 1, "Reprise": 0, "Decrochage": 0}
MOMENTUM = {"Explosion": 1, "Reprise": 1, "Ralentissement": 0, "Decrochage": 0}
DIMENSIONS = {"L": NIVEAU, "M": MOMENTUM}


# ==============================================================================
# 1. EPISODES ET PANEL
# ==============================================================================

def episodes(labels, mapping):
    """Decoupe une sequence de phases en episodes d'un etat binaire.

    Retourne une liste de (debut, fin, etat, observe).
    observe = False si l'episode est censure a droite : soit il court encore en
    fin d'echantillon, soit il est interrompu par un trou (les trois trimestres
    du choc Covid, sortis de la taxonomie).
    """
    out, cur, deb = [], None, None
    for i, lab in enumerate(labels):
        v = mapping.get(lab) if lab else None
        if v is None:
            if cur is not None:
                out.append((deb, i - 1, cur, False))
                cur = None
            continue
        if cur is None:
            cur, deb = v, i
        elif v != cur:
            out.append((deb, i - 1, cur, True))
            cur, deb = v, i
    if cur is not None:
        out.append((deb, len(labels) - 1, cur, False))
    return out


def panel(labels, mapping, covars=None, dmin=3):
    """Panel episode-trimestre : une ligne par (episode, trimestre d'anciennete).

    y = 1 au trimestre ou l'episode se termine, 0 sinon. Un episode censure ne
    produit que des y = 0, ce qui est exactement le traitement correct de la
    censure a droite dans un modele de hasard en temps discret.

    dmin : anciennete minimale conservee. Mettre 3 pour neutraliser la regle de
    duree minimale de l'algorithme de datation.
    """
    lignes = []
    for k, (a, b, v, obs) in enumerate(episodes(labels, mapping)):
        n = b - a + 1
        for d in range(max(1, dmin), n + 1):
            r = {"episode": k, "t": a + d - 1, "d": d, "logd": np.log(d),
                 "etat": float(v), "y": int(obs and d == n)}
            if covars is not None:
                for c in covars.columns:
                    r[c] = covars[c].iloc[a + d - 1]
            lignes.append(r)
    return pd.DataFrame(lignes)


# ==============================================================================
# 2. ESTIMATION DU HASARD
# ==============================================================================

def ajuster(df, covariables=()):
    """cloglog(h) = a + b*log(d) + c*etat + gamma'X

    b > 0  : le risque de sortie croit avec l'anciennete (dependance positive,
             semi-markovien justifie)
    b = 0  : hasard constant, une chaine de Markov simple suffirait
    """
    X = pd.DataFrame({"const": 1.0, "logd": df["logd"], "etat": df["etat"]})
    for c in covariables:
        X[c] = df[c].values
    X = X.dropna()
    y = df.loc[X.index, "y"]
    modele = sm.GLM(y, X,
                    family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
    return modele


def hasard(modele, d, etat, x=None):
    """Probabilite de bascule au trimestre d, sachant qu'on a tenu jusque-la."""
    z = (modele.params["const"]
         + modele.params["logd"] * np.log(d)
         + modele.params["etat"] * etat)
    if x:
        for c, v in x.items():
            if c in modele.params:
                z += modele.params[c] * v
    return 1.0 - np.exp(-np.exp(z))


# ==============================================================================
# 3. SURVIE RESIDUELLE ET RISQUES CONCURRENTS
# ==============================================================================

def survie(modeles, etats, anciennetes, x=None, H=40):
    """P(la phase dure encore au moins s trimestres), pour s = 0..H.

    modeles      {"L": GLM, "M": GLM}
    etats        {"L": 0 ou 1, "M": 0 ou 1}
    anciennetes  {"L": int, "M": int}   les deux horloges, distinctes
    x            valeurs des covariables, supposees constantes sur l'horizon
    """
    S = np.ones(H + 1)
    hL = np.zeros(H + 1)
    hM = np.zeros(H + 1)
    for s in range(1, H + 1):
        hL[s] = hasard(modeles["L"], anciennetes["L"] + s, etats["L"], x)
        hM[s] = hasard(modeles["M"], anciennetes["M"] + s, etats["M"], x)
        S[s] = S[s - 1] * (1 - hL[s]) * (1 - hM[s])
    return S, hL, hM


def synthese(S, hL, hM, etats):
    """Duree residuelle esperee, mediane, et direction probable de la sortie."""
    H = len(S) - 1
    pL = sum(S[s - 1] * hL[s] * (1 - hM[s]) for s in range(1, H + 1))
    pM = sum(S[s - 1] * hM[s] * (1 - hL[s]) for s in range(1, H + 1))
    pB = sum(S[s - 1] * hL[s] * hM[s] for s in range(1, H + 1))
    tot = pL + pM + pB
    L, M = etats["L"], etats["M"]
    nom = lambda l, m: {(1, 1): "Explosion", (1, 0): "Ralentissement",
                        (0, 1): "Reprise", (0, 0): "Decrochage"}[(l, m)]
    return {
        "esperance": float(S[1:].sum()),
        "mediane": next((s for s in range(1, H + 1) if S[s] <= 0.5), np.nan),
        "survie": S,
        "sortie_par_niveau": {"proba": pL / tot, "vers": nom(1 - L, M)},
        "sortie_par_momentum": {"proba": pM / tot, "vers": nom(L, 1 - M)},
        "bascule_double": {"proba": pB / tot, "vers": nom(1 - L, 1 - M)},
    }


def kaplan_meier(labels, mapping, etat, dmin=1):
    """Estimateur non parametrique de la survie, sans hypothese de forme.
    A produire systematiquement comme reference : si le modele parametrique
    s'en ecarte beaucoup, c'est la forme fonctionnelle qui parle, pas les donnees.
    """
    df = panel(labels, mapping, dmin=dmin)
    d = df[df.etat == etat]
    S, out = 1.0, {}
    for dd in sorted(d.d.unique()):
        g = d[d.d == dd]
        if len(g):
            S *= 1 - g.y.sum() / len(g)
        out[int(dd)] = S
    return out


# ==============================================================================
# 4. ESTIMATION COURANTE
# ==============================================================================

def estimer(labels, covars=None, covariables=("spx_rdt12m",), dmin=3, H=40, x=None):
    """Ajuste les deux equations et rend l'estimation pour la phase en cours."""
    modeles, etats, anc = {}, {}, {}
    for cle, mapping in DIMENSIONS.items():
        df = panel(labels, mapping, covars=covars, dmin=dmin)
        modeles[cle] = ajuster(df, covariables)
        a, b, v, _ = episodes(labels, mapping)[-1]
        etats[cle], anc[cle] = int(v), b - a + 1
    if x is None and covars is not None:
        x = {c: float(covars[c].iloc[-1]) for c in covariables}
    S, hL, hM = survie(modeles, etats, anc, x=x, H=H)
    r = synthese(S, hL, hM, etats)
    r.update(modeles=modeles, etats=etats, anciennetes=anc, covariables=x)
    return r


# ==============================================================================
# 5. VALIDATION PSEUDO-HORS-ECHANTILLON
# ==============================================================================

def backtest(labels, covars, debut=140, covariables=("spx_rdt12m",), dmin=3):
    """Fenetre extensible : a chaque trimestre on reestime sur la seule
    information disponible alors, et on compare la duree residuelle predite a
    celle qui s'est realisee. Reference naive : duree moyenne passee de la meme
    phase, moins l'anciennete deja ecoulee.
    """
    res = []
    for i in range(debut, len(labels)):
        p = labels[i]
        if p is None or p == "Choc Covid":
            continue
        j = i
        while j + 1 < len(labels) and labels[j + 1] == p:
            j += 1
        if j == len(labels) - 1:
            continue                      # phase encore en cours : censuree
        reel = j - i + 1
        k = i
        while k > 0 and labels[k - 1] == p:
            k -= 1
        ecoule = i - k + 1
        try:
            r = estimer(labels[:i + 1], covars.iloc[:i + 1], covariables, dmin)
            pred = r["esperance"]
        except Exception:
            continue
        passees = []
        s = 0
        while s < i:
            if labels[s] == p:
                e = s
                while e + 1 <= i and labels[e + 1] == p:
                    e += 1
                if e < i:
                    passees.append(e - s + 1)
                s = e + 1
            else:
                s += 1
        naif = max(np.mean(passees) - ecoule, 0.5) if passees else 4.0
        res.append((i, p, reel, pred, naif))
    R = pd.DataFrame(res, columns=["i", "phase", "reel", "modele", "naif"])
    for c in ("modele", "naif"):
        e = R["reel"] - R[c]
        R.attrs[c] = {"EAM": e.abs().mean(), "REQM": np.sqrt((e ** 2).mean()),
                      "biais": e.mean()}
    return R


# ==============================================================================
# 6. DEMONSTRATION
# ==============================================================================

def main(chemin="phases_et_covariables.csv"):
    D = pd.read_csv(chemin)
    labels = D["phase"].tolist()
    covars = D[["spx_rdt12m", "sahm"]].reset_index(drop=True)

    print("=" * 78)
    print("1. DEPENDANCE A LA DUREE")
    print("=" * 78)
    for nom, mapping in [("NIVEAU", NIVEAU), ("MOMENTUM", MOMENTUM)]:
        for dmin in (1, 3):
            m = ajuster(panel(labels, mapping, dmin=dmin))
            b, se = m.params["logd"], m.bse["logd"]
            print(f"  {nom:9s} dmin={dmin}  beta(log d) = {b:+.3f}"
                  f"  (z = {b/se:+.2f}, p = {m.pvalues['logd']:.3f})")
    print("  dmin=3 neutralise la regle de duree minimale de la datation.")

    print("\n" + "=" * 78)
    print("2. COVARIABLES")
    print("=" * 78)
    for nom, mapping in [("NIVEAU", NIVEAU), ("MOMENTUM", MOMENTUM)]:
        df = panel(labels, mapping, covars=covars, dmin=3)
        base = ajuster(df)
        print(f"  --- {nom} (sorties : {int(df.y.sum())}) ---")
        for cv in [(), ("spx_rdt12m",), ("sahm",), ("spx_rdt12m", "sahm")]:
            m = ajuster(df, cv)
            det = "  ".join(f"{c} = {m.params[c]:+.3f} (z = {m.params[c]/m.bse[c]:+.1f})"
                            for c in cv)
            print(f"    AIC = {m.aic:7.2f}   {det or 'sans covariable'}")

    print("\n" + "=" * 78)
    print("3. PHASE EN COURS")
    print("=" * 78)
    r = estimer(labels, covars)
    print(f"  Phase        : {D['phase'].iloc[-1]}   (dernier trimestre : {D['trimestre'].iloc[-1]})")
    print(f"  Horloge NIVEAU   : etat {r['etats']['L']}, {r['anciennetes']['L']} trimestres")
    print(f"  Horloge MOMENTUM : etat {r['etats']['M']}, {r['anciennetes']['M']} trimestres")
    print(f"  Duree residuelle esperee : {r['esperance']:.1f} trimestres")
    print(f"  Mediane                  : {r['mediane']:.0f} trimestres")
    for cle in ("sortie_par_niveau", "sortie_par_momentum", "bascule_double"):
        print(f"  {cle:22s} {100*r[cle]['proba']:4.0f} %  -> {r[cle]['vers']}")
    S = r["survie"]
    print("  s        " + " ".join(f"{s:5d}" for s in (1, 2, 3, 4, 6, 8, 12)))
    print("  P(>= s)  " + " ".join(f"{S[s]:5.2f}" for s in (1, 2, 3, 4, 6, 8, 12)))

    print("\n" + "=" * 78)
    print("4. VALIDATION PSEUDO-HORS-ECHANTILLON")
    print("=" * 78)
    R = backtest(labels, covars)
    print(f"  {len(R)} trimestres evalues")
    for c in ("modele", "naif"):
        a = R.attrs[c]
        print(f"  {c:8s} EAM = {a['EAM']:5.2f}   REQM = {a['REQM']:5.2f}   biais = {a['biais']:+5.2f}")
    print("  Le biais positif signifie que le modele sous-estime la duree restante.")


if __name__ == "__main__":
    main()
