"""backtest.py — Signaux de seuil sur un score, et choix du decalage d'execution.

    from backtest import merger, signaux, optimiser, simuler, rapport

    df  = merger("scores.xlsx", "spx.csv")            # dates communes
    df  = signaux(df, seuil_entree=15, seuil_sortie=50)
    opt = optimiser(df)                                # quel decalage ?
    a, v, pnl, tx = simuler(df, opt["attente_entree"], opt["attente_sortie"])
    print(rapport(pnl, tx, opt))

Lecture historique, aucune prevision.

L'ATTENTE, ET CE QU'ELLE SIGNIFIE
---------------------------------
`attente_entree = 2` veut dire : le score franchit le seuil en t, on ACHETE en
t+2. Le signal repose donc sur une observation de deux trimestres plus tot.

C'est une regle pleinement APPLICABLE : au moment d'executer, le score est
connu depuis deux trimestres. Rien n'est emprunte au futur.

L'interet vient de ce que le niveau du S&P reagit avec retard sur le score. Si
le creux de marche suit le signal de deux trimestres, attendre permet d'acheter
moins cher. C'est ce decalage que l'optimisation mesure.

POURQUOI UN BOOTSTRAP
----------------------
Avec 6 a 10 signaux sur cinquante ans, le meilleur decalage peut n'etre le
meilleur que par hasard. On reechantillonne les signaux avec remise : si un
decalage gagne dans 80 % des tirages il est robuste, s'il gagne dans 30 % c'est
du bruit.

Dependances : numpy, pandas
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------

def merger(fichier_score, fichier_prix, freq="Q"):
    """Fusionne score et prix sur les dates COMMUNES.

    Chaque fichier : dates en premiere colonne, valeur en deuxieme.
    Excel ou CSV, detecte par l'extension.
    """
    def lire(f):
        d = pd.read_excel(f) if str(f).endswith((".xlsx", ".xls")) else pd.read_csv(f)
        d = d.iloc[:, :2]
        d.columns = ["date", "valeur"]
        d["date"] = pd.PeriodIndex(pd.to_datetime(d["date"].astype(str)), freq=freq)
        return d.dropna().drop_duplicates("date").set_index("date")["valeur"]

    df = pd.DataFrame({"score": lire(fichier_score), "prix": lire(fichier_prix)})
    df = df.dropna().sort_index()
    print(f"{len(df)} dates communes : {df.index.min()} -> {df.index.max()}")
    return df


def signaux(df, seuil_entree=15, seuil_sortie=50):
    """Marque les FRANCHISSEMENTS de seuil.

    entree = 1 le trimestre ou le score passe SOUS seuil_entree
    sortie = 1 le trimestre ou il passe AU-DESSUS de seuil_sortie

    On code le franchissement, pas le niveau : un score qui reste sous 15
    pendant huit trimestres ne doit produire qu'un signal, pas huit.
    """
    d = df.copy()
    p = d["score"].shift(1)
    d["entree"] = ((p >= seuil_entree) & (d["score"] < seuil_entree)).astype(int)
    d["sortie"] = ((p <= seuil_sortie) & (d["score"] > seuil_sortie)).astype(int)
    print(f"{d.entree.sum()} signaux d'entree, {d.sortie.sum()} de sortie")
    return d


# ---------------------------------------------------------------------------

def _prix_apres(df, dates, k):
    """Prix k periodes APRES chaque signal. k=0 : le jour du signal."""
    pos = {d: i for i, d in enumerate(df.index)}
    out = []
    for d in dates:
        j = pos[d] + k
        out.append(float(df["prix"].iloc[j]) if 0 <= j < len(df) else np.nan)
    return np.array(out)


def optimiser(df, attentes=range(0, 5), n_boot=2000, seed=0):
    """Combien de trimestres attendre apres le signal, a l'entree et a la sortie ?

    A l'ENTREE on veut le prix le plus BAS, a la SORTIE le plus HAUT. Les niveaux
    d'indice etant tres differents selon l'epoque (69 en 1974, 4100 en 2022), une
    moyenne de prix bruts serait ecrasee par les dates recentes. On compare donc
    chaque prix decale au prix DU JOUR DU SIGNAL, en pourcentage.

        gain_achat_pct  positif = on achete moins cher qu'au signal
        gain_vente_pct  positif = on vend plus cher qu'au signal
    """
    de = list(df.index[df.entree == 1])
    ds = list(df.index[df.sortie == 1])
    if not de or not ds:
        raise ValueError("Aucun signal : verifiez les seuils.")
    ref_e, ref_s = _prix_apres(df, de, 0), _prix_apres(df, ds, 0)

    ec = {}
    lignes = []
    for k in attentes:
        pe, ps = _prix_apres(df, de, k), _prix_apres(df, ds, k)
        ee = 100 * (pe - ref_e) / ref_e
        es = 100 * (ps - ref_s) / ref_s
        ec[k] = (ee, es)
        lignes.append(dict(
            attente=k,
            n_entrees=int(np.isfinite(ee).sum()),
            gain_achat_pct=round(-float(np.nanmean(ee)), 2) if np.isfinite(ee).any() else np.nan,
            n_sorties=int(np.isfinite(es).sum()),
            gain_vente_pct=round(float(np.nanmean(es)), 2) if np.isfinite(es).any() else np.nan))
    T = pd.DataFrame(lignes)

    # bootstrap sur les signaux : quel k gagne le plus souvent ?
    rng = np.random.default_rng(seed)
    ks = list(attentes)
    ne, ns = len(de), len(ds)
    ge = {k: 0 for k in ks}
    gs = {k: 0 for k in ks}
    for _ in range(n_boot):
        i = rng.integers(0, ne, ne)
        m = {k: np.nanmean(ec[k][0][i]) for k in ks if np.isfinite(ec[k][0][i]).any()}
        if m:
            ge[min(m, key=m.get)] += 1          # entree : ecart le plus NEGATIF
        j = rng.integers(0, ns, ns)
        m = {k: np.nanmean(ec[k][1][j]) for k in ks if np.isfinite(ec[k][1][j]).any()}
        if m:
            gs[max(m, key=m.get)] += 1          # sortie : ecart le plus POSITIF
    T["stabilite_achat"] = T.attente.map(lambda k: round(ge[k] / n_boot, 3))
    T["stabilite_vente"] = T.attente.map(lambda k: round(gs[k] / n_boot, 3))

    ke = int(T.loc[T.gain_achat_pct.idxmax(), "attente"])
    ks_ = int(T.loc[T.gain_vente_pct.idxmax(), "attente"])
    return dict(table=T, attente_entree=ke, attente_sortie=ks_,
                stabilite_entree=float(T.loc[T.attente == ke, "stabilite_achat"].iloc[0]),
                stabilite_sortie=float(T.loc[T.attente == ks_, "stabilite_vente"].iloc[0]))


# ---------------------------------------------------------------------------

def simuler(df, attente_entree=0, attente_sortie=0):
    """Parcourt les signaux et enregistre les allers-retours.

    attente_entree / attente_sortie : nombre de periodes a ATTENDRE apres le
    signal avant d'executer. 0 = execution le jour du signal.

    Une position ouverte en fin d'echantillon n'est pas comptee : la valoriser
    au dernier prix supposerait une vente qui n'a pas eu lieu.
    """
    pos = {d: i for i, d in enumerate(df.index)}
    achats, ventes, pnl, tx = [], [], [], []
    en_position = False
    prix_achat = date_achat = None

    for date, ligne in df.iterrows():
        i = pos[date]
        if not en_position and ligne["entree"] == 1:
            j = i + attente_entree
            if 0 <= j < len(df):
                prix_achat, date_achat = float(df["prix"].iloc[j]), df.index[j]
                achats.append(prix_achat)
                en_position = True
        elif en_position and ligne["sortie"] == 1:
            j = i + attente_sortie
            if 0 <= j < len(df) and df.index[j] > date_achat:
                pv = float(df["prix"].iloc[j])
                ventes.append(pv)
                g = pv - prix_achat
                pnl.append(g)
                tx.append(dict(date_achat=str(date_achat), prix_achat=round(prix_achat, 2),
                               date_vente=str(df.index[j]), prix_vente=round(pv, 2),
                               pnl=round(g, 2),
                               rendement_pct=round(100 * g / prix_achat, 2)))
                en_position = False

    if en_position:
        print(f"position ouverte : achat {date_achat} a {prix_achat:.2f}, hors P&L")
    return achats, ventes, pnl, pd.DataFrame(tx)


def rapport(pnl, tx, opt=None):
    L = []
    if opt is not None:
        L += ["ATTENTES RETENUES",
              f"  achat : {opt['attente_entree']} trimestre(s) apres le signal "
              f"(stabilite {opt['stabilite_entree']:.0%})",
              f"  vente : {opt['attente_sortie']} trimestre(s) apres le signal "
              f"(stabilite {opt['stabilite_sortie']:.0%})", ""]
    if len(tx) == 0:
        return "\n".join(L + ["aucun aller-retour complet"])
    L += [f"{len(tx)} transactions, {(tx.pnl > 0).sum()} gagnantes",
          f"P&L total : {sum(pnl):+.1f}",
          f"rendement moyen par transaction : {tx.rendement_pct.mean():+.1f} %",
          "", tx.to_string(index=False)]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    df = signaux(merger(sys.argv[1], sys.argv[2]),
                 float(sys.argv[3]) if len(sys.argv) > 3 else 15,
                 float(sys.argv[4]) if len(sys.argv) > 4 else 50)
    opt = optimiser(df)
    print("\n" + opt["table"].to_string(index=False))
    a, v, pnl, tx = simuler(df, opt["attente_entree"], opt["attente_sortie"])
    print("\n" + rapport(pnl, tx, opt))
