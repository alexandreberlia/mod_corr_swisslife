"""
features.py — catalogue de features orientées + calibration empirique des poids.

RÔLE : laboratoire de recherche. Tourne une fois par trimestre, pas tous les jours.
Utilise le futur (rendements réalisés) pour MESURER — c'est légitime ici, ça ne
l'est jamais dans portefeuille.py.

PRINCIPE — MONOTONIE
Un indicateur n'entre dans un CLASSEMENT que si sa relation au rendement futur est
monotone : "plus c'est haut, plus c'est haussier", toujours dans le même sens.
    ADX = 60 n'est PAS meilleur que ADX = 30           -> non classable
    momentum élevé EST meilleur que momentum faible    -> classable
Les indicateurs non monotones (ADX, ER, ATR, RVOL) mesurent un CONTEXTE. Pour les
rendre classables on leur adjoint un signe : ER x sign(momentum).

CATALOGUE RÉDUIT
8 features au lieu de 14. Les retirées (mom_3m, pente_ema50, ext_ema200, rev_21,
regularite, vol_confirme) étaient soit corrélées à plus de 0.85 avec une conservée,
soit non significatives à l'IC. Multiplier les candidats corrélés augmente le risque
de retenir par hasard celle qui a le mieux marché in-sample.

CHAÎNE D'USAGE
    features_orientees   -> QUI PEUT concourir      (8 candidats)
    ic_rapport           -> QUI MÉRITE (mesure + intervalle de confiance)
    selection_decorrelee -> on écarte les redondants
    poids_depuis_ic      -> QUI ENTRE, et à quel poids
    -> ParamsPF.poids
"""

import numpy as np
import pandas as pd


# ============================================================================
# 1. Catalogue de features ORIENTÉES
# ============================================================================

def features_orientees(ind, p_atr: int = 14) -> dict:
    """Indicateurs transformés en signaux orientés et normalisés.
    Convention respectée par toutes : 'plus haut = plus haussier'."""
    px = ind.price
    atr = ind.atr(p_atr)
    lr = ind.logret(px, 1)
    signe = np.sign(lr.rolling(20).sum())      # orientation du mouvement récent

    return {
        # --- momentum (monotone par construction) ---
        "mom_12_1":   lr.rolling(252).sum() - lr.rolling(21).sum(),
        "rev_5":     -lr.rolling(5).sum(),          # survendu = signal long
        "mom_sharpe": lr.rolling(126).mean() / lr.rolling(126).std().replace(0, np.nan),

        # --- position vs moyennes (distance signée, normalisée ATR) ---
        "ext_ema50":  (px - ind.ema(50)) / atr,
        "ext_kama":   (px - ind.kama(10, 2, 30)) / atr,

        # --- qualité de tendance ORIENTÉE (ER et ADX seuls n'ont pas de direction) ---
        "er_signe":   ind.er(10) * signe,
        "adx_signe":  (ind.adx(14) / 100) * signe,

        # --- anomalie low-vol : signe NÉGATIF (moins volatil = mieux) ---
        "low_vol":   -(atr / px),
    }


def features_convictions(ind, p_atr: int = 14) -> dict:
    """Features supplementaires exigees par des convictions de trading precises.

    Point cle : "cours AU NIVEAU de la MM mais QUI est en augmentation".
    C'est un profil NON monotone sur la distance (le mieux est d'etre PROCHE, pas
    loin) mais monotone sur la pente. On separe donc en deux features :
        proximite_ema50 = -|prix - EMA50| / ATR   -> plus haut = plus proche
        pente_ema50     = EMA50.diff(20) / ATR    -> plus haut = MM qui monte
    Une extension elevee (ext_ema50) signifie qu'on arrive APRES la hausse ;
    proximite eleve + pente positive = on capte le debut du mouvement.
    """
    px = ind.price
    atr = ind.atr(p_atr)
    lr = ind.logret(px, 1)
    ema50 = ind.ema(50)
    vol = ind.volume

    return {
        # --- momentum : progression du cours et de la MM ---
        "pente_ema50":     ema50.diff(20) / atr,
        "mom_glissant":    lr.rolling(20).sum() / (lr.rolling(20).std() * np.sqrt(20)),

        # --- regime : etre AU NIVEAU de la MM, pas au-dessus ---
        "proximite_ema50": -(px - ema50).abs() / atr,
        "juste_au_dessus": -((px - ema50) / atr - 0.3).abs(),   # optimum a +0.3 ATR

        # --- volatilite EN EXPANSION (et non faible) ---
        "atr_expansion":   ind.atr(p_atr) / ind.atr(p_atr).shift(20) - 1,
        "vol_regime":      ind.rank_pct(atr / px, 252),

        # --- volume EN AUGMENTATION ---
        "vol_expansion":   vol.rolling(5).mean() / vol.rolling(20).mean() - 1,
        "rvol_log":        np.log(ind.rvol(50).clip(0.1, 10)),
    }


def features_completes(ind, p_atr: int = 14) -> dict:
    """Catalogue complet : orientees + convictions + modulateurs bruts.
    C'est ce generateur qu'il faut passer a construire_panels pour explorer
    des combinaisons, car les CONDITIONS ont besoin des indicateurs bruts."""
    d = features_orientees(ind, p_atr)
    d.update(features_convictions(ind, p_atr))
    d.update(modulateurs(ind))
    return d


def modulateurs(ind) -> dict:
    """Indicateurs NON orientés : ils filtrent ou modulent, ils ne se classent pas."""
    return {
        "er":      ind.er(10),
        "er_rk":   ind.rank_pct(ind.er(10), 252),
        "adx":     ind.adx(14),
        "atr_pct": ind.atr(14) / ind.price,
        "rvol":    ind.rvol(50),
    }


# ============================================================================
# 2. Information Coefficient
# ============================================================================
# ATTENTION AU SIGLE : ici IC = Information Coefficient (Grinold & Kahn), une
# corrélation de rang — PAS un intervalle de confiance. L'intervalle de confiance
# est fourni séparément par les colonnes ic_bas / ic_haut.

def ic_serie(panel: pd.DataFrame, close: pd.DataFrame,
             horizon: int = 20, min_titres: int = 10) -> pd.Series:
    """IC = corrélation de rang, à chaque date, entre la feature et le rendement futur.
    Positif => la feature classe correctement les titres."""
    fwd = close.pct_change(horizon).shift(-horizon)
    f, r = panel.align(fwd, join="inner")
    valides = (f.notna() & r.notna()).sum(axis=1)
    return f.corrwith(r, axis=1, method="spearman").where(valides >= min_titres)


def newey_west_se(x: pd.Series, lag: int) -> float:
    """Erreur-type corrigée de l'autocorrélation (noyau de Bartlett).

    INDISPENSABLE : avec un horizon de h jours et des observations quotidiennes, les
    fenêtres de rendement futur SE CHEVAUCHENT — deux IC consécutifs partagent h-1
    jours sur h. L'écart-type naïf suppose l'indépendance et sous-estime l'incertitude
    d'un facteur ~sqrt(h). Vérifié : x1.0 sur bruit i.i.d., x3.7 sur série autocorrélée.
    """
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = (e @ e) / n
    for j in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - j / (lag + 1.0)) * (e[j:] @ e[:-j]) / n
    return np.sqrt(max(s, 0.0) / n)


def ic_rapport(panels: dict, close: pd.DataFrame, features: list,
               horizons=(5, 20, 60), min_titres: int = 10,
               alpha: float = 0.05) -> pd.DataFrame:
    """Tableau IC par feature et par horizon, avec intervalle de confiance.

    IC_moy      : pouvoir prédictif moyen. En equity RÉELLE, 0.02-0.05 est déjà bon.
    IR          : IC_moy / IC_std -> la STABILITÉ, ce qui compte le plus.
    ic_bas/haut : intervalle de confiance. S'il contient 0, la feature n'est PAS
                  significative, quel que soit son IC moyen.
    t_NW        : t corrigé du chevauchement. |t| > 2 -> significatif.
    t_naif      : t non corrigé, pour mesurer l'ampleur de l'illusion.
    n_eff       : taille d'échantillon effective (n_dates / horizon).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    lignes = []

    for f in features:
        if f not in panels:
            continue
        for h in horizons:
            ic = ic_serie(panels[f], close, h, min_titres).dropna()
            if len(ic) < 30:
                continue
            m, s, n = ic.mean(), ic.std(), len(ic)
            se = newey_west_se(ic, lag=h)
            lignes.append({
                "feature": f, "horizon": h, "IC_moy": m, "IC_std": s,
                "IR": m / s if s > 0 else np.nan,
                "se_NW": se, "ic_bas": m - z * se, "ic_haut": m + z * se,
                "t_NW": m / se if se and se > 0 else np.nan,
                "t_naif": m / s * np.sqrt(n) if s > 0 else np.nan,
                "significatif": bool(se and se > 0 and abs(m) > z * se),
                "n_dates": n, "n_eff": int(n / h),
            })

    df = pd.DataFrame(lignes)
    return df.sort_values("IR", key=abs, ascending=False).reset_index(drop=True) if len(df) else df


def poids_depuis_ic(rapport: pd.DataFrame, horizon: int = 20,
                    ir_min: float = 0.05, n_max: int = 5,
                    exiger_significatif: bool = True) -> dict:
    """Poids proportionnels à |IR|, signe repris de l'IC.
    Les poids peuvent être NÉGATIFS : Portefeuille normalise par la somme des
    valeurs absolues et centre les rangs, ce qui gère correctement ce cas."""
    r = rapport[rapport.horizon == horizon]
    if exiger_significatif and "significatif" in r.columns:
        r = r[r.significatif]
    r = r[r.IR.abs() > ir_min]
    r = r.reindex(r.IR.abs().sort_values(ascending=False).index).head(n_max)
    if r.empty:
        return {}
    total = r.IR.abs().sum()
    return {row.feature: round(float(np.sign(row.IC_moy) * abs(row.IR) / total), 3)
            for row in r.itertuples()}


# ============================================================================
# 3. Décorrélation
# ============================================================================

def correlation_features(panels: dict, features: list, date_min=None) -> pd.DataFrame:
    """Corrélation de rang moyenne entre features, calculée cross-sectionnellement.
    Au-delà de |0.8|, deux features apportent la même information."""
    dispo = [f for f in features if f in panels]
    n = len(dispo)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = panels[dispo[i]], panels[dispo[j]]
            if date_min is not None:
                a, b = a.loc[date_min:], b.loc[date_min:]
            M[i, j] = M[j, i] = a.corrwith(b, axis=1, method="spearman").mean()
    return pd.DataFrame(M, index=dispo, columns=dispo)


def selection_decorrelee(rapport: pd.DataFrame, corr: pd.DataFrame,
                         horizon: int = 20, seuil: float = 0.8,
                         exiger_significatif: bool = True) -> list:
    """Garde les features par |IR| décroissant, en écartant toute candidate corrélée
    au-delà du seuil à une feature déjà retenue."""
    r = rapport[rapport.horizon == horizon]
    if exiger_significatif and "significatif" in r.columns:
        r = r[r.significatif]
    ordre = r.reindex(r.IR.abs().sort_values(ascending=False).index).feature.tolist()
    gardees = []
    for f in ordre:
        if f in corr.columns and all(abs(corr.loc[f, g]) < seuil for g in gardees):
            gardees.append(f)
    return gardees


AUX = ("close", "open", "high", "low", "atr", "er_rk", "adx", "dvol")


def calibrer(panels: dict, close: pd.DataFrame, features: list = None,
             horizon: int = 20, split=None, seuil_corr: float = 0.8,
             n_max: int = 5, min_titres: int = 10) -> tuple:
    """Chaîne complète de calibration. Renvoie (poids, rapport, correlations).

    split : date de fin d'échantillon IN-SAMPLE. OBLIGATOIRE en pratique — calibrer
            sur toute la période puis backtester dessus est du look-ahead pur.
    """
    if features is None:
        features = [k for k in panels if k not in AUX]
    pan = {k: (v.loc[:split] if split is not None else v) for k, v in panels.items()}
    cl = close.loc[:split] if split is not None else close

    rapport = ic_rapport(pan, cl, features, horizons=(horizon,), min_titres=min_titres)
    corr = correlation_features(pan, features)
    gardees = selection_decorrelee(rapport, corr, horizon, seuil_corr)
    poids = poids_depuis_ic(rapport[rapport.feature.isin(gardees)], horizon, n_max=n_max)
    return poids, rapport, corr


# ============================================================================
# 4. COMBINAISONS — génération, évaluation binaire, sélection
# ============================================================================
#
# Change de logique par rapport aux sections précédentes :
#   sections 1-3 : "cet indicateur classe-t-il bien ?"        -> IC, corrélation de rang
#   section 4    : "cette COMBINAISON dit-elle correctement
#                   qui monte et qui baisse ?"                -> classification binaire
#
# On ne cherche plus l'ordre exact, seulement la CATÉGORISATION. Un score qui
# désigne les bons "hausse" et les bons "baisse" est bon, même s'il ordonne mal
# à l'intérieur de chaque groupe.
#
# On ne retient PAS "celles qui passent un test" mais LA MEILLEURE — et une
# meilleure par horizon, car un profil qui gagne à 5 jours n'est pas celui qui
# gagne à 12 mois.

FAMILLES = {
    "momentum": [
        "mom_12_1", "mom_sharpe", "mom_glissant", "pente_ema50", "rev_5",
    ],
    "regime": [
        "er_signe", "adx_signe", "ext_ema50", "ext_kama",
        "proximite_ema50", "juste_au_dessus",
    ],
    "volatilite": [
        "atr_expansion", "vol_regime", "low_vol",
    ],
    "volume": [
        "vol_expansion", "rvol_log",
    ],
}

HORIZONS_DEFAUT = {
    "court":  [5, 10],
    "moyen":  [21, 63],
    "long":   [252],
}


# ---------------------------------------------------------------- FONCTION 1

def definir_combinaisons(familles: dict = None,
                         obligatoires=("momentum", "regime", "volatilite"),
                         optionnelles=("volume",),
                         conditions_communes: dict = None,
                         convictions: dict = None,
                         poids_egaux: bool = True,
                         max_combis: int = 300,
                         graine: int = 0) -> dict:
    """Génère toutes les combinaisons respectant une contrainte de composition.

    Chaque combinaison prend EXACTEMENT une feature par famille obligatoire,
    plus éventuellement une par famille optionnelle. C'est ce qui garantit
    qu'aucune combinaison ne soit "trois momentums déguisés".

    convictions : combinaisons écrites à la main, ajoutées telles quelles.
                  Format {nom: {"score": {feat: poids}, "conditions": {...}}}

    Renvoie {nom: {"score": {...}, "conditions": {...}, "familles": {...}}}
    """
    fam = familles or FAMILLES
    cond_c = conditions_communes or {}
    rng = np.random.default_rng(graine)

    # produit cartésien : une feature par famille obligatoire
    listes = [[(f, feat) for feat in fam[f]] for f in obligatoires if f in fam]
    # les familles optionnelles ajoutent aussi la possibilité de ne rien prendre
    for f in optionnelles:
        if f in fam:
            listes.append([(f, feat) for feat in fam[f]] + [(f, None)])

    from itertools import product
    toutes = list(product(*listes))

    if len(toutes) > max_combis:
        idx = rng.choice(len(toutes), max_combis, replace=False)
        toutes = [toutes[i] for i in sorted(idx)]

    combis = {}
    for choix in toutes:
        retenues = {f: feat for f, feat in choix if feat is not None}
        if not retenues:
            continue
        nom = " + ".join(retenues.values())
        poids = {feat: 1.0 for feat in retenues.values()}
        if not poids_egaux:
            poids = {feat: round(float(rng.uniform(0.5, 1.5)), 2)
                     for feat in retenues.values()}
        combis[nom] = {"score": poids, "conditions": dict(cond_c),
                       "familles": retenues}

    if convictions:
        for nom, spec in convictions.items():
            combis[nom] = {"score": dict(spec.get("score", {})),
                           "conditions": dict(spec.get("conditions", {})),
                           "familles": spec.get("familles", {})}

    return combis


# ---------------------------------------------------------------- FONCTION 2

def evaluer_binaire(combi: dict, panels: dict, close: pd.DataFrame,
                    horizons=(5, 10, 21, 63, 252), q: float = 0.10,
                    reference: str = "zero", min_titres: int = 10,
                    alpha: float = 0.05) -> pd.DataFrame:
    """Qualité de la CATÉGORISATION hausse/baisse d'une combinaison, par horizon.

    Mécanique :
      1. la combinaison produit un score, on en tire les rangs percentiles
      2. PRÉDICTION : rang > 1-q  -> "hausse"   |  rang <= q -> "baisse"
         les titres du milieu ne reçoivent aucune prédiction (abstention)
      3. RÉALISÉ : rendement futur > seuil
         reference="zero"    -> hausse = rendement positif (sens commun)
         reference="mediane" -> hausse = mieux que la médiane du panel ce jour-là
                                (neutralise le marché : évite qu'un mois haussier
                                 rende toutes les prédictions "hausse" gagnantes)
      4. on compare, par date, puis on agrège

    Colonnes clés :
      taux_base_%      part réellement en hausse -> le score à battre
      prec_hausse_%    parmi les "hausse" prédits, combien ont monté
      prec_baisse_%    parmi les "baisse" prédits, combien ont baissé
      lift_hausse      prec_hausse / taux_base. >1 = mieux que le hasard
      exactitude_%     % correct sur l'ensemble des titres avec prédiction
      MCC              coefficient de Matthews, -1 à +1. La métrique la plus
                       robuste au déséquilibre des classes. 0 = hasard.
      ecart_repart_pt  |part prédite hausse - part réalisée hausse|, en points.
                       C'est la CALIBRATION : le score annonce-t-il la bonne
                       répartition du panel ?
      t_NW             significativité de (exactitude - hasard), corrigée du
                       chevauchement des fenêtres
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)

    sc = _score_combi(combi, panels, min_titres)
    rangs = sc.rank(axis=1, pct=True)
    pred_h = rangs > (1 - q)
    pred_b = rangs <= q

    lignes = []
    for h in horizons:
        fwd = close.pct_change(h).shift(-h)
        f_h, f_b, r = pred_h.align(fwd, join="inner")[0], None, None
        f_h, r = pred_h.align(fwd, join="inner")
        f_b, _ = pred_b.align(fwd, join="inner")

        if reference == "mediane":
            reel_h = r.gt(r.median(axis=1), axis=0)
        else:
            reel_h = r > 0
        valide = r.notna()
        reel_h = reel_h & valide

        assez = (valide.sum(axis=1) >= min_titres) & (f_h.sum(axis=1) > 0)
        if assez.sum() < 20:
            lignes.append({"horizon": h, "n_dates": int(assez.sum()),
                           "ERREUR": "moins de 20 dates exploitables"})
            continue

        VP = (f_h & reel_h & valide).sum(axis=1)      # prédit hausse, monté
        FP = (f_h & ~reel_h & valide).sum(axis=1)     # prédit hausse, baissé
        VN = (f_b & ~reel_h & valide).sum(axis=1)     # prédit baisse, baissé
        FN = (f_b & reel_h & valide).sum(axis=1)      # prédit baisse, monté

        VP, FP, VN, FN = (s.where(assez).dropna() for s in (VP, FP, VN, FN))
        base = (reel_h.sum(axis=1) / valide.sum(axis=1).replace(0, np.nan))\
            .where(assez).dropna()

        n_h, n_b = VP + FP, VN + FN
        prec_h = (VP / n_h.replace(0, np.nan))
        prec_b = (VN / n_b.replace(0, np.nan))
        exact = ((VP + VN) / (n_h + n_b).replace(0, np.nan))

        # taux de hasard : si on tirait au sort avec la même répartition
        hasard = (base * n_h + (1 - base) * n_b) / (n_h + n_b).replace(0, np.nan)

        # Matthews, agrégé sur l'ensemble de la période
        vp, fp, vn, fn = VP.sum(), FP.sum(), VN.sum(), FN.sum()
        den = np.sqrt((vp+fp) * (vp+fn) * (vn+fp) * (vn+fn))
        mcc = (vp*vn - fp*fn) / den if den > 0 else np.nan

        # calibration : part prédite en hausse vs part réalisée
        part_pred = (n_h / (n_h + n_b).replace(0, np.nan)).mean()
        ecart_rep = abs(part_pred - base.mean()) * 100

        ecart = (exact - hasard).dropna()
        se = newey_west_se(ecart, lag=h)
        t_nw = ecart.mean() / se if se and se > 0 else np.nan

        lignes.append({
            "horizon": h,
            "taux_base_%": base.mean() * 100,
            "prec_hausse_%": prec_h.mean() * 100,
            "prec_baisse_%": prec_b.mean() * 100,
            "lift_hausse": prec_h.mean() / base.mean() if base.mean() > 0 else np.nan,
            "lift_baisse": prec_b.mean() / (1 - base.mean()) if base.mean() < 1 else np.nan,
            "exactitude_%": exact.mean() * 100,
            "hasard_%": hasard.mean() * 100,
            "edge_pt": ecart.mean() * 100,
            "MCC": mcc,
            "ecart_repart_pt": ecart_rep,
            "t_NW": t_nw,
            "significatif": bool(se and se > 0 and abs(ecart.mean()) > z * se),
            "n_titres_pred": (n_h + n_b).mean(),
            "n_dates": len(ecart), "n_eff": int(len(ecart) / h),
        })

    return pd.DataFrame(lignes)


def _score_combi(combi: dict, panels: dict, min_titres: int = 10) -> pd.DataFrame:
    """Score cross-sectionnel d'une combinaison : rangs centrés, pondérés,
    normalisés par la somme des |poids|. Calculé sur les seuls titres éligibles."""
    ref = panels[next(iter(panels))]
    m = pd.DataFrame(True, index=ref.index, columns=ref.columns)

    for feat, (op, seuil) in combi.get("conditions", {}).items():
        x = panels[feat]
        s = _seuil_cs(x, seuil)
        if isinstance(s, pd.Series):
            c = {">": x.gt, ">=": x.ge, "<": x.lt, "<=": x.le}[op](s, axis=0)
        else:
            c = {">": x.gt, ">=": x.ge, "<": x.lt, "<=": x.le}[op](s)
        m &= c.fillna(False)

    acc, total = None, 0.0
    for feat, w in combi["score"].items():
        if w == 0:
            continue
        if feat not in panels:
            raise KeyError(f"'{feat}' absent des panels. Disponibles : {sorted(panels)}")
        xm = panels[feat].where(m)
        ok = xm.notna().sum(axis=1) >= min_titres
        r = xm.rank(axis=1, pct=True).where(ok, np.nan) - 0.5
        acc = r * w if acc is None else acc + r * w
        total += abs(w)

    return (acc / total).where(m, np.nan)


def _seuil_cs(x, seuil):
    if isinstance(seuil, (int, float)):
        return seuil
    if isinstance(seuil, str) and seuil.startswith("cs"):
        return x.quantile(float(seuil[2:]) / 100.0, axis=1)
    if isinstance(seuil, str) and seuil.startswith("ts"):
        return x.rolling(252, min_periods=60).quantile(float(seuil[2:]) / 100.0)
    raise ValueError(f"Seuil non reconnu : {seuil!r}")


# ---------------------------------------------------------------- FONCTION 3

def classer_combinaisons(combis: dict, panels: dict, close: pd.DataFrame,
                         horizons: dict = None, critere: str = "MCC",
                         q: float = 0.10, reference: str = "zero",
                         min_titres: int = 10, verbose: bool = True) -> dict:
    """Fait tourner evaluer_binaire sur toutes les combinaisons et désigne
    LA MEILLEURE PAR GROUPE D'HORIZON (court / moyen / long).

    On ne retient pas "celles qui passent un test" : on classe et on garde la
    tête. Si une combinaison domine à court terme et une autre à long terme,
    les deux sont renvoyées — c'est l'usage qui tranchera.

    critere : "MCC" (robuste au déséquilibre, recommandé) | "edge_pt" |
              "lift_hausse" | "exactitude_%"
    """
    horizons = horizons or HORIZONS_DEFAUT
    tous_h = sorted({h for hs in horizons.values() for h in hs})

    resultats = []
    for i, (nom, combi) in enumerate(combis.items(), 1):
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(combis)}…")
        try:
            df = evaluer_binaire(combi, panels, close, tous_h, q, reference, min_titres)
        except KeyError:
            continue
        if "ERREUR" in df.columns and df[critere if critere in df else "horizon"].isna().all():
            continue
        df.insert(0, "combinaison", nom)
        df["familles"] = " | ".join(f"{k}:{v}" for k, v in
                                    combi.get("familles", {}).items())
        resultats.append(df)

    if not resultats:
        return {"tableau": pd.DataFrame(), "meilleures": {}, "detail": {}}

    tableau = pd.concat(resultats, ignore_index=True)
    if critere not in tableau.columns:
        raise ValueError(f"critère '{critere}' absent. "
                         f"Choix : MCC, edge_pt, lift_hausse, exactitude_%")

    meilleures, detail = {}, {}
    for groupe, hs in horizons.items():
        sous = tableau[tableau.horizon.isin(hs)].dropna(subset=[critere])
        if sous.empty:
            continue
        agg = (sous.groupby("combinaison")
               .agg(critere_moy=(critere, "mean"),
                    critere_min=(critere, "min"),
                    edge_pt=("edge_pt", "mean"),
                    lift_hausse=("lift_hausse", "mean"),
                    exactitude=("exactitude_%", "mean"),
                    ecart_repart=("ecart_repart_pt", "mean"),
                    t_NW=("t_NW", "mean"),
                    n_signif=("significatif", "sum"))
               .sort_values("critere_moy", ascending=False))
        meilleures[groupe] = agg.index[0]
        detail[groupe] = agg

    return {"tableau": tableau, "meilleures": meilleures, "detail": detail}


def resume_selection(res: dict, top: int = 5) -> str:
    """Rapport lisible : la meilleure combinaison par groupe d'horizon."""
    if not res["meilleures"]:
        return "Aucune combinaison exploitable (univers ou historique trop petits)."
    L = []
    for groupe, agg in res["detail"].items():
        L.append("=" * 88)
        L.append(f"HORIZON {groupe.upper()}  —  meilleure : {res['meilleures'][groupe]}")
        L.append("=" * 88)
        L.append(agg.head(top).round(3).to_string())
        L.append("")
    L.append("LECTURE")
    L.append("  MCC          -1 à +1. 0 = hasard. En equity réelle, 0.02-0.05 est déjà bon.")
    L.append("  edge_pt      exactitude moins le taux de hasard, en points de %.")
    L.append("  lift_hausse  précision sur les 'hausse' / taux de base. >1 = mieux que rien.")
    L.append("  ecart_repart calibration : écart entre répartition prédite et réalisée.")
    L.append("  n_signif     nb d'horizons du groupe où l'edge est significatif.")
    return "\n".join(L)
