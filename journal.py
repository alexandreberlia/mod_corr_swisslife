"""
Journal de la stratégie en réel : ce qui a été prédit, décidé, exécuté, et ce que ça rapporte.

Tout est rangé dans un dossier (par défaut "suivi/"), en fichiers lisibles dans Excel :
  etat.json          capital de départ, liquidités, date de début
  portefeuille.json  lignes détenues (lu directement par decisions_portefeuille_du_jour)
  probas.csv         probabilité de CHAQUE titre CHAQUE jour (pour mesurer l'AUC en réel)
  ordres.csv         achats et ventes exécutés (prix, quantité, frais, motif)
  trades.csv         trades clôturés (entrée, sortie, gain)
  valeur.csv         valeur du portefeuille (cash + positions) à chaque jour enregistré
  positions.csv      photo des positions chaque jour enregistré : cours, valeur, P&L latent
  bilan.xlsx         résumé : cash, positions et P&L, ordres, trades, AUC réelle

Utilisation :
  demarrer_strategie("suivi", capital=100_000)             # une seule fois, le premier jour
  portefeuille = charger_portefeuille("suivi")              # chaque jour, avant les décisions
  dec = decisions_portefeuille_du_jour(..., portefeuille, ...)
  enregistrer_jour("suivi", dec, couts_bps=5, max_positions=2)   # APRÈS avoir passé les ordres
  bilan("suivi", prices, horizon=22)                         # quand vous voulez

Relancer enregistrer_jour le même jour remplace les lignes de ce jour (pas de doublon).
Les journées s'enregistrent dans l'ordre du calendrier ; sauter des jours est permis.
Une journée antérieure à la dernière enregistrée est refusée (rien n'est modifié).
  restaurer("suivi", "2026-11-11")    # revenir à l'état de fin de journée du 11, pour rejouer la suite
Chaque enregistrement copie d'abord l'état dans suivi/sauvegardes/ (30 dernières copies).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def _p(dossier, nom):
    return os.path.join(dossier, nom)


def _lire_json(path, defaut):
    if not os.path.exists(path):
        return defaut
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ecrire_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _ajouter_csv(path, df, date):
    """Ajoute les lignes de `df` en remplaçant celles déjà écrites pour `date`."""
    if os.path.exists(path):
        ancien = pd.read_csv(path, sep=";", decimal=",")
        ancien = ancien[ancien["date"] != date]
        df = pd.concat([d for d in (ancien, df) if len(d)], ignore_index=True) \
            if (len(ancien) or len(df)) else df
    for col in df.columns:                      # nombres écrits avec la virgule décimale
        if col not in ("date", "ticker") and not pd.api.types.is_numeric_dtype(df[col]):
            conv = pd.to_numeric(df[col], errors="coerce")
            if conv.notna().sum() == df[col].notna().sum():
                df[col] = conv
    df.to_csv(path, sep=";", decimal=",", index=False)


FICHIERS = ("etat.json", "portefeuille.json", "probas.csv", "ordres.csv", "trades.csv",
            "valeur.csv", "positions.csv")


def _sauvegarde(dossier: str, garder: int = 30) -> None:
    """Copie l'état actuel dans dossier/sauvegardes/<horodatage>/ avant toute modification."""
    import shutil
    from datetime import datetime
    racine = _p(dossier, "sauvegardes")
    cible = os.path.join(racine, datetime.now().strftime("%Y-%m-%d_%H%M%S_%f"))
    os.makedirs(cible, exist_ok=True)
    for f in FICHIERS:
        if os.path.exists(_p(dossier, f)):
            shutil.copy2(_p(dossier, f), os.path.join(cible, f))
    anciens = sorted(os.listdir(racine))[:-garder]
    for a in anciens:
        shutil.rmtree(os.path.join(racine, a), ignore_errors=True)


def _dernier_jour(dossier: str, etat: dict) -> str | None:
    """Dernière journée enregistrée (la plus récente dans le calendrier)."""
    if etat.get("dernier_jour"):
        return etat["dernier_jour"]
    path = _p(dossier, "valeur.csv")
    if os.path.exists(path):
        v = pd.read_csv(path, sep=";", decimal=",")
        if len(v):
            return str(v["date"].max())
    return None


# ---------------------------------------------------------------------------
def demarrer_strategie(dossier: str = "suivi", capital: float = 100_000.0,
                       date: str | None = None, portefeuille: dict | None = None) -> None:
    """Crée le dossier de suivi. À lancer UNE fois. `portefeuille` : lignes déjà détenues
    au démarrage (même format que portefeuille.json), sinon départ 100 % liquide."""
    os.makedirs(dossier, exist_ok=True)
    if os.path.exists(_p(dossier, "etat.json")):
        raise FileExistsError(f"{dossier} contient déjà une stratégie (etat.json). "
                              "Changez de dossier ou supprimez-le pour repartir de zéro.")
    date = str(pd.Timestamp(date or pd.Timestamp.today()).date())
    _ecrire_json(_p(dossier, "etat.json"), {"capital_initial": capital, "liquidites": capital,
                                            "date_debut": date, "historique_etat": {}})
    _ecrire_json(_p(dossier, "portefeuille.json"), portefeuille or {})
    print(f"Stratégie démarrée le {date} avec {capital:,.0f} dans {os.path.abspath(dossier)}")


def charger_portefeuille(dossier: str = "suivi") -> dict:
    """Lignes détenues, au format attendu par decisions_portefeuille_du_jour."""
    return _lire_json(_p(dossier, "portefeuille.json"), {})


# ---------------------------------------------------------------------------
def enregistrer_jour(dossier: str, dec: dict, couts_bps: float = 10.0, max_positions: int = 10,
                     prix_executes: dict | None = None, non_executes: list | None = None,
                     seuil_achat: float | None = None, seuil_sortie: float | None = None,
                     forcer: bool = False, titres_entiers: bool = False) -> dict:
    """Enregistre la journée APRÈS le passage des ordres.

    dec           : sortie de decisions_portefeuille_du_jour
    prix_executes : {"XLK": 182.6} si le prix obtenu diffère du dernier cours (sinon dernier cours)
    non_executes  : titres dont l'ordre proposé n'a PAS été passé, ex. ["CAT"]
    Taille d'un achat = 1/max_positions de la valeur du portefeuille (comme dans le backtest).
    """
    etat = _lire_json(_p(dossier, "etat.json"), None)
    if etat is None:
        raise FileNotFoundError(f"Pas de stratégie dans {dossier} : lancez demarrer_strategie().")
    # état de départ de la journée (permet de ré-enregistrer le même jour sans doublon)
    jour_df = dec["toutes les probabilités"]
    date = str(pd.Timestamp(jour_df.attrs.get("date", pd.Timestamp.today())).date())
    # Garde-fou : les décisions doivent porter sur la séance du jour (cours du moment)
    if "date des données" in jour_df and not forcer:
        aujourd_hui = pd.Timestamp.today().normalize()
        vieux = jour_df.index[pd.to_datetime(jour_df["date des données"]) < aujourd_hui]
        if len(vieux):
            raise ValueError(
                f"Rien n'a été enregistré : {len(vieux)} titre(s) sont évalués sur une séance "
                f"antérieure à aujourd'hui ({', '.join(map(str, vieux[:8]))}), donc sur la clôture "
                "de la veille et pas sur le cours du moment. Vérifiez que ces titres sont dans "
                "cours_now et que la ligne du jour n'a pas de valeur manquante (volume, "
                "indicateurs). Pour enregistrer quand même : forcer=True.")
    dernier = _dernier_jour(dossier, etat)
    hist = etat.get("historique_etat", {}) or {}
    if dernier is not None and date < dernier:
        raise ValueError(
            f"Rien n'a été enregistré : la journée du {date} est antérieure à la dernière "
            f"journée enregistrée ({dernier}). Les journées s'enregistrent dans l'ordre du "
            "calendrier. Pour refaire l'historique à partir d'une date, revenez d'abord à "
            "l'état de la veille : restaurer(dossier, 'AAAA-MM-JJ'), puis rejouez les jours "
            "un par un. Vérifiez aussi DATE_DECISION.")
    _sauvegarde(dossier)
    if date == dernier and date in hist:                       # même jour relancé : on le remplace
        etat["liquidites"] = hist[date]["liquidites_avant"]
        port = hist[date]["portefeuille_avant"]
    elif date == dernier:                                      # même jour, sans état « avant » connu
        raise ValueError(
            f"Rien n'a été enregistré : la journée du {date} est déjà enregistrée et son état "
            "de départ n'est plus disponible. Pour la refaire : restaurer(dossier, "
            "'<journée précédente>') puis relancez.")
    else:                                                      # nouvelle journée (jours sautés permis)
        port = charger_portefeuille(dossier)
    # on ne garde que l'état « avant » de CETTE journée : jamais d'entrée périmée
    etat["historique_etat"] = {date: {"liquidites_avant": etat["liquidites"],
                                      "portefeuille_avant": json.loads(json.dumps(port))}}
    etat["dernier_jour"] = date
    port = json.loads(json.dumps(port))                        # copie
    prix_executes, non_executes = prix_executes or {}, set(non_executes or [])
    c = couts_bps / 1e4
    cours = jour_df["dernier cours"] if "dernier cours" in jour_df else pd.Series(dtype=float)
    px = lambda t: float(prix_executes.get(t, cours.get(t, np.nan)))

    # Valorisation : un titre détenu sans cours aujourd'hui garde son dernier cours connu
    # (sinon sa valeur disparaîtrait du portefeuille)
    dernier_connu: dict = {}
    if os.path.exists(_p(dossier, "positions.csv")):
        anc = pd.read_csv(_p(dossier, "positions.csv"), sep=";", decimal=",")
        anc = anc[anc["date"] < date].dropna(subset=["cours du jour"])
        if len(anc):
            dernier_connu = anc.sort_values("date").groupby("ticker")["cours du jour"].last().to_dict()
    sans_cours: set = set()

    def px_val(t, ligne=None):
        p = px(t)
        if np.isfinite(p):
            return p
        sans_cours.add(t)
        p = dernier_connu.get(t, np.nan)
        if not np.isfinite(p) and ligne is not None:
            p = float(ligne.get("prix", np.nan))                # à défaut : prix d'achat
        return p

    def lignes(t):
        v = port.get(t, [])
        v = v if isinstance(v, list) else [v]
        return [x if isinstance(x, dict) else {"date": x} for x in v]

    # 1. Probabilités du jour (tous les titres)
    pr = jour_df[["proba"]].copy()
    pr["dernier cours"] = cours.reindex(pr.index)
    pr = pr.reset_index().rename(columns={"index": "ticker"})
    pr.insert(0, "date", date)
    pr["seuil_achat"], pr["seuil_sortie"] = seuil_achat, seuil_sortie
    _ajouter_csv(_p(dossier, "probas.csv"), pr, date)

    ordres, trades = [], []
    # 2. Ventes
    for t, row in dec["vendre"].iterrows():
        if t in non_executes:
            continue
        p = px(t)
        reste = []
        for l in lignes(t):
            if str(l["date"]) == str(row["date d'achat"]) and np.isfinite(p):
                q = float(l.get("quantite", np.nan))
                brut = q * p
                etat["liquidites"] += brut * (1 - c)
                ordres.append({"date": date, "ticker": t, "sens": "VENTE", "motif": row["motif"],
                               "prix": p, "quantite": q, "montant": brut, "frais": brut * c})
                trades.append({"date": date, "ticker": t, "date d'entrée": l["date"],
                               "prix d'entrée": l.get("prix"), "date de sortie": date,
                               "prix de sortie": p, "motif": row["motif"],
                               "gain": p / l["prix"] - 1 if l.get("prix") else np.nan,
                               "résultat net": brut * (1 - c) - q * l.get("prix", np.nan) * (1 + c)})
            else:
                reste.append(l)
        if reste:
            port[t] = reste
        else:
            port.pop(t, None)

    # 3. Valeur avant achats (pour la taille des lignes)
    def valeur():
        inv = sum(float(l.get("quantite", 0)) * px_val(t, l) for t in port for l in lignes(t)
                  if np.isfinite(px_val(t, l)))
        return etat["liquidites"] + inv, inv
    v_avant, _ = valeur()

    # 4. Achats
    for t, row in dec["acheter"].iterrows():
        if t in non_executes:
            continue
        p = px(t)
        montant = min(v_avant / max_positions, etat["liquidites"] / (1 + c))
        if not np.isfinite(p) or p <= 0 or montant <= 0:
            print(f"  {t} : achat non enregistré (prix inconnu ou plus de liquidités).")
            continue
        q = montant / p
        if titres_entiers:                       # arrondi à l'inférieur : jamais plus que le cash
            q = float(np.floor(q))
            if q < 1:
                print(f"  {t} : achat non enregistré (montant inférieur au prix d'une action).")
                continue
            montant = q * p
        etat["liquidites"] -= montant * (1 + c)
        port[t] = lignes(t) + [{"date": date, "prix": p, "quantite": q}]
        ordres.append({"date": date, "ticker": t, "sens": "ACHAT", "motif": row.get("ligne", ""),
                       "prix": p, "quantite": q, "montant": montant, "frais": montant * c})

    # 5. Écritures
    _ajouter_csv(_p(dossier, "ordres.csv"), pd.DataFrame(ordres, columns=[
        "date", "ticker", "sens", "motif", "prix", "quantite", "montant", "frais"]), date)
    _ajouter_csv(_p(dossier, "trades.csv"), pd.DataFrame(trades, columns=[
        "date", "ticker", "date d'entrée", "prix d'entrée", "date de sortie", "prix de sortie",
        "motif", "gain", "résultat net"]), date)
    v, inv = valeur()
    # Photo des positions au cours du jour : P&L latent ligne par ligne
    pos = []
    for t in port:
        for l in lignes(t):
            q, pa, pc = float(l.get("quantite", np.nan)), float(l.get("prix", np.nan)), px_val(t, l)
            cout = q * pa * (1 + c)                         # prix payé, frais d'achat compris
            val = q * pc
            pos.append({"date": date, "ticker": t, "date d'achat": l["date"], "prix d'achat": pa,
                        "quantité": q, "cours du jour": pc, "valeur": val, "coût (frais inclus)": cout,
                        "P&L latent": val - cout, "P&L latent %": val / cout - 1 if cout else np.nan,
                        "poids dans le portefeuille": val / v if v else np.nan})
    _ajouter_csv(_p(dossier, "positions.csv"), pd.DataFrame(pos, columns=[
        "date", "ticker", "date d'achat", "prix d'achat", "quantité", "cours du jour", "valeur",
        "coût (frais inclus)", "P&L latent", "P&L latent %", "poids dans le portefeuille"]), date)
    _ajouter_csv(_p(dossier, "valeur.csv"), pd.DataFrame([{
        "date": date, "liquidités": etat["liquidites"], "investi": inv, "valeur": v,
        "lignes": sum(len(lignes(t)) for t in port),
        "performance depuis le début": v / etat["capital_initial"] - 1}]), date)
    _ecrire_json(_p(dossier, "portefeuille.json"), port)
    _ecrire_json(_p(dossier, "etat.json"), etat)
    if sans_cours:
        print(f"  Attention : pas de cours aujourd'hui pour {sorted(sans_cours)} ; valorisé(s) au "
              "dernier cours connu. Vérifiez cours_now.")
    if dernier is not None and date > dernier:
        saut = len(pd.bdate_range(dernier, date)) - 2
        if saut > 0:
            print(f"  Note : {saut} jour(s) ouvré(s) sans enregistrement entre {dernier} et {date} "
                  "(le portefeuille est repris tel quel).")
    print(f"Journée du {date} enregistrée : {sum(o['sens'] == 'ACHAT' for o in ordres)} achat(s), "
          f"{sum(o['sens'] == 'VENTE' for o in ordres)} vente(s), valeur {v:,.0f} "
          f"({v / etat['capital_initial'] - 1:+.2%} depuis le début).")
    return port


# ---------------------------------------------------------------------------
def restaurer(dossier: str, date: str) -> dict:
    """Revient à l'état de FIN de journée `date` (une journée déjà enregistrée) : portefeuille
    et liquidités tels qu'ils étaient ce soir-là, et suppression de tout ce qui a été
    enregistré après. Une sauvegarde est faite avant. Rejouez ensuite les jours suivants
    dans l'ordre."""
    date = str(pd.Timestamp(date).date())
    lire = lambda n: pd.read_csv(_p(dossier, n), sep=";", decimal=",") \
        if os.path.exists(_p(dossier, n)) else pd.DataFrame()
    valeur, positions = lire("valeur.csv"), lire("positions.csv")
    if not len(valeur) or date not in set(valeur["date"]):
        connues = sorted(set(valeur["date"])) if len(valeur) else []
        raise ValueError(f"Aucune journée enregistrée le {date}. Journées disponibles : "
                         f"{connues[-15:]}")
    _sauvegarde(dossier)
    port: dict = {}
    if len(positions):
        for _, r in positions[positions["date"] == date].iterrows():
            port.setdefault(r["ticker"], []).append(
                {"date": str(r["date d'achat"]), "prix": float(r["prix d'achat"]),
                 "quantite": float(r["quantité"])})
    liquidites = float(valeur.loc[valeur["date"] == date, "liquidités"].iloc[-1])
    for n in ("valeur.csv", "positions.csv", "ordres.csv", "trades.csv", "probas.csv"):
        df = lire(n)
        if len(df):
            df[df["date"] <= date].to_csv(_p(dossier, n), sep=";", decimal=",", index=False)
    etat = _lire_json(_p(dossier, "etat.json"), {})
    etat.update({"liquidites": liquidites, "dernier_jour": date, "historique_etat": {}})
    _ecrire_json(_p(dossier, "etat.json"), etat)
    _ecrire_json(_p(dossier, "portefeuille.json"), port)
    n_lignes = sum(len(v) for v in port.values())
    print(f"État restauré à la fin du {date} : {n_lignes} ligne(s), liquidités {liquidites:,.0f}. "
          "Tout ce qui avait été enregistré après a été retiré (copie dans sauvegardes/).")
    return port


# ---------------------------------------------------------------------------
def bilan(dossier: str, prices: dict | None = None, horizon: int | None = None,
          out: str | None = None) -> dict:
    """Résumé du suivi réel. Avec prices + horizon : AUC réelle des probabilités dont
    l'issue est connue (cours horizon séances plus tard). Écrit bilan.xlsx dans le dossier."""
    lire = lambda n: pd.read_csv(_p(dossier, n), sep=";", decimal=",") \
        if os.path.exists(_p(dossier, n)) else pd.DataFrame()
    valeur, trades, ordres, probas, positions = (lire("valeur.csv"), lire("trades.csv"),
                                                 lire("ordres.csv"), lire("probas.csv"),
                                                 lire("positions.csv"))
    etat = _lire_json(_p(dossier, "etat.json"), {})
    # positions actuelles = photo du dernier jour enregistré
    if len(positions) and len(valeur):
        derniere = valeur["date"].max()
        pos_act = positions[positions["date"] == derniere].drop(columns="date")
    else:
        derniere, pos_act = None, pd.DataFrame()
    res = {"positions": pos_act, "valeur": valeur, "trades": trades, "ordres": ordres,
           "historique des positions": positions}
    pnl_latent = pos_act["P&L latent"].sum() if len(pos_act) else 0.0
    pnl_realise = trades["résultat net"].sum() if len(trades) else 0.0
    lignes = {"date de début": etat.get("date_debut"),
              "dernier jour enregistré": derniere,
              "capital initial": etat.get("capital_initial"),
              "liquidités (cash)": etat.get("liquidites"),
              "valeur des positions": pos_act["valeur"].sum() if len(pos_act) else 0.0,
              "valeur actuelle": valeur.sort_values("date")["valeur"].iloc[-1] if len(valeur) else np.nan,
              "P&L latent (positions ouvertes)": pnl_latent,
              "P&L réalisé (trades clôturés)": pnl_realise,
              "achats exécutés": int((ordres["sens"] == "ACHAT").sum()) if len(ordres) else 0,
              "ventes exécutées": int((ordres["sens"] == "VENTE").sum()) if len(ordres) else 0,
              "lignes ouvertes": sum(len(v) if isinstance(v, list) else 1
                                     for v in charger_portefeuille(dossier).values()),
              "trades clôturés (achetés puis revendus)": len(trades),
              "trades gagnants": (trades["gain"] > 0).mean() if len(trades) else np.nan,
              "frais payés": ordres["frais"].sum() if len(ordres) else 0.0}
    lignes["performance"] = lignes["valeur actuelle"] / lignes["capital initial"] - 1 \
        if lignes["capital initial"] else np.nan

    if prices is not None and horizon and len(probas):
        from decision import ohlc
        from train import auc
        cl = ohlc(prices)["Close"]
        issues = []
        for _, r in probas.iterrows():
            d, t = pd.Timestamp(r["date"]), r["ticker"]
            if t not in cl.columns:
                continue
            apres = cl[t].loc[cl.index > d].dropna()
            if len(apres) >= horizon and np.isfinite(r["dernier cours"]):
                issues.append({"date": r["date"], "ticker": t, "proba": r["proba"],
                               "hausse": float(apres.iloc[horizon - 1] > r["dernier cours"])})
        iss = pd.DataFrame(issues)
        res["issues"] = iss
        lignes["probabilités dont l'issue est connue"] = len(iss)
        if len(iss) and iss["hausse"].nunique() == 2:
            lignes["AUC réelle"] = auc(iss["proba"].to_numpy(), iss["hausse"].to_numpy())
            lignes["fréquence de hausse réelle"] = iss["hausse"].mean()
            lignes["probabilité moyenne annoncée"] = iss["proba"].mean()

    res["resume"] = pd.Series(lignes, name="valeur").to_frame()
    out = out or _p(dossier, "bilan.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        res["resume"].to_excel(w, sheet_name="Résumé")
        noms = {"positions": "Positions", "valeur": "Valeur", "trades": "Trades clôturés",
                "ordres": "Ordres", "historique des positions": "Historique positions",
                "issues": "Issues des probas"}
        for k, nom in noms.items():
            if k in res and len(res[k]):
                res[k].to_excel(w, sheet_name=nom, index=False)
    print(res["resume"].to_string())
    if len(pos_act):
        print("\nPOSITIONS :\n", pos_act[["ticker", "date d'achat", "prix d'achat", "cours du jour",
                                         "valeur", "P&L latent", "P&L latent %"]].round(2).to_string(index=False))
    print(f"Bilan écrit dans : {os.path.abspath(out)}")
    return res
