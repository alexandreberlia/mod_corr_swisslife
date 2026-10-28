"""
PIPELINE COMPLÈTE : de l'entraînement à la décision du jour.

Placez ce fichier dans le même dossier que :
  signaux.py, data_prep.py, windows.py, models.py, train.py, resultats.py,
  analyse_regimes.py, decision.py, horizons.py, prevision.py, portefeuille.py

À FAIRE UNE FOIS : compléter les 4 fonctions de la partie 1 (vos données).
ENSUITE : régler la partie 0, choisir les étapes à lancer (ETAPES), exécuter.

Déroulé :
  A. horizons     (facultatif) quel horizon donne le meilleur résultat ? -> validation
  B. entrainement prépare les données, entraîne, sauvegarde modèle + préparation,
                  rapport sur la validation
  C. seuils       choix du seuil d'achat et du seuil de sortie -> validation
                  (enregistrés dans FICHIER_SEUILS, modifiables à la main)
  D. test         backtest du portefeuille sur le test : À NE REGARDER QU'UNE FOIS
  E. aujourdhui   probabilités du jour et décisions : vendre / garder / acheter

Usage courant :
  - première fois : ETAPES = A (optionnel), B, C, D
  - chaque jour   : ETAPES = E seulement (le modèle et les seuils sont rechargés)
"""

from __future__ import annotations

import json
import os

import pandas as pd

from train import TrainConfig

# =============================================================================
# 0. RÉGLAGES
# =============================================================================
DOSSIER = "."                                   # où écrire modèle, préparation, rapports
FICHIER_MODELE = "modele.pt"
FICHIER_PREPARATION = "preparation.pkl"
FICHIER_SEUILS = "seuils.json"

ETAPES = {
    "horizons": False,       # A
    "entrainement": True,    # B  (False = recharger le modèle déjà entraîné)
    "seuils": True,          # C
    "test": True,            # D  (une seule fois, quand tout est figé)
    "aujourdhui": True,      # E
}

# --- Données ---
MACRO_CSV = "score_macro.csv"
MACRO_COL_DATE = "date"
MACRO_DELAI_PUBLICATION = 30                    # jours
CLE_ATR = "atr"                                 # nom de l'ATR dans votre dictionnaire panels

# --- Préparation et découpage (identiques pour toutes les étapes) ---
REGLAGES = dict(
    train_start="2005-01-01", train_end="2016-12-31", val_end="2019-12-31",
    horizon=30,                     # séances : la probabilité porte sur la hausse à 30 séances
    target="direction",
    signaux=True,                   # vos règles de signaux.py
    regime_col="score",             # le score macro comme contexte
    score_min=25,                   # pas de décision si le score est <= 25
    variables_prix=False,           # True = ajouter rendements et volatilité calculés sur les prix
)
FENETRE = 20                                    # séances lues par le modèle
CONFIG_ENTRAINEMENT = TrainConfig(task="proba", d_model=16, dropout=0.2, lr=1e-3,
                                  max_epochs=100, patience=8)

# --- Stratégie (règles de gestion : fixées à l'avance, à ne pas optimiser) ---
OPTIONS = dict(
    couts_bps=10,                   # frais par aller simple (10 = 0,10 %)
    entree="cloture",               # "cloture" = achat au dernier cours disponible
                                    # "ouverture" = achat à l'ouverture suivante
    stop_atr=None,                  # None = stop désactivé ; 2 = stop à 2 ATR
    stop_sur="low",
    max_positions=10,
    max_entrees_jour=2,
    duree_max=REGLAGES["horizon"],  # sortie de secours
)

# --- Votre portefeuille réel (étape E) ---
# {titre: {"date": date d'achat, "prix": prix d'achat}} ; "prix" sert au stop
MON_PORTEFEUILLE = {
    # "MC.PA": {"date": "2026-10-12", "prix": 698.5},
}


# =============================================================================
# 1. VOS DONNÉES : À COMPLÉTER
# =============================================================================
def charger_prix() -> dict:
    """{titre: DataFrame Open/High/Low/Close(/Volume)}, jusqu'au dernier jour disponible."""
    raise NotImplementedError("Complétez charger_prix() avec votre code de téléchargement.")


def mes_indicateurs(prices: dict) -> dict:
    """{indicateur: DataFrame dates x titres}, calculés à partir de `prices`.
    Mêmes noms qu'à l'entraînement et que dans signaux.py, plus l'ATR (CLE_ATR)."""
    raise NotImplementedError("Complétez mes_indicateurs(prices) avec votre calcul des indicateurs.")


def cours_du_moment(prices: dict) -> dict:
    """{titre: cours au moment de la décision}. Dictionnaire vide = utiliser la dernière
    clôture disponible, sans figer de séance."""
    return {}


def charger_macro():
    from data_prep import load_macro
    return load_macro(MACRO_CSV, date_col=MACRO_COL_DATE,
                      publication_lag_days=MACRO_DELAI_PUBLICATION)


# =============================================================================
# 2. ÉTAPES
# =============================================================================
def chemin(nom: str) -> str:
    return os.path.join(DOSSIER, nom)


def titre(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def etape_horizons(prices, panels, macro) -> pd.DataFrame:
    from horizons import comparer_horizons
    titre("A. COMPARAISON DES HORIZONS (validation, régression logistique)")
    opts = {k: OPTIONS[k] for k in ("couts_bps", "entree", "stop_atr", "stop_sur")}
    if opts["stop_atr"] is not None:
        opts["atr"] = panels[CLE_ATR]
    res = comparer_horizons(prices, panels, macro, horizons=range(1, 21), methode="logistique",
                            out_dir=chemin("horizons"), **opts, **REGLAGES)
    print(res.round(3).to_string())
    return res


def etape_entrainement(prices, panels, macro):
    from data_prep import prepare_dataset
    from prevision import sauver_preparation
    from resultats import rapport
    from train import train
    from windows import make_loaders

    titre("B. PRÉPARATION ET ENTRAÎNEMENT")
    data = prepare_dataset(prices, panels, macro, **REGLAGES)
    print(data.df["split"].value_counts().to_string())
    print("variables :", data.feature_cols, "| contexte :", data.regime_cols)
    loaders = make_loaders(data, window=FENETRE, batch_size=256)
    model, _ = train(loaders, CONFIG_ENTRAINEMENT, save_path=chemin(FICHIER_MODELE))
    sauver_preparation(chemin(FICHIER_PREPARATION), data, window=FENETRE, reglages=REGLAGES)
    rapport(chemin(FICHIER_MODELE), loaders, split="val",
            out_dir=chemin("rapport_validation"), data=data)
    return data, loaders, model


def recharger(prices, panels, macro):
    """Modèle déjà entraîné : on le recharge, avec la même préparation."""
    from data_prep import prepare_dataset
    from prevision import charger_preparation
    from train import load_model
    from windows import make_loaders

    titre("B. RECHARGEMENT DU MODÈLE EXISTANT")
    prep = charger_preparation(chemin(FICHIER_PREPARATION))
    data = prepare_dataset(prices, panels, macro, scaler=prep["scaler"], **prep["reglages"])
    loaders = make_loaders(data, window=prep["window"], batch_size=256)
    model, _ = load_model(chemin(FICHIER_MODELE))
    return data, loaders, model


def options_avec_atr(panels) -> dict:
    o = dict(OPTIONS)
    if o.get("stop_atr") is not None:
        o["atr"] = panels[CLE_ATR]
    return o


def etape_seuils(model, loaders, prices, panels) -> dict:
    from portefeuille import choisir_seuils_portefeuille
    from train import predict

    titre("C. CHOIX DES SEUILS (validation)")
    grille = choisir_seuils_portefeuille(predict(model, loaders["val"]), prices,
                                         **options_avec_atr(panels))
    print(grille.round(3).to_string())
    best = grille.iloc[0]
    seuils = {"seuil_achat": float(best["seuil_achat"]),
              "seuil_sortie": None if best["seuil_sortie"] == "aucun (temps seul)"
              else float(best["seuil_sortie"]),
              "options": {k: v for k, v in OPTIONS.items()}}
    with open(chemin(FICHIER_SEUILS), "w", encoding="utf-8") as f:
        json.dump(seuils, f, indent=2, ensure_ascii=False)
    print(f"\nSeuils retenus : achat {seuils['seuil_achat']}, sortie {seuils['seuil_sortie']} "
          f"-> enregistrés dans {chemin(FICHIER_SEUILS)}.")
    print("Si la meilleure ligne est un pic isolé, corrigez les seuils à la main dans ce fichier "
          "(choisissez le centre d'un plateau), AVANT de lancer l'étape D.")
    return seuils


def charger_seuils() -> dict:
    with open(chemin(FICHIER_SEUILS), encoding="utf-8") as f:
        s = json.load(f)
    if s.get("options") and s["options"] != {k: v for k, v in OPTIONS.items()}:
        print("ATTENTION : les OPTIONS actuelles diffèrent de celles utilisées pour choisir les "
              "seuils. Relancez l'étape C.")
    return s


def etape_test(model, loaders, prices, panels, seuils) -> pd.DataFrame:
    from portefeuille import backtest_portefeuille, rapport_portefeuille
    from train import predict

    titre("D. BACKTEST SUR LE TEST (à ne regarder qu'une fois)")
    res = backtest_portefeuille(predict(model, loaders["test"]), prices,
                                seuils["seuil_achat"], seuils["seuil_sortie"],
                                **options_avec_atr(panels))
    stats = rapport_portefeuille(res, out_dir=chemin("backtest_test"))
    print(stats.round(3).to_string())
    return stats


def etape_aujourdhui(prices, macro, seuils) -> dict:
    from data_prep import prepare_dataset
    from portefeuille import decisions_portefeuille_du_jour
    from prevision import charger_preparation, figer_seance
    from train import load_model

    titre("E. DÉCISIONS DU JOUR")
    cours = cours_du_moment(prices)
    if cours:
        prices_jour = figer_seance(prices, cours)
        print(f"Séance du jour figée au cours du moment pour {len(cours)} titre(s).")
    else:
        prices_jour = prices
        print("Pas de cours du moment fourni : dernière clôture disponible utilisée.")
    panels_jour = mes_indicateurs(prices_jour)

    prep = charger_preparation(chemin(FICHIER_PREPARATION))
    data_jour = prepare_dataset(prices_jour, panels_jour, macro,
                                scaler=prep["scaler"], **prep["reglages"])
    model, _ = load_model(chemin(FICHIER_MODELE))

    dec = decisions_portefeuille_du_jour(model, data_jour, prep["window"], MON_PORTEFEUILLE,
                                         seuils["seuil_achat"], seuils["seuil_sortie"],
                                         prices=prices_jour, **options_avec_atr(panels_jour))
    for k in ("vendre", "garder", "acheter"):
        print(f"\n--- À {k.upper()} ---")
        print(dec[k].round(3).to_string() if len(dec[k]) else "(rien)")

    date = dec["toutes les probabilités"].attrs["date"]
    fichier = chemin(f"decisions_{pd.Timestamp(date).date()}.xlsx")
    with pd.ExcelWriter(fichier, engine="openpyxl") as w:
        for k in ("vendre", "garder", "acheter", "toutes les probabilités"):
            dec[k].to_excel(w, sheet_name=k[:31])
    print(f"\nDécisions enregistrées dans {os.path.abspath(fichier)}")
    return dec


# =============================================================================
# 3. LANCEMENT
# =============================================================================
def main():
    os.makedirs(DOSSIER, exist_ok=True)
    prices = charger_prix()
    panels = mes_indicateurs(prices)
    macro = charger_macro()

    if ETAPES.get("horizons"):
        etape_horizons(prices, panels, macro)

    besoin_modele = ETAPES.get("seuils") or ETAPES.get("test")
    model = loaders = None
    if ETAPES.get("entrainement"):
        _, loaders, model = etape_entrainement(prices, panels, macro)
    elif besoin_modele:
        _, loaders, model = recharger(prices, panels, macro)

    seuils = etape_seuils(model, loaders, prices, panels) if ETAPES.get("seuils") else None
    if ETAPES.get("test") or ETAPES.get("aujourdhui"):
        seuils = seuils or charger_seuils()
    if ETAPES.get("test"):
        etape_test(model, loaders, prices, panels, seuils)
    if ETAPES.get("aujourdhui"):
        etape_aujourdhui(prices, macro, seuils)


if __name__ == "__main__":
    main()
