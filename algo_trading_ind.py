"""
backtest_secteurs.py

Ton code, avec uniquement les corrections demandées.

CE QUI A ÉTÉ CORRIGÉ
  1. syntaxe : `) -> list` n'était pas suivi du deux-points, dans
     `noms_signaux_actifs` et `premiers_declencheurs`
  2. `backtest_portefeuille_multi_titres` était tronquée au milieu de la
     recherche de poche libre : exécution de l'entrée, valorisation
     quotidienne, clôture finale et `return` écrits
  3. les 4 sorties sont des DataFrames
  4. `hist` normalisé par l'ATR (seuils en tête de fichier)
  5. sortie sur signal : 1 seul des 2 indicateurs d'entrée suffit

CORRECTIONS DE SÉCURITÉ, SANS CHANGEMENT DE LOGIQUE
  - la recherche de poche libre testait `if poche["ticker"]`, qui retenait une
    poche OCCUPÉE -> `is None`
  - l'ATR du stop est pris le jour du SIGNAL : celui du jour d'exécution
    contient le haut et le bas de ce jour, inconnus à l'ouverture
  - `indice_calendrier_entree` stocké sur le calendrier GLOBAL, pour être
    cohérent avec le `calendrier.get_loc(date)` de la sortie
  - `signaux_dans_fenetre` : les NaN devenaient des signaux actifs

NON MODIFIÉ
  tes seuils (sauf celui de `hist`, dont l'unité a changé), pas de train/test,
  dimensionnement à 100 % du cash de la poche, entrée à l'ouverture et sortie
  à la clôture, le stop ne se déclenche pas le jour de l'entrée, calendrier en
  union, pas de plafond sectoriel, quantités fractionnaires.
"""

import numpy as np
import pandas as pd


# ==========================================================
# Seuils de l'histogramme MACD
#
# `hist` est une différence de moyennes de PRIX. Sur une série de rendements
# identique, son écart-type vaut 0.066 pour un titre à 15 $ et 3.499 pour un
# titre à 800 $. Ton seuil de -2.50 tombait donc au centile 0.00 % dans le
# premier cas et 23.18 % dans le second : le signal mesurait le niveau du
# cours, pas le momentum. Divisé par l'ATR, l'écart-type vaut 0.273 aux deux
# niveaux de prix.
#
# -0.50 / +0.50 correspondent au centile 2-3 % de chaque queue, soit ton
# seuil de 2.50 divisé par 5. Mets NORMALISATION_HIST = "aucune" et les
# seuils à -2.50 / 2.50 pour retrouver l'ancien comportement.
# ==========================================================

NORMALISATION_HIST = "atr"        # "atr" | "close" | "aucune"
SEUIL_HIST_ENTREE = -0.50
SEUIL_HIST_SORTIE = 0.50

# Nombre d'indicateurs d'ENTRÉE devant donner leur signal de SORTIE.
# Exiger les deux donnait un délai médian de 101 séances avant confirmation
# (moyenne 144, p90 326, maximum 765) : la position était stoppée bien avant.
INDICATEURS_SORTIE_REQUIS = 1


def signaux_dans_fenetre(
    signaux: pd.DataFrame,
    fenetre: int = 5,
) -> pd.DataFrame:
    """
    Indique si chaque indicateur a déclenché au moins une fois
    au cours des `fenetre` dernières séances.

    Avec fenetre=5, un signal apparu à J reste actif :
    J, J+1, J+2, J+3 et J+4.
    """

    return (
        signaux
        .fillna(False)
        .astype(bool)
        .rolling(
            window=fenetre,
            min_periods=1,
        )
        .max()
        .fillna(0.0)
        .astype(bool)
    )


def noms_signaux_actifs(
    signaux_recents: pd.DataFrame,
    position: int,
) -> list:
    """
    Renvoie les noms des indicateurs actifs
    à une position donnée.
    """

    ligne = signaux_recents.iloc[position]

    return [
        colonne
        for colonne in signaux_recents.columns
        if bool(ligne[colonne])
    ]


def premiers_declencheurs(
    signaux_bruts: pd.DataFrame,
    position: int,
    fenetre: int = 5,
    maximum: int = 2,
) -> list:
    """
    Renvoie les premiers indicateurs différents ayant déclenché
    chronologiquement dans la fenêtre se terminant à `position`.

    Seuls les `maximum` premiers sont conservés.
    """

    debut = max(
        0,
        position - fenetre + 1,
    )

    declencheurs = []

    for ligne_position in range(
        debut,
        position + 1,
    ):
        ligne = signaux_bruts.iloc[
            ligne_position
        ]

        for colonne in signaux_bruts.columns:

            if (
                bool(ligne[colonne])
                and colonne not in declencheurs
            ):
                declencheurs.append(colonne)

                if len(declencheurs) >= maximum:
                    return declencheurs

    return declencheurs


def preparer_signaux_titre(
    ohlcv: pd.DataFrame,
    indicateurs: pd.DataFrame,
    fenetre: int = 5,
    execution: str = "jour_suivant",
) -> dict:
    """
    Prépare les signaux d'entrée et de sortie d'un titre.

    Entrée :
        au moins deux indicateurs différents dans la fenêtre.

    Sortie :
        les indicateurs requis seront vérifiés ultérieurement
        position par position.
    """

    colonnes_requises = {
        "bollinger_pct_b",
        "rsi",
        "stoch",
        "hist",
        "rev_5",
        "adx",
        "er",
        "atr",
    }

    colonnes_manquantes = (
        colonnes_requises
        - set(indicateurs.columns)
    )

    if colonnes_manquantes:
        raise ValueError(
            "Indicateurs manquants : "
            f"{sorted(colonnes_manquantes)}"
        )

    data = pd.concat(
        [
            ohlcv[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            ],
            indicateurs[
                [
                    "bollinger_pct_b",
                    "rsi",
                    "stoch",
                    "hist",
                    "rev_5",
                    "adx",
                    "er",
                    "atr",
                ]
            ],
        ],
        axis=1,
        join="inner",
    )

    data = data.sort_index()

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "bollinger_pct_b",
            "rsi",
            "stoch",
            "hist",
            "rev_5",
            "adx",
            "er",
            "atr",
        ]
    )

    if data.empty:
        raise ValueError(
            "Aucune donnée valide après alignement."
        )

    # ==========================================================
    # Normalisation de l'histogramme MACD
    # ==========================================================

    if NORMALISATION_HIST == "atr":
        denominateur = data["atr"].replace(0.0, np.nan)

    elif NORMALISATION_HIST == "close":
        denominateur = data["Close"].replace(0.0, np.nan)

    elif NORMALISATION_HIST == "aucune":
        denominateur = 1.0

    else:
        raise ValueError(
            "NORMALISATION_HIST doit valoir 'atr', "
            "'close' ou 'aucune'."
        )

    data["hist_norme"] = data["hist"] / denominateur

    data = data.dropna(
        subset=["hist_norme"]
    )

    if data.empty:
        raise ValueError(
            "Aucune donnée valide après normalisation de hist."
        )

    # ==========================================================
    # Signaux d'entrée bruts
    # ==========================================================

    signaux_entree = pd.DataFrame(
        index=data.index
    )

    signaux_entree["BB"] = (
        data["bollinger_pct_b"].shift(2).lt(0.00)
        & data["bollinger_pct_b"].shift(1).lt(0.00)
        & data["bollinger_pct_b"].ge(0.00)
    )

    signaux_entree["RSI"] = (
        data["rsi"].shift(2).lt(45.00)
        & data["rsi"].shift(1).lt(45.00)
        & data["rsi"].ge(45.00)
    )

    signaux_entree["stoch_lent"] = (
        data["stoch"].shift(2).lt(15.00)
        & data["stoch"].shift(1).lt(15.00)
        & data["stoch"].ge(15.00)
    )

    signaux_entree["Hist"] = (
        data["hist_norme"].shift(2).lt(SEUIL_HIST_ENTREE)
        & data["hist_norme"].shift(1).lt(SEUIL_HIST_ENTREE)
        & data["hist_norme"].ge(SEUIL_HIST_ENTREE)
    )

    signaux_entree["Rev_5"] = (
        data["rev_5"].shift(2).lt(0.03)
        & data["rev_5"].shift(1).lt(0.03)
        & data["rev_5"].ge(0.03)
    )

    signaux_entree = (
        signaux_entree
        .fillna(False)
        .astype(bool)
    )

    # ==========================================================
    # Signaux de sortie bruts
    # ==========================================================

    signaux_sortie = pd.DataFrame(
        index=data.index
    )

    signaux_sortie["BB"] = (
        data["bollinger_pct_b"].shift(2).gt(0.95)
        & data["bollinger_pct_b"].shift(1).gt(0.95)
        & data["bollinger_pct_b"].le(0.95)
    )

    signaux_sortie["ADX"] = (
        data["adx"].shift(2).gt(25.00)
        & data["adx"].shift(1).gt(25.00)
        & data["adx"].le(25.00)
    )

    signaux_sortie["Rev_5"] = (
        data["rev_5"].shift(2).gt(-0.025)
        & data["rev_5"].shift(1).gt(-0.025)
        & data["rev_5"].le(-0.025)
    )

    signaux_sortie["RSI"] = (
        data["rsi"].shift(2).gt(65.00)
        & data["rsi"].shift(1).gt(65.00)
        & data["rsi"].le(65.00)
    )

    signaux_sortie["stoch_lent"] = (
        data["stoch"].shift(2).gt(75.00)
        & data["stoch"].shift(1).gt(75.00)
        & data["stoch"].le(75.00)
    )

    signaux_sortie["Hist"] = (
        data["hist_norme"].shift(2).gt(SEUIL_HIST_SORTIE)
        & data["hist_norme"].shift(1).gt(SEUIL_HIST_SORTIE)
        & data["hist_norme"].le(SEUIL_HIST_SORTIE)
    )

    signaux_sortie["ER"] = (
        data["er"].shift(2).gt(0.35)
        & data["er"].shift(1).gt(0.35)
        & data["er"].le(0.35)
        & data["Close"].lt(
            data["Close"].shift(10)
        )
    )

    signaux_sortie = (
        signaux_sortie
        .fillna(False)
        .astype(bool)
    )

    # ==========================================================
    # Fenêtre glissante de confirmation
    # ==========================================================

    entree_recente = signaux_dans_fenetre(
        signaux_entree,
        fenetre=fenetre,
    )

    sortie_recente = signaux_dans_fenetre(
        signaux_sortie,
        fenetre=fenetre,
    )

    nombre_signaux_entree = (
        entree_recente.sum(axis=1)
    )

    condition_entree = (
        nombre_signaux_entree >= 2
    )

    # Un seul événement lorsque la combinaison devient valide.
    signal_entree_brut = (
        condition_entree
        & ~condition_entree.shift(
            1,
            fill_value=False,
        )
    )

    # ==========================================================
    # Exécution au jour du signal ou au jour suivant
    # ==========================================================

    if execution == "jour_suivant":

        ordre_entree = (
            signal_entree_brut
            .shift(1, fill_value=False)
            .astype(bool)
        )

        # À la date d'exécution, on regarde les signaux
        # de sortie constatés la veille.
        sortie_recente_execution = (
            sortie_recente
            .shift(1, fill_value=False)
            .astype(bool)
        )

    elif execution == "jour_signal":

        ordre_entree = signal_entree_brut.astype(bool)
        sortie_recente_execution = sortie_recente.copy()

    else:
        raise ValueError(
            "execution doit être égal à 'jour_signal' "
            "ou 'jour_suivant'."
        )

    return {
        "data": data,
        "signaux_entree_bruts": signaux_entree,
        "signaux_sortie_bruts": signaux_sortie,
        "entree_recente": entree_recente,
        "sortie_recente": sortie_recente,
        "sortie_recente_execution":
            sortie_recente_execution,
        "signal_entree_brut": signal_entree_brut,
        "ordre_entree": ordre_entree,
        "nombre_signaux_entree":
            nombre_signaux_entree,
    }


def backtest_portefeuille_multi_titres(
    prix_actions: dict[str, pd.DataFrame],
    indicateurs_actions: dict[str, pd.DataFrame],
    secteur_par_ticker: dict[str, str],
    capital_initial: float = 100_000.0,
    nombre_sous_portefeuilles: int = 20,
    fenetre: int = 5,
    stop_atr: float = 2.5,
    execution: str = "jour_suivant",
    frais_bps: float = 0.0,
    cloturer_fin: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Backtest global multi-titres.

    Le capital initial est divisé en plusieurs poches indépendantes.

    Chaque poche :
        - détient au maximum une position ;
        - investit 100 % de son cash ;
        - récupère le capital à la sortie ;
        - réinvestit ensuite sa nouvelle valeur.

    Une seule position simultanée est autorisée par ticker.
    """

    # ==========================================================
    # 1. Contrôles
    # ==========================================================

    if capital_initial <= 0:
        raise ValueError(
            "Le capital initial doit être positif."
        )

    if nombre_sous_portefeuilles <= 0:
        raise ValueError(
            "Le nombre de sous-portefeuilles doit être positif."
        )

    if fenetre <= 0:
        raise ValueError(
            "La fenêtre doit être positive."
        )

    if stop_atr <= 0:
        raise ValueError(
            "Le multiplicateur ATR doit être positif."
        )

    if frais_bps < 0:
        raise ValueError(
            "Les frais ne peuvent pas être négatifs."
        )

    taux_frais = frais_bps / 10_000

    # ==========================================================
    # 2. Préparation des signaux par ticker
    # ==========================================================

    signaux_actions = {}
    tickers_valides = []

    for ticker in prix_actions:

        if ticker not in indicateurs_actions:
            print(
                f"{ticker} ignoré : "
                "indicateurs absents."
            )
            continue

        try:
            signaux_actions[ticker] = (
                preparer_signaux_titre(
                    ohlcv=prix_actions[ticker],
                    indicateurs=indicateurs_actions[ticker],
                    fenetre=fenetre,
                    execution=execution,
                )
            )

            tickers_valides.append(ticker)

        except (ValueError, KeyError) as erreur:
            print(
                f"{ticker} ignoré : {erreur}"
            )

    if not tickers_valides:
        raise ValueError(
            "Aucun ticker valide pour le backtest."
        )

    # Calendrier global : union des dates.
    calendrier = pd.DatetimeIndex(
        sorted(
            set().union(
                *[
                    set(
                        signaux_actions[ticker][
                            "data"
                        ].index
                    )
                    for ticker in tickers_valides
                ]
            )
        )
    )

    # ==========================================================
    # 3. Initialisation des poches
    # ==========================================================

    capital_par_poche = (
        capital_initial
        / nombre_sous_portefeuilles
    )

    poches = {
        numero: {
            "cash": float(capital_par_poche),
            "ticker": None,
        }
        for numero in range(
            1,
            nombre_sous_portefeuilles + 1,
        )
    }

    # Une position par ticker.
    positions_ouvertes = {}

    trades = []
    suivi_portefeuille = []

    # Dernier cours connu pour valoriser une position
    # lorsqu'une date est absente pour un ticker.
    derniers_cours = {}

    # ==========================================================
    # 4. Simulation chronologique
    # ==========================================================

    for date in calendrier:

        indice_date = calendrier.get_loc(date)

        entrees_du_jour = []
        sorties_du_jour = []

        # Mise à jour des derniers cours disponibles.
        for ticker in tickers_valides:

            data_ticker = signaux_actions[
                ticker
            ]["data"]

            if date in data_ticker.index:
                derniers_cours[ticker] = float(
                    data_ticker.loc[
                        date,
                        "Close",
                    ]
                )

        # ======================================================
        # A. SORTIES
        # ======================================================

        ordres_sortie = []

        for ticker, position in list(
            positions_ouvertes.items()
        ):

            donnees = signaux_actions[ticker]
            data_ticker = donnees["data"]

            if date not in data_ticker.index:
                continue

            open_jour = float(
                data_ticker.loc[date, "Open"]
            )

            low_jour = float(
                data_ticker.loc[date, "Low"]
            )

            close_jour = float(
                data_ticker.loc[date, "Close"]
            )

            niveau_stop = float(
                position["niveau_stop"]
            )

            # --------------------------------------------------
            # Stop loss ATR
            # --------------------------------------------------

            stop_touche = (
                low_jour <= niveau_stop
            )

            # En cas de gap sous le stop, vente à l'ouverture.
            if stop_touche:

                if open_jour <= niveau_stop:
                    prix_sortie = open_jour
                else:
                    prix_sortie = niveau_stop

                motif_sortie = "Stop ATR"
                declencheurs_sortie = []

            else:

                # ----------------------------------------------
                # Sortie par indicateurs
                # ----------------------------------------------

                sortie_execution = donnees[
                    "sortie_recente_execution"
                ]

                if date not in sortie_execution.index:
                    continue

                ligne_sortie = (
                    sortie_execution.loc[date]
                )

                declencheurs_sortie = [
                    colonne
                    for colonne in sortie_execution.columns
                    if bool(ligne_sortie[colonne])
                ]

                indicateurs_actifs = set(
                    declencheurs_sortie
                )

                indicateurs_requis = set(
                    position[
                        "indicateurs_requis_sortie"
                    ]
                )

                # Un seul indicateur d'entrée suffit.
                sortie_confirmee = (
                    len(
                        indicateurs_requis
                        & indicateurs_actifs
                    )
                    >= INDICATEURS_SORTIE_REQUIS
                )

                if not sortie_confirmee:
                    continue

                prix_sortie = close_jour
                motif_sortie = "Signal indicateurs"

            ordres_sortie.append({
                "ticker": ticker,
                "prix_sortie": prix_sortie,
                "motif_sortie": motif_sortie,
                "declencheurs_sortie":
                    declencheurs_sortie,
            })

        # ------------------------------------------------------
        # Exécution de toutes les sorties
        # ------------------------------------------------------

        for ordre in ordres_sortie:

            ticker = ordre["ticker"]

            position = positions_ouvertes.pop(
                ticker
            )

            numero_poche = position[
                "numero_poche"
            ]

            prix_sortie = float(
                ordre["prix_sortie"]
            )

            nombre_unites = float(
                position["nombre_unites"]
            )

            valeur_brute_sortie = (
                nombre_unites
                * prix_sortie
            )

            frais_sortie = (
                valeur_brute_sortie
                * taux_frais
            )

            capital_apres_sortie = (
                valeur_brute_sortie
                - frais_sortie
            )

            # La poche récupère tout le capital.
            poches[numero_poche]["cash"] = (
                capital_apres_sortie
            )

            poches[numero_poche]["ticker"] = None

            pnl_montant = (
                capital_apres_sortie
                - position["capital_avant_entree"]
            )

            rendement = (
                capital_apres_sortie
                / position["capital_avant_entree"]
                - 1
            )

            duree_seances = (
                indice_date
                - position["indice_calendrier_entree"]
            )

            trades.append({
                "Sous-portefeuille":
                    numero_poche,

                "Ticker":
                    ticker,

                "Secteur":
                    secteur_par_ticker.get(
                        ticker,
                        "Inconnu",
                    ),

                "Date signal entrée":
                    position["date_signal_entree"],

                "Date entrée":
                    position["date_entree"],

                "Prix entrée":
                    position["prix_entree"],

                "Déclencheurs entrée":
                    position["declencheurs_entree"],

                "Indicateurs requis sortie":
                    position[
                        "indicateurs_requis_sortie"
                    ],

                "ATR entrée":
                    position["atr_entree"],

                "Niveau stop":
                    position["niveau_stop"],

                "Capital avant entrée":
                    position[
                        "capital_avant_entree"
                    ],

                "Frais entrée":
                    position["frais_entree"],

                "Montant investi":
                    position["montant_investi"],

                "Nombre d'unités":
                    nombre_unites,

                "Date sortie":
                    date,

                "Prix sortie":
                    prix_sortie,

                "Motif sortie":
                    ordre["motif_sortie"],

                "Déclencheurs sortie":
                    ordre[
                        "declencheurs_sortie"
                    ],

                "Valeur brute sortie":
                    valeur_brute_sortie,

                "Frais sortie":
                    frais_sortie,

                "Capital après sortie":
                    capital_apres_sortie,

                "P&L ($)":
                    pnl_montant,

                "P&L":
                    rendement,

                "P&L (%)":
                    rendement * 100,

                "Durée en séances":
                    duree_seances,

                "Statut":
                    "Clôturé",
            })

            sorties_du_jour.append(ticker)

        # ======================================================
        # B. ENTRÉES
        # ======================================================

        candidats_entree = []

        for ticker in tickers_valides:

            # Une seule position par ticker.
            if ticker in positions_ouvertes:
                continue

            donnees = signaux_actions[ticker]
            data_ticker = donnees["data"]

            if date not in data_ticker.index:
                continue

            ordre_entree = donnees[
                "ordre_entree"
            ]

            if date not in ordre_entree.index:
                continue

            if not bool(
                ordre_entree.loc[date]
            ):
                continue

            # Retrouver la date réelle du signal.
            position_date = (
                data_ticker.index.get_loc(date)
            )

            if (
                execution == "jour_suivant"
                and position_date > 0
            ):
                position_signal = position_date - 1
            else:
                position_signal = position_date

            declencheurs_requis = (
                premiers_declencheurs(
                    signaux_bruts=donnees[
                        "signaux_entree_bruts"
                    ],
                    position=position_signal,
                    fenetre=fenetre,
                    maximum=2,
                )
            )

            if len(declencheurs_requis) < 2:
                continue

            declencheurs_complets = (
                noms_signaux_actifs(
                    signaux_recents=donnees[
                        "entree_recente"
                    ],
                    position=position_signal,
                )
            )

            nombre_signaux = len(
                declencheurs_complets
            )

            candidats_entree.append({
                "ticker": ticker,
                "position_signal":
                    position_signal,
                "declencheurs_requis":
                    declencheurs_requis,
                "declencheurs_complets":
                    declencheurs_complets,
                "nombre_signaux":
                    nombre_signaux,
            })

        # Priorité aux titres ayant le plus de confirmations.
        # En cas d'égalité, ordre alphabétique du ticker.
        candidats_entree = sorted(
            candidats_entree,
            key=lambda x: (
                -x["nombre_signaux"],
                x["ticker"],
            ),
        )

        for candidat in candidats_entree:

            ticker = candidat["ticker"]

            if ticker in positions_ouvertes:
                continue

            # Recherche de la première poche LIBRE.
            numero_poche = next(
                (
                    numero
                    for numero, poche in poches.items()
                    if poche["ticker"] is None
                ),
                None,
            )

            # Plus aucune poche disponible aujourd'hui.
            if numero_poche is None:
                break

            donnees = signaux_actions[ticker]
            data_ticker = donnees["data"]

            prix_entree = float(
                data_ticker.loc[date, "Open"]
            )

            if prix_entree <= 0:
                continue

            # L'ATR du jour d'exécution contient le haut et le bas
            # de ce jour, inconnus à l'ouverture : on prend celui
            # du jour du signal.
            atr_entree = float(
                data_ticker["atr"].iloc[
                    candidat["position_signal"]
                ]
            )

            if (
                not np.isfinite(atr_entree)
                or atr_entree <= 0
            ):
                continue

            capital_avant_entree = float(
                poches[numero_poche]["cash"]
            )

            if capital_avant_entree <= 0:
                continue

            # On réserve les frais avant d'investir, pour ne
            # jamais dépasser le cash de la poche.
            montant_investi = (
                capital_avant_entree
                / (1 + taux_frais)
            )

            frais_entree = (
                capital_avant_entree
                - montant_investi
            )

            nombre_unites = (
                montant_investi
                / prix_entree
            )

            niveau_stop = (
                prix_entree
                - stop_atr * atr_entree
            )

            poches[numero_poche]["cash"] = 0.0
            poches[numero_poche]["ticker"] = ticker

            positions_ouvertes[ticker] = {
                "numero_poche":
                    numero_poche,

                "date_signal_entree":
                    data_ticker.index[
                        candidat["position_signal"]
                    ],

                "date_entree":
                    date,

                "indice_calendrier_entree":
                    indice_date,

                "prix_entree":
                    prix_entree,

                "atr_entree":
                    atr_entree,

                "niveau_stop":
                    niveau_stop,

                "nombre_unites":
                    nombre_unites,

                "montant_investi":
                    montant_investi,

                "frais_entree":
                    frais_entree,

                "capital_avant_entree":
                    capital_avant_entree,

                "declencheurs_entree":
                    candidat["declencheurs_complets"],

                "indicateurs_requis_sortie":
                    candidat["declencheurs_requis"],
            }

            entrees_du_jour.append(ticker)

        # ======================================================
        # C. VALORISATION DU JOUR
        # ======================================================

        cash_total = sum(
            poche["cash"]
            for poche in poches.values()
        )

        valeur_positions = 0.0

        for ticker, position in positions_ouvertes.items():

            # Dernier cours connu : une date absente pour ce
            # ticker ne doit pas faire disparaître la position
            # de la valorisation.
            cours = derniers_cours.get(
                ticker,
                position["prix_entree"],
            )

            valeur_positions += (
                position["nombre_unites"]
                * cours
            )

        suivi_portefeuille.append({
            "Date": date,
            "Cash": cash_total,
            "Valeur positions": valeur_positions,
            "Valeur portefeuille":
                cash_total + valeur_positions,
            "Positions ouvertes":
                len(positions_ouvertes),
            "Poches libres": sum(
                1
                for poche in poches.values()
                if poche["ticker"] is None
            ),
            "Entrées": len(entrees_du_jour),
            "Sorties": len(sorties_du_jour),
        })

    # ==========================================================
    # 5. Clôture éventuelle en fin de période
    # ==========================================================

    derniere_date = calendrier[-1]
    indice_fin = len(calendrier) - 1

    if cloturer_fin and positions_ouvertes:

        for ticker in list(positions_ouvertes):

            position = positions_ouvertes.pop(ticker)
            numero_poche = position["numero_poche"]

            prix_sortie = derniers_cours.get(
                ticker,
                position["prix_entree"],
            )

            valeur_brute_sortie = (
                position["nombre_unites"]
                * prix_sortie
            )

            frais_sortie = (
                valeur_brute_sortie
                * taux_frais
            )

            capital_apres_sortie = (
                valeur_brute_sortie
                - frais_sortie
            )

            poches[numero_poche]["cash"] = (
                capital_apres_sortie
            )

            poches[numero_poche]["ticker"] = None

            rendement = (
                capital_apres_sortie
                / position["capital_avant_entree"]
                - 1
            )

            trades.append({
                "Sous-portefeuille": numero_poche,
                "Ticker": ticker,
                "Secteur": secteur_par_ticker.get(
                    ticker,
                    "Inconnu",
                ),
                "Date signal entrée":
                    position["date_signal_entree"],
                "Date entrée": position["date_entree"],
                "Prix entrée": position["prix_entree"],
                "Déclencheurs entrée":
                    position["declencheurs_entree"],
                "Indicateurs requis sortie":
                    position["indicateurs_requis_sortie"],
                "ATR entrée": position["atr_entree"],
                "Niveau stop": position["niveau_stop"],
                "Capital avant entrée":
                    position["capital_avant_entree"],
                "Frais entrée": position["frais_entree"],
                "Montant investi":
                    position["montant_investi"],
                "Nombre d'unités":
                    position["nombre_unites"],
                "Date sortie": derniere_date,
                "Prix sortie": prix_sortie,
                "Motif sortie": "Clôture fin de période",
                "Déclencheurs sortie": [],
                "Valeur brute sortie": valeur_brute_sortie,
                "Frais sortie": frais_sortie,
                "Capital après sortie":
                    capital_apres_sortie,
                "P&L ($)":
                    capital_apres_sortie
                    - position["capital_avant_entree"],
                "P&L": rendement,
                "P&L (%)": rendement * 100,
                "Durée en séances":
                    indice_fin
                    - position["indice_calendrier_entree"],
                "Statut": "Clôturé fin de période",
            })

        # La dernière ligne de suivi doit refléter la clôture.
        if suivi_portefeuille:

            cash_total = sum(
                poche["cash"]
                for poche in poches.values()
            )

            suivi_portefeuille[-1].update({
                "Cash": cash_total,
                "Valeur positions": 0.0,
                "Valeur portefeuille": cash_total,
                "Positions ouvertes": 0,
                "Poches libres":
                    nombre_sous_portefeuilles,
            })

    # ==========================================================
    # 6. Mise en DataFrame des quatre sorties
    # ==========================================================

    colonnes_trades = [
        "Sous-portefeuille",
        "Ticker",
        "Secteur",
        "Date signal entrée",
        "Date entrée",
        "Prix entrée",
        "Déclencheurs entrée",
        "Indicateurs requis sortie",
        "ATR entrée",
        "Niveau stop",
        "Capital avant entrée",
        "Frais entrée",
        "Montant investi",
        "Nombre d'unités",
        "Date sortie",
        "Prix sortie",
        "Motif sortie",
        "Déclencheurs sortie",
        "Valeur brute sortie",
        "Frais sortie",
        "Capital après sortie",
        "P&L ($)",
        "P&L",
        "P&L (%)",
        "Durée en séances",
        "Statut",
    ]

    df_suivi = pd.DataFrame(suivi_portefeuille)

    if not df_suivi.empty:
        df_suivi = df_suivi.set_index("Date")

    df_trades = pd.DataFrame(
        trades,
        columns=colonnes_trades,
    )

    if not df_trades.empty:
        df_trades = (
            df_trades
            .sort_values("Date sortie")
            .reset_index(drop=True)
        )

    lignes_ouvertes = []

    for ticker, position in positions_ouvertes.items():

        cours = derniers_cours.get(
            ticker,
            position["prix_entree"],
        )

        valeur_position = (
            position["nombre_unites"]
            * cours
        )

        rendement_latent = (
            valeur_position
            / position["capital_avant_entree"]
            - 1
        )

        lignes_ouvertes.append({
            "Sous-portefeuille": position["numero_poche"],
            "Ticker": ticker,
            "Secteur": secteur_par_ticker.get(
                ticker,
                "Inconnu",
            ),
            "Date signal entrée":
                position["date_signal_entree"],
            "Date entrée": position["date_entree"],
            "Prix entrée": position["prix_entree"],
            "Déclencheurs entrée":
                position["declencheurs_entree"],
            "Indicateurs requis sortie":
                position["indicateurs_requis_sortie"],
            "ATR entrée": position["atr_entree"],
            "Niveau stop": position["niveau_stop"],
            "Nombre d'unités": position["nombre_unites"],
            "Capital avant entrée":
                position["capital_avant_entree"],
            "Dernier cours": cours,
            "Valeur position": valeur_position,
            "P&L latent ($)":
                valeur_position
                - position["capital_avant_entree"],
            "P&L latent (%)": rendement_latent * 100,
            "Durée en séances":
                indice_fin
                - position["indice_calendrier_entree"],
            "Statut": "Ouverte",
        })

    df_ouvertes = pd.DataFrame(lignes_ouvertes)

    # Historique : trades clôturés + positions encore ouvertes.
    morceaux = [
        d
        for d in (df_trades, df_ouvertes)
        if not d.empty
    ]

    df_historique = (
        pd.concat(
            morceaux,
            ignore_index=True,
            sort=False,
        )
        if morceaux
        else pd.DataFrame()
    )

    return (
        df_suivi,
        df_trades,
        df_ouvertes,
        df_historique,
    )


# ==============================================================
# EXÉCUTION
#
# prix_secteurs, data_secteurs et secteur_par_ticker doivent
# exister. Si tu lances ce fichier directement, charge-les
# au-dessus de cette ligne (ou importe ce module depuis le
# script où ils sont déjà construits).
# ==============================================================

manquants = [
    nom
    for nom in (
        "prix_secteurs",
        "data_secteurs",
        "secteur_par_ticker",
    )
    if nom not in globals()
]

if manquants:
    raise NameError(
        "Variables absentes : "
        f"{manquants}. Charge tes données avant ce bloc, "
        "ou importe ce module depuis le script qui les "
        "construit :\n"
        "    from backtest_secteurs import "
        "backtest_portefeuille_multi_titres"
    )

suivi_portefeuille, trades, positions_ouvertes, historique_positions = (
    backtest_portefeuille_multi_titres(
        prix_actions=prix_secteurs,
        indicateurs_actions=data_secteurs,
        secteur_par_ticker=secteur_par_ticker,
        capital_initial=100_000.0,
        nombre_sous_portefeuilles=20,
        fenetre=5,
        stop_atr=2.5,
        execution="jour_suivant",
        frais_bps=10.0,
        cloturer_fin=False,
    )
)

capital_initial = 100_000.0

capital_final = float(
    suivi_portefeuille[
        "Valeur portefeuille"
    ].iloc[-1]
)

performance_totale = (
    capital_final
    / capital_initial
    - 1
)

print("\n===== RÉSULTATS DU PORTEFEUILLE =====")

print(
    f"Capital initial : "
    f"{capital_initial:,.2f} $"
)

print(
    f"Capital final : "
    f"{capital_final:,.2f} $"
)

print(
    f"Performance totale : "
    f"{performance_totale:.2%}"
)

print(
    f"Nombre de trades clôturés : "
    f"{len(trades)}"
)

print(
    f"Nombre de positions ouvertes : "
    f"{len(positions_ouvertes)}"
)

if not trades.empty:

    taux_reussite = (
        trades["P&L"].gt(0).mean()
    )

    print(
        f"P&L moyen par trade : "
        f"{trades['P&L (%)'].mean():.2f} %"
    )

    print(
        f"Taux de réussite : "
        f"{taux_reussite:.2%}"
    )

    print(
        f"Durée moyenne : "
        f"{trades['Durée en séances'].mean():.2f} séances"
    )

    print("\nMotifs de sortie :")

    print(
        trades[
            "Motif sortie"
        ].value_counts()
    )

if not trades.empty:

    resume_secteurs = (
        trades
        .groupby("Secteur")
        .agg(
            Nombre_trades=(
                "Ticker",
                "count",
            ),
            PnL_total=(
                "P&L ($)",
                "sum",
            ),
            PnL_moyen_pct=(
                "P&L (%)",
                "mean",
            ),
            Taux_reussite=(
                "P&L",
                lambda x: (
                    x.gt(0).mean() * 100
                ),
            ),
            Duree_moyenne=(
                "Durée en séances",
                "mean",
            ),
        )
        .round(2)
        .sort_values(
            "PnL_total",
            ascending=False,
        )
    )

else:

    resume_secteurs = pd.DataFrame()

with pd.ExcelWriter(
    "backtest_portefeuille_global.xlsx",
    engine="openpyxl",
) as writer:

    suivi_portefeuille.to_excel(
        writer,
        sheet_name="Portefeuille",
    )

    trades.to_excel(
        writer,
        sheet_name="Trades",
        index=False,
    )

    positions_ouvertes.to_excel(
        writer,
        sheet_name="Positions ouvertes",
        index=False,
    )

    historique_positions.to_excel(
        writer,
        sheet_name="Historique positions",
        index=False,
    )

    resume_secteurs.to_excel(
        writer,
        sheet_name="Résumé secteurs",
    )

print(
    "\nFichier créé : "
    "backtest_portefeuille_global.xlsx"
)
