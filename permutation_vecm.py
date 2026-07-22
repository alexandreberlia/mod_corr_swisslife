import pandas as pd


def permuter_colonnes(df, nouvel_ordre):
    """
    Réordonne les colonnes d'un DataFrame selon l'ordre indiqué.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame original.

    nouvel_ordre : list[str]
        Liste contenant les noms des colonnes dans l'ordre souhaité.

    Returns
    -------
    pandas.DataFrame
        Copie du DataFrame avec les colonnes réordonnées.
    """

    # Vérification des colonnes manquantes
    colonnes_manquantes = [
        colonne
        for colonne in nouvel_ordre
        if colonne not in df.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Les colonnes suivantes n'existent pas dans le DataFrame :\n"
            f"{colonnes_manquantes}\n\n"
            f"Colonnes disponibles :\n{list(df.columns)}"
        )

    # Vérification des doublons
    if len(nouvel_ordre) != len(set(nouvel_ordre)):
        raise ValueError(
            "Le nouvel ordre contient des colonnes en double."
        )

    # Vérification que toutes les colonnes ont été renseignées
    colonnes_oubliees = [
        colonne
        for colonne in df.columns
        if colonne not in nouvel_ordre
    ]

    if colonnes_oubliees:
        raise ValueError(
            "Certaines colonnes du DataFrame sont absentes du nouvel ordre :\n"
            f"{colonnes_oubliees}"
        )

    return df.loc[:, nouvel_ordre].copy()

from itertools import permutations


def generer_permutations_colonnes(df, colonnes=None):
    """
    Génère un dictionnaire contenant un DataFrame pour chaque
    permutation possible des colonnes sélectionnées.

    Attention
    ---------
    Le nombre de permutations augmente très rapidement :
    n variables produisent n! permutations.
    """

    if colonnes is None:
        colonnes = list(df.columns)

    colonnes_manquantes = [
        colonne
        for colonne in colonnes
        if colonne not in df.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Colonnes introuvables :\n"
            f"{colonnes_manquantes}"
        )

    resultats = {}

    for ordre in permutations(colonnes):

        nom_ordre = " -> ".join(ordre)

        resultats[nom_ordre] = (
            df.loc[:, list(ordre)].copy()
        )

    return resultats

from statsmodels.tsa.vector_ar.vecm import VECM


resultats_vecm = {}

for nom_ordre, data_ordonnee in dataframes_ordonnes.items():

    modele = VECM(
        data_ordonnee,
        k_ar_diff=2,
        coint_rank=1,
        deterministic="co"
    )

    resultats_vecm[nom_ordre] = modele.fit()

    print(
        f"VECM estimé pour l'ordre : {nom_ordre}"
    )
