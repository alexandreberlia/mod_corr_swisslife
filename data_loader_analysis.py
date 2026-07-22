from data_loader import df

import numpy as np
import pandas as pd


# Copies de sécurité
df_original_values = df.copy()
df_work = df.copy()


def convert_date_columns(dataframe):
    """
    Convertit en dates les colonnes situées aux positions
    0, 2, 4, etc.
    """
    dataframe = dataframe.copy()

    for i in range(0, len(dataframe.columns) - 1, 2):
        column = dataframe.columns[i]

        dataframe[column] = pd.to_datetime(
            dataframe[column],
            dayfirst=True,
            errors="coerce"
        )

    return dataframe


def count_dfs(dataframe):
    """
    Compte le nombre de groupes de dates distincts.

    Attention : cette fonction repose sur la deuxième ligne
    du DataFrame et peut être fragile.
    """
    if len(dataframe) < 2:
        return 0

    return dataframe.iloc[1, ::2].nunique()


def create_df(dataframe):
    """
    Extrait toutes les séries partageant la même colonne
    de dates que la première colonne du DataFrame.
    """
    if dataframe.empty or dataframe.shape[1] < 2:
        return pd.DataFrame()

    chosen_column = dataframe.columns[0]

    new_df = pd.DataFrame({
        chosen_column: dataframe[chosen_column]
    })

    cols_to_drop = []

    for i in range(0, len(dataframe.columns) - 1, 2):
        date_column = dataframe.columns[i]
        value_column = dataframe.columns[i + 1]

        if dataframe[date_column].equals(
            dataframe[chosen_column]
        ):
            new_df[value_column] = dataframe[value_column]

            cols_to_drop.extend([
                date_column,
                value_column
            ])

    if not cols_to_drop:
        return pd.DataFrame()

    dataframe.drop(
        columns=cols_to_drop,
        inplace=True,
        errors="ignore"
    )

    new_df = new_df.dropna(
        subset=[chosen_column]
    )

    new_df.set_index(
        chosen_column,
        inplace=True
    )

    new_df.index = pd.to_datetime(
        new_df.index,
        errors="coerce"
    )

    new_df = new_df[
        ~new_df.index.isna()
    ]

    new_df = new_df.sort_index()

    return new_df


def generate_dataframes(dataframe):
    """
    Génère les différents DataFrames et les retourne
    dans un dictionnaire.
    """
    dataframe = dataframe.copy()
    generated_dfs = {}

    i = 1

    while dataframe.shape[1] >= 2:
        previous_number_columns = dataframe.shape[1]

        new_df = create_df(dataframe)

        if new_df.empty:
            break

        new_df = new_df.dropna(how="all")

        generated_dfs[f"df{i}"] = new_df

        # Sécurité contre une boucle infinie
        if dataframe.shape[1] == previous_number_columns:
            break

        i += 1

    return generated_dfs


def convert_to_weeks(dict_of_dataframes):
    """
    Convertit chaque DataFrame en fréquence hebdomadaire.
    """
    weekly_dataframes = {}

    for key, dataframe in dict_of_dataframes.items():
        dataframe = dataframe.copy()

        # Conversion des colonnes en valeurs numériques
        dataframe = dataframe.apply(
            pd.to_numeric,
            errors="coerce"
        )

        weekly_df = (
            dataframe
            .resample("W")
            .mean()
            .interpolate(method="linear")
            .round(2)
        )

        weekly_dataframes[key] = weekly_df

    return weekly_dataframes


def materials_block(dict_of_dataframes):
    """
    Assemble horizontalement tous les DataFrames
    contenus dans le dictionnaire.
    """
    if not dict_of_dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(
        list(dict_of_dataframes.values()),
        axis=1,
        join="outer"
    )

    # Suppression éventuelle des colonnes dupliquées
    combined_df = combined_df.loc[
        :,
        ~combined_df.columns.duplicated()
    ]

    return combined_df.sort_index()


# --------------------------------------------------
# Exécution
# --------------------------------------------------

df_work = convert_date_columns(df_work)

dict_of_df = generate_dataframes(df_work)

# Conversion hebdomadaire
dict_of_df = convert_to_weeks(dict_of_df)

# Pas besoin de lire corrected_file.csv
df_original = df_original_values.copy()


# Création du mapping à partir du DataFrame original
mapping = {}

for i in range(0, len(df_original.columns) - 1, 2):
    date_col = df_original.columns[i]
    value_col = df_original.columns[i + 1]

    if str(date_col).startswith("Dates for "):
        mapping[f"Unnamed: {i + 1}"] = value_col


# Renommage des colonnes
for name, dataframe in dict_of_df.items():
    dict_of_df[name] = dataframe.rename(
        columns=mapping
    )


# Assemblage final
materials_df = materials_block(dict_of_df)


print(f"{len(dict_of_df)} DataFrames créés")
print("Dimensions du bloc final :", materials_df.shape)
print("Mapping :", mapping)

display(materials_df.head())
