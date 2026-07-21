import os
import pandas as pd


def load_and_preview_csv(file_name):
    current_directory = os.getcwd()

    file_path = os.path.join(
        current_directory,
        file_name
    )

    df = pd.read_csv(
        file_path,
        sep=";",
        low_memory=False
    )

    # Le fichier contient 58 séries :
    # 58 colonnes de dates + 58 colonnes de valeurs
    df = df.iloc[:, :116].copy()

    return df


df = load_and_preview_csv(
    "Data_in_value.csv"
)
