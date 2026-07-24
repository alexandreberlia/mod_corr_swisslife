"""
data_loader_bloom
=================

Nettoyage des exports CSV Bloomberg.

Format du fichier
-----------------
N séries côte à côte, chacune sur **deux colonnes consécutives** : une
colonne de dates, puis une colonne de valeurs.

    | Dates for GDP US Chained Dollars QoQ SAA (GDP) |      | Dates for ...
    | 31/03/2020                                     | 1,23 | 31/03/2020 ...

Nommage (identique aux anciens scripts)
---------------------------------------
Le nom d'une série est l'en-tête de sa **colonne de valeurs**, lu par pandas
exactement comme le faisait ``pd.read_csv("corrected_file.csv")`` :
"GDP US Chained Dollars QoQ SAA (GDP)" reste
"GDP US Chained Dollars QoQ SAA (GDP)".
C'est le remplacement direct du mapping ``{"Unnamed: 3": "..."}``.
Si cet en-tête est vide, on retombe sur l'en-tête de dates privé de
"Dates for " (sinon la colonne s'appellerait "Unnamed: 3").

Sortie
------
``{nom_de_la_serie: DataFrame}`` — un DataFrame d'**une seule colonne** par
série (jamais de Series : pas de ``Name:`` à l'affichage), indexé par les
dates. Le nom de l'index est l'en-tête de la colonne de dates
("Dates for ..."), comme le produisait ``set_index`` dans l'ancien code.

Aucune conversion de fréquence n'est faite ici : les données restent à leur
temporalité d'origine.

Exemple
-------
    from data_loader_bloom import load_bloomberg_csv

    series = load_bloomberg_csv("export.csv", n_series=58,
                                names_file="corrected_file.csv")
    series["US Industrial Production, YOY S (Economic Dynamic)"].tail()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd

__all__ = [
    "read_columns",
    "series_names",
    "read_raw",
    "clean_pairs",
    "load_bloomberg_csv",
]

DATE_PREFIX = "Dates for "


# ---------------------------------------------------------------------------
# En-têtes
# ---------------------------------------------------------------------------
def read_columns(
    path: str | Path,
    sep: str = ";",
    header_row: int = 0,
    encoding: str | None = None,
) -> list[str]:
    """
    Renvoie les noms de colonnes **tels que pandas les lit**.

    On garde volontairement la lecture pandas (et donc ses éventuels
    "Unnamed: 3" / "X.1") pour que les noms de séries soient rigoureusement
    identiques à ceux des anciens scripts.
    """
    header = pd.read_csv(
        Path(path), sep=sep, nrows=0, skiprows=header_row, encoding=encoding
    )
    return [str(column) for column in header.columns]


def _is_auto_generated(column: str) -> bool:
    return column.startswith("Unnamed:") or column.strip() == ""


def series_names(columns: Sequence[str], n_series: int | None = None) -> list[str]:
    """
    Nom de chaque série = en-tête de sa colonne de valeurs (colonnes impaires).

    Repli sur l'en-tête de dates sans "Dates for " si l'en-tête de valeurs
    est vide.
    """
    cells = list(columns)
    if n_series is not None:
        cells = cells[: 2 * n_series]

    names: list[str] = []
    for position in range(len(cells) // 2):
        date_header = cells[2 * position]
        value_header = cells[2 * position + 1]

        if not _is_auto_generated(value_header):
            name = value_header
        elif not _is_auto_generated(date_header):
            name = (
                date_header[len(DATE_PREFIX):].strip()
                if date_header.startswith(DATE_PREFIX)
                else date_header
            )
        else:
            name = f"serie_{position + 1}"

        if name in names:  # un doublon écraserait une série dans le dictionnaire
            suffix = 2
            while f"{name}.{suffix}" in names:
                suffix += 1
            warnings.warn(
                f"Nom de série en double : '{name}' renommé '{name}.{suffix}'.",
                stacklevel=2,
            )
            name = f"{name}.{suffix}"
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
def _to_numeric(values: pd.Series) -> pd.Series:
    """
    Convertit une colonne de valeurs en float.

    Gère les formats européens : espaces (y compris insécables) comme
    séparateur de milliers, virgule décimale, symbole %. Si un champ contient
    un point ET une virgule, la virgule est traitée comme séparateur de
    milliers ; sinon comme décimale.
    """
    if pd.api.types.is_numeric_dtype(values):
        return values.astype("float64")

    text = (
        values.astype("string")
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    has_both = (
        text.str.contains(".", regex=False, na=False)
        & text.str.contains(",", regex=False, na=False)
    ).any()
    text = (
        text.str.replace(",", "", regex=False)
        if has_both
        else text.str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(text, errors="coerce").astype("float64")


def read_raw(
    path: str | Path,
    sep: str = ";",
    n_series: int | None = None,
    header_row: int = 0,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Lit les données brutes en texte, colonnes numérotées 0..n (sans en-tête)."""
    raw = pd.read_csv(
        Path(path),
        sep=sep,
        header=None,
        skiprows=header_row + 1,
        dtype=str,
        encoding=encoding,
    )

    if n_series is not None:
        return raw.iloc[:, : 2 * n_series].copy()

    # Bloomberg laisse souvent des colonnes vides en fin de fichier.
    non_empty = raw.notna().any(axis=0).to_numpy()
    last = len(non_empty) - 1
    while last >= 0 and not non_empty[last]:
        last -= 1
    raw = raw.iloc[:, : last + 1].copy()

    if raw.shape[1] % 2:
        warnings.warn(
            f"Nombre de colonnes impair ({raw.shape[1]}) : la dernière est ignorée. "
            "Précisez n_series= pour forcer le découpage.",
            stacklevel=2,
        )
        raw = raw.iloc[:, :-1].copy()
    return raw


def clean_pairs(
    raw: pd.DataFrame,
    names: Sequence[str],
    date_headers: Sequence[str] | None = None,
    dayfirst: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Découpe le tableau brut en un DataFrame d'une colonne par série.

    Les NaN de valeurs sont conservés : c'est le calendrier complet qui sert
    ensuite à regrouper les séries entre elles.
    """
    series: dict[str, pd.DataFrame] = {}
    n_pairs = raw.shape[1] // 2

    if len(names) < n_pairs:
        raise ValueError(
            f"{len(names)} noms disponibles pour {n_pairs} séries. "
            "Vérifiez names_file / n_series."
        )

    for position in range(n_pairs):
        name = names[position]
        index_name = (
            date_headers[position]
            if date_headers is not None and position < len(date_headers)
            else f"{DATE_PREFIX}{name}"
        )

        dates = pd.to_datetime(
            raw.iloc[:, 2 * position], dayfirst=dayfirst, errors="coerce"
        )
        values = _to_numeric(raw.iloc[:, 2 * position + 1])

        frame = pd.DataFrame(
            {name: values.to_numpy()},
            index=pd.DatetimeIndex(dates, name=index_name),
        )
        frame = frame[frame.index.notna()]                     # dates illisibles
        frame = frame[~frame.index.duplicated(keep="last")]    # dates en double
        frame = frame.sort_index()

        if frame[name].notna().sum() == 0:
            warnings.warn(f"Série '{name}' sans aucune valeur : ignorée.", stacklevel=2)
            continue
        series[name] = frame

    return series


def load_bloomberg_csv(
    path: str | Path,
    sep: str = ";",
    n_series: int | None = None,
    dayfirst: bool = True,
    names_file: str | Path | None = None,
    names_sep: str = ",",
    names: Sequence[str] | None = None,
    header_row: int = 0,
    encoding: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Chaîne complète : lecture -> découpage -> nettoyage.

    Parameters
    ----------
    path : CSV Bloomberg à charger.
    sep : séparateur du fichier (";" pour les exports européens).
    n_series : nombre de séries attendues (58 dans l'export historique) ;
        None pour détection automatique.
    dayfirst : True pour des dates JJ/MM/AAAA.
    names_file : fichier dont les en-têtes portent les noms des séries
        (typiquement "corrected_file.csv"), si le fichier principal a des
        en-têtes de valeurs vides.
    names_sep : séparateur du fichier de noms.
    names : liste de noms imposée, qui court-circuite tout le reste.
    header_row : index (base 0) de la ligne d'en-tête.

    Returns
    -------
    dict[str, DataFrame] : un DataFrame d'une colonne par série, à la
    fréquence d'origine.
    """
    raw = read_raw(
        path, sep=sep, n_series=n_series, header_row=header_row, encoding=encoding
    )
    date_headers = read_columns(path, sep=sep, header_row=header_row, encoding=encoding)[::2]

    if names is None:
        source, source_sep = (
            (names_file, names_sep) if names_file is not None else (path, sep)
        )
        columns = read_columns(
            source, sep=source_sep, header_row=header_row, encoding=encoding
        )
        names = series_names(columns, n_series=n_series)

    return clean_pairs(raw, names, date_headers=date_headers, dayfirst=dayfirst)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    loaded = load_bloomberg_csv(sys.argv[1])
    print(f"{len(loaded)} séries chargées :")
    for series_name, frame in loaded.items():
        column = frame[series_name]
        print(
            f"  - {series_name:<55} {column.notna().sum():>6} pts  "
            f"{frame.index.min():%Y-%m-%d} -> {frame.index.max():%Y-%m-%d}"
        )
