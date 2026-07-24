"""
data_loader_bloom
=================

Chargement et nettoyage des exports CSV Bloomberg.

Format attendu
--------------
N séries posées côte à côte, chaque série occupant **deux colonnes
consécutives** : une colonne de dates, puis une colonne de valeurs.

    | Dates for GDP US Chained Dollars QoQ SAA (GDP) |       | Dates for ...
    | 31/03/2020                                     | 1,23  | 31/03/2020 ...

Les noms de séries sont repris **à l'identique** du fichier : seuls le
préfixe ``"Dates for "`` et les espaces de bord sont retirés.
"GDP US Chained Dollars QoQ SAA (GDP)" reste exactement
"GDP US Chained Dollars QoQ SAA (GDP)" — virgules et parenthèses comprises.

Si les en-têtes de valeurs sont vides dans le fichier principal, les noms
peuvent être lus dans un fichier de référence (``names_file=``, typiquement
``corrected_file.csv``) : cela remplace l'ancien mapping
``{"Unnamed: 3": "..."}``.

Sortie
------
``{nom_exact_de_la_serie: pandas.Series}`` — index de dates trié, sans NaT,
sans date dupliquée. Les valeurs manquantes restent en NaN pour ne pas
altérer le calendrier de la série ; le regroupement en DataFrames est fait
par :mod:`data_loader_analysis_bloom`.

Exemple
-------
    from data_loader_bloom import load_bloomberg_csv

    series = load_bloomberg_csv("export.csv", n_series=58)
    series["US Industrial Production, YOY S (Economic Dynamic)"].tail()
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd

__all__ = [
    "read_header",
    "series_names",
    "read_raw",
    "clean_pairs",
    "load_bloomberg_csv",
]

DATE_PREFIX = "Dates for "
_MISSING_HEADERS = {"", "nan", "none", "nat"}


# ---------------------------------------------------------------------------
# Lecture des en-têtes (noms de séries)
# ---------------------------------------------------------------------------
def read_header(
    path: str | Path,
    sep: str = ";",
    header_row: int = 0,
    encoding: str | None = None,
) -> list[str]:
    """
    Renvoie la ligne d'en-tête brute, cellule par cellule.

    On passe par le module ``csv`` et non par pandas : pandas renomme les
    doublons ("X" -> "X.1") et les cellules vides ("Unnamed: 3"), ce qui
    corromprait les noms de séries.
    """
    with open(Path(path), newline="", encoding=encoding or "utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter=sep)
        for index, row in enumerate(reader):
            if index == header_row:
                return [cell.strip() for cell in row]
    raise ValueError(f"Ligne d'en-tête {header_row} introuvable dans {path}.")


def _is_missing(header: str) -> bool:
    return header.strip().lower() in _MISSING_HEADERS or header.startswith("Unnamed:")


def series_names(header: Sequence[str], n_series: int | None = None) -> list[str]:
    """
    Déduit le nom de chaque série à partir des paires d'en-têtes.

    Priorité à l'en-tête de la colonne de valeurs (le nom complet y figure
    en général) ; à défaut, l'en-tête de dates privé de "Dates for ".
    Le nom est conservé **tel quel**, sans autre nettoyage.
    """
    cells = list(header)
    if n_series is not None:
        cells = cells[: 2 * n_series]

    names: list[str] = []
    for position in range(len(cells) // 2):
        date_header = cells[2 * position]
        value_header = cells[2 * position + 1]

        if not _is_missing(value_header):
            name = value_header.strip()
        elif not _is_missing(date_header):
            name = date_header.strip()
            if name.startswith(DATE_PREFIX):
                name = name[len(DATE_PREFIX):].strip()
        else:
            name = f"serie_{position + 1}"

        # Un nom en double écraserait une série dans le dictionnaire.
        if name in names:
            suffix = 2
            while f"{name} ({suffix})" in names:
                suffix += 1
            warnings.warn(
                f"Nom de série dupliqué : '{name}' renommé en '{name} ({suffix})'.",
                stacklevel=2,
            )
            name = f"{name} ({suffix})"
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Lecture des données
# ---------------------------------------------------------------------------
def _to_numeric(values: pd.Series) -> pd.Series:
    """
    Convertit une colonne de valeurs en float.

    Gère les formats européens : espaces (y compris insécables) comme
    séparateur de milliers, virgule décimale, symbole %.
    Heuristique : si un champ contient un point ET une virgule, la virgule
    est un séparateur de milliers ; sinon c'est la décimale.
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
    """
    Lit les données brutes (tout en texte), colonnes numérotées 0..n.

    L'en-tête n'est volontairement pas utilisé comme noms de colonnes : les
    vrais noms sont gérés par :func:`series_names`.
    """
    raw = pd.read_csv(
        Path(path),
        sep=sep,
        header=None,
        skiprows=header_row + 1,
        dtype=str,
        encoding=encoding or "utf-8-sig",
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
            f"Nombre de colonnes impair ({raw.shape[1]}) : la dernière est "
            "ignorée. Précisez n_series= pour forcer le découpage.",
            stacklevel=2,
        )
        raw = raw.iloc[:, :-1].copy()
    return raw


def clean_pairs(
    raw: pd.DataFrame,
    names: Sequence[str],
    dayfirst: bool = True,
) -> dict[str, pd.Series]:
    """
    Découpe le tableau brut en une Series propre par paire de colonnes.

    Les NaN de valeurs sont conservés : c'est le calendrier complet de la
    série qui sert ensuite à regrouper les séries entre elles.
    """
    series: dict[str, pd.Series] = {}
    n_pairs = raw.shape[1] // 2

    if len(names) < n_pairs:
        raise ValueError(
            f"{len(names)} noms disponibles pour {n_pairs} séries dans le fichier. "
            "Vérifiez names_file / n_series."
        )

    for position in range(n_pairs):
        name = names[position]
        dates = pd.to_datetime(
            raw.iloc[:, 2 * position], dayfirst=dayfirst, errors="coerce"
        )
        values = _to_numeric(raw.iloc[:, 2 * position + 1])

        clean = pd.Series(
            values.to_numpy(),
            index=pd.DatetimeIndex(dates, name="date"),
            name=name,
        )
        clean = clean[clean.index.notna()]                    # dates illisibles
        clean = clean[~clean.index.duplicated(keep="last")]   # dates en double
        clean = clean.sort_index()

        if clean.notna().sum() == 0:
            warnings.warn(f"Série '{name}' sans aucune valeur : ignorée.", stacklevel=2)
            continue
        series[name] = clean

    return series


def load_bloomberg_csv(
    path: str | Path,
    sep: str = ";",
    n_series: int | None = None,
    dayfirst: bool = True,
    names_file: str | Path | None = None,
    names_sep: str = ",",
    header_row: int = 0,
    encoding: str | None = None,
) -> dict[str, pd.Series]:
    """
    Chaîne complète : lecture -> découpage -> nettoyage.

    Parameters
    ----------
    path : CSV Bloomberg à charger.
    sep : séparateur du fichier (";" pour les exports européens).
    n_series : nombre de séries attendues (58 dans l'export historique) ;
        None pour détection automatique.
    dayfirst : True pour des dates JJ/MM/AAAA.
    names_file : fichier dont les en-têtes portent les noms complets des
        séries, si le fichier principal a des en-têtes de valeurs vides.
    names_sep : séparateur du fichier de noms.
    header_row : index (base 0) de la ligne d'en-tête.

    Returns
    -------
    dict[str, pandas.Series] indexé par le nom exact de chaque série.
    """
    raw = read_raw(
        path, sep=sep, n_series=n_series, header_row=header_row, encoding=encoding
    )

    header_source, header_sep = (
        (names_file, names_sep) if names_file is not None else (path, sep)
    )
    header = read_header(
        header_source, sep=header_sep, header_row=header_row, encoding=encoding
    )
    names = series_names(header, n_series=n_series)

    return clean_pairs(raw, names, dayfirst=dayfirst)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    loaded = load_bloomberg_csv(sys.argv[1])
    print(f"{len(loaded)} séries chargées :")
    for series_name, values in loaded.items():
        print(
            f"  - {series_name:<55} {values.notna().sum():>6} pts  "
            f"{values.index.min():%Y-%m-%d} -> {values.index.max():%Y-%m-%d}"
        )
