"""
data_loader_bloom
=================

Chargement et nettoyage des exports CSV Bloomberg.

Format attendu
--------------
Le fichier contient N séries posées côte à côte, chaque série occupant
**deux colonnes consécutives** : une colonne de dates, puis une colonne de
valeurs.

    | Dates for CPI | (vide)  | Dates for GDP | (vide)  | ...
    | 01/02/2020    | 1,23    | 03/02/2020    | 4,56    | ...

Les en-têtes de valeurs sont souvent vides (pandas les nomme
``Unnamed: 1``, ``Unnamed: 3``, ...) : le nom de la série est alors déduit
de l'en-tête de la colonne de dates ("Dates for CPI" -> "CPI"), ou repris
d'un fichier de référence via ``names_file``.

Sortie
------
Un dictionnaire ``{nom_de_serie: pandas.Series}`` où chaque Series est :
indexée par un DatetimeIndex, triée, sans NaT, sans NaN, sans date dupliquée.

Exemple
-------
    from data_loader_bloom import load_bloomberg_csv

    series = load_bloomberg_csv("export_bloomberg.csv", n_series=58)
    series["CPI YOY"].head()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd

__all__ = [
    "read_raw",
    "extract_series",
    "read_series_names",
    "load_bloomberg_csv",
]

DATE_PREFIX = "Dates for "
_MISSING_HEADERS = {"", "nan", "none", "nat"}


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------
def _clean_header(header: object) -> str:
    """Renvoie un en-tête normalisé, ou "" s'il est vide / auto-généré."""
    text = str(header).strip()
    if text.lower() in _MISSING_HEADERS or text.startswith("Unnamed:"):
        return ""
    return text


def _series_name(date_header: object, value_header: object, position: int) -> str:
    """Déduit le nom d'une série à partir de ses deux en-têtes."""
    value_name = _clean_header(value_header)
    if value_name:
        return value_name

    date_name = _clean_header(date_header)
    if date_name.startswith(DATE_PREFIX):
        date_name = date_name[len(DATE_PREFIX):].strip()
    if date_name:
        return date_name

    return f"serie_{position + 1}"


def _unique_name(name: str, taken: Sequence[str]) -> str:
    """Évite les collisions de noms de séries (suffixe _2, _3, ...)."""
    if name not in taken:
        return name
    suffix = 2
    while f"{name}_{suffix}" in taken:
        suffix += 1
    return f"{name}_{suffix}"


def _to_numeric(values: pd.Series) -> pd.Series:
    """
    Convertit une colonne de valeurs en float.

    Gère les formats européens : espaces / espaces insécables comme
    séparateur de milliers, virgule décimale, symbole %.
    Heuristique : si un même champ contient un point ET une virgule,
    la virgule est un séparateur de milliers ; sinon c'est la décimale.
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


def _trim_columns(raw: pd.DataFrame, n_series: int | None) -> pd.DataFrame:
    """Ne garde que les paires (date, valeur) exploitables."""
    if n_series is not None:
        return raw.iloc[:, : 2 * n_series].copy()

    # Bloomberg ajoute souvent des colonnes vides en fin de fichier.
    non_empty = raw.notna().any(axis=0).to_numpy()
    last = len(non_empty) - 1
    while last >= 0 and not non_empty[last]:
        last -= 1
    raw = raw.iloc[:, : last + 1].copy()

    if raw.shape[1] % 2:
        warnings.warn(
            f"Nombre de colonnes impair ({raw.shape[1]}) : la dernière colonne "
            "est ignorée. Utilisez n_series= pour forcer le découpage.",
            stacklevel=3,
        )
        raw = raw.iloc[:, :-1].copy()
    return raw


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------
def read_raw(
    path: str | Path,
    sep: str = ";",
    n_series: int | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    """
    Lit le CSV brut, sans aucune conversion (tout est lu en texte).

    Parameters
    ----------
    path : chemin du fichier (relatif au dossier courant ou absolu).
    sep : séparateur de colonnes du CSV.
    n_series : nombre de séries à conserver (= 2*n_series colonnes).
        Laisser à None pour détecter automatiquement.
    """
    raw = pd.read_csv(Path(path), sep=sep, dtype=str, encoding=encoding)
    return _trim_columns(raw, n_series)


def extract_series(
    raw: pd.DataFrame,
    dayfirst: bool = True,
    names: Sequence[str] | None = None,
) -> dict[str, pd.Series]:
    """
    Découpe le tableau brut en une série propre par paire de colonnes.

    Parameters
    ----------
    raw : DataFrame issu de :func:`read_raw`.
    dayfirst : True pour les dates au format JJ/MM/AAAA.
    names : noms de séries imposés (un par paire de colonnes). Si None,
        les noms sont déduits des en-têtes.

    Returns
    -------
    dict[str, pandas.Series]
    """
    series: dict[str, pd.Series] = {}
    n_pairs = raw.shape[1] // 2

    if names is not None and len(names) < n_pairs:
        raise ValueError(
            f"{len(names)} noms fournis pour {n_pairs} séries dans le fichier."
        )

    for position in range(n_pairs):
        date_col = raw.columns[2 * position]
        value_col = raw.columns[2 * position + 1]

        name = (
            str(names[position]).strip()
            if names is not None
            else _series_name(date_col, value_col, position)
        )
        name = _unique_name(name, list(series))

        dates = pd.to_datetime(raw[date_col], dayfirst=dayfirst, errors="coerce")
        values = _to_numeric(raw[value_col])

        clean = pd.Series(
            values.to_numpy(),
            index=pd.DatetimeIndex(dates, name="date"),
            name=name,
        )
        clean = clean[clean.index.notna()].dropna()
        clean = clean[~clean.index.duplicated(keep="last")].sort_index()

        if clean.empty:
            warnings.warn(f"Série '{name}' vide après nettoyage : ignorée.", stacklevel=2)
            continue
        series[name] = clean

    return series


def read_series_names(
    path: str | Path,
    sep: str = ",",
    n_series: int | None = None,
    encoding: str | None = None,
) -> list[str]:
    """
    Lit uniquement les en-têtes d'un fichier de référence (type
    ``corrected_file.csv``) et renvoie la liste des noms de séries.

    Remplace l'ancien bricolage ``mapping = {"Unnamed: 3": "..."}``.
    """
    header = pd.read_csv(Path(path), sep=sep, nrows=0, encoding=encoding)
    columns = list(header.columns)
    if n_series is not None:
        columns = columns[: 2 * n_series]
    return [
        _series_name(columns[i], columns[i + 1], i // 2)
        for i in range(0, len(columns) - 1, 2)
    ]


def load_bloomberg_csv(
    path: str | Path,
    sep: str = ";",
    n_series: int | None = None,
    dayfirst: bool = True,
    names_file: str | Path | None = None,
    names_sep: str = ",",
    encoding: str | None = None,
) -> dict[str, pd.Series]:
    """
    Chaîne complète : lecture -> découpage -> nettoyage.

    Parameters
    ----------
    path : le CSV Bloomberg à charger.
    sep : séparateur du CSV (";" pour les exports européens).
    n_series : nombre de séries attendues (58 dans l'export historique).
    dayfirst : True pour des dates JJ/MM/AAAA.
    names_file : fichier de référence dont les en-têtes donnent les noms
        des séries, si le fichier principal a des en-têtes vides.
    names_sep : séparateur du fichier de référence.

    Returns
    -------
    dict[str, pandas.Series] : une série propre par nom.
    """
    raw = read_raw(path, sep=sep, n_series=n_series, encoding=encoding)
    names = (
        read_series_names(names_file, sep=names_sep, n_series=n_series, encoding=encoding)
        if names_file is not None
        else None
    )
    return extract_series(raw, dayfirst=dayfirst, names=names)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    loaded = load_bloomberg_csv(sys.argv[1])
    print(f"{len(loaded)} séries chargées :")
    for series_name, values in loaded.items():
        print(
            f"  - {series_name:<40} {len(values):>6} pts  "
            f"{values.index.min():%Y-%m-%d} -> {values.index.max():%Y-%m-%d}"
        )
