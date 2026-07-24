"""
data_loader_analysis_bloom
==========================

Construction des DataFrames et du dictionnaire, à partir des séries
nettoyées par :mod:`data_loader_bloom`.

Sortie identique aux anciens scripts
------------------------------------
* les séries partageant **exactement le même calendrier** sont regroupées
  dans un DataFrame indexé par la date, une colonne par série ;
* les DataFrames sont nommés ``df1``, ``df2``, ... dans l'**ordre
  d'apparition des calendriers dans le fichier** ;
* ``dict_of_df`` = ``{"df1": DataFrame, "df2": DataFrame, ...}`` ;
* les colonnes portent le nom exact de la série, p.ex.
  ``"GDP US Chained Dollars QoQ SAA (GDP)"`` ;
* l'index garde le nom de la colonne de dates d'origine
  (``"Dates for ..."``), comme le faisait ``set_index`` ;
* **tout est DataFrame**, jamais de Series : pas de ``Name:`` à l'affichage.

Temporalité
-----------
**Aucun rééchantillonnage par défaut** : les données restent à leur
fréquence d'origine. La conversion est explicite et paramétrable :

    weekly    = data.resample("W")    # hebdomadaire
    monthly   = data.resample("ME")   # fin de mois
    quarterly = data.resample("QE")   # fin de trimestre
    daily     = data.resample("D")

Exemple
-------
    from data_loader_analysis_bloom import load_dataset

    data = load_dataset("export.csv", n_series=58,
                        names_file="corrected_file.csv")

    dict_of_df = data.frames                        # {"df1": DataFrame, ...}
    dict_of_df["df1"].head()

    data["GDP US Chained Dollars QoQ SAA (GDP)"]    # DataFrame d'une colonne
    data.find("Industrial Production")              # recherche par mot-clé
    data.resample("ME").frames                      # même dict, en mensuel

    data.to_globals(globals())      # recrée df1, df2, ... pour le code existant

Raccourci en une ligne :

    from data_loader_analysis_bloom import build_dict_of_df

    dict_of_df = build_dict_of_df("export.csv", n_series=58,
                                  names_file="corrected_file.csv")   # brut
    dict_of_df = build_dict_of_df("export.csv", freq="W")            # hebdo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping

import pandas as pd

from data_loader_bloom import load_bloomberg_csv

__all__ = [
    "BloombergData",
    "load_dataset",
    "build_dict_of_df",
    "group_by_calendar",
    "resample_frames",
    "to_wide",
    "summarize",
]


# ---------------------------------------------------------------------------
# Transformations (utilisables seules, sans passer par la classe)
# ---------------------------------------------------------------------------
def group_by_calendar(
    series: Mapping[str, pd.DataFrame],
    prefix: str = "df",
    dropna: str | None = "any",
    index_name: str | None = "original",
) -> dict[str, pd.DataFrame]:
    """
    Regroupe les séries partageant exactement le même index de dates.

    Remplace ``count_dfs`` / ``create_df`` / ``generate_dataframes``.

    Parameters
    ----------
    series : dict {nom de série: DataFrame d'une colonne}.
    prefix : préfixe des clés ("df" -> df1, df2, ...).
    dropna : "any" supprime toute ligne dès qu'une série du groupe a un trou
        (comportement des anciens scripts, valeur par défaut) ;
        "all" ne supprime que les lignes entièrement vides, ce qui préserve
        les autres séries ; None ne supprime rien.
    index_name : "original" garde le nom de la colonne de dates du fichier
        ("Dates for ..."), comme l'ancien ``set_index`` ; sinon une chaîne
        explicite ("date") ou None.

    Returns
    -------
    dict {"df1": DataFrame, "df2": DataFrame, ...}, dans l'ordre
    d'apparition des calendriers dans le fichier.
    """
    buckets: dict[tuple[int, ...], list[pd.DataFrame]] = {}
    for frame in series.values():
        key = tuple(frame.index.asi8.tolist())
        buckets.setdefault(key, []).append(frame)  # dict ordonné = ordre du fichier

    frames: dict[str, pd.DataFrame] = {}
    for position, group in enumerate(buckets.values(), start=1):
        merged = pd.concat(group, axis=1)
        merged.index.name = (
            group[0].index.name if index_name == "original" else index_name
        )
        if dropna:
            merged = merged.dropna(how=dropna)
        if merged.empty:
            continue
        frames[f"{prefix}{position}"] = merged
    return frames


def resample_frames(
    frames: Mapping[str, pd.DataFrame],
    freq: str = "W",
    how: str = "mean",
    interpolate: str | None = "linear",
    decimals: int | None = 2,
) -> dict[str, pd.DataFrame]:
    """
    Change la temporalité de chaque DataFrame.

    Remplace ``convert_to_weeks``, sans ``global`` ni modification en place.

    Parameters
    ----------
    freq : fréquence cible — "D", "W", "ME" (fin de mois), "QE", "YE", ...
    how : agrégation ("mean", "last", "median", "sum", "max", "min", ...).
    interpolate : comblement des trous créés par le rééchantillonnage, ou
        None pour laisser les NaN. L'interpolation reste interne aux
        observations réelles (pas d'extrapolation en début/fin de série).
    decimals : arrondi final, ou None pour ne pas arrondir.
    """
    out: dict[str, pd.DataFrame] = {}
    for key, frame in frames.items():
        index_name = frame.index.name
        resampled = getattr(frame.resample(freq), how)()
        if interpolate:
            resampled = resampled.interpolate(method=interpolate, limit_area="inside")
        if decimals is not None:
            resampled = resampled.round(decimals)
        resampled.index.name = index_name
        out[key] = resampled
    return out


def to_wide(frames: Mapping[str, pd.DataFrame], index_name: str = "date") -> pd.DataFrame:
    """Fusionne tous les DataFrames en un seul (jointure externe sur les dates)."""
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=index_name))
    wide = pd.concat(
        [frame.rename_axis(index_name) for frame in frames.values()], axis=1
    ).sort_index()
    return wide


def summarize(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Inventaire : DataFrame d'appartenance, nb de points, période, stats."""
    rows = []
    for key, frame in frames.items():
        for name in frame.columns:
            column = frame[name].dropna()
            rows.append(
                {
                    "serie": name,
                    "dataframe": key,
                    "n_obs": len(column),
                    "debut": column.index.min() if len(column) else pd.NaT,
                    "fin": column.index.max() if len(column) else pd.NaT,
                    "min": column.min(),
                    "moyenne": column.mean(),
                    "max": column.max(),
                }
            )
    return pd.DataFrame(rows).set_index("serie") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Objet principal
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BloombergData:
    """
    Conteneur immuable autour de ``frames`` = {"df1": DataFrame, ...}.

    Chaque transformation renvoie un nouvel objet ; rien n'est modifié en
    place, et toutes les valeurs renvoyées sont des DataFrames.
    """

    frames: dict[str, pd.DataFrame]

    # -- accès ------------------------------------------------------------
    @property
    def by_name(self) -> dict[str, pd.DataFrame]:
        """{nom de série: DataFrame d'une colonne}, tous groupes confondus."""
        return {
            name: frame[[name]]
            for frame in self.frames.values()
            for name in frame.columns
        }

    @property
    def names(self) -> list[str]:
        """Noms des séries, dans l'ordre du fichier."""
        return [name for frame in self.frames.values() for name in frame.columns]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[str]:
        return iter(self.frames)

    def __contains__(self, key: str) -> bool:
        return key in self.frames or key in self.names

    def __getitem__(self, key: str | list[str]) -> pd.DataFrame:
        """
        ``data["df1"]`` -> le DataFrame du groupe
        ``data["GDP US Chained Dollars QoQ SAA (GDP)"]`` -> DataFrame 1 colonne
        ``data[["serie A", "serie B"]]`` -> DataFrame des deux séries
        """
        if isinstance(key, list):
            return pd.concat([self[name] for name in key], axis=1).sort_index()
        if key in self.frames:
            return self.frames[key]
        for frame in self.frames.values():
            if key in frame.columns:
                return frame[[key]]
        raise KeyError(
            f"'{key}' n'est ni un DataFrame ({', '.join(self.frames)}) ni une "
            f"série connue. Essayez .find('{key[:25]}')."
        )

    def find(self, pattern: str, case: bool = False) -> list[str]:
        """Noms de séries contenant ``pattern`` (pratique avec des noms longs)."""
        needle = pattern if case else pattern.lower()
        return [name for name in self.names if needle in (name if case else name.lower())]

    def frame_of(self, series_name: str) -> str:
        """Clé du DataFrame ("df3") qui contient la série demandée."""
        for key, frame in self.frames.items():
            if series_name in frame.columns:
                return key
        raise KeyError(f"Série inconnue : '{series_name}'.")

    def __repr__(self) -> str:  # pragma: no cover - affichage seulement
        if not self.frames:
            return "BloombergData(vide)"
        wide = self.wide
        return (
            f"BloombergData({len(self.names)} séries, {len(self.frames)} DataFrames, "
            f"{wide.index.min():%Y-%m-%d} -> {wide.index.max():%Y-%m-%d})"
        )

    # -- vues -------------------------------------------------------------
    @property
    def wide(self) -> pd.DataFrame:
        """Toutes les séries dans un seul DataFrame."""
        return to_wide(self.frames)

    def summary(self) -> pd.DataFrame:
        return summarize(self.frames)

    # -- transformations (renvoient un nouvel objet) ----------------------
    def resample(
        self,
        freq: str = "W",
        how: str = "mean",
        interpolate: str | None = "linear",
        decimals: int | None = 2,
    ) -> "BloombergData":
        """Change la temporalité — "D", "W", "ME", "QE", "YE", ..."""
        return BloombergData(
            resample_frames(
                self.frames,
                freq=freq,
                how=how,
                interpolate=interpolate,
                decimals=decimals,
            )
        )

    def between(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "BloombergData":
        return BloombergData(
            {key: frame.loc[start:end] for key, frame in self.frames.items()}
        )

    # -- compatibilité / export -------------------------------------------
    def to_globals(
        self, namespace: MutableMapping[str, object]
    ) -> dict[str, pd.DataFrame]:
        """
        Injecte df1, df2, ... dans un espace de noms, pour le code existant.

        Usage : ``data.to_globals(globals())``.
        """
        namespace.update(self.frames)
        return dict(self.frames)

    def to_csv(self, path: str | Path, **kwargs) -> Path:
        """Écrit la vue large (toutes séries) dans un CSV."""
        destination = Path(path)
        self.wide.to_csv(destination, **kwargs)
        return destination


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------
def load_dataset(
    path: str | Path,
    sep: str = ";",
    n_series: int | None = None,
    dayfirst: bool = True,
    names_file: str | Path | None = None,
    names_sep: str = ",",
    names: list[str] | None = None,
    header_row: int = 0,
    encoding: str | None = None,
    prefix: str = "df",
    dropna: str | None = "any",
    index_name: str | None = "original",
) -> BloombergData:
    """
    Charge un export Bloomberg et renvoie un :class:`BloombergData`.

    Les données restent à leur fréquence d'origine ; utilisez ``.resample()``
    pour changer de temporalité.
    """
    series = load_bloomberg_csv(
        path,
        sep=sep,
        n_series=n_series,
        dayfirst=dayfirst,
        names_file=names_file,
        names_sep=names_sep,
        names=names,
        header_row=header_row,
        encoding=encoding,
    )
    return BloombergData(
        group_by_calendar(series, prefix=prefix, dropna=dropna, index_name=index_name)
    )


def build_dict_of_df(
    path: str | Path,
    freq: str | None = None,
    how: str = "mean",
    interpolate: str | None = "linear",
    decimals: int | None = 2,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Pipeline complet en une ligne : renvoie ``{"df1": DataFrame, ...}``.

    ``freq=None`` (défaut) conserve la fréquence d'origine ;
    ``freq="W"`` / ``"ME"`` / ``"QE"`` applique la conversion correspondante.
    """
    data = load_dataset(path, **kwargs)
    if freq:
        data = data.resample(freq, how=how, interpolate=interpolate, decimals=decimals)
    return data.frames


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    dataset = load_dataset(sys.argv[1])
    print(dataset)
    print(dataset.summary())
    print(f"{len(dataset.frames)} DataFrames : {', '.join(dataset.frames)}")
