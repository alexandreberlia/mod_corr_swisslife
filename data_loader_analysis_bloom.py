"""
data_loader_analysis_bloom
==========================

Construction des DataFrames et du dictionnaire à partir des séries nettoyées
par :mod:`data_loader_bloom`.

Même sortie que les anciens scripts
-----------------------------------
* les séries partageant **exactement le même calendrier** sont regroupées
  dans un DataFrame indexé par la date, une colonne par série ;
* les DataFrames sont nommés ``df1``, ``df2``, ... **dans l'ordre
  d'apparition des calendriers dans le fichier** (identique à l'ancien code) ;
* ``dict_of_df`` = ``{"df1": DataFrame, "df2": DataFrame, ...}`` ;
* les colonnes portent le nom exact de la série, p.ex.
  ``"GDP US Chained Dollars QoQ SAA (GDP)"`` ;
* conversion hebdomadaire : ``resample("W").mean()``, interpolation
  linéaire, arrondi à 2 décimales.

...mais sans les bugs et sans ``globals()``.

Exemple
-------
    from data_loader_analysis_bloom import load_dataset

    data = load_dataset("export.csv", n_series=58,
                        names_file="corrected_file.csv")
    weekly = data.resample("W")

    dict_of_df = weekly.frames                      # {"df1": DataFrame, ...}
    dict_of_df["df1"].head()

    weekly["GDP US Chained Dollars QoQ SAA (GDP)"]  # accès direct par nom
    weekly.wide                                     # tout dans un DataFrame
    weekly.find("Industrial Production")            # recherche par mot-clé

    # pour du code existant qui attend df1, df2, ... en variables globales :
    weekly.to_globals(globals())

Raccourci en une ligne, strictement équivalent aux anciens scripts :

    from data_loader_analysis_bloom import build_dict_of_df

    dict_of_df = build_dict_of_df("export.csv", n_series=58,
                                  names_file="corrected_file.csv")
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
    series: Mapping[str, pd.Series],
    prefix: str = "df",
    dropna: str | None = "all",
) -> dict[str, pd.DataFrame]:
    """
    Regroupe les séries partageant exactement le même index de dates.

    Remplace les anciens ``count_dfs`` / ``create_df`` / ``generate_dataframes``.

    Parameters
    ----------
    series : dict {nom exact: Series}.
    prefix : préfixe des clés ("df" -> df1, df2, ...).
    dropna : "all" supprime les lignes entièrement vides (défaut) ;
        "any" supprime toute ligne dès qu'une série a un trou (comportement
        de l'ancien code, plus destructeur) ; None ne supprime rien.

    Returns
    -------
    dict {"df1": DataFrame, "df2": DataFrame, ...} dans l'ordre d'apparition
    des calendriers dans le fichier.
    """
    buckets: dict[tuple[int, ...], list[pd.Series]] = {}
    for values in series.values():
        key = tuple(values.index.asi8.tolist())
        buckets.setdefault(key, []).append(values)  # dict ordonné = ordre du fichier

    frames: dict[str, pd.DataFrame] = {}
    for position, group in enumerate(buckets.values(), start=1):
        frame = pd.concat(group, axis=1)
        frame.index.name = "date"
        if dropna:
            frame = frame.dropna(how=dropna)
        if frame.empty:
            continue
        frames[f"{prefix}{position}"] = frame
    return frames


def resample_frames(
    frames: Mapping[str, pd.DataFrame],
    rule: str = "W",
    how: str = "mean",
    interpolate: str | None = "linear",
    decimals: int | None = 2,
) -> dict[str, pd.DataFrame]:
    """
    Rééchantillonne chaque DataFrame (hebdomadaire par défaut).

    Remplace ``convert_to_weeks``, sans modification en place ni ``global``.

    Parameters
    ----------
    rule : fréquence cible ("W", "ME", "QE", "D", ...).
    how : agrégation ("mean", "last", "median", "sum", "max", "min", ...).
    interpolate : méthode de comblement des trous créés par le
        rééchantillonnage, ou None. L'interpolation reste interne aux
        observations réelles (pas d'extrapolation en début/fin de série).
    decimals : arrondi final, ou None.
    """
    out: dict[str, pd.DataFrame] = {}
    for key, frame in frames.items():
        resampled = getattr(frame.resample(rule), how)()
        if interpolate:
            resampled = resampled.interpolate(method=interpolate, limit_area="inside")
        if decimals is not None:
            resampled = resampled.round(decimals)
        resampled.index.name = frame.index.name or "date"
        out[key] = resampled
    return out


def to_wide(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Fusionne tous les DataFrames en un seul (jointure externe sur les dates)."""
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    wide = pd.concat(frames.values(), axis=1).sort_index()
    wide.index.name = "date"
    return wide


def summarize(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Inventaire des séries : DataFrame d'origine, nb de points, période, stats."""
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
    Conteneur immuable : ``frames`` = {"df1": DataFrame, "df2": ...}.

    Chaque transformation renvoie un nouvel objet, rien n'est modifié en place.
    """

    frames: dict[str, pd.DataFrame]

    # -- accès ------------------------------------------------------------
    @property
    def series(self) -> dict[str, pd.Series]:
        """{nom exact de la série: Series}, tous DataFrames confondus."""
        return {
            name: frame[name]
            for frame in self.frames.values()
            for name in frame.columns
        }

    @property
    def names(self) -> list[str]:
        """Noms exacts de toutes les séries, dans l'ordre du fichier."""
        return [name for frame in self.frames.values() for name in frame.columns]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[str]:
        return iter(self.frames)

    def __contains__(self, key: str) -> bool:
        return key in self.frames or key in self.series

    def __getitem__(self, key: str | list[str]) -> pd.Series | pd.DataFrame:
        """
        ``data["df1"]``  -> le DataFrame du groupe
        ``data["GDP US Chained Dollars QoQ SAA (GDP)"]`` -> la Series
        ``data[["serie A", "serie B"]]`` -> un DataFrame des deux séries
        """
        if isinstance(key, list):
            return pd.concat([self[name] for name in key], axis=1).sort_index()
        if key in self.frames:
            return self.frames[key]
        series = self.series
        if key in series:
            return series[key]
        raise KeyError(
            f"'{key}' n'est ni un DataFrame ({', '.join(self.frames)}) "
            f"ni une série connue. Essayez .find('{key[:25]}')."
        )

    def find(self, pattern: str, case: bool = False) -> list[str]:
        """Noms de séries contenant ``pattern`` (pratique avec des noms longs)."""
        needle = pattern if case else pattern.lower()
        return [
            name
            for name in self.names
            if needle in (name if case else name.lower())
        ]

    def frame_of(self, series_name: str) -> str:
        """Clé du DataFrame ("df3") contenant la série demandée."""
        for key, frame in self.frames.items():
            if series_name in frame.columns:
                return key
        raise KeyError(f"Série inconnue : '{series_name}'.")

    def __repr__(self) -> str:  # pragma: no cover - affichage seulement
        if not self.frames:
            return "BloombergData(vide)"
        wide = self.wide
        return (
            f"BloombergData({len(self.names)} séries dans {len(self.frames)} "
            f"DataFrames, {wide.index.min():%Y-%m-%d} -> {wide.index.max():%Y-%m-%d})"
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
        rule: str = "W",
        how: str = "mean",
        interpolate: str | None = "linear",
        decimals: int | None = 2,
    ) -> "BloombergData":
        """Conversion hebdo (défaut) — équivalent propre de ``convert_to_weeks``."""
        return BloombergData(
            resample_frames(
                self.frames,
                rule=rule,
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
    def to_globals(self, namespace: MutableMapping[str, object]) -> dict[str, pd.DataFrame]:
        """
        Injecte df1, df2, ... dans un espace de noms, pour du code existant.

        Usage : ``data.to_globals(globals())``. À n'utiliser que pour la
        compatibilité : préférez ``data.frames["df1"]``.
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
    header_row: int = 0,
    encoding: str | None = None,
    prefix: str = "df",
    dropna: str | None = "all",
) -> BloombergData:
    """Charge un export Bloomberg et renvoie un :class:`BloombergData`."""
    series = load_bloomberg_csv(
        path,
        sep=sep,
        n_series=n_series,
        dayfirst=dayfirst,
        names_file=names_file,
        names_sep=names_sep,
        header_row=header_row,
        encoding=encoding,
    )
    return BloombergData(group_by_calendar(series, prefix=prefix, dropna=dropna))


def build_dict_of_df(
    path: str | Path,
    rule: str | None = "W",
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Pipeline complet en une ligne : renvoie ``{"df1": DataFrame, ...}``
    rééchantillonné (hebdo par défaut, ``rule=None`` pour garder la
    fréquence d'origine).

    Équivalent direct des anciens scripts, sans variables globales.
    """
    data = load_dataset(path, **kwargs)
    if rule:
        data = data.resample(rule)
    return data.frames


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    dataset = load_dataset(sys.argv[1]).resample("W")
    print(dataset)
    print(dataset.summary())
    print(f"{len(dataset.frames)} DataFrames : {', '.join(dataset.frames)}")
