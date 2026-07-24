"""
data_loader_analysis_bloom
==========================

Construction des DataFrames et du dictionnaire de séries à partir des
données nettoyées par :mod:`data_loader_bloom`.

Trois représentations, au choix :

* ``dataset.series``  -> dict {nom: Series}, une série = une Series datée
* ``dataset.wide``    -> un seul DataFrame, une colonne par série (jointure
  externe sur les dates)
* ``dataset.group_by_calendar()`` -> dict {"df1": DataFrame, ...}, les séries
  partageant exactement le même calendrier sont regroupées (équivalent
  propre des anciens ``df1``, ``df2``, ... créés dans ``globals()``)

Exemple
-------
    from data_loader_analysis_bloom import load_dataset

    data = load_dataset("export_bloomberg.csv", n_series=58)
    weekly = data.resample("W")            # moyenne hebdo + interpolation

    weekly.wide.head()                     # tout dans un DataFrame
    weekly["CPI YOY"]                      # une série
    weekly.summary()                       # inventaire des séries
    dict_of_df = weekly.group_by_calendar() # {"df1": ..., "df2": ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd

from data_loader_bloom import load_bloomberg_csv

__all__ = [
    "BloombergDataset",
    "load_dataset",
    "to_wide",
    "group_by_calendar",
    "resample_series",
    "summarize",
]


# ---------------------------------------------------------------------------
# Fonctions de transformation (utilisables seules, sans la classe)
# ---------------------------------------------------------------------------
def to_wide(series: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Assemble toutes les séries dans un DataFrame (une colonne par série)."""
    if not series:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    wide = pd.concat(series.values(), axis=1, keys=series.keys())
    wide.index.name = "date"
    return wide.sort_index()


def group_by_calendar(
    series: Mapping[str, pd.Series],
    prefix: str = "df",
) -> dict[str, pd.DataFrame]:
    """
    Regroupe les séries qui partagent exactement le même index de dates.

    Chaque groupe devient un DataFrame sans aucun NaN, nommé
    ``{prefix}1``, ``{prefix}2``, ... (groupes triés du plus grand au plus
    petit nombre de séries).
    """
    buckets: dict[tuple[int, ...], list[pd.Series]] = {}
    for values in series.values():
        key = tuple(values.index.asi8.tolist())
        buckets.setdefault(key, []).append(values)

    ordered = sorted(buckets.values(), key=lambda group: (-len(group), -len(group[0])))
    frames: dict[str, pd.DataFrame] = {}
    for position, group in enumerate(ordered, start=1):
        frame = pd.concat(group, axis=1)
        frame.index.name = "date"
        frames[f"{prefix}{position}"] = frame
    return frames


def resample_series(
    series: Mapping[str, pd.Series],
    rule: str = "W",
    how: str = "mean",
    interpolate: str | None = "linear",
    decimals: int | None = 2,
) -> dict[str, pd.Series]:
    """
    Rééchantillonne chaque série (hebdo par défaut).

    Parameters
    ----------
    rule : fréquence cible ("W", "ME", "QE", "D", ...).
    how : agrégation appliquée ("mean", "last", "median", "sum", "max", ...).
    interpolate : méthode d'interpolation des trous créés par le
        rééchantillonnage, ou None pour laisser les NaN. L'interpolation
        reste strictement interne (pas d'extrapolation avant/après les
        observations réelles).
    decimals : arrondi final, ou None.
    """
    out: dict[str, pd.Series] = {}
    for name, values in series.items():
        resampled = getattr(values.resample(rule), how)()
        if interpolate:
            resampled = resampled.interpolate(method=interpolate, limit_area="inside")
        if decimals is not None:
            resampled = resampled.round(decimals)
        out[name] = resampled.rename(name)
    return out


def summarize(series: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Inventaire : nb de points, première/dernière date, min/max/moyenne."""
    rows = [
        {
            "serie": name,
            "n_obs": len(values),
            "debut": values.index.min(),
            "fin": values.index.max(),
            "min": values.min(),
            "moyenne": values.mean(),
            "max": values.max(),
        }
        for name, values in series.items()
    ]
    return pd.DataFrame(rows).set_index("serie") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Objet principal
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BloombergDataset:
    """Conteneur immuable regroupant les séries nettoyées d'un export."""

    series: dict[str, pd.Series]

    # -- accès ------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self.series)

    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self) -> Iterator[str]:
        return iter(self.series)

    def __getitem__(self, key: str | list[str]) -> pd.Series | pd.DataFrame:
        if isinstance(key, str):
            return self.series[key]
        return to_wide({name: self.series[name] for name in key})

    def __contains__(self, key: str) -> bool:
        return key in self.series

    def __repr__(self) -> str:  # pragma: no cover - affichage seulement
        if not self.series:
            return "BloombergDataset(vide)"
        wide = self.wide
        return (
            f"BloombergDataset({len(self.series)} séries, "
            f"{wide.index.min():%Y-%m-%d} -> {wide.index.max():%Y-%m-%d})"
        )

    # -- vues -------------------------------------------------------------
    @property
    def wide(self) -> pd.DataFrame:
        """Toutes les séries dans un seul DataFrame."""
        return to_wide(self.series)

    def group_by_calendar(self, prefix: str = "df") -> dict[str, pd.DataFrame]:
        """{"df1": DataFrame, "df2": DataFrame, ...} par calendrier commun."""
        return group_by_calendar(self.series, prefix=prefix)

    def summary(self) -> pd.DataFrame:
        return summarize(self.series)

    # -- transformations (renvoient un nouveau dataset) -------------------
    def resample(
        self,
        rule: str = "W",
        how: str = "mean",
        interpolate: str | None = "linear",
        decimals: int | None = 2,
    ) -> "BloombergDataset":
        return BloombergDataset(
            resample_series(
                self.series,
                rule=rule,
                how=how,
                interpolate=interpolate,
                decimals=decimals,
            )
        )

    def select(self, names: list[str]) -> "BloombergDataset":
        return BloombergDataset({name: self.series[name] for name in names})

    def between(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "BloombergDataset":
        return BloombergDataset(
            {name: values.loc[start:end] for name, values in self.series.items()}
        )

    # -- export -----------------------------------------------------------
    def to_csv(self, path: str | Path, **kwargs) -> Path:
        """Écrit la vue large dans un CSV."""
        destination = Path(path)
        self.wide.to_csv(destination, **kwargs)
        return destination


def load_dataset(
    path: str | Path,
    sep: str = ";",
    n_series: int | None = None,
    dayfirst: bool = True,
    names_file: str | Path | None = None,
    names_sep: str = ",",
    encoding: str | None = None,
) -> BloombergDataset:
    """Charge un export Bloomberg et renvoie un :class:`BloombergDataset`."""
    return BloombergDataset(
        load_bloomberg_csv(
            path,
            sep=sep,
            n_series=n_series,
            dayfirst=dayfirst,
            names_file=names_file,
            names_sep=names_sep,
            encoding=encoding,
        )
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    data = load_dataset(sys.argv[1])
    weekly = data.resample("W")
    print(weekly)
    print(weekly.summary())
    print(f"{len(weekly.group_by_calendar())} calendriers distincts")
