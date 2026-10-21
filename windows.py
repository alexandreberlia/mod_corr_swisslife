"""
Étape 4 : fenêtres glissantes et DataLoader PyTorch.

Un échantillon = les `window` derniers jours d'UN ticker, jusqu'à une date t
(le « point d'ancrage »), et la cible de cette date t.

    x      : (window, n_features)   variables des jours t-window+1 ... t
    regime : (window, n_regime)     contexte (score macro), si regime_col est utilisé
    y      : (1,)                   cible de la ligne t

Règles :
  - une fenêtre ne mélange jamais deux tickers ;
  - le split (train / val / test) est celui de la date d'ancrage t ; une fenêtre
    de validation peut donc commencer dans la période d'entraînement, ce qui est
    correct : ces variables étaient connues à la date t ;
  - une fenêtre qui enjambe un trou dans les données (suspension de cotation,
    données manquantes) est écartée.

Les fenêtres ne sont pas copiées en mémoire : on stocke un seul tableau par
ticker et on découpe à la volée dans __getitem__.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data_prep import PreparedData


class WindowDataset(Dataset):
    """Fenêtres glissantes sur un DataFrame long indexé par (date, ticker).

    Args:
        prepared     : sortie de prepare_dataset (étape 3)
        split        : "train", "val" ou "test" (split de la date d'ancrage)
        window       : nombre de jours dans chaque fenêtre
        static       : DataFrame indexé par ticker (une ligne par ticker),
                       par ex. ticker_onehot(prepared.tickers) ; None sinon
        max_gap_days : écart calendaire maximal toléré entre deux lignes
                       consécutives d'une fenêtre (7 = week-ends et jours fériés)
    """

    def __init__(self, prepared: PreparedData, split: str, window: int = 60,
                 static: pd.DataFrame | None = None, max_gap_days: int = 7):
        self.window = window
        self.feature_cols = prepared.feature_cols
        self.regime_cols = list(getattr(prepared, "regime_cols", []) or [])
        df = prepared.df.sort_index()

        self.static = None
        if static is not None:
            missing = set(prepared.tickers) - set(static.index)
            if missing:
                raise ValueError(f"Variables statiques absentes pour : {sorted(missing)}")
            self.static = torch.tensor(static.loc[prepared.tickers].to_numpy(np.float32))

        xs, rs, ys, starts, ticker_ids, anchors = [], [], [], [], [], []
        offset = 0
        for k, ticker in enumerate(prepared.tickers):
            sub = df.xs(ticker, level="ticker")
            n = len(sub)
            if n < window:
                continue
            x = sub[self.feature_cols].to_numpy(np.float32)
            if self.regime_cols:
                rs.append(sub[self.regime_cols].to_numpy(np.float32))
            y = sub["target"].to_numpy(np.float32)

            # Ruptures : écart entre deux dates consécutives > max_gap_days
            gaps = np.diff(sub.index.values).astype("timedelta64[D]").astype(int)
            breaks = np.concatenate([[0], np.cumsum(gaps > max_gap_days)])
            # breaks[i] - breaks[i-window+1] = nb de ruptures dans la fenêtre finissant en i
            end = np.arange(window - 1, n)
            no_break = breaks[end] == breaks[end - window + 1]
            in_split = (sub["split"].to_numpy() == split)[end]
            valid_end = end[no_break & in_split]

            xs.append(x)
            ys.append(y)
            starts.append(offset + valid_end - window + 1)
            ticker_ids.append(np.full(len(valid_end), k))
            anchors.append(pd.MultiIndex.from_arrays(
                [sub.index[valid_end], [ticker] * len(valid_end)], names=["date", "ticker"]))
            offset += n

        if not xs:
            raise ValueError("Aucun ticker n'a assez d'historique pour cette fenêtre.")
        self.x = torch.from_numpy(np.concatenate(xs))
        self.r = torch.from_numpy(np.concatenate(rs)) if rs else None
        self.y = torch.from_numpy(np.concatenate(ys))
        self.starts = np.concatenate(starts)
        self.ticker_ids = np.concatenate(ticker_ids)
        # (date, ticker) de chaque échantillon, pour relier les prédictions aux dates
        self.index = anchors[0].append(anchors[1:]) if len(anchors) > 1 else anchors[0]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict:
        s = self.starts[i]
        e = s + self.window
        item = {"x": self.x[s:e], "y": self.y[e - 1:e]}
        if self.r is not None:
            item["regime"] = self.r[s:e]              # (window, n_regime) : score sur la fenêtre
        if self.static is not None:
            item["static"] = self.static[self.ticker_ids[i]]
        return item


def make_loaders(prepared: PreparedData, window: int = 60, batch_size: int = 128,
                 static: pd.DataFrame | None = None, num_workers: int = 0,
                 seed: int = 0) -> dict[str, DataLoader]:
    """Construit les trois DataLoader.

    Le train est mélangé : chaque fenêtre contient son propre historique, l'ordre
    des échantillons n'a donc plus d'importance pour l'apprentissage, et le
    mélange rend les gradients moins corrélés. Val et test restent dans l'ordre.
    """
    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = WindowDataset(prepared, split, window=window, static=static)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, generator=g)
    return loaders


# ---------------------------------------------------------------------------
# Tests : python windows.py   (nécessite data_prep.py et models.py à côté)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_prep import load_macro, prepare_dataset, ticker_onehot
    from models import QuantileLoss, VSNLSTM

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    tickers = ["AAA", "BBB", "CCC"]
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for t in tickers:
        raw[("Close", t)] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        raw[("Volume", t)] = rng.integers(1e5, 1e6, len(dates))
    # BBB : suspension de cotation de 3 semaines en 2018
    raw.loc["2018-03-01":"2018-03-21", ("Close", "BBB")] = np.nan
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    ind = pd.DataFrame({"RSI": rng.uniform(0, 100, len(idx))}, index=idx)
    months = pd.date_range("2014-06-01", "2024-12-01", freq="MS")
    macro = load_macro(pd.DataFrame({"date": months, "score": rng.normal(0, 1, len(months))}))

    W = 60
    pdata = prepare_dataset(raw, ind, macro, train_end="2020-12-31", val_end="2022-12-31")
    loaders = make_loaders(pdata, window=W, batch_size=64)
    tr, va = loaders["train"].dataset, loaders["val"].dataset

    # 1. Formes
    item = tr[0]
    assert item["x"].shape == (W, len(pdata.feature_cols)) and item["y"].shape == (1,)

    # 2. Le contenu d'une fenêtre correspond exactement aux lignes du DataFrame
    i = 1234
    date, ticker = tr.index[i]
    sub = pdata.df.xs(ticker, level="ticker").sort_index()
    pos = sub.index.get_loc(date)
    expected = sub[pdata.feature_cols].iloc[pos - W + 1:pos + 1].to_numpy(np.float32)
    assert np.allclose(tr[i]["x"].numpy(), expected)
    assert np.isclose(tr[i]["y"].item(), sub["target"].iloc[pos])

    # 3. Les dates d'ancrage sont bien dans leur période
    assert (pdata.df.loc[tr.index, "split"] == "train").all()
    assert (pdata.df.loc[va.index, "split"] == "val").all()

    # 4. La première fenêtre de validation puise son historique dans le train
    first_val = va.index.get_level_values("date").min()
    assert first_val <= pd.Timestamp("2021-01-05")

    # 5. Aucune fenêtre de BBB n'enjambe la suspension de mars 2018
    bbb = tr.index[tr.index.get_level_values("ticker") == "BBB"].get_level_values("date")
    hole_end = pd.Timestamp("2018-03-21")
    assert not ((bbb > hole_end) & (bbb < hole_end + pd.Timedelta(days=90))).any()

    # 6. De bout en bout : un batch passe dans le modèle (avec variables statiques)
    static = ticker_onehot(pdata.tickers)
    loaders_s = make_loaders(pdata, window=W, batch_size=64, static=static)
    batch = next(iter(loaders_s["train"]))
    model = VSNLSTM(n_vars=len(pdata.feature_cols), d_model=16, n_static=len(pdata.tickers))
    out = model(batch["x"], batch["static"])
    loss = QuantileLoss([0.1, 0.5, 0.9])(out["prediction"], batch["y"])
    loss.backward()
    assert out["prediction"].shape == (64, 1, 3) and torch.isfinite(loss)

    print("Étape 4 OK.")
    for s, dl in loaders.items():
        d = dl.dataset.index.get_level_values("date")
        print(f"  {s:<5} : {len(dl.dataset):>5} fenêtres, {len(dl):>3} batchs, "
              f"ancrages du {d.min().date()} au {d.max().date()}")
