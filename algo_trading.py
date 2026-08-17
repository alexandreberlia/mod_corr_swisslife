import numpy as np
import pandas as pd


class Indicateurs:
    """Indicateurs techniques. Toute méthode renvoie une pd.Series alignée sur l'index."""

    def __init__(self, df: pd.DataFrame, burnin: bool = True, source: str = "Close"):
        # yfinance récent renvoie parfois des colonnes MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

        self.df = df
        self.burnin = burnin
        self.price = df[source].astype(float)
        self.high = df["High"].astype(float)
        self.low = df["Low"].astype(float)
        self.close = df["Close"].astype(float)
        self.volume = df["Volume"].astype(float)
        self._cache = {}

    # ---------- infrastructure ----------

    def _get(self, key, compute, warmup: int):
        """Cache + burn-in + copie défensive."""
        if key not in self._cache:
            s = compute()
            if self.burnin and warmup:
                s.iloc[:warmup] = np.nan
            self._cache[key] = s
        return self._cache[key].copy()

    # ---------- moyennes ----------

    def ema(self, n: int) -> pd.Series:
        return self._get(("ema", n),
                         lambda: self._ema(self.price, n).rename(f"ema{n}"),
                         5 * n)

    def sma(self, n: int) -> pd.Series:
        return self._get(("sma", n),
                         lambda: self.price.rolling(n).mean().rename(f"sma{n}"),
                         n)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        """Récurrence explicite, seed = première valeur, NaN-safe."""
        alpha = 2.0 / (n + 1)
        vals = s.to_numpy()
        out = np.full(len(vals), np.nan)
        ema = None
        for i, p in enumerate(vals):
            if np.isnan(p):
                out[i] = ema if ema is not None else np.nan
                continue
            ema = p if ema is None else ema + alpha * (p - ema)
            out[i] = ema
        return pd.Series(out, index=s.index)

    @staticmethod
    def _wilder(s: pd.Series, n: int) -> pd.Series:
        """Lissage de Wilder : alpha = 1/n (≈ 2x plus lent qu'une EMA(n))."""
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    # ---------- volatilité ----------

    def tr(self) -> pd.Series:
        """True Range. Aucun paramètre : une valeur par barre."""
        def _c():
            prev_close = self.close.shift(1)
            return pd.concat([
                self.high - self.low,
                (self.high - prev_close).abs(),
                (self.low - prev_close).abs(),
            ], axis=1).max(axis=1).rename("tr")
        return self._get(("tr",), _c, 1)

    def atr(self, n: int = 14) -> pd.Series:
        return self._get(("atr", n),
                         lambda: self._wilder(self.tr(), n).rename(f"atr{n}"),
                         5 * n)

    # ---------- momentum ----------

    def roc(self, n: int = 14) -> pd.Series:
        """Rate of Change en %, sur n SÉANCES (pas jours calendaires)."""
        return self._get(("roc", n),
                         lambda: (self.price.pct_change(n) * 100).rename(f"roc{n}"),
                         n)

    # ---------- volume ----------

    def rvol(self, n: int = 50) -> pd.Series:
        """Volume du jour / médiane des n jours PRÉCÉDENTS (exclut le jour même)."""
        def _c():
            ref = self.volume.shift(1).rolling(n).median()
            return (self.volume / ref).rename(f"rvol{n}")
        return self._get(("rvol", n), _c, n + 1)

    def avwap(self, ancrage) -> pd.Series:
        """VWAP ancré. `ancrage` : date str/Timestamp, ou Series booléenne (multi-ancres)."""
        hlc3 = (self.high + self.low + self.close) / 3
        pv = hlc3 * self.volume

        if isinstance(ancrage, pd.Series):
            grp = ancrage.astype(bool).cumsum()
            num = pv.groupby(grp).cumsum()
            den = self.volume.groupby(grp).cumsum()
        else:
            mask = self.df.index >= pd.Timestamp(ancrage)
            num = pv.where(mask).cumsum()
            den = self.volume.where(mask).cumsum()

        return (num / den).rename("avwap")

    # ---------- régime ----------

    def adx(self, n: int = 14) -> pd.Series:
        def _c():
            up = self.high.diff()
            down = -self.low.diff()          # attention au signe

            plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0),
                                index=self.df.index)
            minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0),
                                 index=self.df.index)

            atr = self._wilder(self.tr(), n)
            plus_di = 100 * self._wilder(plus_dm, n) / atr
            minus_di = 100 * self._wilder(minus_dm, n) / atr

            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            return self._wilder(dx, n).rename(f"adx{n}")
        return self._get(("adx", n), _c, 5 * n)
