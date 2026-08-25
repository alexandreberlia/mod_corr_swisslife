import numpy as np
import pandas as pd


class Indicateurs:
    def __init__(self, df, burnin=True, burnin_mult=3, source="Close"):
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        self.df = df
        self.burnin = burnin
        self.burnin_mult = burnin_mult
        self.price = df[source].astype(float)
        self.high = df["High"].astype(float)
        self.low = df["Low"].astype(float)
        self.close = df["Close"].astype(float)
        self.volume = df["Volume"].astype(float)
        self._cache = {}

    def _get(self, key, compute, warmup):
        if key not in self._cache:
            s = compute()
            if self.burnin and warmup:
                s.iloc[:warmup] = np.nan
            self._cache[key] = s
        return self._cache[key].copy()

    def ema(self, n):
        return self._get(("ema", n), lambda: self._ema(self.price, n).rename(f"ema{n}"),
                         self.burnin_mult * n)

    @staticmethod
    def _ema(s, n):
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
    def _wilder(s, n):
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    def tr(self):
        def _c():
            pc = self.close.shift(1)
            return pd.concat([self.high - self.low,
                              (self.high - pc).abs(),
                              (self.low - pc).abs()], axis=1).max(axis=1).rename("tr")
        return self._get(("tr",), _c, 1)

    def atr(self, n=14):
        return self._get(("atr", n), lambda: self._wilder(self.tr(), n).rename(f"atr{n}"),
                         self.burnin_mult * n)

    def roc(self, n=14):
        return self._get(("roc", n), lambda: (self.price.pct_change(n) * 100).rename(f"roc{n}"), n)

    def rvol(self, n=50):
        def _c():
            ref = self.volume.shift(1).rolling(n).median()
            return (self.volume / ref).rename(f"rvol{n}")
        return self._get(("rvol", n), _c, n + 1)

    def adx(self, n=14):
        def _c():
            up = self.high.diff()
            down = -self.low.diff()
            pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=self.df.index)
            ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=self.df.index)
            atr = self._wilder(self.tr(), n)
            pdi = 100 * self._wilder(pdm, n) / atr
            ndi = 100 * self._wilder(ndm, n) / atr
            dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
            return self._wilder(dx, n).rename(f"adx{n}")
        return self._get(("adx", n), _c, self.burnin_mult * n)

    def er(self, n=10):
        def _c():
            direction = self.price.diff(n).abs()
            vol = self.price.diff().abs().rolling(n).sum()
            return (direction / vol.replace(0, np.nan)).rename(f"er{n}")
        return self._get(("er", n), _c, n)

    def kama(self, n=10, fast=2, slow=30):
        def _c():
            er = self.er(n).fillna(0)
            fa, sa = 2 / (fast + 1), 2 / (slow + 1)
            sc = (er * (fa - sa) + sa) ** 2
            vals, scs = self.price.to_numpy(), sc.to_numpy()
            out = np.full(len(vals), np.nan)
            k = None
            for i, (p, c) in enumerate(zip(vals, scs)):
                if np.isnan(p):
                    out[i] = k if k is not None else np.nan
                    continue
                k = p if k is None else k + c * (p - k)
                out[i] = k
            return pd.Series(out, index=self.price.index).rename(f"kama{n}")
        return self._get(("kama", n, fast, slow), _c, self.burnin_mult * n)

    @staticmethod
    def zscore(s, n):
        m = s.rolling(n).mean()
        sd = s.rolling(n).std()
        return ((s - m) / sd.replace(0, np.nan)).rename(f"{s.name}_z{n}")

    @staticmethod
    def rank_pct(s, n):
        return s.rolling(n).rank(pct=True).rename(f"{s.name}_rk{n}")

    @staticmethod
    def logret(s, n=1):
        return np.log(s).diff(n).rename(f"logret{n}")

    def ext(self, n_ema=50, n_atr=14):
        return ((self.price - self.ema(n_ema)) / self.atr(n_atr)).rename(f"ext{n_ema}")

    def atr_pct(self, n=14):
        return (self.atr(n) / self.price).rename(f"atrpct{n}")
