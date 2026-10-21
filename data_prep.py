"""
Étape 3 : préparation des données pour le modèle VSN + LSTM.

Entrées attendues :
  - prix journaliers yfinance (colonnes MultiIndex (champ, ticker) ou (ticker, champ))
  - indicateurs techniques journaliers (MultiIndex en lignes ou en colonnes)
  - score macro mensuel (CSV : une colonne date + une ou plusieurs colonnes de valeurs)

Sortie : un DataFrame « long », indexé par (date, ticker), avec
  - les variables explicatives normalisées,
  - la cible,
  - une colonne "split" (train / val / test).

Règle d'or : à la date t, chaque ligne ne contient que de l'information
réellement disponible à la clôture de t.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


# ---------------------------------------------------------------------------
# 1. Mise au format long (date, ticker)
# ---------------------------------------------------------------------------
def prices_dict_to_long(prices: dict) -> pd.DataFrame:
    """{titre: DataFrame Open/High/Low/Close/Volume} -> format long (date, ticker).

    Gère aussi le cas où chaque DataFrame a des colonnes MultiIndex (Price, Ticker),
    ce que renvoient les versions récentes de yf.download pour un seul titre.
    """
    frames = {}
    for t, df in prices.items():
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            lvl = 0 if set(df.columns.get_level_values(0)) & PRICE_FIELDS else 1
            df.columns = df.columns.get_level_values(lvl)
        frames[t] = df
    long = pd.concat(frames, names=["ticker", "date"]).swaplevel()
    long.columns.name = None
    return _clean_long_index(long)


def _clean_long_index(out: pd.DataFrame) -> pd.DataFrame:
    """Index (date, ticker) : dates sans heure ni fuseau, trié."""
    out.index = out.index.set_names(["date", "ticker"])
    dates = pd.to_datetime(out.index.get_level_values("date"))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    out.index = pd.MultiIndex.from_arrays(
        [dates.normalize(), out.index.get_level_values("ticker")], names=["date", "ticker"])
    return out.sort_index()


def to_long(df, field_level: int | None = None) -> pd.DataFrame:
    """Convertit des prix ou indicateurs en format long indexé par (date, ticker).

    Cas gérés :
      - dictionnaire {titre: DataFrame Open/High/Low/Close/Volume}
      - colonnes MultiIndex (champ, ticker) : format par défaut de yf.download
      - colonnes MultiIndex (ticker, champ) : yf.download(..., group_by="ticker")
      - lignes MultiIndex (date, ticker) ou (ticker, date)

    field_level : niveau des colonnes qui contient les noms de champs (0 ou 1).
                  Laisser None pour une détection automatique sur les champs OHLCV ;
                  à préciser pour un DataFrame d'indicateurs en colonnes MultiIndex.
    """
    if isinstance(df, dict):
        return prices_dict_to_long(df)
    if isinstance(df.columns, pd.MultiIndex):
        if field_level is None:
            lvl0 = set(df.columns.get_level_values(0))
            lvl1 = set(df.columns.get_level_values(1))
            if lvl0 & PRICE_FIELDS:
                field_level = 0
            elif lvl1 & PRICE_FIELDS:
                field_level = 1
            else:
                raise ValueError(
                    "Impossible de deviner quel niveau de colonnes contient les champs : "
                    "précisez field_level=0 ou field_level=1."
                )
        ticker_level = 1 - field_level
        out = df.stack(level=ticker_level)
        out.index = out.index.set_names(["date", "ticker"])
    elif isinstance(df.index, pd.MultiIndex):
        out = df.copy()
        # le niveau date est celui dont les valeurs sont des dates
        date_pos = next(
            i for i in range(out.index.nlevels)
            if pd.api.types.is_datetime64_any_dtype(out.index.get_level_values(i))
        )
        if date_pos != 0:
            out = out.swaplevel(0, 1)
        out.index = out.index.set_names(["date", "ticker"])
    else:
        raise ValueError("Le DataFrame doit avoir un MultiIndex en lignes ou en colonnes.")

    out.columns.name = None
    out = out.sort_index()
    # dates sans heure ni fuseau, pour pouvoir fusionner proprement
    dates = pd.to_datetime(out.index.get_level_values("date"))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    out.index = pd.MultiIndex.from_arrays(
        [dates.normalize(), out.index.get_level_values("ticker")], names=["date", "ticker"]
    )
    return out


def panel_to_long(panels: dict, tickers=None) -> pd.DataFrame:
    """Convertit un dictionnaire d'indicateurs en format long (date, ticker).

    Formats acceptés (détectés automatiquement) :
      {titre: DataFrame}            lignes = dates, colonnes = indicateurs
      {indicateur: DataFrame}       lignes = dates, colonnes = titres
      {titre: {indicateur: Series}} ou {indicateur: {titre: Series}}

    tickers : liste des titres (par ex. ceux de vos prix). Facultatif, mais
              permet de savoir à coup sûr quel niveau du dictionnaire contient
              les titres.
    """
    known = set(tickers) if tickers is not None else None

    def is_ticker_level(keys, inner_cols=None) -> bool:
        keys = set(map(str, keys))
        if known is not None:
            k = set(map(str, known))
            if keys & k:
                return True
            if inner_cols is not None and set(map(str, inner_cols)) & k:
                return False
        # sans liste de titres : les noms d'indicateurs ressemblent rarement à "MC.PA"
        indicator_like = {"RSI", "BB_pct", "rev_5", "stoch_long", "MACD_hist", "ER"}
        if keys & indicator_like:
            return False
        if inner_cols is not None and set(map(str, inner_cols)) & indicator_like:
            return True
        raise ValueError("Impossible de savoir si les clés du dictionnaire sont des titres ou "
                         "des indicateurs : passez tickers=[...] (par ex. la liste de vos titres).")

    frames = {}
    for key, val in panels.items():
        if isinstance(val, dict):                       # dictionnaire de Series
            val = pd.DataFrame(val)
        elif isinstance(val, pd.Series):
            val = val.to_frame()
        frames[key] = val

    inner_cols = frames[next(iter(frames))].columns
    keys_are_tickers = is_ticker_level(panels.keys(), inner_cols)

    if keys_are_tickers:        # {titre: DataFrame dates x indicateurs}
        long = pd.concat(frames, names=["ticker", "date"])
    else:                       # {indicateur: DataFrame dates x titres}
        long = pd.concat({ind: df.stack() for ind, df in frames.items()}, axis=1)
        long.index = long.index.set_names(["date", "ticker"])
        long = long.swaplevel()
        long.index = long.index.set_names(["ticker", "date"])
    long = long.swaplevel().sort_index()
    long.index = long.index.set_names(["date", "ticker"])
    # même nettoyage des dates que to_long
    dates = pd.to_datetime(long.index.get_level_values("date"))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    long.index = pd.MultiIndex.from_arrays(
        [dates.normalize(), long.index.get_level_values("ticker")], names=["date", "ticker"])
    long.columns.name = None
    return long.sort_index()


def indicators_to_long(indicators, tickers=None, field_level: int | None = None) -> pd.DataFrame:
    """Accepte un dictionnaire (panel) ou un DataFrame MultiIndex."""
    if isinstance(indicators, dict):
        return panel_to_long(indicators, tickers)
    if _is_long(indicators):
        return indicators
    return to_long(indicators, field_level)


# ---------------------------------------------------------------------------
# 2. Variables issues des prix + cible
# ---------------------------------------------------------------------------
def price_features(prices: pd.DataFrame, horizon: int = 1,
                   target: str = "return") -> pd.DataFrame:
    """Transforme les prix (non stationnaires) en variables stationnaires + cible.

    prices : format long (date, ticker) avec au moins Close (ou Adj Close) et Volume.
    horizon: h, la cible porte sur la période t -> t+h.
    target : "return"    -> log(P_{t+h} / P_t)
             "direction" -> 1 si ce rendement est positif, sinon 0

    Les prix bruts NE sont PAS conservés comme variables : leur niveau dépend
    de l'action et de l'époque, et le modèle s'en servirait mal.
    """
    close_col = "Adj Close" if "Adj Close" in prices.columns else "Close"
    g = prices.groupby(level="ticker", group_keys=False)
    log_close = np.log(prices[close_col])
    lc = log_close.groupby(level="ticker", group_keys=False)

    feats = pd.DataFrame(index=prices.index)
    feats["ret_1d"] = lc.diff(1)
    feats["ret_5d"] = lc.diff(5)
    feats["ret_20d"] = lc.diff(20)
    feats["vol_20d"] = feats["ret_1d"].groupby(level="ticker", group_keys=False) \
        .apply(lambda s: s.rolling(20).std())
    if "Volume" in prices.columns:
        vol = prices["Volume"].replace(0, np.nan)
        vol_ma = g["Volume"].apply(lambda s: s.replace(0, np.nan).rolling(20).mean())
        feats["volume_rel"] = np.log(vol / vol_ma)          # volume du jour vs moyenne 20 j

    # Cible : uniquement du futur, décalé de -h à l'intérieur de chaque ticker
    fwd = lc.shift(-horizon) - log_close
    feats["target"] = (fwd > 0).astype(float).where(fwd.notna()) if target == "direction" else fwd
    return feats


# ---------------------------------------------------------------------------
# 3. Données macro mensuelles -> journalières, sans regarder le futur
# ---------------------------------------------------------------------------
def load_macro(path_or_df, date_col: str = "date", publication_lag_days: int = 30,
               date_is_period_start: bool = True, add_changes: bool = True) -> pd.DataFrame:
    """Charge un fichier macro mensuel et calcule sa date de disponibilité.

    Le score du mois M n'est connu qu'après la fin de M, plus le délai de
    publication des indicateurs qui le composent. On ne l'utilise qu'à partir
    de : fin du mois M + publication_lag_days.

    date_is_period_start : True si la date du CSV désigne le mois de référence
                           (ex. 2024-03-01 pour le score de mars).
                           False si la date est déjà la date de publication.
    add_changes          : ajoute la variation d'un mois sur l'autre (suffixe _chg).
    """
    macro = read_macro_table(path_or_df, date_col)
    macro = macro.sort_values(date_col).set_index(date_col)
    macro = macro.select_dtypes("number")

    if add_changes:
        for c in list(macro.columns):
            macro[f"{c}_chg"] = macro[c].diff()

    if date_is_period_start:
        month_end = macro.index.to_period("M").to_timestamp(how="end").normalize()
        available = month_end + pd.Timedelta(days=publication_lag_days)
    else:
        available = macro.index
    macro.index = pd.DatetimeIndex(available, name="available_date")
    return macro


def read_macro_table(path_or_df, date_col: str = "date") -> pd.DataFrame:
    """Lit le CSV macro de façon tolérante aux formats français.

    - séparateur détecté automatiquement (« , » ou « ; ») ;
    - décimales avec virgule (« 45,3 ») converties en nombres ;
    - dates « 31/01/2010 » lues jour/mois/année ;
    - date en index plutôt qu'en colonne acceptée.
    """
    if isinstance(path_or_df, (str, os.PathLike)):
        macro = pd.read_csv(path_or_df, sep=None, engine="python")
    else:
        macro = path_or_df.copy()
    macro.columns = [str(c).strip() for c in macro.columns]
    if date_col not in macro.columns:
        if isinstance(macro.index, pd.DatetimeIndex) or macro.index.name == date_col:
            macro = macro.reset_index().rename(columns={macro.index.name or "index": date_col})
        else:
            raise ValueError(f"Colonne date {date_col!r} introuvable. Colonnes du fichier : "
                             f"{list(macro.columns)} -> précisez date_col=...")
    raw = macro[date_col]
    dayfirst = (not pd.api.types.is_datetime64_any_dtype(raw)
                and raw.astype(str).str.contains("/").any())
    macro[date_col] = pd.to_datetime(raw, dayfirst=bool(dayfirst))
    for c in macro.columns:
        if c != date_col and not pd.api.types.is_numeric_dtype(macro[c]):
            conv = pd.to_numeric(macro[c].astype(str).str.replace(",", ".", regex=False)
                                 .str.replace(" ", "", regex=False).str.strip(),
                                 errors="coerce")
            if conv.notna().mean() > 0.8:
                macro[c] = conv
    return macro


def align_macro(dates: pd.DatetimeIndex, macro: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque date journalière, prend la dernière valeur macro DÉJÀ disponible."""
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates).unique().sort_values()})
    right = macro.reset_index().rename(columns={"available_date": "macro_date"})
    out = pd.merge_asof(left, right, left_on="date", right_on="macro_date",
                        direction="backward")
    return out.set_index("date").drop(columns="macro_date")


# ---------------------------------------------------------------------------
# 4. Découpage temporel et normalisation
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    """Moyennes et écarts-types appris sur la seule période d'entraînement."""
    mean: pd.Series = field(default_factory=pd.Series)
    std: pd.Series = field(default_factory=pd.Series)
    clip: float = 5.0

    def fit(self, df: pd.DataFrame) -> "Scaler":
        self.mean = df.mean()
        self.std = df.std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # z-score puis écrêtage à ±clip : limite l'effet des queues épaisses
        return ((df - self.mean) / self.std).clip(-self.clip, self.clip)


def time_split(dates: pd.DatetimeIndex, train_end: str, val_end: str,
               horizon: int, train_start: str | None = None) -> pd.Series:
    """Attribue train / val / test selon la date, avec une purge.

    La cible d'une ligne datée t utilise le prix de t+h. Les h derniers jours
    d'entraînement (et de validation) ont donc une cible qui « déborde » sur la
    période suivante : on les retire (label "purge").

    train_start : les dates antérieures sont étiquetées "warmup". Elles ne
    servent jamais de point de prédiction, mais leurs variables peuvent remplir
    l'historique des premières fenêtres d'entraînement.
    """
    udates = pd.DatetimeIndex(dates).unique().sort_values()
    split = pd.Series("test", index=udates)
    split[udates <= pd.Timestamp(val_end)] = "val"
    split[udates <= pd.Timestamp(train_end)] = "train"
    if train_start is not None:
        split[udates < pd.Timestamp(train_start)] = "warmup"
    for name in ("train", "val"):
        last = split.index[split == name]
        if len(last):
            split[last[-horizon:]] = "purge"
    return split


# ---------------------------------------------------------------------------
# 5. Pipeline complet
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    df: pd.DataFrame            # index (date, ticker) ; features + régime + "target" + "split"
    feature_cols: list[str]     # variables lues par le VSN
    scaler: Scaler
    tickers: list[str]
    regime_cols: list[str] = field(default_factory=list)   # contexte (score macro)
    signal_cols: list[str] = field(default_factory=list)   # variables de franchissement


def prepare_dataset(prices,
                    indicators=None,
                    macro: pd.DataFrame | None = None,
                    train_start: str | None = None,
                    train_end: str = "2019-12-31",
                    val_end: str = "2021-12-31",
                    horizon: int = 1,
                    target: str = "return",
                    indicator_field_level: int | None = None,
                    scale_target: bool = False,
                    signaux: list[dict] | bool | None = None,
                    garder_niveaux: bool = False,
                    demi_vie: float = 2.0,
                    max_jours: int = 10,
                    regime_col: str | None = None,
                    seuils_regime: tuple[float, float] | None = None,
                    variables_prix: bool = True) -> PreparedData:
    """Assemble prix, indicateurs et macro en un jeu de données propre.

    prices     : yf.download, dictionnaire {titre: DataFrame}, ou format long
    indicators : dictionnaire `panels` ou DataFrame MultiIndex
    macro      : sortie de load_macro

    Règles de décision (franchissements) :
      signaux        : True pour utiliser la liste SIGNAUX de signaux.py (ou une liste à vous).
                       Chaque signal devient une variable « fraîcheur » : 1 le jour du
                       franchissement, puis décroissante (demi_vie), 0 après max_jours.
      garder_niveaux : False (défaut) = les indicateurs bruts (RSI = 52...) sont retirés,
                       le modèle ne voit que les règles. True = il voit les deux.

    Score macro comme régime (contexte) plutôt que comme variable :
      regime_col     : nom de la colonne du score (ex. "score"). Toutes les colonnes macro
                       passent alors en contexte : elles ne sont plus lues par le VSN comme
                       une variable, elles modifient la lecture des autres variables.
      seuils_regime  : (25, 75) par défaut : ajoute deux indicatrices 0/1, régime bas et élevé.

    variables_prix : False pour retirer les rendements / volatilité calculés sur les prix.
    """
    from signaux import SEUIL_BAS, SEUIL_HAUT, SIGNAUX, calcul_signaux, memoire_signaux

    prices_long = prices if _is_long(prices) else to_long(prices)
    data = price_features(prices_long, horizon=horizon, target=target)
    if not variables_prix:
        data = data[["target"]]
    no_scale: list[str] = []
    signal_cols: list[str] = []

    if indicators is not None:
        tickers = prices_long.index.get_level_values("ticker").unique()
        ind = indicators_to_long(indicators, tickers, indicator_field_level)
        ind = ind.select_dtypes("number")

        if signaux:
            liste = SIGNAUX if signaux is True else signaux
            close_col = "Adj Close" if "Adj Close" in prices_long.columns else "Close"
            base = ind.join(prices_long[[close_col]].rename(columns={close_col: "Close"}),
                            how="inner")
            ev = calcul_signaux(base, liste, par="id")
            mem = memoire_signaux(ev, demi_vie=demi_vie, max_jours=max_jours)
            mem.columns = [f"sig_{c}" for c in mem.columns]
            signal_cols = list(mem.columns)
            no_scale += signal_cols          # déjà entre 0 et 1 : pas de normalisation
            data = data.join(mem, how="left")
            if garder_niveaux:
                data = data.join(ind, how="left")
        else:
            data = data.join(ind, how="left")

    regime_cols: list[str] = []
    if macro is not None:
        dates = data.index.get_level_values("date")
        macro_daily = align_macro(dates, macro)
        data = data.join(macro_daily, on="date")
        if regime_col is not None:
            if regime_col not in macro_daily.columns:
                raise ValueError(f"regime_col={regime_col!r} absent des colonnes macro "
                                 f"{list(macro_daily.columns)}")
            regime_cols = list(macro_daily.columns)
            bas, haut = seuils_regime or (SEUIL_BAS, SEUIL_HAUT)
            sc = data[regime_col]
            data["regime_bas"] = (sc < bas).astype(float).where(sc.notna())
            data["regime_eleve"] = (sc > haut).astype(float).where(sc.notna())
            regime_cols += ["regime_bas", "regime_eleve"]
            no_scale += ["regime_bas", "regime_eleve"]

    # Nettoyage : lignes incomplètes (début des moyennes glissantes, fin de cible, etc.)
    feature_cols = [c for c in data.columns if c != "target" and c not in regime_cols]
    if signal_cols:
        data[signal_cols] = data[signal_cols].fillna(0.0)
    data = data.replace([np.inf, -np.inf], np.nan) \
        .dropna(subset=feature_cols + regime_cols + ["target"])

    # Découpage
    split = time_split(data.index.get_level_values("date"), train_end, val_end, horizon,
                       train_start=train_start)
    data["split"] = split.reindex(data.index.get_level_values("date")).values
    # Les lignes "purge" sont gardées : leurs variables restent utilisables comme
    # historique dans les fenêtres de la période suivante. Seule leur cible est
    # exclue (elles ne servent jamais de point de prédiction, cf. étape 4).

    # Normalisation apprise sur le train uniquement
    train = data["split"] == "train"
    scale_cols = [c for c in feature_cols + regime_cols if c not in no_scale] \
        + (["target"] if scale_target and target == "return" else [])
    scaler = Scaler().fit(data.loc[train, scale_cols])
    data[scale_cols] = scaler.transform(data[scale_cols])

    tickers = sorted(data.index.get_level_values("ticker").unique())
    return PreparedData(df=data, feature_cols=feature_cols, scaler=scaler, tickers=tickers,
                        regime_cols=regime_cols, signal_cols=signal_cols)


def _is_long(df) -> bool:
    return (isinstance(df, pd.DataFrame) and isinstance(df.index, pd.MultiIndex)
            and list(df.index.names) == ["date", "ticker"])


def ticker_onehot(tickers: list[str]) -> pd.DataFrame:
    """Variables statiques simples pour un panel : un indicateur 0/1 par ticker."""
    return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Tests sur des données simulées au format yfinance : python data_prep.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    tickers = ["AAA", "BBB", "CCC"]

    # Prix au format yf.download : colonnes (champ, ticker)
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"], tickers],
                                      names=["Price", "Ticker"])
    raw = pd.DataFrame(index=pd.DatetimeIndex(dates, name="Date"), columns=cols, dtype=float)
    for t in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        raw[("Close", t)] = close
        raw[("High", t)] = close * 1.01
        raw[("Low", t)] = close * 0.99
        raw[("Open", t)] = close
        raw[("Volume", t)] = rng.integers(1e5, 1e6, len(dates))
    raw.iloc[:50, raw.columns.get_loc(("Close", "CCC"))] = np.nan     # introduction tardive

    # Indicateurs en lignes MultiIndex (Date, Ticker)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    indicators = pd.DataFrame({"RSI": rng.uniform(0, 100, len(idx)),
                               "BB_pct": rng.normal(0.5, 0.3, len(idx)),
                               "Stoch_slow": rng.uniform(0, 100, len(idx))}, index=idx)

    # Score macro mensuel : la date = début du mois de référence
    months = pd.date_range("2014-06-01", "2024-12-01", freq="MS")
    macro_csv = pd.DataFrame({"date": months, "macro_score": rng.normal(0, 1, len(months))})
    macro = load_macro(macro_csv, publication_lag_days=30)

    pdata = prepare_dataset(raw, indicators, macro,
                            train_end="2020-12-31", val_end="2022-12-31", horizon=1)
    df = pdata.df

    # 1. Plus aucune valeur manquante, colonnes attendues
    assert not df[pdata.feature_cols + ["target"]].isna().any().any()
    assert {"ret_1d", "vol_20d", "RSI", "macro_score", "macro_score_chg"} <= set(df.columns)
    assert "Close" not in df.columns

    # 2. Macro sans fuite : la valeur utilisée le 2023-03-15 est celle de janvier 2023
    #    (fin janvier + 30 jours = 2 mars <= 15 mars ; février n'est dispo que le 30 mars)
    jan = macro_csv.loc[macro_csv.date == "2023-01-01", "macro_score"].item()
    row = df.loc[(pd.Timestamp("2023-03-15"), "AAA")]
    expected = (jan - pdata.scaler.mean["macro_score"]) / pdata.scaler.std["macro_score"]
    assert abs(row["macro_score"] - expected) < 1e-9

    # 3. Cible = log-rendement du lendemain (vérifiée sur une ligne avant normalisation)
    long_prices = to_long(raw)
    c = long_prices["Close"].xs("AAA", level="ticker")
    d0 = pd.Timestamp("2021-06-01")
    d1 = c.index[c.index.get_loc(d0) + 1]
    assert abs(df.loc[(d0, "AAA"), "target"] - np.log(c[d1] / c[d0])) < 1e-12

    # 4. Normalisation apprise sur le train : moyenne ~0 et écart-type ~1 sur le train
    tr = df[df.split == "train"][pdata.feature_cols]
    assert tr.mean().abs().max() < 0.05 and (tr.std() - 1).abs().max() < 0.05

    # 5. Pas de chevauchement de dates entre périodes, purge appliquée
    last_train = df[df.split == "train"].index.get_level_values("date").max()
    first_val = df[df.split == "val"].index.get_level_values("date").min()
    assert last_train < pd.Timestamp("2020-12-31") < first_val

    print("Étape 3 OK.")
    print(f"  {len(df):,} lignes, {len(pdata.feature_cols)} variables : {pdata.feature_cols}")
    print("  " + df["split"].value_counts().to_string().replace("\n", " | "))
    print(f"  dernier jour train : {last_train.date()}, premier jour val : {first_val.date()}")
