"""
Étape 5 : entraînement du VSNLSTM sur une cible de rendement.

Contenu :
  - TrainConfig   : tous les hyperparamètres au même endroit
  - train(...)    : boucle avec Adam, gradient clipping, réduction du pas,
                    early stopping et sauvegarde du meilleur modèle
  - predict(...)  : prédictions remises à l'échelle, indexées par (date, ticker)
  - load_model(...), plot_history(...)

Mise à l'échelle de la cible : un rendement journalier vaut ~0,01. Le modèle
apprend sur y / y_scale (y_scale = écart-type des cibles du train), ce qui
donne des pertes et des gradients d'ordre 1. predict() remet tout à l'échelle
d'origine : ses sorties sont des rendements log.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import QuantileLoss, VSNLSTM


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Modèle
    d_model: int = 32
    n_layers: int = 1
    dropout: float = 0.2
    task: str = "quantile"            # "quantile", "point" ou "proba" (probabilité de hausse)
    quantiles: list[float] | None = field(default_factory=lambda: [0.1, 0.5, 0.9])
    point_loss: str = "huber"         # si task="point" : "huber" ou "mse"
    # Optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-5        # légère pénalité L2 (cf. ridge, partie 1.6)
    grad_clip: float = 1.0            # norme max du gradient (cf. partie 3.3)
    max_epochs: int = 100
    patience: int = 10                # early stopping : époques sans amélioration
    lr_patience: int = 4              # époques sans amélioration avant de diviser le pas
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    # Divers
    seed: int = 42
    device: str = "auto"              # "auto", "cpu", "cuda" ou "mps"
    verbose: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Une époque (entraînement ou évaluation)
# ---------------------------------------------------------------------------
def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, y_scale: float,
              device: torch.device, optimizer: torch.optim.Optimizer | None = None,
              grad_clip: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Renvoie la perte moyenne, les prédictions (échelle réduite) et les cibles."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    preds, targets = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device) / y_scale                 # (B, 1)
            static = batch["static"].to(device) if "static" in batch else None
            regime = batch["regime"].to(device) if "regime" in batch else None
            pred = model(x, static=static, regime=regime)["prediction"]   # (B, 1, Q)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    return total / n, torch.cat(preds).numpy(), torch.cat(targets).numpy()


class PointLoss(nn.Module):
    """Perte pour une prévision ponctuelle : (B, 1, 1) contre (B, 1)."""

    def __init__(self, kind: str = "huber"):
        super().__init__()
        self.fn = nn.HuberLoss() if kind == "huber" else nn.MSELoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fn(y_hat.squeeze(-1), y)


class BinaryLoss(nn.Module):
    """Entropie croisée binaire sur un logit : (B, 1, 1) contre (B, 1) en 0/1."""

    def __init__(self):
        super().__init__()
        self.fn = nn.BCEWithLogitsLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fn(y_hat.squeeze(-1), y)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Aire sous la courbe ROC : probabilité qu'un jour de hausse reçoive une
    probabilité plus élevée qu'un jour de baisse. 0,5 = hasard, 1 = parfait."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def proba_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """Indicateurs d'une probabilité de hausse."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    base = y.mean()
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_base = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return {"auc": auc(p, y),
            "logloss": float(ll),
            "gain_logloss": float(ll_base - ll),     # > 0 : mieux que la fréquence moyenne
            "brier": float(np.mean((p - y) ** 2)),
            "acc": float(np.mean((p > 0.5) == (y > 0.5))),
            "base": float(base)}


def val_metrics(pred: np.ndarray, y: np.ndarray, quantiles: list[float] | None,
                task: str = "quantile") -> dict:
    """Indicateurs lisibles en plus de la perte.

    ic       : corrélation de rang (Spearman) entre prévision centrale et réalisé,
               l'« information coefficient » des gérants quantitatifs
    hit      : part des cas où le signe prévu est le bon
    coverage : part des réalisations dans [q_bas, q_haut] (cible : q_haut - q_bas)
    """
    pred = pred[:, 0, :]                                        # (N, Q)
    y = y[:, 0]
    if task == "proba":
        m = proba_metrics(1 / (1 + np.exp(-pred[:, 0])), y)
        return {k: m[k] for k in ("auc", "gain_logloss", "acc")}
    mid = pred[:, len(quantiles) // 2] if quantiles else pred[:, 0]
    out = {
        "ic": float(pd.Series(mid).corr(pd.Series(y), method="spearman")),
        "hit": float(np.mean(np.sign(mid) == np.sign(y))),
    }
    if quantiles and len(quantiles) >= 2:
        out["coverage"] = float(np.mean((y >= pred[:, 0]) & (y <= pred[:, -1])))
    return out


# ---------------------------------------------------------------------------
# Entraînement complet
# ---------------------------------------------------------------------------
def train(loaders: dict[str, DataLoader], cfg: TrainConfig,
          save_path: str | None = "best_model.pt") -> tuple[VSNLSTM, pd.DataFrame]:
    """Entraîne le modèle et renvoie (meilleur modèle, historique par époque)."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_ds = loaders["train"].dataset

    # Échelle de la cible, apprise sur le train uniquement (inutile pour une probabilité)
    y_train = train_ds.y[torch.as_tensor(train_ds.starts + train_ds.window - 1)]
    if cfg.task == "proba":
        y_scale = 1.0
        if not set(y_train.unique().tolist()) <= {0.0, 1.0}:
            raise ValueError("task='proba' attend une cible 0/1 : utilisez "
                             "prepare_dataset(..., target='direction').")
    else:
        y_scale = float(y_train.std())

    n_static = train_ds.static.shape[1] if train_ds.static is not None else None
    n_regime = len(train_ds.regime_cols) if getattr(train_ds, "regime_cols", None) else None
    quantiles = cfg.quantiles if cfg.task == "quantile" else None
    model_kwargs = dict(n_vars=len(train_ds.feature_cols), d_model=cfg.d_model,
                        n_layers=cfg.n_layers, horizon=1, quantiles=quantiles,
                        n_static=n_static, dropout=cfg.dropout,
                        n_regime=n_regime, task=cfg.task)
    model = VSNLSTM(**model_kwargs).to(device)

    if cfg.task == "proba":
        loss_fn = BinaryLoss()
    elif cfg.task == "quantile":
        loss_fn = QuantileLoss(cfg.quantiles)
    else:
        loss_fn = PointLoss(cfg.point_loss)
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)

    if cfg.verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{n_params:,} paramètres | {len(train_ds):,} fenêtres d'entraînement | "
              f"appareil : {device} | y_scale = {y_scale:.5f}")

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, y_scale, device,
                                  optimizer, cfg.grad_clip)
        va_loss, va_pred, va_y = run_epoch(model, loaders["val"], loss_fn, y_scale, device)
        metrics = val_metrics(va_pred, va_y, quantiles, cfg.task)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_loss)

        improved = va_loss < best_loss - 1e-6
        if improved:
            best_loss, best_epoch, wait = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "lr": lr, **{f"val_{k}": v for k, v in metrics.items()}})
        if cfg.verbose:
            extra = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            print(f"époque {epoch:>3} | train {tr_loss:.4f} | val {va_loss:.4f} | {extra} | "
                  f"lr {lr:.1e} | {time.time() - t0:.1f}s{'  *' if improved else ''}")

        if wait >= cfg.patience:
            if cfg.verbose:
                print(f"Arrêt anticipé : pas d'amélioration depuis {cfg.patience} époques.")
            break

    model.load_state_dict(best_state)
    model.y_scale = y_scale
    history = pd.DataFrame(history).set_index("epoch")
    if cfg.verbose:
        print(f"Meilleur modèle : époque {best_epoch}, perte de validation {best_loss:.4f}")

    if save_path:
        torch.save({"state_dict": best_state, "model_kwargs": model_kwargs,
                    "y_scale": y_scale, "feature_cols": train_ds.feature_cols,
                    "regime_cols": list(getattr(train_ds, "regime_cols", []) or []),
                    "config": asdict(cfg), "best_epoch": best_epoch,
                    "history": history.reset_index().to_dict("list")}, save_path)
        if cfg.verbose:
            print(f"Sauvegardé dans {save_path}")
    return model, history


def load_model(path: str, device: str = "auto") -> tuple[VSNLSTM, dict]:
    """Recharge un modèle sauvegardé par train()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = VSNLSTM(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.y_scale = ckpt["y_scale"]
    model.to(get_device(device)).eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Prédictions
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model: VSNLSTM, loader: DataLoader, with_weights: bool = False) -> pd.DataFrame:
    """Prédictions en rendements log, indexées par (date, ticker).

    Colonnes :
      - q0.1, q0.5, q0.9 (quantiles), "pred" (valeur) ou "proba" (probabilité de hausse)
      - "y" : réalisé (rendement, ou 1/0 pour une hausse)
      - si with_weights : poids du VSN de chaque variable au dernier jour de la
        fenêtre (préfixe "w_"), et le contexte de régime de ce jour (préfixe "ctx_")
    Le loader doit être non mélangé (val ou test).
    """
    device = next(model.parameters()).device
    model.eval()
    task = getattr(model, "task", "quantile")
    preds, ys, weights, regs = [], [], [], []
    for batch in loader:
        static = batch["static"].to(device) if "static" in batch else None
        regime = batch["regime"].to(device) if "regime" in batch else None
        out = model(batch["x"].to(device), static=static, regime=regime)
        p = out["prediction"][:, 0, :].cpu()
        preds.append(torch.sigmoid(p) if task == "proba" else p * model.y_scale)
        ys.append(batch["y"][:, 0])
        if with_weights:
            weights.append(out["vsn_weights"][:, -1].cpu())
            if regime is not None:
                regs.append(batch["regime"][:, -1])
    ds = loader.dataset
    if task == "proba":
        cols = ["proba"]
    else:
        cols = [f"q{q:g}" for q in model.quantiles] if model.quantiles else ["pred"]
    df = pd.DataFrame(torch.cat(preds).numpy(), index=ds.index, columns=cols)
    df["y"] = torch.cat(ys).numpy()
    if with_weights:
        w = pd.DataFrame(torch.cat(weights).numpy(), index=ds.index,
                         columns=[f"w_{c}" for c in ds.feature_cols])
        df = df.join(w)
        if regs:
            r = pd.DataFrame(torch.cat(regs).numpy(), index=ds.index,
                             columns=[f"ctx_{c}" for c in ds.regime_cols])
            df = df.join(r)
    return df


def plot_history(history: pd.DataFrame, path: str | None = None):
    """Courbes de perte train / validation et IC de validation."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history.index, history["train_loss"], label="train")
    ax1.plot(history.index, history["val_loss"], label="validation")
    best = history["val_loss"].idxmin()
    ax1.axvline(best, color="grey", ls="--", lw=1, label=f"meilleure époque ({best})")
    ax1.set_xlabel("époque")
    ax1.set_ylabel("perte")
    ax1.set_title("Pertes")
    ax1.legend()
    if "val_auc" in history.columns:
        ax2.plot(history.index, history["val_auc"], color="tab:green")
        ax2.axhline(0.5, color="grey", lw=1, label="hasard")
        ax2.set_ylabel("AUC")
        ax2.set_title("AUC sur la validation")
        ax2.legend()
    else:
        ax2.plot(history.index, history["val_ic"], color="tab:green")
        ax2.axhline(0, color="grey", lw=1)
        ax2.set_ylabel("IC (Spearman)")
        ax2.set_title("Corrélation de rang sur la validation")
    ax2.set_xlabel("époque")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ---------------------------------------------------------------------------
# Test sur données simulées avec un signal caché : python train.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_prep import load_macro, prepare_dataset, ticker_onehot
    from windows import make_loaders

    # Le rendement de demain dépend de l'indicateur "signal" d'aujourd'hui :
    #   r_{t+1} = 1 % x (0,3 x signal_t + bruit)   -> corrélation théorique ~0,29
    # Les autres variables sont du bruit. Le modèle doit retrouver le signal.
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2012-01-01", "2024-12-31")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    sig = {}
    for t in tickers:
        s = rng.normal(0, 1, len(dates))
        r = 0.01 * (0.3 * np.roll(s, 1) + rng.normal(0, 1, len(dates)))
        raw[("Close", t)] = 100 * np.exp(np.cumsum(r))
        raw[("Volume", t)] = rng.integers(1e5, 1e6, len(dates))
        sig[t] = s
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    ind = pd.DataFrame({"signal": np.stack([sig[t] for t in tickers], 1).ravel(),
                        "RSI": rng.uniform(0, 100, len(idx))}, index=idx)
    months = pd.date_range("2011-06-01", "2024-12-01", freq="MS")
    macro = load_macro(pd.DataFrame({"date": months, "score": rng.normal(0, 1, len(months))}))

    pdata = prepare_dataset(raw, ind, macro, train_end="2020-12-31", val_end="2022-12-31")
    loaders = make_loaders(pdata, window=20, batch_size=128,
                           static=ticker_onehot(pdata.tickers))

    cfg = TrainConfig(d_model=16, max_epochs=40, patience=8)
    model, history = train(loaders, cfg, save_path="test_model.pt")
    plot_history(history, "test_history.png")

    # Rechargement et prédictions sur le test
    model2, ckpt = load_model("test_model.pt", device="cpu")
    test = predict(model2, loaders["test"], with_weights=True)
    ic = test["q0.5"].corr(test["y"], method="spearman")
    cov = ((test["y"] >= test["q0.1"]) & (test["y"] <= test["q0.9"])).mean()
    top_var = test.filter(like="w_").mean().sort_values(ascending=False)

    print("\nSur le test :")
    print(f"  IC (Spearman) = {ic:.3f}   (maximum théorique ~0,29)")
    print(f"  couverture [q0,1 ; q0,9] = {cov:.0%}   (cible 80 %)")
    print("  poids moyens du VSN :", ", ".join(f"{k[2:]} {v:.2f}" for k, v in top_var.items()))
    assert ic > 0.15, "le modèle aurait dû retrouver le signal"
    assert top_var.index[0] == "w_signal", "le VSN aurait dû privilégier la variable signal"
    assert 0.65 < cov < 0.95
    print("Étape 5 OK.")
