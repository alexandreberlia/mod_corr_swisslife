"""
Étape 1 : briques du modèle VSN + LSTM (GLU, GRN, VSN).

Conventions de dimensions utilisées partout :
    B = taille du batch
    T = longueur de la fenêtre temporelle (nombre de pas)
    m = nombre de variables d'entrée
    d = d_model, dimension interne du modèle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit : GLU(x) = sigmoid(W_gate x + b) * (W_value x + b).

    - value : le contenu proposé (non borné)
    - gate  : la part de ce contenu qu'on laisse passer (entre 0 et 1)
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.gate = nn.Linear(d_in, d_out)
        self.value = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * self.value(x)


class GRN(nn.Module):
    """Gated Residual Network (Lim et al., 2021).

        eta2 = ELU(W2 a + W3 c + b2)          # c : contexte optionnel
        eta1 = W1 eta2 + b1
        GRN(a, c) = LayerNorm(skip(a) + GLU(eta1))

    Args:
        d_in      : dimension de l'entrée a
        d_hidden  : dimension de la couche cachée
        d_out     : dimension de sortie
        d_context : dimension du contexte c (None = pas de contexte)
        dropout   : taux de dropout appliqué avant le GLU
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 d_context: int | None = None, dropout: float = 0.1):
        super().__init__()
        # Connexion résiduelle : projection seulement si les dimensions diffèrent
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc2 = nn.Linear(d_in, d_hidden)
        self.fc_context = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        eta2 = self.fc2(a)
        if self.fc_context is not None and c is not None:
            eta2 = eta2 + self.fc_context(c)
        eta2 = F.elu(eta2)
        eta1 = self.dropout(self.fc1(eta2))
        return self.norm(self.skip(a) + self.glu(eta1))


class VSN(nn.Module):
    """Variable Selection Network pour m variables continues.

    À chaque pas de temps :
      1. chaque variable j est projetée en un vecteur de dimension d    -> xi_j
      2. un GRN regarde toutes les variables et produit m poids (softmax)
      3. chaque xi_j passe dans son propre GRN                          -> xi_tilde_j
      4. sortie = somme pondérée des xi_tilde_j

    Entrée  : x de forme (B, T, m)
    Sorties : out (B, T, d) et poids (B, T, m), les poids sommant à 1
    """

    def __init__(self, n_vars: int, d_model: int,
                 d_context: int | None = None, dropout: float = 0.1,
                 context_in_vars: bool = False):
        """context_in_vars : le contexte entre aussi dans le GRN de chaque variable.
        C'est ce qui permet une LECTURE différente d'une même variable selon le
        contexte (terme W3 c du GRN), en plus d'une sélection différente."""
        super().__init__()
        self.n_vars = n_vars
        self.context_in_vars = context_in_vars and d_context is not None
        # Étape 1 : une projection linéaire 1 -> d par variable
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_vars)])
        # Étape 2 : GRN de sélection, entrée = toutes les variables concaténées
        self.grn_weights = GRN(n_vars * d_model, d_model, n_vars,
                               d_context=d_context, dropout=dropout)
        # Étape 3 : un GRN propre à chaque variable (poids non partagés)
        d_ctx_vars = d_context if self.context_in_vars else None
        self.grn_vars = nn.ModuleList(
            [GRN(d_model, d_model, d_model, d_context=d_ctx_vars, dropout=dropout)
             for _ in range(n_vars)]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        if x.shape[-1] != self.n_vars:
            raise ValueError(f"Attendu {self.n_vars} variables, reçu {x.shape[-1]}")

        # 1. Projection : liste de m tenseurs (B, T, d)
        xi = [proj(x[..., j:j + 1]) for j, proj in enumerate(self.proj)]

        # 2. Poids de sélection : (B, T, m)
        flat = torch.cat(xi, dim=-1)                     # (B, T, m*d)
        if c is not None and c.dim() == 2:               # contexte statique (B, d_c)
            c = c.unsqueeze(1).expand(-1, x.shape[1], -1)  # -> (B, T, d_c)
        weights = torch.softmax(self.grn_weights(flat, c), dim=-1)

        # 3. Traitement individuel : (B, T, m, d)
        c_vars = c if self.context_in_vars else None
        xi_tilde = torch.stack([grn(e, c_vars) for grn, e in zip(self.grn_vars, xi)], dim=-2)

        # 4. Combinaison pondérée : (B, T, d)
        out = (weights.unsqueeze(-1) * xi_tilde).sum(dim=-2)
        return out, weights


class GateAddNorm(nn.Module):
    """Saut à porte : LayerNorm(residual + GLU(x)).

    Si la porte se ferme, le bloc précédent (ici le LSTM) est court-circuité
    et seule la connexion résiduelle passe.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + self.glu(self.dropout(x)))


# ---------------------------------------------------------------------------
# Étape 2 : le modèle complet
# ---------------------------------------------------------------------------
class VSNLSTM(nn.Module):
    """VSN -> LSTM -> saut à porte -> GRN -> tête de sortie.

    Args:
        n_vars        : nombre de variables temporelles (colonnes de x)
        d_model       : dimension interne (16 à 64 en général)
        n_layers      : nombre de couches LSTM empilées
        horizon       : nombre de pas futurs prédits (1 = J+1 seulement)
        quantiles     : liste de quantiles, ex. [0.1, 0.5, 0.9] ;
                        None = une seule valeur par horizon (perte MSE)
        n_static      : nombre de variables statiques (None = pas de statiques)
        n_regime      : nombre de variables de contexte qui évoluent dans le temps,
                        typiquement le score macro (None = pas de contexte)
        task          : "quantile", "point" ou "proba" (probabilité de hausse) ;
                        None = "quantile" si quantiles, sinon "point"
        dropout       : taux de dropout

    Le contexte de régime c_t (issu du score) agit à trois endroits :
      - la sélection des variables du VSN (quelles variables comptent),
      - la lecture de chaque variable (GRN de chaque variable, terme W3 c),
      - l'enrichissement final avant la sortie.
    Un même signal peut ainsi être lu différemment selon le niveau du score.

    Entrées du forward :
        x      : (B, T, n_vars)      variables temporelles
        static : (B, n_static)       variables statiques (optionnel)
        regime : (B, T, n_regime)    contexte de régime (optionnel)

    Sortie : dictionnaire
        "prediction"  : (B, horizon, n_outputs)   quantiles, valeur, ou LOGIT si task="proba"
                        (probabilité = sigmoïde du logit)
        "vsn_weights" : (B, T, n_vars)            importance des variables
    """

    def __init__(self, n_vars: int, d_model: int = 32, n_layers: int = 1,
                 horizon: int = 1, quantiles: list[float] | None = (0.1, 0.5, 0.9),
                 n_static: int | None = None, dropout: float = 0.1,
                 n_regime: int | None = None, task: str | None = None):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_layers = n_layers
        self.horizon = horizon
        self.task = task or ("quantile" if quantiles else "point")
        if self.task == "proba":
            quantiles = None
        self.quantiles = list(quantiles) if quantiles is not None else None
        self.n_outputs = len(self.quantiles) if self.quantiles else 1
        self.n_regime = n_regime

        # Encodeur de régime (optionnel) : score macro -> contexte c_t à chaque pas
        if n_regime:
            self.regime_proj = nn.Linear(n_regime, d_model)
            self.grn_regime = GRN(d_model, d_model, d_model, dropout=dropout)
        has_ctx = bool(n_static) or bool(n_regime)

        # Encodeur statique (optionnel) : produit 3 contextes, comme dans le TFT
        #   c_s : guide la sélection de variables du VSN
        #   c_h, c_c : initialisent l'état caché et la cellule du LSTM
        self.n_static = n_static
        if n_static:
            self.static_proj = nn.Linear(n_static, d_model)
            self.grn_cs = GRN(d_model, d_model, d_model, dropout=dropout)
            self.grn_ch = GRN(d_model, d_model, d_model, dropout=dropout)
            self.grn_cc = GRN(d_model, d_model, d_model, dropout=dropout)

        # 1. Sélection de variables
        self.vsn = VSN(n_vars, d_model, d_context=d_model if has_ctx else None,
                       dropout=dropout, context_in_vars=bool(n_regime))

        # 2. Dynamique temporelle
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)

        # 3. Saut à porte autour du LSTM
        self.post_lstm = GateAddNorm(d_model, dropout)

        # 4. Dernier traitement non linéaire, puis tête de sortie
        self.grn_out = GRN(d_model, d_model, d_model,
                           d_context=d_model if n_regime else None, dropout=dropout)
        self.head = nn.Linear(d_model, horizon * self.n_outputs)

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None,
                regime: torch.Tensor | None = None) -> dict:
        B, T = x.shape[0], x.shape[1]

        # Contextes statiques
        c_s, state = None, None
        if self.n_static:
            if static is None:
                raise ValueError("Le modèle attend des variables statiques (n_static)")
            s = self.static_proj(static)                       # (B, d)
            c_s = self.grn_cs(s)                               # (B, d)
            h0 = self.grn_ch(s).unsqueeze(0).repeat(self.n_layers, 1, 1)  # (L, B, d)
            c0 = self.grn_cc(s).unsqueeze(0).repeat(self.n_layers, 1, 1)
            state = (h0.contiguous(), c0.contiguous())

        # Contexte de régime : (B, T, d)
        c_r = None
        if self.n_regime:
            if regime is None:
                raise ValueError("Le modèle attend le contexte de régime (n_regime)")
            c_r = self.grn_regime(self.regime_proj(regime))

        # Contexte total vu par le VSN
        if c_s is not None and c_r is not None:
            ctx = c_s.unsqueeze(1).expand(-1, T, -1) + c_r
        else:
            ctx = c_r if c_r is not None else c_s

        # 1. VSN : (B, T, d) + poids (B, T, n_vars)
        xi, weights = self.vsn(x, ctx)

        # 2. LSTM : (B, T, d)
        h, _ = self.lstm(xi, state)

        # 3. Saut à porte : si le LSTM n'apporte rien, xi passe directement
        h = self.post_lstm(h, residual=xi)

        # 4. On garde le dernier pas de la fenêtre, qui résume tout le passé
        z = self.grn_out(h[:, -1], c_r[:, -1] if c_r is not None else None)   # (B, d)
        pred = self.head(z).view(B, self.horizon, self.n_outputs)

        # Quantiles toujours ordonnés (q0,1 <= q0,5 <= q0,9) : évite les croisements
        if self.quantiles and self.n_outputs > 1:
            pred, _ = torch.sort(pred, dim=-1)

        return {"prediction": pred, "vsn_weights": weights}


class QuantileLoss(nn.Module):
    """Perte pinball moyennée sur les quantiles, les horizons et le batch.

    y_hat : (B, horizon, Q)   y : (B, horizon)
    """

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        err = y.unsqueeze(-1) - y_hat                          # (B, H, Q)
        return torch.max(self.q * err, (self.q - 1) * err).mean()


# ---------------------------------------------------------------------------
# Tests rapides : python models.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, m, d = 8, 60, 12, 16

    # GRN : formes, avec et sans projection résiduelle, avec contexte
    grn = GRN(d_in=5, d_hidden=d, d_out=d, d_context=3)
    out = grn(torch.randn(B, T, 5), torch.randn(B, T, 3))
    assert out.shape == (B, T, d)

    # GRN : si la porte du GLU est fermée, GRN(a) = LayerNorm(a)  (partie 5.3)
    grn_off = GRN(d, d, d).eval()
    with torch.no_grad():
        grn_off.glu.gate.bias.fill_(-50.0)
    a = torch.randn(B, T, d)
    assert torch.allclose(grn_off(a), grn_off.norm(a), atol=1e-5)

    # VSN : formes et poids qui somment à 1
    vsn = VSN(n_vars=m, d_model=d)
    out, w = vsn(torch.randn(B, T, m))
    assert out.shape == (B, T, d)
    assert w.shape == (B, T, m)
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-6)
    assert (w >= 0).all()

    # VSN avec contexte statique (par exemple un embedding d'actif de dimension 4)
    vsn_ctx = VSN(n_vars=m, d_model=d, d_context=4)
    out, w = vsn_ctx(torch.randn(B, T, m), torch.randn(B, 4))
    assert out.shape == (B, T, d)

    # Le gradient remonte bien jusqu'aux entrées
    x = torch.randn(B, T, m, requires_grad=True)
    vsn(x)[0].sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0

    n_params = sum(p.numel() for p in vsn.parameters())
    print(f"Étape 1 OK. VSN({m} variables, d={d}) : {n_params:,} paramètres")

    # ----- Étape 2 : VSNLSTM -----
    # Cas 1 : un seul actif, J+1, 3 quantiles
    model = VSNLSTM(n_vars=m, d_model=d)
    out = model(torch.randn(B, T, m))
    assert out["prediction"].shape == (B, 1, 3)
    assert out["vsn_weights"].shape == (B, T, m)
    p = out["prediction"]
    assert (p[..., 0] <= p[..., 1]).all() and (p[..., 1] <= p[..., 2]).all()

    # Cas 2 : panel avec 5 variables statiques, 2 couches LSTM, horizon 5, point
    model2 = VSNLSTM(n_vars=m, d_model=d, n_layers=2, horizon=5,
                     quantiles=None, n_static=5)
    out2 = model2(torch.randn(B, T, m), torch.randn(B, 5))
    assert out2["prediction"].shape == (B, 5, 1)

    # Un pas d'entraînement complet : la perte baisse sur un batch fixe
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xb, yb = torch.randn(B, T, m), torch.randn(B, 1)
    model.train()
    first = loss_fn(model(xb)["prediction"], yb).item()
    for _ in range(50):
        loss = loss_fn(model(xb)["prediction"], yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first, "la perte devrait baisser sur un batch fixe"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Étape 2 OK. VSNLSTM({m} variables, d={d}) : {n_params:,} paramètres ; "
          f"perte sur batch fixe {first:.3f} -> {loss.item():.3f}")
