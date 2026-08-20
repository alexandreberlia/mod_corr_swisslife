# Moteur cross-sectionnel — actions

## Fichiers

| Fichier | Rôle | Fréquence |
|---|---|---|
| `indicateurs.py` | Briques de base : EMA, ATR, ADX, ER, KAMA, RVOL… | — |
| `features.py` | **Laboratoire** : catalogue de features orientées + calibration par IC | 1× / trimestre |
| `portefeuille.py` | **Production** : score, sélection, allocation, backtest | tous les jours |
| `pipeline_complet.py` | Exemple de bout en bout sur données réelles | — |
| `test_validation.py` | 31 contrôles automatiques | après chaque modif |
| `diagnostic.py` | Trouve à quel étage un book vide se vide | en cas de problème |

## Installation

```bash
pip install pandas numpy scipy yfinance
python test_validation.py     # doit afficher 31/31
python pipeline_complet.py
```

## Usage minimal

```python
from indicateurs import Indicateurs
from features import features_orientees, calibrer
from portefeuille import ParamsPF, Portefeuille, construire_panels, stats

p = ParamsPF(n_long=5, rang_entree=0.70, min_titres=8)

# generateur=features_orientees OBLIGATOIRE si les poids viennent de l'IC
panels = construire_panels(prix, Indicateurs, p, generateur=features_orientees)
close  = panels["close"]

# calibration IN-SAMPLE uniquement — sinon look-ahead
split = close.index[int(len(close) * 0.60)]
poids, rapport, corr = calibrer(panels, close, split=split, horizon=20)

p.poids = poids
pf  = Portefeuille(panels, p, secteurs)
res = pf.backtest(10_000)

print(stats(res, close=close))
print(pf.book(equity=10_000))      # les ordres à passer
```

## Séparation des rôles

`features.py` utilise le futur (`shift(-horizon)`) pour **mesurer** quelles features
prédisent. C'est légitime en recherche. `portefeuille.py` ne regarde que le passé.
Le seul objet qui transite entre les deux est le dictionnaire de poids.

Supprimer `features.py` : `portefeuille.py` tourne quand même, avec des poids
codés en dur. Le laboratoire remplace le jugement par une mesure, il n'est pas
un composant nécessaire du système.

## Trois règles de dimensionnement

1. `n_long <= (1 - rang_entree) × taille_univers`, avec marge pour l'éligibilité
2. `min_titres <=` la moitié de l'univers
3. Les filtres d'éligibilité se **multiplient** : trois filtres à 40 % ne laissent
   passer que quelques titres

En cas de book vide : `from diagnostic import diagnostic; diagnostic(pf)`

## Comment lire un backtest

Le seul chiffre probant est **`alpha_vs_bh_%` sur la période out-of-sample**
(après `split`). Comparer à zéro ne prouve rien : sur une marche aléatoire en log,
le prix dérive à la hausse (inégalité de Jensen), donc une stratégie long-only
gagne mécaniquement sans aucun edge.

Contrôle négatif intégré : sur du bruit pur, l'alpha moyen doit être négatif.
Mesuré : **-3,87 %** sur 6 graines.


## Simulation walk-forward

« Si j'avais lancé l'algo le 2025-01-01, où en serais-je aujourd'hui ? »

```python
from simulation import simuler
sim = simuler(prix, "2025-01-01", Indicateurs, p, secteurs, capital=10_000)
print(sim.rapport())
print(sim.pnl)          # décomposition réalisé / latent / frais
sim.positions           # titres encore détenus, valorisés au dernier cours
sim.trades              # tout ce qui a été vendu
```

Ou directement : `python lancer_simulation.py` (éditer `DATE_DEBUT` en tête).

**Recalibrage walk-forward.** Les poids sont réestimés tous les 63 jours en
n'utilisant que les données antérieures à la date de recalibrage. Sans cela, des
poids calibrés sur toute la période contiendraient le futur — c'est le seul endroit
du système où le look-ahead peut se glisser, les indicateurs et les rangs étant
causaux par construction.

**Historique d'amorçage.** Télécharger au moins 3-4 ans AVANT la date de début :
`mom_12_1` exige 252 séances et la calibration IC davantage. Cet historique amorce
les indicateurs, il n'est jamais tradé.

**Contrôle de causalité.** `test_causalite()` vérifie qu'à une date T le carnet est
identique selon qu'on dispose de l'historique complet ou tronqué à T. Une divergence
signale une fuite d'information future.

## Limites connues

- **Long only.** La vente à découvert n'est pas implémentée.
- **Pas d'univers point-in-time.** Biais du survivant, sévère sur du momentum.
- **Coûts forfaitaires** (10 bps). Sous-estimé sur les valeurs peu liquides.
- **~21 paramètres.** N'en optimiser que 2 ou 3 (`rang_entree`, `trail_atr`),
  chercher des **plateaux** et non des pics.
- Les IC des tests synthétiques (~0,20) sont irréalistes. En equity réelle,
  0,02–0,05 est déjà bon.
