# Indicateurs avancés & scoring macro

Deux outils indépendants, construits sur le même panel de séries trimestrielles américaines.

1. **Modèle d'indicateurs avancés** — classer les variables selon leur capacité à anticiper une cible (PIB, chômage…), et déterminer avec quelle avance.
2. **Scoring macro** — résumer l'état du cycle en un score 0-100, dans le prolongement du découpage en quatre phases.

---

## Installation

```bash
pip install numpy pandas scipy statsmodels
```

Tous les fichiers `.py` dans le même dossier.

## Format des données

Deux objets, indexés par trimestre :

```python
import pandas as pd

# Le panel : une colonne par série
panel = pd.read_csv("panel_trimestriel.csv", index_col=0)
panel.index = pd.PeriodIndex(panel.index, freq="Q")

# Les phases : une colonne 'phase' obligatoire
phases = pd.read_csv("phases_et_covariables.csv")
phases.index = pd.PeriodIndex(phases["trimestre"].str.replace("T", "Q"), freq="Q")
phases = phases.drop(columns=["trimestre"])
phases = phases[phases["phase"] != "Choc Covid"]   # 3 trimestres : trop court
```

L'index doit être un `PeriodIndex('Q')` des deux côtés. C'est la seule contrainte stricte, mais elle ne souffre aucune exception : tout l'appariement se fait sur les dates, jamais par position.

---

# Partie 1 — Indicateurs avancés

## L'idée

Pour chaque variable candidate, on teste : *à quel horizon anticipe-t-elle le mieux la cible, et cette avance est-elle fiable ?*

Le modèle compare, pour chaque horizon `k` de 1 à 8 trimestres :

> cible dans k trimestres = f( cible aujourd'hui , variable aujourd'hui )

La cible d'aujourd'hui figure dans l'équation à dessein : le coefficient mesure alors ce que la variable apporte **en plus** de ce que la cible dit déjà d'elle-même. Une variable simplement procyclique, qui ne ferait que suivre le mouvement, ressort à zéro.

L'horizon retenu est celui où le signal est le plus net. On rejoue ensuite l'estimation sur des milliers d'échantillons rééchantillonnés pour savoir à quel point cet horizon est stable.

## Utilisation

### Trois étapes, dans l'ordre

```python
from diagnostics import screen_panel
from lead_rank import rank_leads

# 1. Cribler — écarter les séries impropres à la régression
crible = screen_panel(panel)
ok = crible[crible.verdict == "stationnaire"].serie.tolist()
ok = [c for c in ok if c != "GDP CYOY Index"]     # ne pas garder la cible

# 2. Classer
R = rank_leads(phases, panel[ok], panel["GDP CYOY Index"],
               grouping="momentum", horizons=range(1, 9), n_boot=400)

# 3. Vérifier le haut du classement
from diagnostics import rapport
from lead_rank import apply_grouping
rapport(panel["LEI YOY Index"], panel["GDP CYOY Index"], 2,
        apply_grouping(phases, "momentum"), "Deceleration")
```

**L'étape 1 n'est pas optionnelle.** Une série qui dérive sans revenir vers une moyenne — un indice boursier, un prix de matière première — produit des corrélations élevées avec n'importe quoi d'autre qui dérive aussi. Sur le panel initial, 25 séries sur 57 étaient dans ce cas, et quatre d'entre elles occupaient le haut du classement avant filtrage. Elles ont toutes disparu une fois converties en rendements.

### L'argument `grouping`

Détermine sur quel sous-échantillon les régressions sont estimées :

| valeur | découpage | à utiliser quand |
|---|---|---|
| `"global"` | tout l'échantillon | vue d'ensemble |
| `"momentum"` | accélération / décélération | **le plus informatif ici** |
| `"niveau"` | sous / au-dessus du potentiel | complément |
| `"phase"` | les 4 phases | trop peu d'observations par cellule |

Plus on découpe, plus on capte la spécificité de chaque régime, moins on a de puissance statistique. Le regroupement par paires est le bon compromis.

### Les autres réglages

- `horizons=range(1,9)` — horizons testés, en trimestres.
- `n_boot=400` — nombre de rééchantillonnages. 150 pour dégrossir, 1000 pour conclure.
- `min_obs=100` — écarte les séries trop courtes.

## Lire la sortie

```
      groupe      variable  avance   beta     t  prob_avance  ipd_bas  ipd_haut  p_max   FDR   n
Deceleration LEI YOY Index       1  1.849  4.42        0.905        1         1 0.0249  True   88
Deceleration OUTFGAF Index       1  1.275  3.47        0.755        1         6 0.0896 False   88
```

| colonne | signification |
|---|---|
| `avance` | horizon retenu, en trimestres |
| `beta` | effet sur la cible d'un écart-type de la variable |
| `t` | rapport signal/bruit du coefficient |
| `prob_avance` | à quel point l'horizon est stable au rééchantillonnage |
| `ipd_bas`–`ipd_haut` | fourchette des horizons plausibles |
| `p_max` | significativité, corrigée du fait qu'on a testé 8 horizons |
| `FDR` | survit à la correction pour le nombre de variables testées |

**Ordre de lecture.** D'abord `n` et le groupe — un résultat sur 88 observations n'a pas le poids d'un résultat sur 300. Puis `p_max` : au-delà de 0,15, la ligne ne dit rien. Puis le **signe de `beta`** : conforme à l'intuition économique ? Puis la largeur de l'intervalle : `[1;1]` est une avance déterminée, `[1;6]` ne l'est pas. Enfin `beta` pour l'ampleur.

Deux réserves. Le test rejette un peu trop facilement — lisez `p_max = 0,05` comme « environ 0,08 réel ». Et une avance de 1 trimestre est rarement exploitable : le PIB paraît un mois après la fin du trimestre et continue d'être révisé.

## Ce que ça a donné

Résultats sur le panel, cible par cible :

| cible | variables retenues | qualité d'ajustement |
|---|---|---|
| **Chômage** (variation 4T) | 8 | forte |
| PIB (glissement annuel) | 1 | moyenne |
| Sentiment entreprises | 1 | faible |
| Taux Fed Funds | 0 | moyenne |
| Sentiment ménages | 0 | faible |
| Consommation | 0 | très faible |

Le chômage est de loin la cible la plus prévisible. La consommation ne l'est pratiquement pas — ce qui est attendu : sous lissage intertemporel, la consommation suit approximativement une marche aléatoire, et sa variation ne devrait pas être prévisible.

Trois motifs récurrents :

- **En décélération**, les indicateurs réels d'activité (LEI, commandes, emploi, permis) anticipent à 1-2 trimestres, avec des ajustements élevés.
- **En accélération**, presque rien n'anticipe à court terme, mais les enquêtes manufacturières ressortent à 6 trimestres **avec un signe négatif** : une activité très forte aujourd'hui précède une dégradation un an et demi plus tard. Signal de retournement par surchauffe, retrouvé sur trois cibles différentes.
- **La pente des taux** ne prédit pas l'activité, mais elle prédit la **politique monétaire** (avance de 4 trimestres, en décélération). Elle anticipe les décisions de la Fed, pas la croissance.

## Limites

Corrélation, pas causalité. Relation supposée stable sur toute la période, ce qui est douteux sur 50 ans. Et la datation des phases provient d'un filtre appliqué en plein échantillon : elle utilise donc une information indisponible en temps réel. Acceptable en descriptif, à refaire en récursif pour toute prétention prédictive.

---

# Partie 2 — Scoring macro

## L'idée

Un score de **0 à 100** résumant l'état du cycle, dans le prolongement du découpage en quatre phases mais en continu.

Deux sous-scores plutôt qu'un seul, parce que le découpage repose sur le croisement de deux dimensions :

- **Niveau** — où en est l'activité par rapport à sa tendance (activité constatée)
- **Momentum** — dans quelle direction elle va (indicateurs avancés)

Le score global les moyenne, mais l'information est dans le couple : Reprise et Ralentissement ont des scores globaux quasi identiques puisque ce sont les deux coins opposés de la grille.

## Comment lire un score

Le score est un **percentile historique**. Un score de 40 signifie : *l'activité est plus faible que dans 60 % des trimestres depuis 1970*.

Conséquence pratique : l'échelle n'est pas linéaire. Un mouvement de 45 à 55 traverse la zone où se concentrent la plupart des trimestres et marque un vrai changement d'état. Un mouvement de 85 à 95 se produit dans la queue de la distribution : plus spectaculaire en apparence, moins significatif.

| score | lecture |
|---|---|
| 85+ | Surchauffe |
| 70-85 | Expansion soutenue |
| 55-70 | Expansion modérée |
| 45-55 | Proche de la tendance |
| 30-45 | Ralentissement |
| 15-30 | Contraction |
| < 15 | Contraction sévère |

**Repères calibrés** sur le découpage en phases (valeur médiane) :

| phase | médiane | fourchette courante |
|---|---|---|
| Décrochage | **15** | 3 – 27 |
| Ralentissement | **49** | 37 – 63 |
| Reprise | **53** | 42 – 70 |
| Explosion | **72** | 59 – 80 |

Au trimestre d'entrée en récession NBER, le score valait 9 (1980), 14 (1981), 12 (1990), 13 (2001), 10 (2020). **Un score sous 15 signale une récession en cours.**

## Utilisation

```python
from cycle_score import build_score_poids_fixes, libelle, decomposer

S = build_score_poids_fixes(panel)
print(S.dropna(subset=["score_global"]).tail(8))
```

Sortie : `score_niveau`, `score_momentum`, `score_global`, plus `couv_*` (part des composantes disponibles).

### Choisir sa pondération

Chaque composante s'écrit `nom: (signe, poids)`. Le signe vaut `+1` si une hausse signale une économie forte, `−1` sinon. Les poids sont renormalisés — seules leurs proportions comptent.

```python
mes_poids = {
    "CFNAI Index":   (+1, 3.0),
    "USURTOT Index": (-1, 2.0),    # chômage : signe négatif
    "IP  YOY Index": (+1, 2.0),
    "NFP TCH Index": (+1, 2.0),
    "NAPMPMI Index": (+1, 1.0),
}
S = build_score_poids_fixes(panel, poids_niveau_spec=mes_poids)
```

Autres réglages : `poids_niveau=0.5` (part du niveau dans le global), `rolling=40` (statistiques glissantes sur 10 ans, indispensable pour un backtest), `couverture_min=0.60`.

### Comprendre un mouvement

```python
decomposer(S, "2023Q1", "niveau")
```

```
               poids  contribution  z_composante
CFNAI Index      0.3       -0.204         -0.68
NAPMPMI Index    0.1       -0.095         -0.95
IP  YOY Index    0.2       -0.075         -0.37
NFP TCH Index    0.2       -0.012         -0.06
USURTOT Index    0.2       +0.310         +1.55
```

Lecture de 2023Q1 : activité dégradée sur toute la ligne, mais le chômage tirait le score vers le haut et compensait presque le reste — la configuration atypique de 2023, ralentissement industriel sans dégradation de l'emploi.

Lisez la colonne `contribution`, pas les pourcentages : quand contributions positives et négatives se compensent, les parts relatives deviennent ininterprétables.

## Points d'attention

**Le panel s'arrête au deuxième trimestre 2024** pour la quasi-totalité des séries. Un garde-fou empêche de publier un score calculé sur trop peu de composantes : les trimestres postérieurs sortent vides plutôt que faux. Dernière valeur fiable : **26,6 en 2024Q2** — contraction.

**Le score ne reproduit pas le découpage en phases** — environ la moitié de concordance. Les Décrochages et Explosions sont bien retrouvés, Reprise et Ralentissement se confondent. C'est normal : le découpage vient d'un filtre sur le PIB, le score d'indicateurs d'activité. Ce sont deux mesures complémentaires, pas deux versions de la même chose.

**Le PIB n'est volontairement pas dans le score.** Il paraît en retard et est révisé pendant des mois ; l'intérêt d'un score composite est justement de situer le cycle sans l'attendre.

---

## Fichiers

| fichier | rôle |
|---|---|
| `lead_rank.py` | classement des indicateurs avancés |
| `diagnostics.py` | crible de fiabilité — **à lancer avant tout classement** |
| `cycle_score.py` | scoring macro |
| `leadlag.py` | variante pour cible binaire (récession oui/non) |
| `bbg_loader.py` | lecture de l'export Bloomberg brut |
| `panel_trimestriel.csv` | panel prêt à l'emploi |
| `classement_*.csv` | résultats par cible |
| `cycle_score_poids_fixes.csv` | historique du score |
