# Indicateurs avancés & scoring macro

Deux outils indépendants, construits sur le même panel de séries trimestrielles américaines.

1. **Modèle d'indicateurs avancés** : classer les variables selon leur capacité à anticiper une cible (PIB, chômage…), et déterminer avec quelle avance.
2. **Scoring macro** : résumer l'état du cycle en un score 0-100, dans le prolongement du découpage en quatre phases.

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

Trois motifs :

- **En décélération**, les indicateurs réels d'activité (LEI, commandes, emploi, permis) anticipent à 1-2 trimestres, avec des ajustements élevés.
- **En accélération**, presque rien n'anticipe à court terme, mais les enquêtes manufacturières ressortent à 6 trimestres **avec un signe négatif** : une activité très forte aujourd'hui précède une dégradation un an et demi plus tard. Signal de retournement par surchauffe, retrouvé sur trois cibles différentes.
- **La pente des taux** ne prédit pas l'activité, mais elle prédit la **politique monétaire** (avance de 4 trimestres, en décélération). Elle anticipe les décisions de la Fed, pas la croissance.

## Limites

Corrélation, pas causalité. Relation supposée stable sur toute la période. Et la datation des phases provient d'un filtre appliqué en plein échantillon : elle utilise donc une information indisponible en temps réel.

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


# La courbe des taux : le résultat qui a demandé trois tentatives

Cette section documente le seul résultat solide obtenu sur les taux d'intérêt, et surtout les deux échecs qui l'ont précédé. Ces échecs ne venaient pas de la variable mais de la **cible** — c'est la leçon à retenir.

---

## Le résultat

**Spread 5 ans − 3 mois, contre l'entrée en récession NBER, données mensuelles :**

| | |
|---|---|
| avance | **12 mois** |
| pseudo-R² de McFadden | **0,17** |
| coefficient | **−0,644** |
| p globale (corrigée des horizons) | **0,010** |
| événements | 12 |

Les quatre conditions d'un résultat exploitable sont réunies : signe correct, avance conforme à la littérature, ajustement dans la fourchette attendue, significativité qui survit à la correction.

Après Bonferroni sur les trois spreads testés, p = 0,03. Et comme le test sur-rejette légèrement (mesuré à 15 % au seuil nominal de 10 %), la lecture prudente est « environ 0,015 ». Le résultat tient dans les deux cas.

### Lecture du coefficient

Le coefficient probit ne s'interprète pas directement : il agit sur un indice latent, pas sur la probabilité. Avec un taux de base de 1,4 % par mois (12 entrées sur ~870 mois) :

| spread 5 ans − 3 mois | P(entrée en récession ce mois) | effet marginal |
|---|---|---|
| +3,0 pt | 0,002 % | −0,005 pp |
| +2,0 pt | 0,024 % | −0,058 pp |
| +1,0 pt | 0,22 % | −0,45 pp |
| 0,0 pt (courbe plate) | 1,4 % | −2,27 pp |
| **−1,0 pt (inversée)** | **5,9 %** | **−7,61 pp** |

Cumulé sur douze mois, à spread constant :

| spread | P(au moins une entrée dans l'année) |
|---|---|
| +2 pt | 0,3 % |
| 0 pt | 15 % |
| **−1 pt** | **52 %** |

Une inversion d'un point fait passer la probabilité de récession à un an de quasi nulle à une chance sur deux.

**L'effet marginal n'est pas constant** — il vaut −0,06 pp à +2 pt de spread et −7,6 pp à −1 pt. C'est la non-linéarité du probit : quand la situation est déjà tendue, une dégradation supplémentaire pèse beaucoup plus lourd.

---

## Les deux échecs qui ont précédé

### Échec 1 — cible : la croissance du PIB

```
T10Y3M   pic +1 trim.   rho = −0,08   p_max = 0,97
```

Rien. Interprétation initiale : la pente ne fonctionne pas sur ces données.

**Interprétation correcte : la pente ne prédit pas un taux de croissance.** Elle prédit des *retournements*. Le PIB en glissement annuel est une variable continue et bruitée ; chercher un lien linéaire avec elle revient à poser la mauvaise question.

Preuve indirecte : contre la variation du Fed Funds, la même pente ressortait avec l'IPD le plus serré du tableau ([3;5] trimestres) et le β le plus élevé. Elle prédit bien quelque chose — la politique monétaire, pas l'activité.

### Échec 2 — cible : toute transition de phase

```
T10Y3M   avance +6   pseudo-R² = 0,009   p = 0,558
```

Toujours rien. Mais cette fois le diagnostic est net :

| cible | événements | avance | pseudo-R² | p |
|---|---|---|---|---|
| toute transition | 40 | +6 | **0,009** | 0,558 |
| entrée en Décrochage seulement | 8 | +3 | **0,261** | 0,018 |
| entrée en récession NBER | 10 | +3 | 0,185 | 0,039 |

Le pseudo-R² est multiplié par **trente** en ciblant les seules transitions vers le bas.

**Pourquoi.** L'indicatrice « toute transition » regroupe les quatre types : Explosion→Ralentissement, Reprise→Explosion, Décrochage→Reprise, Ralentissement→Décrochage. La pente n'en prédit qu'un seul — la dégradation. On lui demandait d'annoncer aussi bien un décrochage qu'une sortie de récession, deux configurations de courbe opposées. Les 8 événements qu'elle sait prédire étaient noyés dans 32 qu'elle ne prédit pas.

C'est le même mécanisme que les risques concurrents sur l'ISM : **un signal qui agit en sens opposés selon la destination se moyenne à zéro quand on agrège.**

### Le piège qui restait — 3 événements

Le pseudo-R² de 0,261 ci-dessus semblait valider l'approche. Il ne valait rien.

Les 8 entrées en Décrochage sont : 1953Q3, 1955Q4, 1969Q4, 1974Q1, 1980Q1, 1990Q3, 2001Q3, 2007Q4. Or `T10Y3M` du panel Bloomberg ne commence qu'en **1982Q1**. Après alignement, il restait **trois** événements : 1990Q3, 2001Q3, 2007Q4.

Un probit à deux paramètres sur trois événements n'est pas une estimation.

| série | début | entrées en Décrochage exploitables |
|---|---|---|
| T10Y3M (panel) | 1982Q1 | **3** |
| NAPMPMI | 1970Q1 | 5 |
| NHSPATOT | 1970Q1 | 5 |
| GS5 − TB3MS (reconstruit) | 1953 | **12** (NBER) |

**Vérification à faire systématiquement avant toute conclusion :**

```python
print(cible_binaire.reindex(serie.dropna().index).sum())
```

---

## Les deux corrections décisives

### 1. Cibler l'événement, pas l'état

Une indicatrice de **niveau** (`USREC` brut, qui vaut 1 pendant toute la récession) déplace le pic d'ajustement au milieu de l'épisode. Pour mesurer une avance, il faut l'indicatrice de **transition** :

```python
from usrec import usrec_entree
y = usrec_entree("M")      # 1 au seul mois d'entrée en récession — 12 événements
```

### 2. Récupérer l'historique long

Le spread reconstruit depuis FRED (`GS5`, `TB3MS`, disponibles depuis 1953) porte **12 événements** au lieu des 3 du panel Bloomberg. C'est le seul geste de toute la démarche qui *multiplie* l'information disponible plutôt que de la redistribuer.

Bénéfice supplémentaire : le NBER est une cible **externe**, indépendante du filtre de Hamilton dont dérivent les phases. Toute validation menée contre la datation maison est partiellement circulaire.

---

## Reproduire

```python
import pandas as pd
from usrec import usrec_entree
from leadlag import leadlag_probit
from diagnostics import screen_panel

# 1. Séries FRED (GS5, GS10, GS2, TB3MS), mensuelles
taux = pd.read_csv("taux_fred.csv", index_col=0, parse_dates=True)
taux.index = pd.PeriodIndex(taux.index, freq="M")

# 2. Spreads — jamais les niveaux bruts, qui sont intégrés
taux["spread_5y_ff"] = taux["GS5"] - taux["TB3MS"]
taux["pente_2_5"]    = taux["GS5"] - taux["GS2"]
taux["papillon"]     = 2*taux["GS5"] - taux["GS2"] - taux["GS10"]

# 3. Crible de stationnarité
ok = screen_panel(taux).query("verdict == 'stationnaire'").serie.tolist()

# 4. Cible : entrée en récession
y = usrec_entree("M")

# 5. Test
for c in ok:
    n_ev = int(y.reindex(taux[c].dropna().index).sum())
    r = leadlag_probit(taux[c], y, max_lag=24, min_lag=3, n_boot=200)
    j = list(r.lags).index(r.best_lag)
    print(f"{c:16s} n_ev={n_ev:2d}  avance={r.best_lag:3d} mois  "
          f"R2={r.best_stat:.3f}  p={r.p_global:.3f}  "
          f"coef={r.detail['coef'][j]:+.4f}")
```

`min_lag=3` élimine les décalages inexploitables une fois le délai de publication déduit. `max_lag=24` couvre la fourchette attendue de 12 à 18 mois.

---

## Ce qui reste à vérifier

**La stabilité post-2008.** L'assouplissement quantitatif a comprimé la prime de terme, et l'inversion de 2022-2024 n'a été suivie d'aucune récession à ce jour. Réestimez sur 1953-2007 et comparez : si le coefficient s'affaiblit nettement en incluant la période récente, le signal est sous tension. C'est le débat en cours dans la littérature, et cela conditionne l'usage qu'on peut en faire aujourd'hui.

Deux correctifs existent si c'est le cas : soustraire la prime de terme estimée (Adrian-Crump-Moench) pour ne garder que la composante anticipations, ou passer au *near-term forward spread* d'Engstrom-Sharpe (2018).

**L'intégration au modèle de durée.** Le spread devient une covariable :

```python
ex = pd.DataFrame({"spread5": spread.shift(4)})    # 12 mois = 4 trimestres
m = CycleModel(mutualise=True).fit(phases["phase"], exog=ex)
m.calibrate(m.backtest(phases["phase"], debut="1970Q1", H=4, exog=ex))
```

**Le critère de succès est chiffré** : le plafond de calibration doit dépasser **44 %**, valeur mesurée sans covariable. Si le spread apporte réellement de la discrimination, la tranche haute doit monter. S'il ne bouge pas malgré une significativité au probit, le problème n'est pas le choix de covariable mais le nombre de transitions — et il faudra basculer le modèle de durée lui-même sur les récessions NBER plutôt que sur les phases maison.

---

## Ce que cet épisode enseigne

Trois principes, applicables au-delà des taux.

**La cible fait le résultat.** La même variable donne p = 0,97, p = 0,558 ou p = 0,010 selon ce qu'on lui demande de prédire. Avant de conclure qu'une variable ne fonctionne pas, vérifier qu'on lui pose la bonne question.

**Agréger des événements hétérogènes détruit le signal.** Quatre types de transition mélangés, un signal qui n'en prédit qu'un : le pseudo-R² tombe de 0,26 à 0,009. Le même mécanisme rendait l'ISM invisible dans le hasard agrégé.

**Compter les événements avant de lire les p-values.** Un pseudo-R² de 0,26 sur trois événements ne vaut rien, et rien dans la sortie ne le signale. L'effectif effectif après alignement est le premier chiffre à vérifier, pas le dernier.

# Faut-il prévoir le score ? Non — et voici pourquoi

Le module `prevision.py` contient une fonction `forecast_score` qui projette le score d'activité à un horizon donné, avec intervalle de prévision. Elle fonctionne techniquement. Elle n'est pas utile.

Cette note documente les trois mesures qui mènent à cette conclusion, et ce qu'il faut faire à la place.

---

## Le raisonnement qui semblait naturel

On dispose d'un score 0-100 qui résume l'état du cycle. Il paraît logique de vouloir sa valeur future : *« le score sera-t-il à 30 ou à 60 dans un an ? »*

L'ajustement en échantillon donne d'ailleurs des chiffres encourageants :

| horizon | R² |
|---|---|
| 1 trimestre | 0,79 |
| 2 trimestres | 0,53 |
| 3 trimestres | 0,30 |
| 4 trimestres | 0,11 |

Le problème est que ces R² ne mesurent presque rien : à l'horizon 1, le score futur ressemble au score présent, et n'importe quel modèle qui recopie la valeur courante obtient 0,79.

---

## Mesure 1 — le plancher de bruit

Le score n'est pas une quantité observée, c'est une construction. En retirant au hasard une composante de chaque bloc et en recalculant tout, on obtient sa **précision intrinsèque** :

| | |
|---|---|
| écart-type du score (bruit de composition) | **3,8 pt** |
| erreur absolue minimale atteignable | **4,3 pt** |

Ce second chiffre est le résultat décisif. Même avec une prévision **parfaite** de la « vraie » valeur, l'écart mesuré avec le score effectivement calculé serait de 4,3 points en moyenne — parce que la cible elle-même est bruitée.

Autrement dit : **un score de 40 est en réalité 40 ± 6**, et prévoir sa valeur à l'unité près n'a pas de sens.

---

## Mesure 2 — le gain sur la persistance, une fois le plancher déduit

Backtest à fenêtre croissante, réestimation à chaque date :

| horizon | erreur modèle | erreur naïve | plancher | **gain réel** |
|---|---|---|---|---|
| 1 trim. | 8,4 | 8,6 | 4,3 | **4 %** |
| 2 trim. | 12,8 | 13,4 | 4,3 | **7 %** |
| 4 trim. | 17,0 | 20,3 | 4,3 | **21 %** |

La référence naïve est « le score ne bouge pas ». Le gain brut semble modeste ; le gain *réel* — après avoir retiré la part d'erreur qu'aucun modèle ne peut éliminer — est de 4 à 7 % aux horizons courts.

Un modèle qui améliore la persistance de 4 % ne justifie pas d'exister.

Le gain de 21 % à quatre trimestres est plus substantiel, mais il porte sur une erreur de 17 points sur une échelle de 100 : la prévision est alors trop imprécise pour distinguer un ralentissement d'une expansion.

---

## Mesure 3 — prévoir le score dégrade l'information

Le test qui tranche. Question identique : *« y aura-t-il une récession dans k trimestres ? »*, résolue de deux façons.

| horizon | **prévoir** le score, puis le lire | **utiliser** le score actuel |
|---|---|---|
| 1 trim. | AUC 0,824 | AUC 0,826 |
| 2 trim. | AUC 0,661 | **AUC 0,704** |
| 3 trim. | AUC 0,530 | **AUC 0,603** |
| 4 trim. | AUC 0,366 | **AUC 0,502** |

**La voie indirecte est systématiquement moins bonne**, et l'écart se creuse avec l'horizon. À quatre trimestres, prévoir le score donne une AUC de 0,366 — *pire que le hasard*, c'est-à-dire que la prévision inverse le signal.

L'explication est simple. Le score courant contient déjà l'information avancée : c'est précisément ce que fait le bloc « avancé ». Le prévoir revient à faire passer cette information par un modèle autorégressif, qui la lisse vers la moyenne et détruit ce qui la rendait utile.

---

## Ce qu'il faut faire à la place

### Pour situer le présent

Lisez le score tel quel, **par tranche**, jamais à l'unité.

```python
from cycle_score import build_score_3blocs, libelle
S = build_score_3blocs(panel)
print(S.score_global.dropna().tail())
```

| score | lecture |
|---|---|
| < 15 | contraction sévère — récession en cours |
| 15 – 30 | contraction |
| 30 – 45 | ralentissement |
| 45 – 55 | proche de la tendance |
| 55 – 70 | expansion modérée |
| > 70 | expansion soutenue |

Un passage de 44 à 52 n'est pas un signal : c'est du bruit de composition.

### Pour regarder devant

Utilisez le **bloc avancé seul**, avec `poids_globaux=(1, 0, 0)`. Au-delà d'un trimestre, il domine tout mélange incluant le coïncident :

| horizon | score global | bloc avancé |
|---|---|---|
| 1 | 0,793 | **0,843** |
| 2 | 0,668 | **0,735** |
| 3 | 0,563 | **0,633** |

### Pour prévoir un changement de régime

Ne passez pas par le score. Modélisez directement l'événement :

```python
from cycle_model import CycleModel
m = CycleModel(mutualise=True).fit(phases["phase"], exog=ex)
m.calibrate(m.backtest(phases["phase"], debut="1985-01", H=12, exog=ex))
print(m.explain(H=12, x={"ism": 48.7}))
```

Le modèle de durée répond à la bonne question — *quand cela va-t-il changer, et vers quoi* — sans passer par un intermédiaire continu qui perd de l'information.

---

## Le principe général

**Prévoir un indicateur avancé est un contresens.** Un indicateur avancé vaut par ce qu'il dit du futur ; le projeter dans le futur revient à demander à un modèle statistique de faire le travail que l'indicateur faisait déjà, en moins bien.

La règle : si l'on veut savoir quelque chose sur $t+k$, on régresse directement cette chose sur l'information disponible en $t$. On n'interpose pas une variable intermédiaire qu'il faudrait elle-même prévoir.

C'est exactement l'argument des prévisions **directes** contre les prévisions **itérées** (Marcellino-Stock-Watson, 2006) : chaque étape intermédiaire ajoute son erreur de spécification sans rien apporter.

---

## Faut-il supprimer `forecast_score` ?

Non, elle garde deux usages.

**Mesurer la persistance du score.** Le R² de 0,79 à un trimestre quantifie à quel point le cycle est inerte — information utile en soi.

**Fabriquer un banc d'essai.** Toute méthode prétendant prévoir l'état du cycle doit battre cette référence. Elle sert de plancher, pas d'outil.

Ce qu'il ne faut pas faire : publier « le score sera à 47 dans un an » comme un résultat. L'intervalle à 80 % de cette prévision couvre [18 ; 73], soit plus de la moitié de l'échelle.

# Le score et le S&P 500 : pourquoi ça ne marche pas

Question testée : le score d'activité peut-il servir de signal sur les actions ? Par exemple *« score > 80 → perspective haussière de x % sur 3 trimestres »*.

**Réponse : non.** Trois mesures, et une nuance qui sauve une lecture descriptive.

---

## 1. La relation existe, mais dans l'autre sens

Rendement du S&P à 3 trimestres, par quartile de score (n = 212) :

| quartile de score | rendement moyen | médiane |
|---|---|---|
| **Q1 (score bas)** | **+10,3 %** | +11,5 % |
| Q2 | +8,0 % | +9,3 % |
| Q3 | +5,2 % | +5,7 % |
| **Q4 (score haut)** | **+4,2 %** | +5,4 % |

La relation est monotone et **inverse de l'intuition** : un score élevé n'annonce pas une hausse, mais des rendements ultérieurs plus faibles. Le motif tient à tous les horizons (+3,4 % contre +1,1 % à 1 trimestre ; +12,9 % contre +4,8 % à 4).

Ce n'est pas une anomalie, c'est le résultat standard sur les **primes de risque contracycliques**

---

## 2. L'écart n'est pas significatif

Écart Q1 − Q4 = +6,0 points. **p = 0,180** par rotation circulaire — méthode qui casse le lien tout en préservant l'autocorrélation.

Les *t* de 2,3 à 4,3 qu'un test naïf produirait sont trompeurs : les rendements à 3 trimestres **se chevauchent**, donc les 212 observations ne portent pas 212 informations indépendantes.

Deuxième indice dans le même sens : le bloc coïncident fait aussi bien que le bloc avancé (+7,1 contre +6,1 points d'écart). Ce n'est donc pas de l'anticipation, mais une association contemporaine avec le niveau de prime.

---

## 3. Optimiser les poids sur le S&P aggrave les choses

Le test décisif. Pondération des trois blocs optimisée sur 1971-1997, appliquée telle quelle à 1998-2023 :

| | in-sample | **hors échantillon** |
|---|---|---|
| poids optimisés sur le S&P | −0,295 | **−0,144** |
| score par défaut, non optimisé | −0,250 | **−0,172** |

L'optimisation perd la moitié de son avantage, et **fait moins bien que le score non optimisé**. Les poids appris captent le bruit de la première période, pas une relation stable. Surajustement caractérisé.

---

## Pourquoi c'était prévisible

Le S&P est le **seul objet négociable** du panel. Toute prévisibilité identifiée serait immédiatement exploitée et arbitrée : si l'on savait qu'un score bas annonce +10 % dans neuf mois, les prix intégreraient cette information aujourd'hui et l'écart disparaîtrait. C'est Samuelson-Fama.

Le chômage, lui, n'est pas négociable. Personne ne peut arbitrer sa prévisibilité, donc elle persiste — d'où les 8 variables au FDR et les R² de 0,61 à 0,84 obtenus sur cette cible.

Le résultat converge avec le classement d'indicateurs mené séparément : sur 19 variables testées contre le rendement du S&P, **zéro** ne passait le FDR. Tous les coefficients significatifs étaient négatifs (ISM à −5,98 avec 4 trimestres d'avance, Chicago PMI à −5,03) — le même motif de prime contracyclique, et la même absence de significativité après correction.

---

## La piste inverse

Plutôt que de faire prédire le S&P par le score, **intégrer le S&P au score**.

Son rendement sur 12 mois est stationnaire, couvre 1949-2026 dans le fichier `spx500.csv` — soit bien plus que le panel Bloomberg — et les marchés actions figurent dans la plupart des indices avancés composites, y compris le LEI du Conference Board.

```python
mon_bloc_avance = {
    "NAPMPMI Index": (+1, 2.5, None),
    "spx_rdt_12m":   (+1, 2.0, None),   # rendement, jamais le niveau
    "OUTFGAF Index": (+1, 1.5, None),
}
S = build_score_3blocs(panel, avance=mon_bloc_avance)
```

Le S&P est un bon **indicateur avancé** de l'économie réelle — il regarde en avant par nature. Il est en revanche une mauvaise **cible**, pour la raison qui fait précisément sa qualité d'indicateur : il intègre déjà toute l'information disponible.
