VAR / VECM MODULE
================

Ce module a pour but de modéliser les interactions dynamiques entre plusieurs
variables macroéconomiques à l'aide de modèles VAR (Vector Autoregression)
et VECM (Vector Error Correction Model).

Contrairement aux modèles de régression classiques qui étudient une variable
expliquée par plusieurs variables explicatives, les modèles VAR et VECM
considèrent que l'ensemble des variables du système peuvent s'influencer
mutuellement au cours du temps.

Chaque variable est donc à la fois variable expliquée et variable explicative.

Plusieurs blocs économiques ont été définis afin de faciliter la construction
de systèmes cohérents :

    - Growth Block :
        PIB, production industrielle, ventes au détail, revenu personnel

    - Inflation Block :
        CPI, PPI, PCE, croissance salariale

    - Employment Block :
        Chômage, emploi non agricole, ADP

    - Leading Indicators Block :
        PMI, LEI, Chicago Fed Activity Index

    - Housing Block :
        Permis de construire, ventes immobilières

    - Consumer Block :
        Confiance des consommateurs, consommation, ventes au détail

    - Macro Core Block :
        PIB, inflation, chômage et production industrielle

    - Macro Policy Block :
        PIB, inflation, chômage et taux FED Funds

Le Macro Core Block constitue généralement le meilleur point de départ pour
les analyses VAR/VECM car il représente les principaux moteurs du cycle
économique américain.

L'objectif général de ce module est de fournir un cadre modulaire,
réutilisable et économiquement interprétable pour l'analyse dynamique
des interactions entre les variables macroéconomiques.


POURQUOI LE MACRO CORE BLOCK EST LE MEILLEUR CANDIDAT POUR UN VAR
------------------------------------------------------------------

Le bloc Macro Core contient :

    - GDP US Chained Dollars YoY SA (GDP)
    - US CPI Urban Consumers Less Fo (Inflation)
    - U-3 US Unemployment Rate Total (Employment)
    - US Industrial Production YOY S (Economic Dynamic)

Il représente les quatre principaux piliers de l'économie :

    - Croissance
    - Inflation
    - Emploi
    - Production

Les tests de stationnarité montrent que ces variables sont principalement
stationnaires ou quasi-stationnaires.

Un VAR est particulièrement adapté lorsque les variables sont I(0),
ou peuvent facilement être rendues stationnaires.

Le Macro Core Block est donc le candidat naturel pour :

    - VAR
    - Granger Causality
    - Impulse Response Functions
    - Forecasting
    - Forecast Error Variance Decomposition


POURQUOI UTILISER UN VECM SUR D'AUTRES BLOCS
--------------------------------------------

Le VECM est utilisé lorsque :

    1. Les variables sont I(1)
    2. Les variables sont cointégrées

Autrement dit :

    Variable non stationnaire
            +
      Relation long terme
            =
           VECM

Les résultats des tests de stationnarité montrent notamment que les groupes
suivants contiennent plusieurs séries I(1) :

    - PMI
    - Consumer Confidence
    - Financial Markets
    - Commodities
    - Crypto Assets
    - Sub-indices du S&P500

Les résultats des tests de cointégration indiquent également l'existence
de nombreuses relations de long terme entre certaines de ces variables.

Dans ce contexte :

    VAR
        -> dynamique de court terme

    VECM
        -> dynamique de court terme
           + mécanisme de retour à l'équilibre long terme

Le VECM est donc plus pertinent pour :

    - Financial Markets Block
    - Commodities Block
    - Consumer Confidence Block
    - Sub S&P500 Indicators Block
    - Toute combinaison de séries I(1) présentant une cointégration


INTERPRÉTATION DU TABLEAU DE SÉLECTION DES RETARDS
--------------------------------------------------

La fonction :

    select_optimal_lag()

calcule le nombre optimal de retards selon plusieurs critères.

Exemple :

    Criterion                     Growth Block

    AIC                                  15
    BIC                                  10
    HQIC                                 15
    FPE                                  15

Chaque critère répond à une logique différente.


AIC (Akaike Information Criterion)
----------------------------------

L'AIC privilégie généralement :

    - le pouvoir explicatif
    - les modèles plus riches

Il tend à sélectionner davantage de retards.

Avantage :

    meilleure capacité prédictive.

Inconvénient :

    risque de sur-paramétrisation.


BIC (Bayesian Information Criterion)
------------------------------------

Le BIC pénalise fortement :

    - les paramètres inutiles
    - les modèles trop complexes

Il sélectionne souvent moins de retards.

Avantage :

    modèles plus robustes.

Inconvénient :

    peut parfois sous-estimer la dynamique.


HQIC (Hannan-Quinn Information Criterion)
------------------------------------------

Le HQIC est un compromis entre :

    AIC
        et
    BIC

Il constitue souvent un bon choix intermédiaire.


FPE (Final Prediction Error)
----------------------------

Le FPE mesure directement :

    l'erreur de prévision future.

Il est fréquemment proche de l'AIC.


RÈGLE PRATIQUE
--------------

Pour les analyses académiques :

    BIC

est souvent privilégié.

Pour les modèles prédictifs :

    AIC
    ou
    FPE

sont généralement privilégiés.

Pour la plupart des travaux macroéconomiques :

    HQIC

constitue souvent un compromis intéressant.


L'objectif final est d'obtenir une compréhension dynamique des relations
macroéconomiques et de produire des scénarios cohérents de prévision.



INTERPRÉTATION ET UTILISATION DES MODÈLES VAR / VECM
===================================================

Cette section décrit les principales fonctions du module, leur utilité
économique et la manière d'interpréter leurs résultats.


1. SÉLECTION DU NOMBRE OPTIMAL DE RETARDS


Fonction :

    optimal_lag()

Utilisation :

    optimal_lag(dict_of_df)

Objectif :

    Déterminer le nombre optimal de retards du système.

Sortie :

    Criterion           Optimal Lag

    AIC                    15
    BIC                    10
    HQIC                   12
    FPE                    15


2. ESTIMATION D'UN VAR


Fonction :

    estimate_var()

Utilisation :

    var_model = estimate_var(
        data=macro_core,
        lag=...,
        block_name="Macro Core Block"
    )

Objectif :

    Modéliser les interactions dynamiques entre plusieurs variables
    stationnaires.

Le modèle estimé est :

    Yt = A1Yt-1 + A2Yt-2 + ... + ApYt-p + εt

Chaque variable du système dépend :

    - de ses propres retards
    - des retards des autres variables du système


3. INTERPRÉTATION DES ÉQUATIONS VAR

Fonction :

    display_var_equations(var_model)

Objectif :

    Afficher les coefficients estimés pour chacune des équations du système.

Exemple :

    Variable dépendante :
    GDP US Chained Dollars YoY SA (GDP)

    const                                   0.007
    L1.GDP                                  1.826
    L1.CPI                                 -0.007
    L1.Unemployment                        -0.036
    L1.Industrial Production                0.037
    ...

Lecture :

    L1.X signifie :

        X(t−1)

    c'est-à-dire la valeur observée une période auparavant.

L'équation se lit donc :

    GDP(t)

    =
    0.007
    +
    1.826 × GDP(t−1)
    − 0.007 × CPI(t−1)
    − 0.036 × Unemployment(t−1)
    + 0.037 × Industrial Production(t−1)
    + ...

Interprétation :

    La croissance actuelle du PIB est influencée :

        - par son propre passé ;
        - par l'inflation passée ;
        - par le chômage passé ;
        - par la production industrielle passée.

Dans un VAR :

    chaque variable du système possède sa propre équation.

Les coefficients mesurent uniquement les effets dynamiques de court et
moyen terme.


4. MODELE VECM

Le modèle estimé est :

    ΔYt
    =
    αβ'Yt-1
    +
    Γ1ΔYt-1
    +
    ...
    +
    ΓpΔYt-p
    +
    εt

5. TEST DE JOHANSEN

Fonction :

    johansen_summary()

Utilisation :

    johansen_summary(
        consumer_block,
        k_ar_diff=5
    )

Objectif :

    Déterminer le nombre de relations de cointégration.

Sortie :

    Rank      Trace Statistic

    0             56.2
    1             22.4
    2              7.1

Interprétation :

    Rank = 0
        aucune relation de long terme

    Rank = 1
        une relation de long terme

    Rank = 2
        deux relations de long terme

Le rang estimé est utilisé dans :

    estimate_vecm()

sous la forme :

    coint_rank

6. ESTIMATION D'UN VECM

Fonction :

    estimate_vecm()

Utilisation :

    vecm_model = estimate_vecm(
        data=consumer_block,
        k_ar_diff=...,
        coint_rank=1,
        block_name="Consumer Block"
    )

Objectif :

    Modéliser simultanément :

        - les ajustements de court terme
        - les relations de long terme

Entre des variables non stationnaires.

7. RELATION DE COINTÉGRATION (BETA)

Fonction :

    display_cointegration_relation()

Utilisation :

    display_cointegration_relation(vecm_model)

Objectif :

    Afficher les vecteurs β.

Exemple :

    U-3 Unemployment                     1.000
    Payrolls                             0.030
    ADP                                 -0.013

La relation de long terme estimée est :

    Unemployment
    +
    0.030 Payrolls
    -
    0.013 ADP
    =
    0

Cette équation ne décrit pas les variations d'une période à l'autre.

Elle représente :

    l'équilibre de long terme du système.

L'interprétation économique est :

    lorsque cette relation est rompue,
    le système tend à revenir vers cette relation d'équilibre.

Si plusieurs colonnes apparaissent :

    β1
    β2
    β3

alors plusieurs relations de long terme ont été détectées
par le test de Johansen.


8. TERMES DE CORRECTION D'ERREUR (ALPHA)

Fonction :

    display_error_correction_terms()

Utilisation :

    display_error_correction_terms(vecm_model)

Objectif :

    Identifier quelles variables corrigent les déséquilibres.

Exemple :

    Unemployment      -0.24
    Payrolls           0.01
    ADP               -0.05


Lecture :

Chaque coefficient α mesure la réaction de la variable
à un déséquilibre de long terme.

Exemple :

    Unemployment = -0.24

signifie :

    Lorsque le système s'éloigne de l'équilibre,
    le taux de chômage réduit une partie de cet écart
    à la période suivante à hauteur de 24%.

Règle générale :

    α proche de 0

        →
        la variable participe peu au réajustement.

    |α| élevé

        →
        la variable est fortement impliquée
        dans le retour à l'équilibre.

Signe :

    α négatif

        →
        correction dans le sens du retour à l'équilibre.

    α positif

        →
        ajustement dans le sens opposé.

Les coefficients α sont souvent le résultat économiquement le plus
important d'un VECM.

Ils répondent à la question :

    "Qui corrige les déséquilibres de long terme ?"


9. DYNAMIQUE DE COURT TERME (GAMMA)

Fonction :

    display_gamma_matrices()

Utilisation :

    display_gamma_matrices(vecm_model)

Objectif :

    Étudier la dynamique de court terme.


Exemple :

    Gamma_1

                                Confidence  Michigan  Retail Sales  Consumption

    Confidence                   0.824      0.065      -0.001        0.074

    Michigan                     0.040      0.782       0.020       -0.025

    Retail Sales                 0.012     -0.012       0.850        0.019

    Consumption                  0.006     -0.004       0.016        0.855

Règle de lecture :

    Ligne
        =
        variable expliquée.

    Colonne
        =
        variable explicative retardée.

Première ligne :

    ΔConfidence(t)

    =
    0.824 × ΔConfidence(t−1)
    +
    0.065 × ΔMichigan(t−1)
    −
    0.001 × ΔRetailSales(t−1)
    +
    0.074 × ΔConsumption(t−1)

Interprétation :

    Les variations passées de la confiance des consommateurs
    ont un fort impact sur les variations futures de cette même variable.

Observation typique :

    Des coefficients élevés sur la diagonale indiquent
    une forte inertie propre des séries.

Les matrices Γ décrivent :

    la dynamique de court terme.

Elles répondent à la question :

    "Comment les variations passées influencent-elles
    les variations présentes ?"


10. RÉSUMÉ DE L'INTERPRÉTATION

VAR :

    A_i

        → Effets dynamiques entre variables

VECM :

    β

        → Équilibre de long terme

    α

        → Vitesse de retour vers l'équilibre

    Γ

        → Dynamique de court terme

En pratique :

    VAR :

        "Comment les variables réagissent-elles les unes aux autres ?"

    VECM :

        "Comment les variables réagissent-elles les unes aux autres,
        tout en respectant une relation économique de long terme ?"


TESTS DE STABILITÉ, TESTS PORTMANTEAU ET TESTS DE ARCH
-------------------------------------------------------

Afin de s'assurer de la robustesse et significativité des modèles et de la fiabilité de leurs résulats pour un dataset donné, il est nécessaire de bien vérifier leur stabilité ainsi que la normalité et la non corrélation de leurs résidus. 

1. TEST DE STABILITÉ D'UN VAR
   
   Un modèle VAR est dit stable lorsqu'un choc affectant le système aujourd'hui
voit son impact diminuer progressivement au fil du temps, conduisant à un
retour vers l'équilibre de long terme.

Mathématiquement, la stabilité est assurée lorsque toutes les valeurs propres
de la matrice compagnon sont à l'intérieur du cercle unité :

    |λ| < 1

Cependant, la fonction `roots` de Statsmodels ne renvoie pas directement les
valeurs propres, mais leurs inverses. La condition de stabilité devient donc :

    np.abs(var_results.roots) > 1

Le modèle est considéré comme stable si toutes les racines retournées par
Statsmodels ont un module strictement supérieur à 1.

Exemple :

    Root      Modulus
    5.17      5.17
    2.84      2.84
    1.65      1.65
    1.02      1.02

Toutes les racines étant supérieures à 1, le système est stable.

Un modèle stable permet une interprétation fiable des :

    - fonctions de réponse impulsionnelle (IRF),
    - décompositions de variance (FEVD),
    - prévisions (Forecasting).

À l'inverse, si au moins une racine possède un module inférieur ou égal à 1,
les chocs peuvent ne pas se dissiper correctement, rendant l'interprétation
économique du modèle plus délicate.

2.VALIDATION D'UN VECM

   Contrairement au VAR, la stabilité d'un VECM ne se résume pas à l'étude des
racines du système.

Un VECM est considéré comme correctement spécifié lorsque :

    1. Les variables sont majoritairement I(1).
    2. Le test de Johansen identifie au moins une relation de cointégration.
    3. Le rang de cointégration utilisé dans le VECM est cohérent avec celui
       estimé par le test de Johansen.
    4. Les coefficients de correction d'erreur (alpha) sont présents,
       traduisant un mécanisme de retour à l'équilibre.
    5. Les résidus se comportent comme un bruit blanc
       (Portmanteau/Hosking non rejeté).

Un cas particulier mérite attention :

    Rank = Nombre de variables

Cela indique généralement que toutes les variables sont déjà stationnaires.
Dans cette situation, un VAR est souvent plus approprié qu'un VECM.

La fonction dédiée est :

```python
vecm_portmanteau_test()
```

### Utilisation

```python
consumer_portmanteau = vecm_portmanteau_test(
    vecm_results=consumer_vecm,
    nlags=30,
    k_ar_diff=15
)
```

La fonction :

1. récupère les résidus du VECM ;
2. construit un DataFrame avec les noms des variables ;
3. applique le test multivarié de Hosking ;
4. ajuste les degrés de liberté à partir de l’ordre du VAR sous-jacent.

Le résultat est un dictionnaire contenant notamment :

```python
consumer_portmanteau["pvalue"]
```

---

## 11. Rapport de validation d’un VECM

La fonction :

```python
vecm_validation_report()
```

regroupe les principaux critères nécessaires à la validation d’un VECM.

### Étape 1 : test de Johansen

```python
consumer_johansen = johansen_summary(
    data=consumer_block,
    k_ar_diff=15,
    block_name="Consumer Block"
)
```

### Étape 2 : estimation du VECM

```python
consumer_vecm = estimate_vecm(
    data=consumer_block,
    k_ar_diff=15,
    coint_rank=consumer_johansen["rank"],
    block_name="Consumer Block"
)
```

### Étape 3 : Portmanteau multivarié

```python
consumer_portmanteau = vecm_portmanteau_test(
    vecm_results=consumer_vecm,
    nlags=30,
    k_ar_diff=15
)
```

### Étape 4 : rapport de validation

```python
consumer_validation = vecm_validation_report(
    vecm_results=consumer_vecm,
    johansen_results=consumer_johansen,
    portmanteau_results=consumer_portmanteau,
    alpha_signif=0.05
)
```

### Diagnostics vérifiés

#### Valid Cointegration Rank

Le rang doit satisfaire :

```text
0 < r < K
```

Si :

```text
r = 0
```

aucune relation de cointégration n’est détectée.

Si :

```text
r = K
```

le système est de rang plein et les variables sont déjà stationnaires. Un VAR en niveau est alors généralement plus approprié.

#### Johansen Consistency

Le rang utilisé dans le VECM doit être identique au rang estimé par le test de Johansen.

#### Error Correction Mechanism

Au moins un coefficient d’ajustement `alpha` doit être statistiquement significatif.

Cela montre qu’au moins une variable participe au retour vers l’équilibre de long terme.

#### Residual Whiteness

La p-value du Portmanteau multivarié doit être supérieure au seuil de significativité.

---


3. TEST PORTMANTEAU

L'absence d'autocorrélation des résidus constitue une condition essentielle à la validité d'un modèle VAR ou VECM. Si les résidus restent autocorrélés après l'estimation, cela signifie qu'une partie de la dynamique temporelle des données n'a pas été capturée par le modèle.

### Hypothèse nulle

H0 : les résidus ne sont pas autocorrélés jusqu'au retard h.

Autrement dit :

```text
E(εt ε't-k) = 0
pour tout k = 1,...,h
```

### Statistique de test

Le test repose sur la somme des autocorrélations croisées des résidus.

Q(h) ~ χ²(K²(h-p))
```
avec :

- K = nombre de variables ;
- p = nombre de retards du modèle.

```text
statistique < valeur critique d'une χ²(K²(h-p))
```

↓

```text
On ne rejette pas H0.
```

↓

```text
Les résidus sont assimilables à un bruit blanc.
```

À l'inverse :

```text
statistique > valeur critique d'une χ²(K²(h-p))
```

↓

```text
Des autocorrélations résiduelles persistent.
```


La fonction :

```python
portmanteau_test()
```

utilise directement la méthode `test_whiteness()` de Statsmodels.

### Utilisation

```python
portmanteau_results = portmanteau_test(
    model=production_var,
    nlags=24,
    adjusted=True
)
```

L’hypothèse nulle est :

```text
H0 : absence d’autocorrélation résiduelle jusqu’à l’horizon testé
```

### Interprétation

Si :

```text
P-value > 5 %
```

alors l’hypothèse nulle n’est pas rejetée :

```text
les résidus sont compatibles avec un bruit blanc.
```

Si :

```text
P-value < 5 %
```

alors l’hypothèse nulle est rejetée :

```text
des autocorrélations résiduelles persistent.
```

L’horizon doit respecter :

```text
nlags > p
```

où `p` est l’ordre du VAR.

---

4. TEST DE HOSKING

Le test de Hosking est une version ajustée du test Portmanteau adaptée aux systèmes multivariés.

Le Portmanteau classique sous-pondère les autocorrélations d'ordre élevé puisque seules :

```text
T - h
```

observations sont disponibles pour les estimer. La statistique est ainsi trop petite en moyenne. 

Hosking corrige ce biais en appliquant une pondération spécifique à chaque retard.


### Hypothèse nulle

```text
H0 :
Absence d'autocorrélation résiduelle.
```

### Interprétation

```text
statistique < valeur critique d'une χ²(K²(h-p))
```

↓

```text
Les résidus sont compatibles avec un bruit blanc multivarié.
```

Le test de Hosking est généralement plus robuste que le Portmanteau classique lorsque :

- le nombre de variables est élevé ;
- le nombre de retards est important ;
- l'échantillon est de taille limitée.



La fonction :

```python
hosking_portmanteau()
```

applique un test Portmanteau multivarié directement aux résidus du modèle.

### Pour un VAR

```python
hosking_results = hosking_portmanteau(
    residuals=production_var.resid,
    nlags=24,
    n_model_lags=production_var.k_ar,
    adjusted=True
)
```

### Pour un VECM

Il est préférable d’utiliser le wrapper dédié :

```python
vecm_hosking_results = vecm_portmanteau_test(
    vecm_results=consumer_vecm,
    nlags=24,
    k_ar_diff=15
)
```

Dans le cas d’un VECM :

```text
p = k_ar_diff + 1
```

correspond à l’ordre du VAR sous-jacent.

Par conséquent, il faut respecter :

```text
nlags > k_ar_diff + 1
```

Par exemple, pour :

```text
k_ar_diff = 15
```

utiliser :

```python
vecm_portmanteau_test(
    consumer_vecm,
    nlags=30,
    k_ar_diff=15
)
```

### Résultat retourné

La fonction renvoie un dictionnaire :

```python
{
    "Q": ...,
    "df": ...,
    "pvalue": ...,
    "nlags": ...,
    "n_model_lags": ...,
    "adjusted": ...
}
```

Pour récupérer uniquement la p-value :

```python
hosking_results["pvalue"]
```

### Interprétation

```text
P-value > 5 %
```

indique que les résidus peuvent être assimilés à un bruit blanc multivarié.

```text
P-value < 5 %
```

indique que le modèle laisse subsister une structure temporelle dans les résidus.

---

5. TEST DE LI-MCLEOD

Le test de Li-McLeod applique le principe du Portmanteau aux résidus au carré.


### Hypothèse nulle

```text
H0 :
Absence d'autocorrélation des résidus au carré.
```

### Objectif

Détecter une dépendance dans la volatilité du processus.

Une p-value faible indique souvent :

```text
Présence potentielle d'effets ARCH/GARCH.
```


```python
li_mcleod_portmanteau()
```

applique la correction additive de Li-McLeod au test Portmanteau multivarié.

### Utilisation avec un VAR

```python
li_mcleod_results = li_mcleod_portmanteau(
    residuals=production_var.resid,
    nlags=24,
    n_model_lags=production_var.k_ar
)
```

### Utilisation avec un VECM

```python
li_mcleod_results = li_mcleod_portmanteau(
    residuals=consumer_vecm.resid,
    nlags=30,
    n_model_lags=consumer_vecm.k_ar
)
```

L’hypothèse nulle est :

```text
H0 : absence d’autocorrélation multivariée dans les résidus en niveau
```

### Interprétation

```text
P-value > 5 %
```

indique que les résidus sont compatibles avec un bruit blanc.

```text
P-value < 5 %
```

indique la présence d’autocorrélation résiduelle.

Le test de Li-McLeod sur les niveaux ne doit pas être confondu avec le test de McLeod-Li sur les résidus au carré.

---

6. COMPARAISON DES TESTS PORTMANTEAU
7. 
a fonction :

```python
compare_portmanteau()
```

compare plusieurs versions du test sur les mêmes résidus et avec le même horizon.

### Utilisation

```python
comparison_table = compare_portmanteau(
    model=production_var,
    nlags=24
)
```

Cette fonction compare :

- le Portmanteau brut de Statsmodels ;
- le Portmanteau ajusté de Statsmodels ;
- la version Hosking brute développée dans le module ;
- la version Hosking ajustée ;
- la correction additive de Li-McLeod.

Le tableau produit contient :

```text
Test
Stat
P-value
```

### Contrôle de cohérence

Les implémentations internes et celles de Statsmodels doivent produire des résultats proches lorsqu’elles utilisent :

- les mêmes résidus ;
- le même horizon ;
- les mêmes degrés de liberté ;
- la même correction.

Une différence très importante entre les résultats peut signaler :

- un horizon incorrect ;
- une mauvaise valeur de `n_model_lags` ;
- une différence de correction ;
- un problème d’implémentation.

Le paramètre doit respecter :

```text
nlags > model.k_ar
```


7. TEST ARCH (AUTOREGRESSIVE CONDITIONAL HETEROSKEDASTICITY)

Un modèle peut produire des résidus non autocorrélés tout en présentant une variance dépendante du passé.

Dans ce cas :

```text
εt
```

est un bruit blanc mais :

```text
εt²
```

reste autocorrélé.

### Hypothèses

```text
H0 :
Variance conditionnelle constante
(homoscédasticité)
```

contre :

```text
H1 :
Présence d'effets ARCH
```

### Régression auxiliaire

Le test repose sur :

```text
εt²
=
α0
+
α1 ε²t-1
+
...
+
αq ε²t-q
+
ut
```

La statistique de test est :

```text
LM = T × R²
```

où :

- T est le nombre d'observations ;
- R² est obtenu à partir de la régression auxiliaire.

Sous H0 :

```text
LM ~ χ²(q)
```

### Interprétation

```text
P-value > seuil
```

↓

```text
Absence d'effet ARCH.
```

```text
P-value < seuil
```

↓

```text
Présence d'effets ARCH.
```

Cela signifie que les périodes de forte volatilité tendent à être suivies d'autres périodes de forte volatilité, phénomène connu sous le nom de **volatility clustering**.


La fonction :

```python
arch_test()
```

applique le test ARCH-LM d’Engle séparément à chaque équation du système.

### Pour un VAR

```python
arch_results = arch_test(
    residuals=production_var.resid,
    names=production_var.names,
    nlags=4
)
```

### Pour un VECM

```python
arch_results = arch_test(
    residuals=consumer_vecm.resid,
    names=consumer_vecm.names,
    nlags=4
)
```

L’hypothèse nulle est :

```text
H0 : absence d’effet ARCH
```

c’est-à-dire :

```text
variance conditionnelle constante.
```

### Interprétation

```text
P-value > 5 %
```

indique l’absence d’effet ARCH détectable.

```text
P-value < 5 %
```

indique que la variance des résidus dépend de leur passé.

La colonne :

```text
No ARCH
```

indique directement le résultat du test pour chaque équation.


---

## 8. Test de McLeod-Li sur les résidus au carré

La fonction :

```python
mcleod_li_arch_test()
```

applique un Portmanteau multivarié aux résidus au carré.

### Pour un VAR

```python
mcleod_li_results = mcleod_li_arch_test(
    residuals=production_var.resid,
    nlags=12
)
```

### Pour un VECM

```python
mcleod_li_results = mcleod_li_arch_test(
    residuals=consumer_vecm.resid,
    nlags=12
)
```

Ce test cherche une dépendance temporelle dans :

```text
epsilon_t²
```

et non directement dans :

```text
epsilon_t
```

L’hypothèse nulle est :

```text
H0 : absence d’autocorrélation dans les résidus au carré
```

### Interprétation

```text
P-value > 5 %
```

indique qu’aucune dynamique de volatilité significative n’est détectée.

```text
P-value < 5 %
```

indique la présence potentielle d’effets ARCH ou GARCH.

---

## 9. Test de normalité

La fonction :

```python
normality_test()
```

applique le test multivarié de normalité disponible dans Statsmodels.

### Utilisation

```python
normality_results = normality_test(
    production_var
)
```

L’hypothèse nulle est :

```text
H0 : les résidus suivent une distribution normale multivariée
```

### Interprétation

```text
P-value > 5 %
```

indique que la normalité des résidus n’est pas rejetée.

```text
P-value < 5 %
```

indique que les résidus s’écartent significativement d’une distribution normale.

Le rejet de la normalité est fréquent sur des données macroéconomiques et financières, notamment en présence de :

- crises ;
- valeurs extrêmes ;
- asymétrie ;
- queues épaisses.

Il ne rend pas automatiquement le modèle inutilisable, mais invite à interpréter avec prudence les intervalles de confiance classiques.

---

      


TEST DE CAUSALITÉ DE GRANGER
--------------------------------

Les tests de causalité de Granger permettent d'identifier les relations
prédictives entre les variables d'un système VAR ou VECM.

Important :

    Une causalité de Granger n'est pas une causalité économique.

Elle signifie uniquement que :

    les valeurs passées d'une variable améliorent significativement
    la prévision d'une autre variable.

Autrement dit :

    X Granger-cause Y

signifie :

    les retards de X apportent une information utile
    pour prévoir Y.

1. CAUSALITÉ DE GRANGER DANS UN VAR

Fonction :

    granger_causality_matrix(var_model)

Utilisation :

    granger_results = granger_causality_matrix(
        macro_core_var
    )

Résultat :

    Cause                    Effect
    Industrial Production    GDP
    GDP                      Unemployment
    ...

Chaque ligne teste :

    H0 :

        X ne Granger-cause pas Y

contre :

    H1 :

        X Granger-cause Y

La décision est prise à partir de la p-value.

Si :

    p-value < seuil de significativité

alors :

    H0 est rejetée

et :

    X Granger-cause Y

Exemple :

    Cause:
        Industrial Production

    Effect:
        GDP

    p-value:
        0.003

Interprétation :

    La production industrielle contient une information prédictive
    significative pour expliquer l'évolution future du PIB.


2. CAUSALITÉ DE GRANGER DANS UN VECM

Fonction :

    vecm_granger_matrix(vecm_model)

Utilisation :

    vecm_granger_matrix(
        consumer_vecm,
        significance_level=0.10
    )

Résultat :

                                Retail Sales
    Consumer Confidence           0.003

Lecture :

    Ligne
        =
        variable explicative (Cause)

    Colonne
        =
        variable expliquée (Effect)

    Valeur
        =
        p-value du test de Granger

Dans l'exemple :

    Consumer Confidence → Retail Sales

    p-value = 0.003

Comme :

    0.003 < 0.10

on rejette :

    H0 :
    Consumer Confidence ne Granger-cause pas Retail Sales

Conclusion :

    La confiance des consommateurs améliore significativement
    la prévision des ventes au détail.


3. COMMENT INTERPRÉTER LES P-VALUES

Hypothèse nulle :

    H0 :
    absence de causalité de Granger

Si :

    p-value < α

alors :

    H0 est rejetée

et une relation de Granger est retenue.

Exemple :

    p-value = 0.001

Avec :

    α = 5 %

ou

    α = 10 %

Conclusion :

    relation significative.

Inversement :

    p-value = 0.45

Interprétation :

    les données ne permettent pas d'affirmer
    l'existence d'un contenu prédictif.


4. COMMENT UTILISER LES RÉSULTATS

Les résultats de Granger permettent :

    - d'identifier les variables les plus informatives ;
    - de comprendre la chaîne de transmission économique ;
    - de préparer l'analyse IRF ;
    - de construire des modèles de prévision plus pertinents.

Exemple :

    Retail Sales
            ↓
    Consumption

signifie :

    les ventes au détail aident à prévoir
    la consommation future.

Autre exemple :

    GDP
            ↓
    Unemployment

signifie :

    la croissance apporte une information utile
    pour prévoir l'évolution du chômage.


5. LIMITES DU TEST DE GRANGER

Le test ne démontre pas :

    qu'une variable provoque directement une autre.

Il démontre uniquement :

    qu'une variable contient une information prédictive
    utile pour anticiper une autre variable.

Ainsi :

    Consumer Confidence
            ↓
    Retail Sales

doit être interprété comme :

    La confiance des consommateurs améliore la prévision
    des ventes au détail.

et non comme :

    La confiance des consommateurs est nécessairement
    la cause économique directe des ventes au détail.


IMPULSE RESPONSE FUNCTIONS (IRF)
--------------------------------

Les fonctions de réponse impulsionnelle mesurent
la réaction dynamique des variables du système
suite à un choc exogène sur l'une d'elles.

Question économique :

    Que se passe-t-il lorsqu'une variable
    subit un choc inattendu ?

Exemple :

    Industrial Production
            ↓
          GDP

L'IRF permet de mesurer :

    - la direction du choc
    - son amplitude
    - sa durée
    - sa vitesse de dissipation

Lecture :

    Axe horizontal
        → horizon temporel

    Axe vertical
        → réponse de la variable

Une réponse positive signifie :

    la variable augmente après le choc.

Une réponse négative signifie :

    la variable diminue après le choc.

L'IRF constitue généralement l'outil principal
d'interprétation économique des modèles VAR.


FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)
--------------------------------

La Forecast Error Variance Decomposition (FEVD) permet de mesurer
la contribution relative de chaque choc du système à l'erreur de prévision
d'une variable donnée.

Le FEVD répond à la question :

    D'où proviennent les fluctuations observées ?

ou encore :

    Quelles variables expliquent la variance future
    d'une variable donnée ?


1. ESTIMATION DU FEVD

Fonction :

    compute_fevd()

Utilisation :

    fevd = compute_fevd(
        var_model,
        periods=24
    )

Objectif :

    Décomposer la variance de prévision
    des variables du système à différents horizons.

2. VISUALISATION DU FEVD

Fonction :

    plot_fevd_stacked()

Utilisation :

    plot_fevd_stacked(
        macro_core_var,
        variable,
        periods=24
    )

Résultat :

    Graphique représentant la contribution
    relative de chaque choc à travers le temps.

Lecture :

    Axe horizontal

        →
        horizon de prévision

    Axe vertical

        →
        part expliquée de la variance

Chaque couleur représente un choc différent.


3. TABLEAU FEVD À UN HORIZON DONNÉ

Fonction :

    fevd_horizon()

Utilisation :

    fevd_horizon(
        macro_core_var,
        horizon=12
    )

Exemple :

                            GDP     CPI     UNEMP    INDUS

    GDP                     58       8       7        27

    CPI                     12      74       5         9

    UNEMP                   18       3      69        10

Lecture :

Première ligne :

    GDP

La variance de prévision du PIB à l'horizon 12 est expliquée par :

    58 % → chocs propres du PIB

     8 % → chocs d'inflation

     7 % → chocs de chômage

    27 % → chocs de production industrielle

La somme des contributions est égale à :

    100 %


4. INTERPRÉTATION ÉCONOMIQUE

Supposons :

    GDP

        Industrial Production = 27 %

Interprétation :

    27 % de l'incertitude future du PIB
    provient des chocs de production industrielle.

La production industrielle constitue donc
un moteur important des fluctuations du PIB.

Autre exemple :

    Unemployment

        GDP = 35 %

Interprétation :

    Une part importante des fluctuations futures du chômage
    est expliquée par les chocs de croissance économique.


5. COMPARAISON AVEC LES IRF

IRF :

    Question :

        Que se passe-t-il lorsqu'un choc apparaît ?

Exemple :

    Choc positif sur la production industrielle

        ↓

    Hausse du PIB

        ↓

    Amplitude maximale : +1.35

Le FEVD répond à une question complémentaire :

    Quelle part des fluctuations du PIB
    est attribuable à ce type de choc ?

Exemple :

    Production industrielle

        ↓

    Explique 27 % de la variance du PIB.


6. COMPARAISON AVEC LE TEST DE GRANGER

Granger :

    Industrial Production
                ↓
                GDP

Interprétation :

    La production industrielle améliore la prévision du PIB.

FEVD :

    Industrial Production
                ↓
            27 % du PIB

Interprétation :

    Les chocs de production industrielle expliquent
    27 % des fluctuations futures du PIB.

Ainsi :

    Granger
        mesure l'existence d'un lien prédictif.

    FEVD
        mesure l'importance quantitative du lien.


7. IDENTIFICATION DES VARIABLES DOMINANTES

Le FEVD permet d'identifier les variables
les plus influentes du système.

Une variable est considérée comme dominante lorsqu'elle explique
une part importante de la variance des autres variables.

Exemple :

    Industrial Production explique :

        27 % du PIB

        18 % du chômage

        12 % de l'inflation

Conclusion :

    La production industrielle constitue un facteur central
    dans la dynamique du système macroéconomique.


