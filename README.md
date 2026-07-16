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
"""

"""
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
"""
