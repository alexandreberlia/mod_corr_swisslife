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
