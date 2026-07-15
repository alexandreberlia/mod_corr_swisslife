VAR / VECM MODULE
================

Ce module a pour objectif d'étendre les analyses statistiques et économétriques
réalisées dans la branche Analysis en introduisant des modèles multivariés
de séries temporelles (VAR et VECM).

La philosophie retenue consiste à construire des systèmes économiques cohérents
plutôt que d'estimer des modèles sur plusieurs variantes d'un même indicateur.
Par exemple, un système composé de PIB, inflation, chômage et production
industrielle apportera généralement davantage d'information économique qu'un
système contenant uniquement plusieurs mesures du PIB.

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
