# Présentation du projet

Nous avons disposé d'une base de données de Saint-Gobain, qui recense les caractéristiques d'un matériaux lors de certains tests. Le projet consiste à créer des modèles de régression et des réseaux de neurones afin de prédire certains résultats, tout en jouant un rôle de vulgarisateur pour que des chercheurs en physique puisse utiliser l'outil. Notre code doit donc être facilement adaptable à une autre base de donnnées et proposer une interface utilisateur intuitive.

# Déroulé du projet
Nous avons commencé par faire le pre-processing des données : étant réparties sur trois feuilles de calcul différentes, il a fallu trouver un moyen de join les dataframes, pour constituer une dataframe globale utilisable pour l'ensemble des opérations.  
Après avoir centré et normalisé les donnés numériques, et transformé les donnés textuelles en binaire, nous avons commencé nos modèles. Nous avons codé :
- Une **ACP**, pour étudier l'influence des différentes caractéristiques
- Une **régression linéaire**
- Une **régression polynomiale**
- Un **random forest**
- Un modèle utilisant le **gradient boosting**

# Structure des fichiers
Tout d'abord, nous avons commencé par pre-process les données (join, normaliser (MinMax) (on a ensuite changé en StandardScaler)).

Il y a eu deux étapes pour les modèles.  
Tout d'abord, il a fallu fit des modèles sur trois simulations dont on nous a donnés les paramètres. Cela correspond aux fichiers :
- ACP, qui présente l'ACP de la première simulation ainsi que la régression polynommiales sur les trois simulations
- AL_model_1/2/3 qui présente les régressions linéaires des trois simulations

Puis nous avons construit l'interface graphique et des fonctions qui fit les différents modèles (toujours les régressions, mais aussi random forest et XGboost) selon les paramètres d'entrée et de sortie données par l'utilisateur.

# Difficultés et solutions

Pour des soucis de confidentialité, les données de Saint-Gobain ont été anonymisées (on ne sait pas à quoi elles correspondent : les noms ne sont pas explicites). Ainsi, il est difficile de faire des interprétations, et notamment lorsque l'on a des résultats étranges, il n'était pas possible d'utiliser notre sens physique pour expliquer la cohérence (ou non-cohérence) de ces derniers.  

De plus, lors du pre-processing des données, il a été difficile de trouver les colonnes sur lesquelles join les feuilles. Il n'y a en effet aucune colonne d'identification commune aux quatre feuilles, à part les essais, mais ceux-ci ne sont pas uniques : il peut y avoir un essai XXX dans la feuille P, pour plus d'une centaine dans une autre feuille. En faisant quelques hypothèses raisonnables sur les données (la feuille P correspond aux conditions de test, ce qui explique l'unique ligne par essai), et en traitant quelques données (la feuille S est join aux autres en faisant la moyenne des S1 sur un essai), on a quand même pu s'en sortir.

Lors du développement des différents modèles, on s'est également confrontés au problème des valeurs manquantes (plus de 80% de données manquantes pour certaines colonnes). Hors, pour les modèles, on ne peut utiliser les lignes qui ont une valeur manquante. Au départ, on a donc utilisé seulement les lignes où il n'y avait pas de données manquantes (en excluant P1_réel, car c'était la variable avec le plus de données manquantes et avec l'ACP, on a vu que la contribution de cette variable était minime). On a donc dû se contenter de quelques centaines de lignes au lieu de quelques milliers. Après discussion avec l'encadrante, nous avons pu reconstruire certaines valeurs manquantes, en utilisant l'indépendance des variables entre elles, ce qui a grandement augmenté les performances de nos différents modèles.

Après avoir fix ce problème, on a eu des résultats très impressionnants (r2 supérieurs à 0.999, rmse inférieur à 0.1). On s'est rendus compte que, en scindant les données en jeu de test et en jeu d'entraînement, on n'avait pas pris en compte que les essais apparaissaient plusieurs fois (à cause des différentes mesures dans la pratique) : ainsi, le même essai (avec des valeurs un peu différentes, certes, mais quand même fortement corrélées) se retrouve à la fois dans les données de test et dans les données d'entraînement. Une fois que cela a été changé en faisant un group by sur les essais, on a eu des résultats bien moins bons, mais beaucoup plus cohérents.