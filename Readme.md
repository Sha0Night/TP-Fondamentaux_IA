TP1 — Analyse d’un algorithme en fonctionnement sur le dataset Iris
Ce document synthétise les résultats obtenus lors du TP sur le dataset Iris. L’objectif était d’observer un pipeline de machine learning de bout en bout, de comparer une baseline simple à un modèle plus complexe, de détecter l’overfitting et d’introduire l’explicabilité avec SHAP.
Le jeu de données Iris contient 150 échantillons répartis en 3 classes équilibrées (setosa, versicolor, virginica), avec 4 variables botaniques et aucune valeur manquante. Les variables petal_length et petal_width apparaissent comme les plus discriminantes dès l’exploration initiale.
1. Exploration des données
Le dataset a une forme de 150 lignes et 5 colonnes, avec une distribution parfaitement équilibrée : 50 échantillons par espèce. Aucune valeur manquante n’a été détectée.
L’analyse visuelle montre que setosa est nettement séparée des deux autres classes, tandis que versicolor et virginica présentent un léger chevauchement. Les variables liées aux pétales offrent la séparation la plus nette.

2. Entraînement des modèles
Le split utilisé est de 80 % pour l’entraînement et 20 % pour le test, soit 120 échantillons d’entraînement et 30 de test. Deux modèles ont été entraînés : un arbre de décision limité en profondeur comme baseline, puis une forêt aléatoire comme modèle principal.
Commencer par une baseline simple permet de vérifier rapidement si le problème est déjà bien résolu avec un modèle interprétable et peu coûteux.
3. Résultats de performance
Le Decision Tree baseline obtient une accuracy de 96,7 % et un F1-score pondéré de 0,967, tandis que le Random Forest atteint 93,3 % d’accuracy et un F1-score pondéré de 0,933.
Le rapport détaillé du Random Forest montre une classification parfaite pour setosa, mais une confusion légère entre versicolor et virginica.
Modèle	Accuracy	F1-score pondéré
Decision Tree baseline	96,7 %	0,967
Random Forest	93,3 %	0,933
 
4. Overfitting et tuning
L’étude du nombre d’arbres par validation croisée montre que les performances se stabilisent autour de 50 à 100 arbres, avec peu de gain au-delà. Un choix de 100 arbres constitue donc un bon compromis entre performance et coût de calcul.
L’analyse de la profondeur maximale met en évidence un début d’overfitting à partir d’environ max_depth = 5, moment où le score d’entraînement atteint 100 % alors que le score de test tombe autour de 90 %.
 
5. Explicabilité des variables
Selon l’analyse SHAP, petal_length est la variable la plus déterminante, suivie de petal_width. Ce résultat est cohérent avec le pairplot et avec l’intuition métier.
La comparaison entre l’importance MDI de scikit-learn et SHAP montre une hiérarchie globale similaire, avec une contribution beaucoup plus faible pour sepal_width.
 
 
6. Trois insights clés
Insight 1 — Performance
Observation : sur ce dataset, un modèle simple atteint déjà d’excellents résultats, et le Decision Tree surpasse même légèrement le Random Forest sur le split observé.
Explication : le dataset Iris est simple, équilibré, et les classes sont bien séparées, en particulier setosa.
Implication pratique : il faut toujours commencer par une baseline simple avant de passer à un modèle plus lourd.
Insight 2 — Généralisation
Observation : à partir de certaines profondeurs, le score d’entraînement devient parfait tandis que le score test baisse.
Explication : cela traduit un phénomène d’overfitting, où le modèle apprend trop précisément les exemples vus à l’entraînement.
Implication pratique : il faut limiter la complexité du modèle, surveiller l’écart train/test et utiliser la validation croisée.
Insight 3 — Explicabilité
Observation : les variables de pétales dominent très clairement l’explication du modèle.
Explication : elles capturent les différences morphologiques les plus marquées entre les espèces d’Iris.
Implication pratique : l’explicabilité est essentielle pour vérifier que le modèle apprend des critères pertinents, notamment dans les domaines réglementés.
Conclusion
Ce TP montre qu’un pipeline ML complet peut être analysé avec des outils simples mais rigoureux : exploration, baseline, métriques, tuning et explicabilité. Le principal enseignement est qu’un modèle plus complexe n’est pas toujours meilleur, et qu’il faut équilibrer performance, généralisation et interprétabilité.
