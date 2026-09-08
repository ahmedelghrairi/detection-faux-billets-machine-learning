# Détection de faux billets par machine learning

Projet de classification supervisée réalisé pendant ma formation Data Analyst (OpenClassrooms), sur un scénario de mission pour l'ONCFM, organisme de lutte contre le faux-monnayage. Objectif : à partir de six dimensions géométriques mesurées par une machine, prédire si un billet est vrai ou faux, et livrer une application utilisable par les équipes de terrain.

## Le contexte et le besoin

L'ONCFM fournit 1 500 billets déjà mesurés (1 000 vrais, 500 faux) et un cahier des charges précis : comparer quatre méthodes de prédiction (régression logistique, k-means, KNN, random forest), évaluer les erreurs par matrice de confusion, puis livrer un algorithme dans un script autonome, séparé du notebook d'analyse. Le critère métier est clair : rater le moins de faux billets possible.

## Ce que contient le dépôt

- `notebook_analyse.ipynb` : toute la démarche, de l'exploration au choix du modèle
- `app.py` : l'application de prédiction, utilisable avec un fichier CSV ou en saisie manuelle
- `modele_billets.joblib` : le pipeline complet sauvegardé (imputation, standardisation, modèle, seuil de décision et colonnes attendues)
- `requirements.txt` : les dépendances

## La démarche en bref

Exploration : les faux billets sont en moyenne plus courts, avec une marge basse plus grande. Deux variables, length et margin_low, portent l'essentiel de l'information (81 % de l'importance selon le random forest). Le fichier contient 37 valeurs manquantes sur margin_low ; un test statistique confirme que ces trous ne sont pas liés à la nature du billet, et ils sont complétés par régression linéaire sur les cinq autres mesures.

Modélisation : les quatre algorithmes demandés sont entraînés et comparés sur 300 billets de test jamais vus. Les trois modèles supervisés arrivent à égalité (99 % de bonnes réponses), le k-means, non supervisé, reste en retrait. La validation croisée et des critères pratiques (simplicité, interprétabilité, probabilités exploitables) départagent : régression logistique retenue.

Réglage du seuil : un faux billet manqué coûte plus cher qu'une fausse alerte. Plutôt que le seuil par défaut de 0,5, j'ai cherché le seuil qui minimise ce coût métier : à 0,85, le modèle attrape les 100 faux billets du jeu de test sans en laisser passer un seul, au prix de 5 vérifications manuelles supplémentaires.

## L'application
```
pip install -r requirements.txt
python app.py billets_a_verifier.csv
```

Sans argument, `app.py` passe en saisie manuelle des six mesures. Le script gère les cas réels : séparateur de colonnes variable, virgule décimale, colonne d'identifiants absente, mesures manquantes (complétées par le pipeline), et affiche pour chaque billet la probabilité et le verdict.

## Les données

Les fichiers de billets sont fournis dans le cadre de la formation et ne sont pas publiés. L'application attend un CSV avec six colonnes numériques : diagonal, height_left, height_right, margin_low, margin_up, length, et une colonne id facultative.

## Limites assumées

L'imputation des valeurs manquantes a été ajustée sur l'ensemble du fichier avant la séparation entraînement / test : sur 2,5 % des lignes, l'effet est négligeable, mais je l'ai identifié et documenté dans le notebook, et le pipeline final refait l'imputation proprement. Le seuil de 0,85 repose sur une hypothèse de coût (un faux passé vaut cinq vérifications manuelles) qui devrait être validée avec le métier. Enfin, le modèle ne connaît que les contrefaçons présentes dans le fichier : un nouveau procédé de contrefaçon demanderait un réentraînement.

## Auteur

Ahmed El Ghrairi, Data Analyst à Marseille.
[Portfolio](https://ahmedelghrairi.github.io) · [LinkedIn](https://www.linkedin.com/in/ahmed-el-ghrairi-a581ba177/)
