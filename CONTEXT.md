# Exemples de Classification

Application Shiny for Python qui visualise les frontières de décision de dix
classificateurs sur des jeux de données synthétiques en deux dimensions.

Le vocabulaire est en français ; les identifiants du code restent en anglais. Chaque
terme ci-dessous indique son identifiant correspondant.

## Les données

**Jeu de données** (`dataset`) :
Ensemble de points synthétiques en deux dimensions réparti en deux classes. Trois
formes existent : les **lunes**, les **cercles** et le **linéairement séparable**.
_Éviter_ : dataset, données, échantillon

**Bruit** (`noise`) :
Écart-type du bruit gaussien appliqué à la génération d'un jeu de données. Plus il
est élevé, moins les deux classes sont nettement séparées.
_Éviter_ : perturbation, dispersion

**Nombre d'échantillons** (`n_samples`) :
Nombre total de points d'un jeu de données.
_Éviter_ : taille, volume, effectif

**Graine** (`seed`) :
Valeur qui rend une session reproductible de bout en bout : elle détermine le jeu de
données, sa répartition en ensembles, et l'ajustement des classificateurs
stochastiques. Trois des dix classificateurs sont déterministes et ne dépendent
d'aucune graine.
_Éviter_ : seed, germe, aléa

**Ensemble d'entraînement** (`train_set`) :
Partie du jeu de données sur laquelle un classificateur est ajusté.
_Éviter_ : échantillon d'apprentissage, données d'apprentissage

**Ensemble de test** (`test_set`) :
Partie du jeu de données réservée à l'évaluation, jamais vue pendant l'ajustement.
_Éviter_ : échantillon de validation, jeu de validation

## La visualisation

**Classificateur** (`classifier`) :
Un des dix algorithmes de classification comparés, avec ses hyperparamètres fixés.
Les hyperparamètres font partie du dispositif constant : ils ne sont pas réglables.
_Éviter_ : modèle, algorithme, estimateur

**Vignette** (`tile`) :
Emplacement de la grille où est tracée la frontière de décision d'un classificateur.
Il y en a dix, à positions fixes, toutes de même hauteur.
_Éviter_ : tuile, case, cellule, carte, panneau

**Frontière de décision** (`decision_boundary`) :
Partition du plan selon la classe qu'un classificateur prédit en chaque point. C'est
l'objet représenté dans chacune des dix vignettes.
_Éviter_ : figure, graphique, visualisation, tracé

**Score** (`score`) :
Proportion de points de l'ensemble de test correctement classés, affichée dans un
coin de la vignette.
_Éviter_ : exactitude, précision, taux de bonne classification, performance

## Le comportement de l'interface

**Mise à jour continue** (`continuous_update`) :
Contrat d'interaction : toute modification d'un paramètre déclenche le recalcul, sans
validation explicite. Continue, et non instantanée — le recalcul prend un temps
perceptible.
_Éviter_ : temps réel, mise à jour instantanée, réactivité

**Anti-rebond** (`debounce`) :
Délai d'inactivité observé avant de déclencher un recalcul, pour qu'un réglage
parcouru de bout en bout n'en déclenche qu'un seul. Aussi appelé _debounce_.
_Éviter_ : temporisation, throttle, limitation

**Rendu progressif** (`progressive_render`) :
Affichage de chaque vignette dès que son résultat est disponible, plutôt qu'en un
seul bloc. Les positions dans la grille restent fixes : une vignette donnée occupe
toujours le même emplacement.
_Éviter_ : rendu incrémental, affichage différé, streaming

**Rechargement à chaud** (`hot_reload`) :
Prise en compte automatique des modifications du code source pendant le
développement. Concerne l'outillage, jamais l'interface : sans rapport avec la
**mise à jour continue**.
_Éviter_ : temps réel, rechargement automatique, live reload
