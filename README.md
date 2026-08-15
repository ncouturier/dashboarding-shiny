# Exemples de Classification (Shiny for Python)

Application Shiny for Python interactive pour explorer les algorithmes de classification automatique.

## Présentation

**Exemples de Classification** est une application web éducative qui permet aux utilisateurs de visualiser et comparer différents algorithmes de classification. Construite avec Shiny for Python et scikit-learn, elle offre un environnement interactif pour comprendre comment les classificateurs créent des frontières de décision sur divers jeux de données synthétiques.

### Fonctionnalités

- **10 algorithmes de classification** : Découvrez les plus proches voisins, SVM linéaire, SVM RBF, processus gaussien, arbre de décision, forêt aléatoire, réseau de neurones, AdaBoost, Naive Bayes et QDA.
- **3 jeux de données synthétiques** : Lunes, cercles et données linéairement séparables avec des paramètres configurables.
- **Contrôles interactifs** : Ajustez le niveau de bruit, le nombre d'échantillons et la graine.
- **Mise à jour continue** : Les frontières de décision se recalculent au fil de vos ajustements, sans validation, à positions fixes.
- **Rendu progressif** : chaque vignette s'affiche dès que son résultat est prêt, un indicateur d'attente occupant sa place entretemps. Shiny rassemblant les valeurs de toutes les sorties d'un même cycle dans un unique message, ce résultat demande une tâche étendue par vignette — là où Dash y parvient avec un callback par vignette et Streamlit avec des emplacements réservés.
- **Score par classificateur** : Chaque vignette porte la proportion de points de l'ensemble de test correctement classés.
- **Réactivité puissante** : Grâce au système de calculs réactifs de Shiny for Python.
- **Objectif pédagogique** : Comprenez le comportement des algorithmes grâce à des expérimentations pratiques.

## Utilisation

1. **Sélectionner un jeu de données** : Choisissez entre Lunes, Cercles ou Séparable linéairement
2. **Ajuster les paramètres** :
   - Niveau de bruit : Contrôle la difficulté de séparation des classes
   - Nombre d'échantillons : Nombre de points à générer
   - Graine : Détermine le jeu de données, sa répartition en ensembles et
     l'ajustement des sept classificateurs stochastiques ; les trois classificateurs
     déterministes n'en dépendent pas
3. **Observer** : Les visualisations se mettent à jour d'elles-mêmes au fil de vos ajustements
4. **Voir les résultats** : Observez comment chaque classificateur crée des frontières de décision
5. **Expérimenter** : Testez différentes combinaisons pour comprendre le comportement des algorithmes

## Déploiement sur Posit Connect Cloud

L'application est publiée sur [Posit Connect Cloud](https://connect.posit.cloud), qui
déploie directement depuis ce dépôt GitHub. Aucun conteneur n'est nécessaire : Connect
Cloud reconnaît nativement Shiny for Python.

Trois conditions, toutes remplies par ce dépôt :

| Condition | État |
| --- | --- |
| Dépôt GitHub public | `ncouturier/dashboarding-shiny` |
| Fichier principal | `app.py` à la racine |
| `requirements.txt` à la racine | engendré depuis `uv.lock` par l'intégration continue |

À la création du contenu, **choisir Python 3.13** dans la liste déroulante : la version
par défaut est 3.11, alors que `pyproject.toml` exige `>=3.13`.

### requirements.txt

Ce fichier n'est jamais modifié à la main. Le flux
[`.github/workflows/requirements.yml`](.github/workflows/requirements.yml) le
réengendre depuis `uv.lock` à chaque poussée sur `main` et le publie s'il a changé.
`uv.lock` reste donc la seule source des dépendances.

Pour le reconstruire localement :

```bash
uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt
```

## Développement local

```bash
# Assurez-vous d'avoir uv installé
# Installation : https://docs.astral.sh/uv/

# Synchroniser les dépendances
uv sync

# Lancer l'application
uv run shiny run app.py

# Accéder à http://localhost:8000
```

### Avec Docker

```bash
docker build -t exemples-classification .
docker run --rm -p 8000:8000 exemples-classification
```

## Flux de développement

```bash
# Tests
uv run pytest

# Vérification de types
uv run mypy .

# Formatage
uv run black .
uv run isort .

# Dépendances de développement
uv sync --all-extras
```

## Structure du projet

```
app.py                 # Application Shiny for Python principale
classifiers.py         # Définitions des classificateurs avec hyperparamètres
datasets.py            # Fonctions de génération de jeux de données
pipeline.py            # Socle de calcul, sans dépendance de dashboarding
visualizations.py      # Fonctions de visualisation Plotly
tests/                 # Suite de tests
www/styles.css         # Styles CSS personnalisés
requirements.txt       # Engendré depuis uv.lock, lu par Connect Cloud
Dockerfile             # Exécution locale en conteneur, et repli vers un autre hébergeur
.github/workflows/     # Intégration continue
CONTEXT.md             # Glossaire du domaine
docs/adr/              # Décisions d'architecture
```

## Configuration

- **Python 3.13+**, déclaré dans `pyproject.toml` (`requires-python = ">=3.13"`)
- Dépendances principales : **shiny**, **plotly**, **scikit-learn**, **numpy**, **pandas**, **shinyswatch**
- Dépendances de développement : **pytest**, **pytest-asyncio**, **mypy**, **black**, **isort**

## Différences avec les versions Dash et Streamlit

- **Framework** : Shiny for Python, avec un modèle de programmation réactive
- **Port** : port 8000 (au lieu de 8050 pour Dash ou 8501 pour Streamlit)
- **Réactivité** : calculs réactifs avec `@reactive.calc` au lieu de callbacks (Dash) ou top-down (Streamlit)
- **Identifiants** : underscores dans les identifiants (`dataset_selector`) au lieu de tirets
- **CSS** : fichiers statiques dans `www/` au lieu de `assets/` (Dash) ou `.streamlit/` (Streamlit)

## Origine

Ce dépôt est un embranchement de l'implémentation `shiny/` du dépôt de comparaison
« Exemples de Classification », qui confronte Dash, Shiny for Python et Streamlit sur
une tâche identique. Il n'est plus soumis à la parité stricte qui gouverne ce dépôt —
voir [`docs/adr/0001-embranchement-du-depot-de-comparaison.md`](docs/adr/0001-embranchement-du-depot-de-comparaison.md).

## Remerciements

Inspiré de l'[exemple de comparaison de classificateurs scikit-learn](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).
