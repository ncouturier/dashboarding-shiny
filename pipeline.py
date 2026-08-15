"""
Socle de calcul partagé par les trois implémentations.

Prépare une session — jeu de données et répartition en ensemble d'entraînement et
ensemble de test — puis construit les frontières de décision une vignette à la fois, de
sorte qu'une implémentation puisse les afficher au fil des résultats.

Ce module ne doit importer aucune bibliothèque de dashboarding : il est identique dans
les trois implémentations et c'est cette identité qui rend la comparaison valide.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from classifiers import DEFAULT_SEED, get_classifiers
from datasets import generate_circles, generate_linearly_separable, generate_moons
from visualizations import create_decision_boundary_plot

# Proportion de l'ensemble de test, suivant l'exemple de scikit-learn
TEST_SIZE = 0.4

# Bornes de la graine, identiques aux contrôles des trois interfaces
SEED_MIN = 0
SEED_MAX = 9999

# Résolution de la grille utilisée pour tracer la frontière de décision
GRID_RESOLUTION = 0.02

# Hauteur d'une vignette, en pixels
TILE_HEIGHT = 300


@dataclass(frozen=True)
class Run:
    """
    Une session de calcul : le jeu de données généré, sa répartition, et la graine.

    La graine est conservée parce qu'elle porte au-delà de la génération des données :
    elle détermine aussi l'ajustement des classificateurs stochastiques.

    Attributes:
        X_train: Caractéristiques de l'ensemble d'entraînement
        y_train: Labels de l'ensemble d'entraînement
        X_test: Caractéristiques de l'ensemble de test
        y_test: Labels de l'ensemble de test
        seed: Graine de la session
    """

    X_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int64]
    seed: int


def normalise_seed(seed: Optional[int]) -> int:
    """
    Ramener une graine venue de l'interface dans un intervalle exploitable.

    Un champ numérique vidé transmet None, et les attributs min/max d'un champ HTML ne
    sont que des indications : une valeur absente ou hors bornes atteint le serveur.
    Sans cette garde, une graine absente rendrait la session non déterministe — chaque
    vignette préparerait la sienne et la grille afficherait dix jeux de données
    différents — et une valeur hors bornes ferait échouer scikit-learn.

    Args:
        seed: Graine telle que transmise par l'interface, éventuellement absente

    Returns:
        Une graine comprise entre SEED_MIN et SEED_MAX
    """
    if seed is None:
        return DEFAULT_SEED
    return max(SEED_MIN, min(SEED_MAX, int(seed)))


def prepare_run(
    dataset_type: str, noise: float, n_samples: int, seed: Optional[int]
) -> Run:
    """
    Générer un jeu de données et le répartir en ensembles d'entraînement et de test.

    Args:
        dataset_type: Forme du jeu de données ('moons', 'circles' ou 'linear')
        noise: Écart-type du bruit gaussien
        n_samples: Nombre total d'échantillons
        seed: Graine de la génération du jeu de données et de sa répartition

    Returns:
        La session de calcul correspondante

    Raises:
        ValueError: Si la forme de jeu de données est inconnue
    """
    seed = normalise_seed(seed)

    if dataset_type == "moons":
        X, y = generate_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif dataset_type == "circles":
        X, y = generate_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=seed
        )
    elif dataset_type == "linear":
        X, y = generate_linearly_separable(
            n_samples=n_samples, noise=noise, random_state=seed
        )
    else:
        raise ValueError(f"Forme de jeu de données inconnue: {dataset_type}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )

    return Run(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, seed=seed
    )


def build_decision_boundary(run: Run, classifier_name: str) -> go.Figure:
    """
    Ajuster un classificateur et construire la vignette de sa frontière de décision.

    Args:
        run: La session de calcul préparée par prepare_run
        classifier_name: Nom du classificateur, tel que listé par get_classifier_names

    Returns:
        La figure Plotly de la vignette, ou une vignette de repli si l'ajustement échoue

    Raises:
        ValueError: Si le nom de classificateur est inconnu
    """
    classifiers = get_classifiers(seed=run.seed)
    if classifier_name not in classifiers:
        raise ValueError(f"Classificateur inconnu: {classifier_name}")

    classifier = classifiers[classifier_name]

    try:
        classifier.fit(run.X_train, run.y_train)

        figure = create_decision_boundary_plot(
            classifier=classifier,
            X_train=run.X_train,
            y_train=run.y_train,
            X_test=run.X_test,
            y_test=run.y_test,
            classifier_name=classifier_name,
            resolution=GRID_RESOLUTION,
        )
        figure.update_layout(
            height=TILE_HEIGHT,
            margin=dict(l=40, r=40, t=50, b=40),
            showlegend=False,  # Masquer la légende pour un rendu plus propre
        )
        return figure

    except Exception as erreur:
        return _build_fallback_tile(classifier_name, erreur)


def _build_fallback_tile(classifier_name: str, erreur: Exception) -> go.Figure:
    """
    Construire la vignette affichée lorsqu'un classificateur échoue.

    Args:
        classifier_name: Nom du classificateur en échec
        erreur: L'exception levée

    Returns:
        Une figure Plotly portant le message d'erreur
    """
    figure = go.Figure()
    figure.add_annotation(
        text=f"{classifier_name}<br><br>Erreur: {str(erreur)}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=12, color="red"),
    )
    figure.update_layout(
        title=classifier_name,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=TILE_HEIGHT,
    )
    return figure
