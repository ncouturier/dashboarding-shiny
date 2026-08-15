"""
Fonctions de génération de datasets pour les exemples de classification.

Fournit des datasets synthétiques (moons, cercles, linéairement séparables) avec
des paramètres configurables pour le bruit, le nombre d'échantillons et la graine.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_circles, make_classification, make_moons


def generate_moons(
    n_samples: int = 300,
    noise: float = 0.3,
    random_state: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Générer le dataset des lunes (deux demi-cercles imbriqués).

    Args:
        n_samples: Nombre d'échantillons à générer
        noise: Écart-type du bruit Gaussien ajouté aux données
        random_state: Graine, pour la reproductibilité

    Returns:
        Tuple de (X, y) où X est la matrice de caractéristiques et y est le vecteur de labels cibles
    """
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)  # type: ignore[no-any-return]


def generate_circles(
    n_samples: int = 300,
    noise: float = 0.3,
    factor: float = 0.5,
    random_state: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Générer le dataset des cercles (grand cercle contenant un petit cercle).

    Args:
        n_samples: Nombre d'échantillons à générer
        noise: Écart-type du bruit Gaussien ajouté aux données
        factor: Facteur d'échelle entre le grand cercle et le petit cercle
        random_state: Graine, pour la reproductibilité

    Returns:
        Tuple de (X, y) où X est la matrice de caractéristiques et y est le vecteur de labels cibles
    """
    return make_circles(  # type: ignore[no-any-return]
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )


def generate_linearly_separable(
    n_samples: int = 300,
    noise: float = 0.3,
    random_state: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Générer le dataset linéairement séparable.

    Args:
        n_samples: Nombre d'échantillons à générer
        noise: Écart-type du bruit Gaussien (affecte la séparation des classes)
        random_state: Graine, pour la reproductibilité

    Returns:
        Tuple de (X, y) où X est la matrice de caractéristiques et y est le vecteur de labels cibles
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=noise * 0.2,  # Convertir le bruit en probabilité de flip
        random_state=random_state,
    )
    return X, y
