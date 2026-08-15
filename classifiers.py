"""
Définitions et fonctions d'entraînement des classificateurs.

Implémente tous les 10 classificateurs de l'exemple de comparaison des classificateurs de scikit-learn
avec les hyperparamètres exacts spécifiés dans le référentiel.
"""

from typing import Any, Dict, List

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Graine par défaut : celle de l'exemple de scikit-learn dont ce dépôt s'inspire
DEFAULT_SEED = 42


def get_classifiers(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """
    Obtenir tous les classificateurs avec les hyperparamètres de l'exemple de scikit-learn.

    Seuls les hyperparamètres aléatoires dépendent de la graine ; tous les autres
    (C, gamma, max_depth, alpha…) appartiennent au dispositif constant et restent figés.

    Trois classificateurs sont déterministes et n'acceptent aucune graine : Nearest
    Neighbors, Naive Bayes et QDA. Ce n'est pas un oubli — leur ajustement ne comporte
    aucune part d'aléatoire.

    Args:
        seed: Graine des sept classificateurs stochastiques

    Returns:
        Dictionnaire associant les noms des classificateurs aux objets classificateurs instanciés
    """
    classifiers: Dict[str, Any] = {
        "Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Linear SVM": SVC(kernel="linear", C=0.025, random_state=seed),
        "RBF SVM": SVC(gamma=2, C=1, random_state=seed),
        "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "Random Forest": RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=seed
        ),
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000, random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }
    return classifiers


def get_classifier_names() -> List[str]:
    """
    Obtenir la liste de tous les noms des classificateurs.

    L'ordre est stable et ne dépend pas de la graine : c'est lui qui fixe la position
    de chaque vignette dans la grille.

    Returns:
        Liste de noms des classificateurs dans l'ordre d'affichage
    """
    return list(get_classifiers().keys())
