"""Tests pour les fonctions de classificateurs."""

import pytest
from sklearn.base import BaseEstimator

from classifiers import get_classifier_names, get_classifiers


def test_get_classifiers_returns_dict() -> None:
    """Teste que get_classifiers renvoie un dictionnaire."""
    classifiers = get_classifiers()
    assert isinstance(classifiers, dict)


def test_get_classifiers_count() -> None:
    """Teste que get_classifiers renvoie exactement 10 classificateurs."""
    classifiers = get_classifiers()
    assert len(classifiers) == 10


def test_get_classifiers_all_estimators() -> None:
    """Teste que tous les classificateurs sont des estimateurs de scikit-learn."""
    classifiers = get_classifiers()
    for name, clf in classifiers.items():
        assert isinstance(clf, BaseEstimator), f"{name} is not a BaseEstimator"


def test_get_classifier_names_count() -> None:
    """Teste que get_classifier_names renvoie 10 noms."""
    names = get_classifier_names()
    assert len(names) == 10


def test_get_classifier_names_match_classifiers() -> None:
    """Teste que les noms des classificateurs correspondent aux clés du dictionnaire des classificateurs."""
    names = get_classifier_names()
    classifiers = get_classifiers()
    assert set(names) == set(classifiers.keys())


def test_get_classifiers_graine_par_defaut() -> None:
    """Teste que la graine par défaut reste 42."""
    for classifier in get_classifiers().values():
        parametres = classifier.get_params()
        if "random_state" in parametres:
            assert parametres["random_state"] == 42


def test_get_classifiers_propage_la_graine() -> None:
    """Teste que la graine atteint les classificateurs qui l'acceptent."""
    classifiers = get_classifiers(seed=7)

    stochastiques = [
        nom
        for nom, classifier in classifiers.items()
        if "random_state" in classifier.get_params()
    ]
    assert len(stochastiques) == 7

    for nom in stochastiques:
        assert classifiers[nom].get_params()["random_state"] == 7, nom


def test_get_classifiers_trois_deterministes() -> None:
    """Teste que trois classificateurs n'acceptent aucune graine."""
    classifiers = get_classifiers(seed=7)

    deterministes = [
        nom
        for nom, classifier in classifiers.items()
        if "random_state" not in classifier.get_params()
    ]
    assert sorted(deterministes) == ["Naive Bayes", "Nearest Neighbors", "QDA"]
