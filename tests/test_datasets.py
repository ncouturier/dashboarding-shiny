"""Tests pour les fonctions de génération de datasets."""

import numpy as np
import pytest

from datasets import generate_circles, generate_linearly_separable, generate_moons


def test_generate_moons_shape() -> None:
    """Teste que generate_moons renvoie le bon shape."""
    X, y = generate_moons(n_samples=100, noise=0.1, random_state=42)
    assert X.shape == (100, 2)
    assert y.shape == (100,)


def test_generate_moons_labels() -> None:
    """Teste que generate_moons renvoie des labels binaires."""
    X, y = generate_moons(n_samples=100, noise=0.1, random_state=42)
    assert set(y) == {0, 1}


def test_generate_moons_reproducibility() -> None:
    """Teste que generate_moons est reproductible avec le même seed."""
    X1, y1 = generate_moons(n_samples=100, noise=0.1, random_state=42)
    X2, y2 = generate_moons(n_samples=100, noise=0.1, random_state=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_circles_shape() -> None:
    """Teste que generate_circles renvoie le bon shape."""
    X, y = generate_circles(n_samples=100, noise=0.1, random_state=42)
    assert X.shape == (100, 2)
    assert y.shape == (100,)


def test_generate_circles_labels() -> None:
    """Teste que generate_circles renvoie des labels binaires."""
    X, y = generate_circles(n_samples=100, noise=0.1, random_state=42)
    assert set(y) == {0, 1}


def test_generate_linearly_separable_shape() -> None:
    """Teste que generate_linearly_separable renvoie le bon shape."""
    X, y = generate_linearly_separable(n_samples=100, noise=0.1, random_state=42)
    assert X.shape == (100, 2)
    assert y.shape == (100,)


def test_generate_linearly_separable_labels() -> None:
    """Teste que generate_linearly_separable renvoie des labels binaires."""
    X, y = generate_linearly_separable(n_samples=100, noise=0.1, random_state=42)
    assert set(y) == {0, 1}
