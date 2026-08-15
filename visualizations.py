"""
Fonctions de visualisation pour les frontières de décision et les résultats de classification.

Crée des figures Plotly montrant les frontières de décision, les données d'entraînement et les données de test
pour chaque classificateur.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from sklearn.base import BaseEstimator


def create_decision_boundary_plot(
    classifier: BaseEstimator,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    classifier_name: str,
    resolution: float = 0.02,
) -> go.Figure:
    """
    Créer une visualisation de la frontière de décision pour un classificateur.

    La figure porte, dans un coin, le score du classificateur sur l'ensemble de test.
    Le classificateur doit donc être déjà ajusté : la fonction l'évalue, en plus de
    prédire sur la grille.

    Args:
        classifier: Classificateur déjà ajusté sur l'ensemble d'entraînement
        X_train: Matrice de caractéristiques d'entraînement
        y_train: Labels d'entraînement
        X_test: Matrice de caractéristiques de test
        y_test: Labels de test
        classifier_name: Nom du classificateur pour le titre
        resolution: Résolution de la grille pour la frontière de décision

    Returns:
        Objet Figure Plotly avec la visualisation de la frontière de décision
    """
    # Créer la grille de points
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    # Prédire sur la grille de points
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Créer la figure
    fig = go.Figure()

    # Ajouter le contour de la frontière de décision
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=Z,
            colorscale=[[0, "lightblue"], [1, "lightcoral"]],
            showscale=False,
            opacity=0.6,
            hoverinfo="skip",
        )
    )

    # Ajouter les données d'entraînement
    for label in [0, 1]:
        mask = y_train == label
        fig.add_trace(
            go.Scatter(
                x=X_train[mask, 0],
                y=X_train[mask, 1],
                mode="markers",
                name=f"Entraînement {label}",
                marker=dict(
                    size=8,
                    color="blue" if label == 0 else "red",
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
            )
        )

    # Ajouter les données de test
    for label in [0, 1]:
        mask = y_test == label
        fig.add_trace(
            go.Scatter(
                x=X_test[mask, 0],
                y=X_test[mask, 1],
                mode="markers",
                name=f"Test {label}",
                marker=dict(
                    size=10,
                    color="blue" if label == 0 else "red",
                    symbol="x",
                    line=dict(width=2),
                ),
            )
        )

    # Annoter le score : proportion de points de l'ensemble de test correctement
    # classés, dans un coin plutôt que dans le titre — sous mise à jour continue, une
    # valeur qui bouge dans le titre rend la lecture instable.
    fig.add_annotation(
        x=0.99,
        y=0.02,
        xref="paper",
        yref="paper",
        text=f"{classifier.score(X_test, y_test):.2f}",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=14, color="black"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        borderpad=2,
    )

    # Mettre à jour le layout
    fig.update_layout(
        title=classifier_name,
        xaxis_title="Caractéristique 1",
        yaxis_title="Caractéristique 2",
        showlegend=True,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig
