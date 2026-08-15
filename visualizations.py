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
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Classe positive du rappel et de la précision. Les jeux de données n'ont que deux
# classes, 0 et 1 ; les deux métriques demandent laquelle elles rapportent,
# contrairement à l'exactitude.
CLASSE_POSITIVE = 1

# Définition rappelée au survol de chaque métrique. Le texte reprend celui du glossaire
# — deux formulations pour une même notion en feraient deux notions. Les retours à la
# ligne sont explicites : une infobulle laissée libre s'étale sur toute la vignette.
DEFINITIONS = {
    "Exactitude": (
        "Proportion de points de l'ensemble de test<br>"
        "correctement classés, toutes classes confondues."
    ),
    "Rappel": (
        "Proportion des points de la classe positive<br>"
        "(classe 1) que le classificateur retrouve.<br>"
        "Ne voit pas les points qu'il y range à tort."
    ),
    "Précision": (
        "Proportion des points annoncés dans la classe<br>"
        "positive (classe 1) qui en relèvent réellement.<br>"
        "Compte les erreurs que le rappel ignore."
    ),
}

# Pas vertical entre deux métriques empilées, en fraction de la hauteur du tracé.
PAS_METRIQUE = 0.09

# Nom anglais accolé à chaque libellé français : c'est celui que porte la littérature,
# et celui des identifiants du code. Il est grisé — le vocabulaire de l'interface reste
# le français, l'anglais n'est qu'un repère.
NOMS_ANGLAIS = {
    "Exactitude": "accuracy",
    "Rappel": "recall",
    "Précision": "precision",
}
COULEUR_NOM_ANGLAIS = "#808495"


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

    La figure porte, dans un coin, le score du classificateur sur l'ensemble de test :
    exactitude, rappel et précision, chacun sous son libellé et rappelant sa définition
    au survol. Le classificateur doit donc être déjà ajusté : la fonction l'évalue, en
    plus de prédire sur la grille.

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

    # Annoter le score dans un coin plutôt que dans le titre — sous mise à jour
    # continue, une valeur qui bouge dans le titre rend la lecture instable.
    #
    # Une seule prédiction sert aux deux métriques. classifier.score prédirait une
    # seconde fois sur le même ensemble, pour retrouver l'exactitude.
    y_pred = classifier.predict(X_test)

    # Chaque métrique porte son libellé : des nombres nus les uns sous les autres ne se
    # distingueraient pas, et c'est leur écart qui instruit.
    #
    # Les trois se complètent, et aucune paire ne suffirait. La répartition en ensembles
    # ne garantit pas l'équilibre des classes, et la classe positive est souvent la plus
    # facile : le rappel sature alors à 1 sur presque toutes les vignettes. Ce qu'il
    # ignore — les points rangés à tort dans la classe positive — est précisément ce que
    # la précision compte. Un SVM linéaire sur des cercles le montre : rappel parfait,
    # précision au ras de la proportion de la classe positive.
    #
    # zero_division couvre les cas indéfinis — ensemble de test dépourvu de classe
    # positive pour le rappel, aucune prédiction positive pour la précision. La métrique
    # vaut alors 0 plutôt que d'avertir à chaque vignette.
    exactitude = accuracy_score(y_test, y_pred)
    rappel = recall_score(y_test, y_pred, pos_label=CLASSE_POSITIVE, zero_division=0)
    precision = precision_score(
        y_test, y_pred, pos_label=CLASSE_POSITIVE, zero_division=0
    )

    # Une annotation par métrique, et non trois lignes dans une seule : une annotation
    # ne porte qu'un texte de survol, et c'est la définition de la métrique survolée
    # qu'il faut montrer. Les trois sont empilées dans l'ordre de lecture, la première
    # nommée en haut — d'où le parcours à l'envers, les positions se comptant depuis le
    # bas de la vignette.
    metriques = (
        ("Exactitude", exactitude),
        ("Rappel", rappel),
        ("Précision", precision),
    )
    for rang, (libelle, valeur) in enumerate(reversed(metriques)):
        fig.add_annotation(
            x=0.99,
            y=0.02 + rang * PAS_METRIQUE,
            xref="paper",
            yref="paper",
            text=(
                f"{libelle} "
                f'<span style="color:{COULEUR_NOM_ANGLAIS}">'
                f"{NOMS_ANGLAIS[libelle]}</span> "
                f"{valeur:.2f}"
            ),
            # Renseigner hovertext suffit : Plotly active alors de lui-même la capture
            # des événements de souris sur la boîte de l'annotation.
            hovertext=DEFINITIONS[libelle],
            # hoverlabel d'annotation n'accepte que bgcolor, bordercolor et font :
            # ni align, ni les autres propriétés du hoverlabel des traces.
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#d5d6d8",
                font=dict(size=11, color="black"),
            ),
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            align="right",
            font=dict(size=11, color="black"),
            bgcolor="rgba(255, 255, 255, 0.75)",
            borderpad=3,
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
