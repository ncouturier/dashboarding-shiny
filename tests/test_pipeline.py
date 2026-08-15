"""Tests du socle de calcul partagé."""

import re
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import pytest

from classifiers import get_classifier_names
from pipeline import build_decision_boundary, prepare_run

LIBELLES = ("Exactitude", "Rappel", "Précision")
NOMS_ANGLAIS = {"Exactitude": "accuracy", "Rappel": "recall", "Précision": "precision"}

# Le nom anglais s'intercale entre le libellé et la valeur, dans une balise dont la
# couleur porte elle-même des chiffres — d'où l'ancrage sur la fin du texte plutôt que
# sur le premier nombre rencontré. Chaque annotation ne porte qu'une métrique.
LIBELLE_ET_VALEUR = rf"^({'|'.join(LIBELLES)})\b.*\s(\d\.\d{{2}})$"


def metriques_affichees(figure: go.Figure) -> Optional[Dict[str, float]]:
    """Lire les métriques annotées sur une vignette, ou None si elle n'en porte pas.

    Le libellé fait partie de ce qui est vérifié : un nombre nu ne dirait pas de quelle
    métrique il s'agit. Chaque métrique vit dans son annotation, pour porter son propre
    texte de survol : il faut donc les parcourir toutes.
    """
    trouvees: Dict[str, float] = {}
    for annotation in figure.layout.annotations:
        for libelle, valeur in re.findall(LIBELLE_ET_VALEUR, str(annotation.text)):
            trouvees[libelle] = float(valeur)
    return trouvees or None


def test_prepare_run_est_deterministe() -> None:
    """Teste que deux appels à paramètres égaux donnent les mêmes données."""
    premier = prepare_run("moons", noise=0.3, n_samples=100, seed=42)
    second = prepare_run("moons", noise=0.3, n_samples=100, seed=42)

    np.testing.assert_array_equal(premier.X_train, second.X_train)
    np.testing.assert_array_equal(premier.y_train, second.y_train)
    np.testing.assert_array_equal(premier.X_test, second.X_test)
    np.testing.assert_array_equal(premier.y_test, second.y_test)


def test_prepare_run_respecte_la_proportion() -> None:
    """Teste que la répartition suit la proportion 60/40."""
    for n_samples in (100, 200, 300):
        run = prepare_run("moons", noise=0.3, n_samples=n_samples, seed=42)
        assert len(run.X_test) == int(n_samples * 0.4)
        assert len(run.X_train) == n_samples - int(n_samples * 0.4)


@pytest.mark.parametrize("dataset_type", ["moons", "circles", "linear"])
def test_prepare_run_accepte_les_trois_formes(dataset_type: str) -> None:
    """Teste que les trois formes de jeu de données produisent des données valides."""
    run = prepare_run(dataset_type, noise=0.3, n_samples=100, seed=42)

    assert run.X_train.shape[1] == 2
    assert run.X_test.shape[1] == 2
    assert set(np.unique(run.y_train)) <= {0, 1}
    assert len(run.X_train) == len(run.y_train)
    assert len(run.X_test) == len(run.y_test)


def test_prepare_run_graine_absente_retombe_sur_la_valeur_par_defaut() -> None:
    """Teste qu'une graine absente ne rend pas la session non déterministe.

    Un champ numérique vidé par l'utilisateur transmet None. Sans garde, chaque vignette
    préparerait sa propre session et la grille afficherait dix jeux de données
    différents.
    """
    sans_graine = prepare_run("moons", noise=0.3, n_samples=100, seed=None)
    par_defaut = prepare_run("moons", noise=0.3, n_samples=100, seed=42)

    np.testing.assert_array_equal(sans_graine.X_train, par_defaut.X_train)
    assert sans_graine.seed == 42


def test_prepare_run_borne_une_graine_hors_limites() -> None:
    """Teste qu'une graine hors des bornes du contrôle est ramenée dans l'intervalle.

    Les attributs min/max d'un champ HTML ne sont que des indications : une valeur
    négative ou démesurée atteint le serveur et ferait échouer scikit-learn.
    """
    assert prepare_run("moons", noise=0.3, n_samples=100, seed=-1).seed == 0
    assert prepare_run("moons", noise=0.3, n_samples=100, seed=10**12).seed == 9999


def test_prepare_run_refuse_une_forme_inconnue() -> None:
    """Teste qu'un type de jeu de données inconnu échoue explicitement."""
    with pytest.raises(ValueError):
        prepare_run("inexistant", noise=0.3, n_samples=100, seed=42)


def test_build_decision_boundary_rend_une_vignette_par_classificateur() -> None:
    """Teste qu'une figure est produite pour chacun des dix classificateurs."""
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)
    noms = get_classifier_names()

    assert len(noms) == 10
    for nom in noms:
        figure = build_decision_boundary(run, nom)
        assert isinstance(figure, go.Figure)
        assert figure.layout.title.text == nom


def test_build_decision_boundary_refuse_un_nom_inconnu() -> None:
    """Teste qu'un nom de classificateur inconnu échoue explicitement."""
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)

    with pytest.raises(ValueError):
        build_decision_boundary(run, "Classificateur Imaginaire")


def test_build_decision_boundary_est_deterministe() -> None:
    """Teste que deux appels à paramètres égaux produisent la même figure."""
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)

    premier = build_decision_boundary(run, "Decision Tree")
    second = build_decision_boundary(run, "Decision Tree")

    assert premier.to_json() == second.to_json()


def test_prepare_run_la_graine_atteint_la_repartition() -> None:
    """Teste que la répartition dépend de la graine, et non d'une valeur figée.

    À jeu de données identique, une répartition seedée par 7 doit différer de la
    répartition figée à 42 qui prévalait auparavant.
    """
    from sklearn.model_selection import train_test_split

    from datasets import generate_moons

    X, y = generate_moons(n_samples=100, noise=0.3, random_state=7)
    X_train_figee, _, _, _ = train_test_split(X, y, test_size=0.4, random_state=42)

    run = prepare_run("moons", noise=0.3, n_samples=100, seed=7)

    # Même jeu de données de départ...
    np.testing.assert_array_equal(
        np.sort(X, axis=0), np.sort(np.vstack([run.X_train, run.X_test]), axis=0)
    )
    # ...mais une répartition différente, donc seedée.
    assert not np.array_equal(X_train_figee, run.X_train)


def test_prepare_run_conserve_la_graine() -> None:
    """Teste que la session porte la graine, pour l'ajustement des classificateurs."""
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=7)
    assert run.seed == 7


def test_build_decision_boundary_stochastiques_suivent_la_graine() -> None:
    """Teste que la graine modifie l'ajustement d'un classificateur stochastique."""
    premier = prepare_run("moons", noise=0.3, n_samples=100, seed=42)
    second = prepare_run("moons", noise=0.3, n_samples=100, seed=7)

    figure_a = build_decision_boundary(premier, "Random Forest")
    figure_b = build_decision_boundary(second, "Random Forest")

    assert figure_a.to_json() != figure_b.to_json()


def test_build_decision_boundary_deterministes_insensibles_a_la_graine() -> None:
    """Teste qu'à données égales, un classificateur déterministe ignore la graine."""
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)

    from dataclasses import replace

    meme_donnees_autre_graine = replace(run, seed=7)

    for nom in ("Nearest Neighbors", "Naive Bayes", "QDA"):
        figure_a = build_decision_boundary(run, nom)
        figure_b = build_decision_boundary(meme_donnees_autre_graine, nom)
        assert figure_a.to_json() == figure_b.to_json(), nom


def test_build_decision_boundary_annote_les_metriques_sur_chaque_vignette() -> None:
    """Teste que les dix vignettes portent les trois métriques, libellées et bornées."""
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)

    for nom in get_classifier_names():
        metriques = metriques_affichees(build_decision_boundary(run, nom))
        assert metriques is not None, f"{nom} ne porte pas de métriques"
        assert set(metriques) == set(LIBELLES), nom
        for libelle, valeur in metriques.items():
            assert 0.0 <= valeur <= 1.0, f"{nom} : {libelle} hors bornes ({valeur})"


def test_chaque_metrique_porte_son_nom_anglais() -> None:
    """Teste que le nom anglais accompagne le libellé français, à sa droite.

    C'est le nom de la littérature et celui de l'identifiant du code : il fait le pont
    entre la vignette et le glossaire.
    """
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)
    figure = build_decision_boundary(run, "Random Forest")

    for annotation in figure.layout.annotations:
        texte = str(annotation.text)
        libelle = texte.split()[0]
        assert libelle in NOMS_ANGLAIS, libelle

        nom_anglais = NOMS_ANGLAIS[libelle]
        assert nom_anglais in texte, f"{libelle} ne porte pas son nom anglais"
        # À droite du libellé français, et à gauche de la valeur.
        assert texte.index(libelle) < texte.index(nom_anglais), libelle
        assert texte.index(nom_anglais) < texte.rindex(" "), libelle


def test_chaque_metrique_rappelle_sa_definition_au_survol() -> None:
    """Teste que chaque métrique porte sa définition en infobulle.

    La définition vit sur l'annotation de la métrique, et non sur un bloc commun : une
    annotation ne porte qu'un texte de survol, et c'est celle qu'on survole qu'il faut
    définir.
    """
    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)
    figure = build_decision_boundary(run, "Random Forest")

    # Les retours à la ligne relèvent de la présentation : les neutraliser pour que les
    # vérifications portent sur la formulation seule.
    definitions = {
        str(annotation.text)
        .split()[0]: str(annotation.hovertext or "")
        .replace("<br>", " ")
        for annotation in figure.layout.annotations
    }

    for libelle in LIBELLES:
        assert libelle in definitions, f"{libelle} n'a pas d'annotation propre"
        assert definitions[
            libelle
        ].strip(), f"{libelle} n'a pas de définition au survol"

    # Une définition recopiée d'une métrique sur l'autre passerait le test précédent.
    assert len(set(definitions[libelle] for libelle in LIBELLES)) == len(LIBELLES)

    # Le texte doit venir du glossaire, pas d'une reformulation parallèle.
    assert "classe positive" in definitions["Rappel"]
    assert "classe positive" in definitions["Précision"]
    assert "toutes classes confondues" in definitions["Exactitude"]


def test_le_score_porte_sur_l_ensemble_de_test() -> None:
    """Teste que le score annoté est celui de l'ensemble de test, pas d'entraînement.

    Un arbre de décision profond sur des données bruitées sur-apprend : son score
    d'entraînement dépasse son score de test. C'est ce qui donne du mordant au test.
    """
    from classifiers import get_classifiers

    run = prepare_run("moons", noise=0.5, n_samples=100, seed=42)

    classifier = get_classifiers(seed=run.seed)["Decision Tree"]
    classifier.fit(run.X_train, run.y_train)
    score_entrainement = classifier.score(run.X_train, run.y_train)
    score_test = classifier.score(run.X_test, run.y_test)

    # L'écart doit dépasser largement l'arrondi à deux décimales, sinon la dernière
    # assertion ne distinguerait plus les deux scores.
    assert score_entrainement - score_test > 0.05, (
        "cas de test mal choisi : l'écart entre les deux scores est trop faible "
        f"({score_entrainement:.3f} contre {score_test:.3f})"
    )

    metriques = metriques_affichees(build_decision_boundary(run, "Decision Tree"))
    assert metriques is not None
    annote = metriques["Exactitude"]

    # Le score annoté est arrondi à deux décimales : la tolérance doit l'absorber.
    assert annote == pytest.approx(score_test, abs=0.01)
    assert abs(annote - score_test) < abs(annote - score_entrainement)


def test_le_score_est_reproductible_a_graine_egale() -> None:
    """Teste que la même graine redonne les mêmes métriques."""
    premier = prepare_run("circles", noise=0.2, n_samples=150, seed=7)
    second = prepare_run("circles", noise=0.2, n_samples=150, seed=7)

    for nom in get_classifier_names():
        assert metriques_affichees(
            build_decision_boundary(premier, nom)
        ) == metriques_affichees(build_decision_boundary(second, nom)), nom


def test_le_rappel_ne_recopie_pas_l_exactitude() -> None:
    """Teste que le rappel est bien celui de la classe positive, sous son libellé.

    Les deux métriques coïncident sur la plupart des vignettes : les distinguer demande
    un cas où elles divergent. Un SVM linéaire sur des cercles ne peut pas séparer deux
    classes concentriques — il range tout l'ensemble de test dans la classe positive.
    Le rappel est alors parfait tandis que l'exactitude s'effondre, ce qui est
    exactement ce que les libellés donnent à lire.
    """
    from sklearn.metrics import accuracy_score, recall_score

    from classifiers import get_classifiers

    run = prepare_run("circles", noise=0.5, n_samples=100, seed=8)
    classifier = get_classifiers(seed=run.seed)["Linear SVM"]
    classifier.fit(run.X_train, run.y_train)
    y_pred = classifier.predict(run.X_test)

    exactitude = accuracy_score(run.y_test, y_pred)
    rappel = recall_score(run.y_test, y_pred, pos_label=1, zero_division=0)
    assert rappel - exactitude > 0.2, (
        "cas de test mal choisi : les deux métriques ne divergent plus assez "
        f"(exactitude {exactitude:.3f}, rappel {rappel:.3f})"
    )

    metriques = metriques_affichees(build_decision_boundary(run, "Linear SVM"))
    assert metriques is not None
    assert metriques["Exactitude"] == pytest.approx(exactitude, abs=0.01)
    assert metriques["Rappel"] == pytest.approx(rappel, abs=0.01)


def test_build_decision_boundary_rend_une_vignette_de_repli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teste qu'un classificateur qui échoue produit une vignette de repli."""

    class ClassificateurDefaillant:
        """Classificateur qui lève une exception à l'ajustement."""

        def fit(self, X: Any, y: Any) -> None:
            raise RuntimeError("ajustement impossible")

    import pipeline

    def get_classifiers_defaillants(seed: int = 42) -> Dict[str, Any]:
        return {"Decision Tree": ClassificateurDefaillant()}

    monkeypatch.setattr(pipeline, "get_classifiers", get_classifiers_defaillants)

    run = prepare_run("moons", noise=0.3, n_samples=100, seed=42)
    figure = build_decision_boundary(run, "Decision Tree")

    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Decision Tree"
    assert "ajustement impossible" in figure.layout.annotations[0].text
    assert (
        metriques_affichees(figure) is None
    ), "la vignette de repli ne doit pas porter de métriques"
