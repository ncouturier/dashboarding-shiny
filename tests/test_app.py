"""Tests pour l'application Shiny principale."""

import os

import pytest


def test_app_imports() -> None:
    """Teste que tous les modules nécessaires peuvent être importés."""
    try:
        from app import app, app_ui, server
        from classifiers import get_classifiers
        from datasets import generate_moons
        from visualizations import create_decision_boundary_plot

        assert True
    except ImportError as e:
        pytest.fail(f"Échec de l'import: {e}")


def test_app_ui_contains_title() -> None:
    """Teste que l'UI contient le titre de l'application."""
    from app import app_ui

    assert "Exemples de Classification" in str(app_ui)


def test_app_na_plus_de_bouton_de_generation() -> None:
    """Teste que le bouton 'Générer' a disparu : la mise à jour est continue.

    Inversion volontaire de test_app_has_generate_button : la disparition du bouton
    doit rester vérifiée, pas simplement cesser de l'être.
    """
    from app import app_ui

    interface = str(app_ui)
    assert "generate_button" not in interface
    assert "Générer les Visualisations" not in interface


def test_app_utilise_le_vocabulaire_du_glossaire() -> None:
    """Teste que les libellés suivent CONTEXT.md."""
    from app import app_ui

    interface = str(app_ui)
    assert "Type de jeu de données" in interface
    assert "Graine" in interface
    assert "Type de Dataset" not in interface
    assert "Seed Aléatoire" not in interface


def _app_source() -> str:
    """Lire le source de l'application."""
    from pathlib import Path

    return (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")


def test_app_construit_chaque_vignette_independamment() -> None:
    """Teste que le calcul partagé s'arrête à la session, pas aux dix figures.

    Chaque vignette est construite séparément : c'est le préalable au rendu progressif,
    et cela évite que chaque sortie recalcule les dix figures.
    """
    source = _app_source()

    assert "def session_de_calcul() -> Run:" in source
    assert "trained_classifiers" not in source


def test_app_calcule_chaque_vignette_dans_une_tache_etendue() -> None:
    """Teste le rendu progressif : chaque vignette sort du graphe réactif.

    Shiny rassemble les valeurs de toutes les sorties d'un même cycle dans un unique
    message de flush. Une sortie qui calcule pendant le cycle fait donc attendre les
    neuf autres. Une tâche étendue rend la main immédiatement et déclenche, à son
    achèvement, un cycle pour sa seule sortie — d'où dix peintures successives.
    """
    source = _app_source()

    assert "@reactive.extended_task" in source
    # Le calcul est bloquant pour le processeur : sans passage en fil d'exécution
    # séparé, les dix tâches se sérialiseraient sur la boucle d'événements.
    assert "run_in_executor" in source


def test_app_annule_les_taches_avant_d_en_relancer() -> None:
    """Teste la supersession : aucune tâche périmée ne peut peindre."""
    source = _app_source()

    assert ".cancel()" in source


def test_app_utilise_un_pool_de_fils_borne() -> None:
    """Teste que les calculs ne partent pas dans le pool partagé par défaut.

    Le pool par défaut est partagé par tout le processus : un afflux de calculs
    abandonnés y priverait les autres sessions de fils d'exécution.
    """
    source = _app_source()

    assert "ThreadPoolExecutor" in source
    assert "max_workers" in source
    # asyncio.to_thread passe par le pool partagé : il ne doit plus être utilisé.
    assert "asyncio.to_thread" not in source


def test_app_abandonne_un_calcul_devenu_caduc_avant_de_le_lancer() -> None:
    """Teste qu'un calcul encore en file n'est pas lancé si les paramètres ont changé.

    Un fil déjà parti n'est pas interruptible, mais un travail encore en file ne l'est
    pas non plus : le vérifier au moment de sa prise en charge suffit à libérer
    immédiatement le fil.
    """
    source = _app_source()

    assert "session_courante" in source


def test_app_echappe_la_fermeture_de_balise() -> None:
    """Teste que le contenu injecté ne peut pas fermer la balise script par accident.

    La vignette de repli interpole le message d'une exception quelconque dans la figure ;
    un message contenant </script> terminerait le script et casserait la vignette.
    """
    source = _app_source()

    assert '"</"' in source and '"<\\\\/"' in source


def test_app_reverifie_la_session_apres_le_calcul() -> None:
    """Teste qu'un résultat devenu périmé pendant le calcul n'est pas peint.

    Vérifier la session avant le calcul ne couvre pas le cas où elle change pendant :
    la tâche achevée porterait alors la frontière des paramètres précédents.
    """
    source = _app_source()

    corps = source[source.index("def travail()") : source.index("boucle = asyncio")]
    assert corps.count('session_courante["run"] is not run') == 2


def test_app_sérialise_la_figure_hors_de_la_boucle() -> None:
    """Teste que la sérialisation se fait sur le fil de calcul.

    Une frontière de décision compte des dizaines de milliers de points : sérialiser
    dans la fonction de rendu bloquerait la boucle d'événements, donc toutes les
    sessions, dix fois par ajustement.
    """
    source = _app_source()

    corps = source[source.index("def travail()") : source.index("boucle = asyncio")]
    assert "to_json()" in corps


def test_app_pool_borne_par_les_coeurs() -> None:
    """Teste que le pool suit les cœurs, et non le nombre de vignettes.

    Un pool dimensionné sur la salve réclamait dix fils par session : la deuxième
    connexion épuisait la machine. Borné par les cœurs, le nombre total de fils ne
    dépend plus du nombre de sessions.
    """
    import app

    assert app.MAX_WORKERS <= app.NOMBRE_DE_VIGNETTES
    assert app.MAX_WORKERS <= max(2, os.process_cpu_count() or 2)


def test_app_pool_garde_un_plancher() -> None:
    """Teste qu'il reste un fil libre pour la salve courante.

    Un calcul périmé n'est pas interruptible : sur une machine à un cœur, un pool d'un
    seul fil ferait attendre la salve courante la fin d'un calcul dont plus personne
    n'a l'usage.
    """
    import app

    assert app.MAX_WORKERS >= 2


def test_app_reutilise_le_graphique_plotly() -> None:
    """Teste qu'aucun objet Plotly n'est abandonné à chaque mise à jour.

    Réinjecter un fragment HTML complet recrée le graphique et laisse le précédent
    orphelin. Un emplacement stable mis à jour par Plotly.react réutilise le même
    objet.
    """
    source = _app_source()

    assert "Plotly.react" in source
    # Les emplacements doivent vivre dans l'interface statique, hors des sorties :
    # une sortie voit son contenu intégralement remplacé à chaque mise à jour.
    assert 'ui.div(id=f"vignette_{i}"' in source or "ui.div(id=f'vignette_{i}'" in source
    assert "to_html" not in source


def test_app_affiche_un_indicateur_d_attente() -> None:
    """Teste qu'une vignette en cours de calcul montre son état plutôt que du vide."""
    source = _app_source()

    assert "SilentOperationInProgressException" in source
    # L'indicateur doit avoir la hauteur d'une vignette, sinon la grille se décale
    # lorsqu'un résultat arrive.
    assert "TILE_HEIGHT" in source


def test_app_has_all_inputs() -> None:
    """Teste que tous les inputs sont présents."""
    from app import app_ui

    ui_str = str(app_ui)
    assert "dataset_selector" in ui_str
    assert "noise_slider" in ui_str
    assert "samples_slider" in ui_str
    assert "seed_input" in ui_str


def test_app_has_all_classifier_outputs() -> None:
    """Teste que tous les outputs pour les 10 classificateurs sont présents."""
    from app import app_ui

    ui_str = str(app_ui)
    for i in range(10):
        assert f"classifier_{i}" in ui_str
