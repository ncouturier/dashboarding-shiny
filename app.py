"""
Application Shiny avec exemples de Classification.

Visualisation interactive des algorithmes d'apprentissage automatique pour la classification.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from plotly.offline import get_plotlyjs_version
from shiny import App, reactive, render, ui
from shiny.types import SilentException, SilentOperationInProgressException

from classifiers import get_classifier_names
from pipeline import TILE_HEIGHT, Run, build_decision_boundary, prepare_run

# Plotly est chargé une seule fois dans l'en-tête de la page, et non réinjecté par
# chaque vignette. Une vignette qui embarquerait sa propre balise <script> vers le CDN
# exécuterait son appel de tracé avant la fin du chargement de la bibliothèque : au
# premier affichage, les dix emplacements resteraient vides. Le bouton « Générer »
# masquait ce défaut, le second rendu trouvant la bibliothèque déjà chargée.
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"

# Nombre de vignettes d'une salve, déduit du dispositif plutôt que réécrit à la main.
NOMBRE_DE_VIGNETTES = len(get_classifier_names())

# Pool de fils dédié et borné. Le pool par défaut d'asyncio est partagé par tout le
# processus : un afflux de calculs abandonnés y priverait les autres sessions de fils.
#
# La borne suit les cœurs disponibles, et non le nombre de vignettes. L'ajustement d'un
# classificateur puis la sérialisation de sa figure sont bornés par le processeur :
# au-delà d'un fil par cœur, les calculs se disputent la même machine sans que le débit
# augmente. Deux conséquences, l'une et l'autre recherchées sur un hébergement partagé.
#
# Le nombre total de fils cesse de dépendre du nombre de sessions. Dix fils par session
# épuisaient la machine dès la deuxième connexion, chaque salve en réclamant dix ;
# désormais la file d'attente absorbe les sessions supplémentaires et le débit reste
# celui de la machine, quel que soit le nombre de spectateurs.
#
# Le rendu progressif y gagne. Dix calculs menés de front sur deux cœurs progressent
# ensemble et s'achèvent ensemble : la grille se peint d'un coup, en fin de cycle. Deux
# calculs à la fois s'achèvent l'un après l'autre — c'est précisément l'échelonnement
# que l'application donne à voir.
#
# os.process_cpu_count respecte l'affinité du processus. Il ignore en revanche un quota
# cgroup : un conteneur bridé à un demi-cœur sur une machine qui en compte seize en
# verra seize. Le plancher de deux fils évite qu'un calcul périmé, qu'aucune annulation
# n'interrompt, occupe le seul fil dont la salve courante a besoin.
MAX_WORKERS = max(2, min(NOMBRE_DE_VIGNETTES, os.process_cpu_count() or 2))
EXECUTEUR = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="vignette")

# Définir l'UI
app_ui = ui.page_fluid(
    # En-tête
    ui.row(
        ui.column(
            12,
            ui.h1("Exemples de Classification", class_="text-center mb-2 mt-3"),
            ui.p(
                "Exploration interactive des algorithmes d'apprentissage automatique pour la classification",
                class_="text-center text-muted mb-4",
            ),
        )
    ),
    ui.hr(),
    # Contenu principal: Sidebar + Grille
    ui.row(
        # Sidebar (colonne 3)
        ui.column(
            3,
            ui.h4("Configuration du jeu de données", class_="mb-4"),
            # Card 1 : forme du jeu de données
            ui.card(
                ui.card_header("Type de jeu de données", class_="fw-bold mb-2"),
                ui.input_select(
                    "dataset_selector",
                    label=None,
                    choices={
                        "moons": "🌙 Lunes",
                        "circles": "⭕ Cercles",
                        "linear": "📏 Linéairement Séparable",
                    },
                    selected="moons",
                ),
                class_="mb-3",
            ),
            # Card 2: Paramètres
            ui.card(
                ui.card_body(
                    ui.p("Niveau de Bruit", class_="fw-bold mb-2"),
                    ui.input_slider(
                        "noise_slider",
                        label=None,
                        min=0.0,
                        max=1.0,
                        value=0.3,
                        step=0.05,
                    ),
                    ui.hr(),
                    ui.p("Nombre d'Échantillons", class_="fw-bold mb-2 mt-3"),
                    ui.input_slider(
                        "samples_slider",
                        label=None,
                        min=50,
                        max=500,
                        value=100,
                        step=50,
                    ),
                    ui.hr(),
                    ui.p(
                        "Graine",
                        class_="fw-bold mb-2 mt-3",
                    ),
                    ui.input_numeric(
                        "seed_input",
                        label=None,
                        value=42,
                        min=0,
                        max=9999,
                    ),
                ),
                class_="mb-3",
            ),
            # Carte d'information
            ui.card(
                ui.card_body(
                    ui.h6("ℹ️ Comment Utiliser", class_="mb-2"),
                    ui.p(
                        "1. Sélectionnez une forme de jeu de données",
                        ui.br(),
                        "2. Ajustez le bruit et le nombre d'échantillons",
                        ui.br(),
                        "3. Définissez une graine",
                        ui.br(),
                        "4. Les frontières de décision se mettent à jour au fil de "
                        "vos ajustements",
                        class_="small mb-0",
                    ),
                ),
                class_="bg-light",
            ),
            class_="bg-light p-3",
        ),
        # Grille 2x5 (colonne 9)
        ui.column(
            9,
            ui.row(
                *[
                    ui.column(
                        6,  # 2 colonnes par ligne (Bootstrap: 12/6 = 2)
                        # L'emplacement du graphique vit dans l'interface statique, et
                        # non dans la sortie : le contenu d'une sortie est intégralement
                        # remplacé à chaque mise à jour, ce qui recréerait le graphique
                        # et laisserait le précédent orphelin. Ici le même nœud est
                        # réutilisé d'une mise à jour à l'autre par Plotly.react.
                        ui.div(id=f"vignette_{i}", style=f"height: {TILE_HEIGHT}px;"),
                        ui.output_ui(f"classifier_{i}"),
                        class_="mb-3",
                    )
                    for i in range(NOMBRE_DE_VIGNETTES)
                ],
                class_="g-3",
            ),
        ),
    ),
    # Pied de page
    ui.hr(class_="mt-4"),
    ui.row(
        ui.column(
            12,
            ui.p(
                "Basé sur l'exemple de ",
                ui.a(
                    "comparaison d'algorithmes de classification de scikit-learn",
                    href="https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html",
                    target="_blank",
                ),
                " | Construit avec Shiny for Python & scikit-learn",
                class_="text-center text-muted small mb-3",
            ),
        ),
    ),
    # Inclure CSS personnalisé
    ui.include_css("www/styles.css"),
    ui.head_content(ui.tags.script(src=PLOTLY_CDN)),
    title="Exemples de Classification",
)


# Définir le serveur
def server(input, output, session):
    """
    Logique serveur pour l'application Shiny.

    Gère la réactivité et la génération des visualisations.
    """

    # Mise à jour continue — comment Shiny for Python s'y prend
    #
    # Anti-rebond : reactive.calc ne recalcule qu'une fois par salve d'invalidations,
    # et les curseurs de Shiny n'émettent qu'au relâchement. La session n'est donc
    # préparée qu'une fois par ajustement, pas une fois par valeur intermédiaire.
    #
    # Supersession : chaque tâche en cours est annulée avant qu'une nouvelle ne soit
    # lancée. L'annulation suffit à écarter un travail encore en file, qu'asyncio retire
    # alors du pool avant son démarrage.
    #
    # Elle ne suffit pas pour un travail déjà parti : un fil n'est pas interruptible en
    # Python et va jusqu'à son terme. C'est une limite du langage, pas un oubli. La
    # session est donc revérifiée à deux moments — avant le calcul, pour ne pas lancer
    # ce qui est déjà caduc, et après, pour ne pas peindre un résultat devenu caduc
    # pendant le calcul. Reste qu'un ajustement enchaîné sur un grand nombre
    # d'échantillons attend la fin des ajustements non interruptibles déjà lancés.
    #
    # Rendu progressif : obtenu par une tâche étendue par vignette. Shiny rassemble les
    # valeurs de toutes les sorties d'un même cycle dans un unique message de flush ;
    # une sortie qui calcule pendant le cycle fait donc attendre les neuf autres.
    # Une tâche étendue sort du graphe réactif : la sortie rend immédiatement un
    # indicateur d'attente, et l'achèvement de la tâche déclenche un nouveau cycle pour
    # cette seule sortie. Dix tâches produisent dix cycles, donc dix peintures
    # successives. Le calcul étant bloquant pour le processeur, il part dans un pool de
    # fils dédié : sans cela les dix tâches se sérialiseraient sur la boucle d'événements
    # et le serveur cesserait de répondre pendant le calcul.

    # Session en vigueur. Un travail pris en charge alors que cette valeur a changé est
    # devenu inutile : il rend la main sans calculer.
    session_courante: Dict[str, Optional[Run]] = {"run": None}

    @reactive.calc
    def session_de_calcul() -> Run:
        """
        Préparer la session : jeu de données et répartition en ensembles.

        Partagée par les dix vignettes, recalculée une seule fois par ajustement.

        Returns:
            La session de calcul correspondant aux paramètres courants
        """
        return prepare_run(
            dataset_type=input.dataset_selector(),
            noise=input.noise_slider(),
            n_samples=input.samples_slider(),
            seed=input.seed_input(),
        )

    def indicateur_d_attente(index: int, classifier_name: str) -> ui.HTML:
        """
        Masquer l'emplacement du graphique et afficher un indicateur à sa place.

        L'indicateur a la hauteur d'une vignette : plus court, il ferait descendre toute
        la grille à l'arrivée de chaque résultat.

        Args:
            index: Position de la vignette dans la grille
            classifier_name: Nom du classificateur en cours de calcul

        Returns:
            Le fragment HTML de l'indicateur
        """
        return ui.HTML(
            f'<script>(function(){{var e=document.getElementById("vignette_{index}");'
            f'if(e)e.style.display="none";}})();</script>'
            f'<div style="height: {TILE_HEIGHT}px; display: flex; '
            "align-items: center; justify-content: center; color: #808495; "
            'border: 1px dashed #d5d6d8; border-radius: 8px;">'
            f"⏳ {classifier_name}</div>"
        )

    def tracer(index: int, figure_json: str) -> ui.HTML:
        """
        Mettre à jour l'emplacement stable de la vignette avec une nouvelle figure.

        Plotly.react réutilise le graphique déjà présent sur ce nœud au lieu d'en créer
        un autre : aucun objet Plotly n'est laissé orphelin, quel que soit le nombre de
        mises à jour.

        Args:
            index: Position de la vignette dans la grille
            figure_json: La figure déjà sérialisée par le fil de calcul

        Returns:
            Le fragment de script qui met à jour l'emplacement
        """
        return ui.HTML(
            "<script>(function(){"
            f'var e=document.getElementById("vignette_{index}");'
            "if(!e)return;"
            # Plotly indisponible (CDN injoignable) : échouer visiblement plutôt que
            # de laisser l'emplacement masqué et la grille se replier en silence.
            "if(!window.Plotly){"
            f'console.error("Plotly indisponible : vignette {index} non tracée");'
            "return;}"
            'e.style.display="";'
            f"var f={figure_json};"
            "Plotly.react(e, f.data, f.layout, {displayModeBar:false});"
            "})();</script>"
        )

    def enregistrer_vignette(index: int, classifier_name: str) -> None:
        """
        Déclarer la tâche et la sortie d'une vignette, à position fixe dans la grille.

        Args:
            index: Position de la vignette dans la grille
            classifier_name: Nom du classificateur affiché à cette position
        """

        @reactive.extended_task
        async def calculer(run: Run) -> Optional[str]:
            """
            Calculer la vignette hors du graphe réactif et hors de la boucle d'événements.

            La session est passée en argument : une tâche étendue ne doit lire aucune
            valeur réactive, sans quoi elle recréerait la dépendance qu'elle sert à
            rompre.

            La figure est sérialisée ici, sur le fil de calcul. Une frontière de
            décision compte des dizaines de milliers de points : sérialiser dans la
            fonction de rendu bloquerait la boucle d'événements, et donc toutes les
            sessions connectées, dix fois par ajustement.

            Args:
                run: La session pour laquelle ce calcul a été demandé

            Returns:
                La figure sérialisée, ou None si la session n'est plus en vigueur
            """

            def travail() -> Optional[str]:
                # Un travail encore en file alors que la session a changé rend la main
                # sans calculer.
                if session_courante["run"] is not run:
                    return None

                figure = build_decision_boundary(run, classifier_name)

                # La session a pu changer pendant le calcul : ne pas peindre un
                # résultat devenu périmé, la tâche relancée s'en chargera.
                if session_courante["run"] is not run:
                    return None

                # Le contenu est injecté dans une balise <script> : une chaîne
                # contenant </script> la fermerait par accident. Le cas est atteignable
                # — la vignette de repli interpole le message d'une exception quelconque.
                return figure.to_json().replace("</", "<\\/")

            boucle = asyncio.get_running_loop()
            return await boucle.run_in_executor(EXECUTEUR, travail)

        @reactive.effect
        def relancer() -> None:
            """Annuler la tâche en cours, puis la relancer sur la session courante."""
            run = session_de_calcul()
            session_courante["run"] = run
            calculer.cancel()
            calculer(run)

        @output(id=f"classifier_{index}")
        @render.ui
        def vignette() -> ui.HTML:
            try:
                figure_json = calculer.result()
            except (SilentOperationInProgressException, SilentException):
                # Tâche en cours, jamais lancée ou annulée : même attente côté
                # utilisateur.
                return indicateur_d_attente(index, classifier_name)

            if figure_json is None:
                # Travail abandonné parce que la session avait changé ; la tâche
                # relancée peindra.
                return indicateur_d_attente(index, classifier_name)

            return tracer(index, figure_json)

    for index, classifier_name in enumerate(get_classifier_names()):
        enregistrer_vignette(index, classifier_name)


# Créer l'application
app = App(app_ui, server)
