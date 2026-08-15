# Embranchement du dépôt de comparaison

Ce dépôt part de l'implémentation `shiny/` du dépôt « Exemples de Classification »,
qui confronte Dash, Shiny for Python et Streamlit sur une tâche identique, et s'en
sépare définitivement pour être publié en ligne. Le dépôt d'origine reste la référence
de la comparaison ; celui-ci ne l'alimente plus et n'en reçoit rien.

L'hébergeur retenu est **Posit Connect Cloud**, qui déploie Shiny for Python
nativement depuis un dépôt GitHub public. Hugging Face Spaces avait été envisagé
d'abord, puis écarté : Shiny n'y a pas de SDK dédié, le SDK Docker y est la seule
voie, et son hébergement demande depuis peu un abonnement PRO — la création du Space
échoue en `402 Payment Required`. Le `Dockerfile` écrit pour cette tentative est
conservé : il sert à l'exécution locale en conteneur et garde ouverte la porte d'un
hébergeur générique.

La conséquence est directe et assumée : `classifiers.py`, `datasets.py`,
`visualizations.py`, `pipeline.py` et les trois fichiers de tests qui les couvrent
existent désormais en deux exemplaires que rien ne rapproche. L'ADR 0001 du dépôt
d'origine impose leur identité octet pour octet entre les trois implémentations,
vérifiée par `check-parity.sh` et par l'intégration continue ; cette vérification ne
voit pas ce dépôt et ne le verra jamais. Une correction appliquée ici ne parviendra
pas à `dash/` ni à `streamlit/`, et réciproquement.

Cet embranchement est cohérent avec le raisonnement de l'ADR d'origine, qui a écarté
le paquet partagé précisément pour que « chaque implémentation reste copiable telle
quelle comme point de départ d'un nouveau projet ». Ce qu'il ajoute, c'est que la
copie continue de vivre.

## Options considérées

- **Publier le sous-arbre `shiny/` du dépôt d'origine** (`git subtree push`). Écarté :
  les fichiers propres au déploiement — `requirements.txt` engendré, `Dockerfile`,
  flux d'intégration continue dédié — devraient alors vivre dans le dépôt de
  comparaison, où ils n'ont pas de sens pour les deux autres implémentations.
- **Miroir engendré** — le dépôt d'origine reste la source, un script recopie ici les
  six modules sous parité, ce dépôt n'ajoute que l'écart propre au Space. Écarté :
  demande un script de synchronisation et la discipline de ne jamais corriger un bogue
  dans le dépôt où on l'a constaté.
- **Ne rien décider** et laisser les deux exemplaires diverger sans le dire. Écarté :
  c'est le régime sous lequel `dash/` avait dérivé sans que personne s'en aperçoive,
  et que l'ADR d'origine a été écrit pour interdire.

## Conséquences

- La comparaison à trois du dépôt d'origine reste valide : elle porte sur `dash/`,
  `shiny/` et `streamlit/` tels qu'ils y vivent. Ce dépôt n'en fait pas partie et ne
  doit pas être présenté comme la version Shiny de cette comparaison.
- Une correction sur un module de calcul faite ici doit être reportée à la main dans le
  dépôt d'origine si elle vaut pour lui — ou délibérément pas.
- Le vocabulaire peut évoluer librement. Le renommage de **tuile** en **vignette**, que
  la parité aurait rendu triple et coûteux, a été fait ici seul : `CONTEXT.md` disait
  déjà vignette et le code avait dérivé.
- Les fichiers d'outillage local du dépôt d'origine — `docker-compose.yml`, scripts de
  lancement, `LAUNCHER_GUIDE.md` — ne sont pas repris : Connect Cloud déploie depuis le
  dépôt sans conteneur, et le `Dockerfile` conservé se construit seul.
- `requirements.txt` devient un fichier engendré, jamais modifié à la main : `uv.lock`
  reste la seule source des dépendances, et l'intégration continue dérive l'un de
  l'autre à chaque poussée sur `main`.
