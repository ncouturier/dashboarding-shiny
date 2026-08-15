# syntax=docker/dockerfile:1

# Image autonome, pour l'exécution locale en conteneur et pour un hébergeur générique.
# Le déploiement en production passe par Posit Connect Cloud, qui n'utilise pas ce
# fichier. L'outillage du dépôt d'origine — docker-compose.yml, scripts de lancement —
# n'a pas été repris : une seule image, sans orchestration.

FROM python:3.13-slim

# uv est installé avant le changement d'utilisateur : /bin appartient à root, et le
# binaire n'a besoin que d'être exécutable.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Spaces exécute le conteneur sous l'identifiant 1000. L'utilisateur est créé et le
# répertoire de travail fixé avant tout COPY : des fichiers appartenant à root
# seraient illisibles pour le processus.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/user/app/.venv

WORKDIR $HOME/app

# Les dépendances sont installées avant le code : une modification de l'application
# ne réinvalide pas la couche d'installation.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY --chown=user . .

# 8000 est le port par défaut de Shiny, et l'un des points de comparaison documentés
# face à Dash (8050) et Streamlit (8501). Un hébergeur qui impose son port le
# transmettra par la variable PORT ; il faudra alors adapter cette ligne et la commande.
EXPOSE 8000

# --no-sync : l'environnement est déjà construit, ne pas le résoudre au démarrage.
CMD ["uv", "run", "--no-sync", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
