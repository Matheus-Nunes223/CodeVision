#!/usr/bin/env python
"""Utilitário de linha de comando do Django para tarefas administrativas."""

import os
import sys
from pathlib import Path


def main():
    """Executa comandos administrativos do Django."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

    # Adiciona a raiz do projeto ao caminho de importação do Python.
    raiz_projeto = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(raiz_projeto))

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Não foi possível importar o Django. "
            "Verifique se ele está instalado no ambiente virtual "
            "e se o ambiente está ativado."
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()