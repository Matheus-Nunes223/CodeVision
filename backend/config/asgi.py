"""
Configuração ASGI do projeto CodeVision.

Expõe a aplicação Django como uma variável chamada `application`.
"""

import os
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao caminho de importação do Python.
# Estrutura esperada:
# Codevision/
# ├── ia/
# └── backend/
#     └── config/
#         └── asgi.py
raiz_projeto = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(raiz_projeto))

# Define o módulo de configurações do Django.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.asgi import get_asgi_application

# Aplicação ASGI utilizada pelo servidor.
application = get_asgi_application()