"""
Configuração WSGI do projeto CodeVision.

Expõe a aplicação Django por meio da variável `application`,
utilizada por servidores compatíveis com WSGI.
"""

import os
import sys
from pathlib import Path

# Localiza a raiz do projeto:
# Codevision/
# ├── ia/
# └── backend/
#     └── config/
#         └── wsgi.py
raiz_projeto = Path(__file__).resolve().parent.parent.parent

# Permite importar módulos localizados na raiz do projeto,
# incluindo os módulos de inteligência artificial da pasta ia/.
if str(raiz_projeto) not in sys.path:
    sys.path.insert(0, str(raiz_projeto))

# Define onde estão as configurações do Django.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.wsgi import get_wsgi_application

# Aplicação carregada pelo servidor WSGI.
application = get_wsgi_application()