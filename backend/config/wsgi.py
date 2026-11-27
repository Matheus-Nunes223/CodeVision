import os
import sys
from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent.parent

if str(raiz_projeto) not in sys.path:
    sys.path.insert(0, str(raiz_projeto))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()