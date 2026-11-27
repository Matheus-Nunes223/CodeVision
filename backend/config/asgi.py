import os
import sys
from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(raiz_projeto))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.asgi import get_asgi_application

application = get_asgi_application()