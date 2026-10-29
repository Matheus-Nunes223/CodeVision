"""
Configuração da aplicação de reconhecimento do CodeVision.
"""

from django.apps import AppConfig


class ReconhecimentoConfig(AppConfig):
    """Configuração principal do aplicativo de reconhecimento."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "reconhecimento"
    verbose_name = "Reconhecimento de Imagens"