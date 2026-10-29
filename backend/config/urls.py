"""
Rotas principais do backend do CodeVision.
"""

from django.contrib import admin
from django.urls import include, path
from django.http import JsonResponse


def health_check(request):
    """
    Verifica se o backend está respondendo.
    """
    return JsonResponse({
        "status": "online",
        "projeto": "CodeVision",
        "mensagem": "Backend funcionando corretamente.",
    })


urlpatterns = [
    # Painel administrativo do Django.
    path("admin/", admin.site.urls),

    # Verificação de disponibilidade do backend.
    path("api/health/", health_check, name="health_check"),

    # Rotas da aplicação de reconhecimento de imagens.
    path("api/reconhecimento/", include("reconhecimento.urls")),
]