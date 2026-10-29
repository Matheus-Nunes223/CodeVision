"""
Rotas da aplicação de reconhecimento do CodeVision.
"""

from django.urls import path

from reconhecimento import views


app_name = "reconhecimento"

urlpatterns = [
    # Recebe uma imagem e solicita o reconhecimento visual.
    path(
        "",
        views.reconhecer_imagem,
        name="reconhecer_imagem",
    ),

    # Verifica se o serviço de reconhecimento está disponível.
    path(
        "status/",
        views.status_reconhecimento,
        name="status_reconhecimento",
    ),
]