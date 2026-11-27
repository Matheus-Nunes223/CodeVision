

from django.urls import path

from reconhecimento import views


app_name = "reconhecimento"

urlpatterns = [
    
    path(
        "",
        views.reconhecer_imagem,
        name="reconhecer_imagem",
    ),

   
    path(
        "status/",
        views.status_reconhecimento,
        name="status_reconhecimento",
    ),
]