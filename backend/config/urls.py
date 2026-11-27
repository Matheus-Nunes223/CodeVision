

from django.contrib import admin
from django.urls import include, path
from django.http import JsonResponse


def health_check(request):
    
    return JsonResponse({
        "status": "online",
        "projeto": "CodeVision",
        "mensagem": "Backend funcionando corretamente.",
    })


urlpatterns = [
    
    path("admin/", admin.site.urls),

    
    path("api/health/", health_check, name="health_check"),

    
    path("api/reconhecimento/", include("reconhecimento.urls")),
]