"""
Configurações do projeto CodeVision.

Projeto desenvolvido com Django e Django REST Framework,
integrado a módulos de inteligência artificial para
reconhecimento visual de mensagens codificadas.
"""

import os
from pathlib import Path


# ============================================================
# 1. CAMINHOS DO PROJETO
# ============================================================

# Diretório backend/
BASE_DIR = Path(__file__).resolve().parent.parent

# Raiz do projeto CodeVision/
RAIZ_PROJETO = BASE_DIR.parent

# Garante que os módulos da pasta ia/ possam ser encontrados.
# Estrutura:
# Codevision/
# ├── ia/
# ├── frontend/
# └── backend/
#     ├── manage.py
#     └── config/
sys_path = str(RAIZ_PROJETO)

import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


# ============================================================
# 2. SEGURANÇA
# ============================================================

# Em produção, defina SECRET_KEY por variável de ambiente.
# O valor padrão abaixo serve apenas para desenvolvimento local.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-codevision-chave-temporaria-apenas-desenvolvimento"
)

# Ative DEBUG somente no ambiente de desenvolvimento.
DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in {
    "true",
    "1",
    "yes",
    "sim",
}

# Hosts permitidos. Em desenvolvimento local, localhost e
# 127.0.0.1 são suficientes.
_hosts = os.environ.get(
    "DJANGO_ALLOWED_HOSTS",
    "localhost,127.0.0.1"
)

ALLOWED_HOSTS = [
    host.strip()
    for host in _hosts.split(",")
    if host.strip()
]


# ============================================================
# 3. APLICATIVOS INSTALADOS
# ============================================================

INSTALLED_APPS = [
    # Aplicativos nativos do Django
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Aplicativos de terceiros
    "rest_framework",
    "corsheaders",

    # Aplicativo principal do CodeVision
    "reconhecimento",
]


# ============================================================
# 4. MIDDLEWARE
# ============================================================

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",

    # Deve ficar antes do CommonMiddleware.
    "corsheaders.middleware.CorsMiddleware",

    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# ============================================================
# 5. CONFIGURAÇÃO PRINCIPAL
# ============================================================

ROOT_URLCONF = "config.urls"

WSGI_APPLICATION = "config.wsgi.application"

ASGI_APPLICATION = "config.asgi.application"


# ============================================================
# 6. TEMPLATES
# ============================================================

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# ============================================================
# 7. BANCO DE DADOS
# ============================================================

# SQLite é suficiente para o desenvolvimento inicial.
# Ainda não precisamos configurar um banco externo.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# ============================================================
# 8. VALIDAÇÃO DE SENHAS
# ============================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "MinimumLengthValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "CommonPasswordValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "NumericPasswordValidator"
        ),
    },
]


# ============================================================
# 9. IDIOMA, FUSO HORÁRIO E INTERNACIONALIZAÇÃO
# ============================================================

LANGUAGE_CODE = "pt-br"

TIME_ZONE = "America/Sao_Paulo"

USE_I18N = True

USE_TZ = True


# ============================================================
# 10. ARQUIVOS ESTÁTICOS
# ============================================================

STATIC_URL = "static/"

STATIC_ROOT = BASE_DIR / "staticfiles"


# ============================================================
# 11. ARQUIVOS ENVIADOS PELO USUÁRIO
# ============================================================

MEDIA_URL = "media/"

MEDIA_ROOT = BASE_DIR / "media"


# ============================================================
# 12. LIMITES DE UPLOAD
# ============================================================

# Limite máximo de 10 MB por arquivo enviado.
# A validação específica de formato e conteúdo será feita
# também pelo aplicativo de reconhecimento.
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

# Limita o número de campos multipart processados por requisição.
DATA_UPLOAD_MAX_NUMBER_FIELDS = 1000

# Limita a quantidade de arquivos enviados em uma requisição.
DATA_UPLOAD_MAX_NUMBER_FILES = 10


# ============================================================
# 13. DJANGO REST FRAMEWORK
# ============================================================

REST_FRAMEWORK = {
    # Formatos de entrada aceitos pela API.
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.MultiPartParser",
    ],

    # Formato padrão das respostas da API.
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],

    # Não exige autenticação nesta fase inicial.
    # A autenticação poderá ser configurada posteriormente.
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],

    # Formato das mensagens de erro de validação.
    "EXCEPTION_HANDLER": (
        "rest_framework.views.exception_handler"
    ),
}


# ============================================================
# 14. CORS — COMUNICAÇÃO COM O FRONTEND
# ============================================================

# Permite configurar as origens autorizadas por variável
# de ambiente, separadas por vírgulas.
#
# Exemplo:
# DJANGO_CORS_ALLOWED_ORIGINS=http://localhost:5173
#
# Por padrão, autorizamos apenas origens locais comuns
# durante o desenvolvimento.
_origins = os.environ.get(
    "DJANGO_CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _origins.split(",")
    if origin.strip()
]

# Não habilitamos CORS_ALLOW_ALL_ORIGINS:
# outras origens precisam ser autorizadas explicitamente.


# ============================================================
# 15. CONFIGURAÇÕES ADICIONAIS
# ============================================================

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Evita incluir barras finais automaticamente nas rotas
# quando o cliente faz requisições à API.
APPEND_SLASH = False