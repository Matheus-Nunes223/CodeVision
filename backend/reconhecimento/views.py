"""
Views da API de reconhecimento de imagens do CodeVision.

Endpoints:
- POST /api/reconhecimento/       -> reconhece símbolos em uma imagem.
- GET  /api/reconhecimento/status/ -> verifica o estado do serviço de IA.
"""

import logging

from rest_framework import status
from rest_framework.decorators import (
    api_view,
    parser_classes,
    permission_classes,
)
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from reconhecimento.exceptions import (
    ErroProcessamentoImagem,
    ErroReconhecimento,
    ErroValidacaoImagem,
    ModeloIAIndisponivel,
)
from reconhecimento.serializers import (
    ImagemReconhecimentoSerializer,
    ResultadoReconhecimentoSerializer,
)
from reconhecimento.services import servico_reconhecimento
from reconhecimento.validators import validar_imagem


logger = logging.getLogger(__name__)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])
def reconhecer_imagem(request):
    """
    Recebe uma imagem e solicita o reconhecimento dos símbolos.

    A imagem deve ser enviada pelo campo 'imagem', usando
    multipart/form-data.

    Retorna:
        200: imagem processada com sucesso.
        400: requisição ou imagem inválida.
        422: não foi possível reconhecer os símbolos.
        500: erro interno durante o processamento.
        503: modelo de IA indisponível.
    """
    serializer_entrada = ImagemReconhecimentoSerializer(
        data=request.data
    )

    if not serializer_entrada.is_valid():
        return Response(
            {
                "sucesso": False,
                "mensagem": "Os dados enviados são inválidos.",
                "erros": serializer_entrada.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    arquivo = serializer_entrada.validated_data["imagem"]

    try:
        # Aplica as validações específicas da imagem antes de chamar a IA.
        validar_imagem(arquivo)

        # O serviço coordena o processamento e o reconhecimento.
        resultado = servico_reconhecimento.reconhecer_imagem(arquivo)

        # Confere se o resultado possui a estrutura esperada pela API.
        serializer_saida = ResultadoReconhecimentoSerializer(
            data=resultado
        )

        if not serializer_saida.is_valid():
            logger.error(
                "O serviço de reconhecimento retornou dados incompatíveis "
                "com o contrato da API: %s",
                serializer_saida.errors,
            )

            return Response(
                {
                    "sucesso": False,
                    "mensagem": (
                        "O reconhecimento foi processado, mas o resultado "
                        "não possui o formato esperado pela API."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            serializer_saida.validated_data,
            status=status.HTTP_200_OK,
        )

    except ErroValidacaoImagem as erro:
        return Response(
            {
                "sucesso": False,
                "mensagem": str(erro),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    except ModeloIAIndisponivel:
        logger.warning(
            "Solicitação de reconhecimento recebida, mas o modelo "
            "de IA está indisponível."
        )

        return Response(
            {
                "sucesso": False,
                "mensagem": (
                    "O modelo de IA não está disponível no momento. "
                    "Verifique se ele foi treinado e configurado."
                ),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    except ErroProcessamentoImagem as erro:
        return Response(
            {
                "sucesso": False,
                "mensagem": str(erro),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    except ErroReconhecimento:
        logger.exception(
            "Ocorreu um erro durante o reconhecimento da imagem."
        )

        return Response(
            {
                "sucesso": False,
                "mensagem": (
                    "Não foi possível concluir o reconhecimento da imagem."
                ),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    except Exception:
        logger.exception(
            "Erro inesperado no endpoint de reconhecimento."
        )

        return Response(
            {
                "sucesso": False,
                "mensagem": (
                    "Ocorreu um erro interno ao processar a imagem."
                ),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def status_reconhecimento(request):
    """
    Informa se o backend está funcionando e se o modelo está disponível.

    Este endpoint não carrega o modelo na memória: apenas verifica
    sua disponibilidade e se já foi carregado pelo serviço.
    """
    try:
        caminho_modelo = servico_reconhecimento.caminho_modelo

        modelo_disponivel = caminho_modelo.is_file()
        modelo_carregado = servico_reconhecimento.reconhecedor is not None

        return Response(
            {
                "status": "online",
                "servico": "CodeVision",
                "modelo_disponivel": modelo_disponivel,
                "modelo_carregado": modelo_carregado,
            },
            status=status.HTTP_200_OK,
        )

    except Exception:
        logger.exception(
            "Não foi possível consultar o estado do serviço de IA."
        )

        return Response(
            {
                "status": "erro",
                "servico": "CodeVision",
                "mensagem": (
                    "Não foi possível verificar o estado do serviço de IA."
                ),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )