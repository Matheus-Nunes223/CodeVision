"""
Serializadores da API do CodeVision.

Responsáveis por validar os dados recebidos e padronizar
as respostas do sistema de reconhecimento de imagens.
"""

from rest_framework import serializers


# ============================================================
# CONSTANTES
# ============================================================

TAMANHO_MAXIMO_IMAGEM = 10 * 1024 * 1024  # 10 MB

FORMATOS_IMAGEM_PERMITIDOS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
}


# ============================================================
# SERIALIZADOR DE ENTRADA
# ============================================================

class ImagemReconhecimentoSerializer(serializers.Serializer):
    """
    Valida a imagem enviada pelo usuário para reconhecimento.

    Campo esperado:
        imagem: arquivo de imagem enviado por multipart/form-data.
    """

    imagem = serializers.ImageField(
        required=True,
        allow_empty_file=False,
        error_messages={
            "required": "Envie uma imagem para realizar o reconhecimento.",
            "invalid": "O arquivo enviado não é uma imagem válida.",
            "empty": "A imagem enviada está vazia.",
        },
    )

    def validate_imagem(self, imagem):
        """
        Verifica o tamanho, o formato e as dimensões da imagem.
        """

        # Verifica o tamanho do arquivo.
        if imagem.size > TAMANHO_MAXIMO_IMAGEM:
            raise serializers.ValidationError(
                "A imagem não pode ultrapassar 10 MB."
            )

        # Verifica o formato detectado pelo Pillow.
        formato = getattr(imagem, "image", None)

        if formato is None:
            raise serializers.ValidationError(
                "Não foi possível verificar o formato da imagem."
            )

        formato = getattr(formato, "format", None)

        if formato not in FORMATOS_IMAGEM_PERMITIDOS:
            formatos = ", ".join(sorted(FORMATOS_IMAGEM_PERMITIDOS))
            raise serializers.ValidationError(
                f"Formato não permitido. Formatos aceitos: {formatos}."
            )

        # Verifica as dimensões da imagem.
        largura, altura = imagem.image.size

        if largura <= 0 or altura <= 0:
            raise serializers.ValidationError(
                "A imagem possui dimensões inválidas."
            )

        # Limita dimensões excessivas para evitar processamento abusivo.
        if largura > 10000 or altura > 10000:
            raise serializers.ValidationError(
                "As dimensões da imagem não podem ultrapassar "
                "10000 pixels em cada lado."
            )

        return imagem


# ============================================================
# SERIALIZADOR DE SAÍDA
# ============================================================

class SimboloReconhecidoSerializer(serializers.Serializer):
    """
    Representa um símbolo identificado pelo modelo de IA.
    """

    simbolo = serializers.CharField(
        required=True,
        allow_blank=False,
    )

    classe = serializers.IntegerField(
        required=True,
        min_value=0,
        max_value=2,
    )

    confianca = serializers.FloatField(
        required=True,
        min_value=0.0,
        max_value=1.0,
    )

    probabilidades = serializers.DictField(
        child=serializers.FloatField(
            min_value=0.0,
            max_value=1.0,
        ),
        required=False,
    )


class ResultadoReconhecimentoSerializer(serializers.Serializer):
    """
    Estrutura a resposta completa do reconhecimento de uma imagem.
    """

    sucesso = serializers.BooleanField(
        required=True,
    )

    mensagem = serializers.CharField(
        required=True,
    )

    texto_reconhecido = serializers.CharField(
        required=False,
        allow_blank=True,
    )

    mensagem_decodificada = serializers.CharField(
        required=False,
        allow_blank=True,
    )

    quantidade_simbolos = serializers.IntegerField(
        required=True,
        min_value=0,
    )

    simbolos = SimboloReconhecidoSerializer(
        many=True,
        required=True,
    )

    tempo_processamento_ms = serializers.FloatField(
        required=False,
        min_value=0.0,
    )