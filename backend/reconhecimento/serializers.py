from rest_framework import serializers


TAMANHO_MAXIMO_IMAGEM = 10 * 1024 * 1024

FORMATOS_IMAGEM_PERMITIDOS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
}


class ImagemReconhecimentoSerializer(serializers.Serializer):

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

        if imagem.size > TAMANHO_MAXIMO_IMAGEM:
            raise serializers.ValidationError(
                "A imagem não pode ultrapassar 10 MB."
            )

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

        largura, altura = imagem.image.size

        if largura <= 0 or altura <= 0:
            raise serializers.ValidationError(
                "A imagem possui dimensões inválidas."
            )

        if largura > 10000 or altura > 10000:
            raise serializers.ValidationError(
                "As dimensões da imagem não podem ultrapassar "
                "10000 pixels em cada lado."
            )

        return imagem


class CoordenadasSimboloSerializer(serializers.Serializer):
    x = serializers.IntegerField(min_value=0)
    y = serializers.IntegerField(min_value=0)
    largura = serializers.IntegerField(min_value=1)
    altura = serializers.IntegerField(min_value=1)


class DimensoesImagemSerializer(serializers.Serializer):
    largura = serializers.IntegerField(min_value=1)
    altura = serializers.IntegerField(min_value=1)


class SimboloReconhecidoSerializer(serializers.Serializer):

    simbolo = serializers.CharField(
        required=True,
        allow_blank=False,
    )

    classe = serializers.IntegerField(
        required=True,
        min_value=0,
        max_value=2,
    )

    nome_classe = serializers.CharField(required=False)

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

    coordenadas = CoordenadasSimboloSerializer(required=False)


class ResultadoDetalhadoSerializer(serializers.Serializer):
    indice = serializers.IntegerField(min_value=0)
    linha = serializers.IntegerField(min_value=0, required=False)
    coordenadas = CoordenadasSimboloSerializer()
    area = serializers.IntegerField(min_value=1)
    classe = serializers.IntegerField(
        min_value=0, max_value=2, allow_null=True
    )
    nome_classe = serializers.CharField()
    simbolo = serializers.CharField(allow_null=True, allow_blank=True)
    confianca = serializers.FloatField(
        min_value=0.0, max_value=1.0, allow_null=True
    )
    probabilidades = serializers.DictField(
        child=serializers.FloatField(min_value=0.0, max_value=1.0),
    )
    status = serializers.ChoiceField(
        choices=("fundo", "baixa_confianca", "reconhecido", "separador"),
    )


class ResultadoReconhecimentoSerializer(serializers.Serializer):

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

    dimensoes_imagem = DimensoesImagemSerializer(required=False)
    resultados_detalhados = ResultadoDetalhadoSerializer(
        many=True,
        required=False,
    )