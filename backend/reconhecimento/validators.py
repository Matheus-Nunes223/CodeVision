from PIL import Image, UnidentifiedImageError

from reconhecimento.exceptions import (
    ErroValidacaoImagem,
    FormatoImagemNaoSuportado,
    TamanhoImagemExcedido,
)


TAMANHO_MAXIMO_IMAGEM = 10 * 1024 * 1024
LARGURA_MAXIMA_IMAGEM = 10_000
ALTURA_MAXIMA_IMAGEM = 10_000

FORMATOS_PERMITIDOS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
}


def validar_tamanho_arquivo(arquivo):
    tamanho = getattr(arquivo, "size", None)

    if tamanho is None:
        raise ErroValidacaoImagem(
            "Não foi possível determinar o tamanho da imagem."
        )

    if tamanho <= 0:
        raise ErroValidacaoImagem(
            "O arquivo enviado está vazio."
        )

    if tamanho > TAMANHO_MAXIMO_IMAGEM:
        raise TamanhoImagemExcedido(
            "A imagem ultrapassa o limite permitido de 10 MB."
        )


def validar_formato_imagem(arquivo):
    try:
        arquivo.seek(0)

        with Image.open(arquivo) as imagem:
            formato = imagem.format

            if formato not in FORMATOS_PERMITIDOS:
                raise FormatoImagemNaoSuportado(
                    "Formato de imagem não suportado. "
                    "Utilize PNG, JPEG, WEBP ou BMP."
                )

            return formato

    except FormatoImagemNaoSuportado:
        raise

    except (UnidentifiedImageError, OSError, ValueError) as erro:
        raise ErroValidacaoImagem(
            "O arquivo enviado não é uma imagem válida ou está corrompido."
        ) from erro

    finally:
        arquivo.seek(0)


def validar_dimensoes_imagem(arquivo):
    try:
        arquivo.seek(0)

        with Image.open(arquivo) as imagem:
            largura, altura = imagem.size

            if largura <= 0 or altura <= 0:
                raise ErroValidacaoImagem(
                    "A imagem possui dimensões inválidas."
                )

            if (
                largura > LARGURA_MAXIMA_IMAGEM
                or altura > ALTURA_MAXIMA_IMAGEM
            ):
                raise ErroValidacaoImagem(
                    "A imagem ultrapassa as dimensões máximas permitidas "
                    f"de {LARGURA_MAXIMA_IMAGEM} × "
                    f"{ALTURA_MAXIMA_IMAGEM} pixels."
                )

            return largura, altura

    except ErroValidacaoImagem:
        raise

    except (UnidentifiedImageError, OSError, ValueError) as erro:
        raise ErroValidacaoImagem(
            "Não foi possível verificar as dimensões da imagem."
        ) from erro

    finally:
        arquivo.seek(0)


def validar_integridade_imagem(arquivo):
    try:
        arquivo.seek(0)

        with Image.open(arquivo) as imagem:
            imagem.verify()

        arquivo.seek(0)

        with Image.open(arquivo) as imagem:
            imagem.load()

    except (UnidentifiedImageError, OSError, ValueError) as erro:
        raise ErroValidacaoImagem(
            "A imagem está corrompida ou não pode ser processada."
        ) from erro

    finally:
        arquivo.seek(0)


def validar_imagem(arquivo):
    if arquivo is None:
        raise ErroValidacaoImagem(
            "Nenhuma imagem foi enviada."
        )

    validar_tamanho_arquivo(arquivo)
    formato = validar_formato_imagem(arquivo)
    largura, altura = validar_dimensoes_imagem(arquivo)
    validar_integridade_imagem(arquivo)

    return {
        "formato": formato,
        "largura": largura,
        "altura": altura,
        "tamanho_bytes": arquivo.size,
    }