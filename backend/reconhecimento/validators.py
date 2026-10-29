"""
Validadores de imagens utilizados pelo backend do CodeVision.

Responsabilidades:
- Validar o tamanho máximo dos arquivos.
- Verificar os formatos de imagem permitidos.
- Validar as dimensões das imagens.
- Detectar arquivos corrompidos ou que não sejam imagens válidas.
- Garantir que a imagem possa ser processada antes de enviá-la à IA.
"""

from PIL import Image, UnidentifiedImageError

from reconhecimento.exceptions import (
    ErroValidacaoImagem,
    FormatoImagemNaoSuportado,
    TamanhoImagemExcedido,
)


# Limites de validação
TAMANHO_MAXIMO_IMAGEM = 10 * 1024 * 1024  # 10 MB
LARGURA_MAXIMA_IMAGEM = 10_000
ALTURA_MAXIMA_IMAGEM = 10_000

# Formatos aceitos pelo CodeVision
FORMATOS_PERMITIDOS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
}


def validar_tamanho_arquivo(arquivo):
    """
    Verifica se o arquivo respeita o limite máximo de tamanho.

    Args:
        arquivo: arquivo recebido pelo backend.

    Raises:
        TamanhoImagemExcedido: se o arquivo ultrapassar o limite.
        ErroValidacaoImagem: se o tamanho não puder ser determinado.
    """
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
    """
    Verifica o formato real da imagem utilizando o Pillow.

    A validação não depende apenas da extensão ou do Content-Type
    informado pelo cliente.

    Args:
        arquivo: arquivo recebido pelo backend.

    Returns:
        str: formato identificado, como PNG ou JPEG.

    Raises:
        FormatoImagemNaoSuportado: se o formato não for permitido.
        ErroValidacaoImagem: se o arquivo não for uma imagem válida.
    """
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
    """
    Verifica se a largura e a altura da imagem estão dentro dos limites.

    Returns:
        tuple[int, int]: largura e altura da imagem.

    Raises:
        ErroValidacaoImagem: se as dimensões forem inválidas.
    """
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
    """
    Verifica se a imagem pode ser carregada integralmente.

    O método verify() do Pillow identifica diversos problemas estruturais,
    mas não substitui a validação de formato e dimensões.
    """
    try:
        arquivo.seek(0)

        with Image.open(arquivo) as imagem:
            imagem.verify()

        # verify() exige que o arquivo seja reaberto para carregar os pixels.
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
    """
    Executa todas as validações necessárias antes do reconhecimento.

    Args:
        arquivo: arquivo de imagem recebido pelo backend.

    Returns:
        dict: informações básicas sobre a imagem validada.
    """
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