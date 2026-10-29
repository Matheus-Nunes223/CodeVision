"""
Pré-processamento de imagens do CodeVision.

Responsabilidades:
- Carregar imagens a partir de arquivos ou arrays.
- Converter imagens para escala de cinza.
- Melhorar o contraste quando necessário.
- Preparar imagens para segmentação posterior.
- Preparar recortes individuais de símbolos Morse para a CNN.

Contrato com modelo.py:
- Cada símbolo é representado por um tensor de formato (1, 32, 32).
- Os pixels são normalizados para o intervalo [0, 1].
- A dimensão de canal é única, pois a rede utiliza imagens em escala de cinza.

Este módulo NÃO realiza:
- Detecção dos símbolos em uma imagem completa.
- Classificação dos símbolos pela CNN.
- Decodificação de mensagens Morse.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError


CaminhoImagem = Union[str, Path]
TAMANHO_SIMBOLO = (32, 32)


class ErroPreprocessamento(Exception):
    """Erro específico durante o pré-processamento de imagens."""


class PreprocessadorCodeVision:
    """Prepara imagens para as etapas seguintes do CodeVision."""

    def __init__(
        self,
        tamanho_simbolo: tuple[int, int] = TAMANHO_SIMBOLO,
        melhorar_contraste: bool = True,
        inverter_automaticamente: bool = True,
    ):
        """
        Args:
            tamanho_simbolo: dimensões (largura, altura) dos símbolos.
            melhorar_contraste: aplica melhoria de contraste às imagens.
            inverter_automaticamente: tenta manter os símbolos escuros
                sobre fundo claro, quando essa escolha for evidente.
        """
        if (
            not isinstance(tamanho_simbolo, tuple)
            or len(tamanho_simbolo) != 2
            or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                for dim in tamanho_simbolo
            )
        ):
            raise ValueError(
                "tamanho_simbolo deve ser uma tupla de dois inteiros positivos."
            )

        self.tamanho_simbolo = tamanho_simbolo
        self.melhorar_contraste = melhorar_contraste
        self.inverter_automaticamente = inverter_automaticamente

    @staticmethod
    def _validar_caminho(caminho: CaminhoImagem) -> Path:
        """Valida o caminho de uma imagem."""
        if not isinstance(caminho, (str, Path)):
            raise TypeError("O caminho da imagem deve ser uma string ou Path.")

        arquivo = Path(caminho).expanduser()

        if not arquivo.exists():
            raise FileNotFoundError(
                f"Imagem não encontrada: {arquivo}"
            )

        if not arquivo.is_file():
            raise ErroPreprocessamento(
                f"O caminho informado não é um arquivo: {arquivo}"
            )

        return arquivo

    @staticmethod
    def _converter_para_imagem(imagem: Image.Image) -> Image.Image:
        """Converte uma imagem para escala de cinza de 8 bits."""
        if not isinstance(imagem, Image.Image):
            raise TypeError("O objeto informado não é uma imagem PIL.")

        return ImageOps.grayscale(imagem)

    @staticmethod
    def _inverter_se_necessario(imagem: Image.Image) -> Image.Image:
        """
        Tenta detectar imagens com fundo predominantemente escuro.

        A inversão é uma heurística simples, não uma garantia de que
        os símbolos estejam corretamente identificados.
        """
        pixels = np.asarray(imagem, dtype=np.float32)

        if pixels.size == 0:
            raise ErroPreprocessamento("A imagem está vazia.")

        brilho_medio = float(pixels.mean())

        if brilho_medio < 127:
            return ImageOps.invert(imagem)

        return imagem

    def carregar_imagem(self, caminho: CaminhoImagem) -> Image.Image:
        """
        Carrega e prepara uma imagem completa.

        Returns:
            Imagem PIL em escala de cinza, pronta para etapas posteriores.
        """
        arquivo = self._validar_caminho(caminho)

        try:
            with Image.open(arquivo) as imagem_original:
                imagem = ImageOps.exif_transpose(imagem_original)
                imagem.load()
                imagem = self._converter_para_imagem(imagem)

        except (UnidentifiedImageError, OSError, ValueError) as erro:
            raise ErroPreprocessamento(
                f"Não foi possível abrir a imagem '{arquivo}': {erro}"
            ) from erro

        if imagem.width <= 0 or imagem.height <= 0:
            raise ErroPreprocessamento("A imagem possui dimensões inválidas.")

        if self.inverter_automaticamente:
            imagem = self._inverter_se_necessario(imagem)

        if self.melhorar_contraste:
            imagem = ImageEnhance.Contrast(imagem).enhance(1.5)

        return imagem

    def carregar_array(self, imagem: np.ndarray) -> Image.Image:
        """
        Converte um array NumPy em uma imagem PIL em escala de cinza.

        Aceita:
        - Imagens em tons de cinza (H, W).
        - Imagens RGB (H, W, 3).
        - Imagens RGBA (H, W, 4).

        Arrays float devem conter valores entre 0 e 1.
        Arrays inteiros devem conter valores entre 0 e 255.
        """
        if not isinstance(imagem, np.ndarray):
            raise TypeError("A imagem deve ser um array NumPy.")

        if imagem.size == 0:
            raise ErroPreprocessamento("O array da imagem está vazio.")

        if imagem.dtype.kind not in "uif":
            raise ErroPreprocessamento(
                "O array deve conter valores numéricos reais."
            )

        if not np.isfinite(imagem).all():
            raise ErroPreprocessamento(
                "O array contém valores NaN ou infinitos."
            )

        if imagem.ndim not in (2, 3):
            raise ErroPreprocessamento(
                "A imagem deve ter duas dimensões (cinza) "
                "ou três dimensões (RGB/RGBA)."
            )

        if imagem.ndim == 3 and imagem.shape[2] not in (3, 4):
            raise ErroPreprocessamento(
                "A imagem colorida deve possuir 3 canais (RGB) "
                "ou 4 canais (RGBA)."
            )

        if imagem.dtype.kind == "f":
            if imagem.min() < 0 or imagem.max() > 1:
                raise ErroPreprocessamento(
                    "Arrays float devem ter valores entre 0 e 1."
                )

            array = np.round(imagem * 255).astype(np.uint8)

        else:
            if imagem.min() < 0 or imagem.max() > 255:
                raise ErroPreprocessamento(
                    "Os valores do array devem estar entre 0 e 255."
                )

            array = imagem.astype(np.uint8)

        try:
            imagem_pil = Image.fromarray(array)
            imagem_pil = ImageOps.grayscale(imagem_pil)

            if self.inverter_automaticamente:
                imagem_pil = self._inverter_se_necessario(imagem_pil)

            if self.melhorar_contraste:
                imagem_pil = ImageEnhance.Contrast(imagem_pil).enhance(1.5)

            return imagem_pil

        except (OSError, ValueError) as erro:
            raise ErroPreprocessamento(
                f"Não foi possível converter o array: {erro}"
            ) from erro

    def preparar_simbolo(
        self,
        imagem: Union[Image.Image, np.ndarray],
        inverter: bool = False,
    ) -> torch.Tensor:
        """
        Prepara um recorte que contém um único símbolo Morse.

        Args:
            imagem: recorte do ponto ou traço, em PIL ou NumPy.
            inverter: inverte explicitamente os tons da imagem.

        Returns:
            Tensor float32 com formato (1, 32, 32), normalizado para [0, 1].

        Importante:
            O método pressupõe que o recorte já contenha um símbolo.
            Ele não localiza nem separa símbolos de uma mensagem completa.
        """
        if isinstance(imagem, np.ndarray):
            imagem_pil = self.carregar_array(imagem)
        elif isinstance(imagem, Image.Image):
            imagem_pil = self._converter_para_imagem(imagem)
        else:
            raise TypeError(
                "A imagem deve ser um objeto PIL.Image ou um array NumPy."
            )

        if inverter:
            imagem_pil = ImageOps.invert(imagem_pil)

        if self.melhorar_contraste:
            imagem_pil = ImageEnhance.Contrast(imagem_pil).enhance(1.5)

        imagem_pil = imagem_pil.resize(
            self.tamanho_simbolo,
            Image.Resampling.LANCZOS,
        )

        array = np.asarray(imagem_pil, dtype=np.float32) / 255.0

        # O modelo recebe um canal de imagem: (1, altura, largura).
        tensor = torch.from_numpy(array.copy()).unsqueeze(0)

        return tensor.to(dtype=torch.float32)

    def preparar_arquivo_simbolo(
        self,
        caminho: CaminhoImagem,
    ) -> torch.Tensor:
        """
        Carrega um arquivo e prepara seu conteúdo como símbolo individual.

        Use este método somente quando o arquivo já contiver um recorte
        com um único símbolo, e não uma mensagem Morse inteira.
        """
        imagem = self.carregar_imagem(caminho)
        return self.preparar_simbolo(imagem)

    def salvar_imagem_preprocessada(
        self,
        imagem: Union[Image.Image, np.ndarray],
        destino: CaminhoImagem,
    ) -> Path:
        """
        Salva uma imagem em escala de cinza após o pré-processamento.

        Útil para inspecionar visualmente o resultado durante os testes.
        """
        if isinstance(imagem, np.ndarray):
            imagem_pil = self.carregar_array(imagem)
        elif isinstance(imagem, Image.Image):
            imagem_pil = self._converter_para_imagem(imagem)
        else:
            raise TypeError(
                "A imagem deve ser um objeto PIL.Image ou um array NumPy."
            )

        caminho_destino = Path(destino).expanduser()
        caminho_destino.parent.mkdir(parents=True, exist_ok=True)

        try:
            imagem_pil.save(caminho_destino)
        except (OSError, ValueError) as erro:
            raise ErroPreprocessamento(
                f"Não foi possível salvar a imagem: {erro}"
            ) from erro

        return caminho_destino