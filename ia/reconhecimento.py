"""
Reconhecimento visual de símbolos Morse do CodeVision.

Responsabilidades:
- Receber imagens completas ou recortes individuais.
- Localizar componentes visuais que possam representar símbolos.
- Preparar os recortes com PreprocessadorCodeVision.
- Classificar os recortes usando ModeloCodeVision.
- Preservar coordenadas, ordem e confiança das classificações.

Este módulo não decodifica a mensagem Morse em texto.
A decodificação será responsabilidade de decodificacao.py.

Dependências do projeto:
- ia/modelo.py
- ia/preprocessamento.py

Dependências externas:
- opencv-python
- numpy
- pillow
- torch
"""

from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ia.modelo import ModeloCodeVision
from ia.preprocessamento import PreprocessadorCodeVision


CaminhoImagem = Union[str, Path]

CLASSES = {
    0: "fundo",
    1: "ponto",
    2: "traco",
}


class ErroReconhecimento(Exception):
    """Erro específico durante o reconhecimento visual."""


class ReconhecedorCodeVision:
    """
    Localiza e classifica possíveis símbolos Morse em imagens.

    Observação:
        A segmentação implementada é baseada em componentes conectados.
        Ela funciona melhor com símbolos escuros, bem separados e sobre
        um fundo claro. Não é um detector universal de Morse.
    """

    def __init__(
        self,
        modelo: ModeloCodeVision,
        preprocessador: Optional[PreprocessadorCodeVision] = None,
        confianca_minima: float = 0.60,
        area_minima: int = 3,
        area_maxima: Optional[int] = None,
        limiar_binario: int = 0,
    ):
        """
        Args:
            modelo: instância de ModeloCodeVision com pesos carregados.
            preprocessador: instância do preprocessador. Se omitida,
                uma instância padrão será criada.
            confianca_minima: confiança mínima para aceitar uma previsão.
            area_minima: área mínima, em pixels, de um componente.
            area_maxima: área máxima permitida para um componente.
                None significa sem limite superior explícito.
            limiar_binario: 0 utiliza limiarização automática de Otsu.
                Valores entre 1 e 254 usam um limiar fixo.
        """
        if not isinstance(modelo, ModeloCodeVision):
            raise TypeError(
                "modelo deve ser uma instância de ModeloCodeVision."
            )

        if not isinstance(confianca_minima, (int, float)):
            raise TypeError("confianca_minima deve ser numérica.")

        if not 0.0 <= confianca_minima <= 1.0:
            raise ValueError(
                "confianca_minima deve estar entre 0 e 1."
            )

        if (
            not isinstance(area_minima, int)
            or isinstance(area_minima, bool)
            or area_minima < 1
        ):
            raise ValueError("area_minima deve ser um inteiro positivo.")

        if area_maxima is not None:
            if (
                not isinstance(area_maxima, int)
                or isinstance(area_maxima, bool)
                or area_maxima < area_minima
            ):
                raise ValueError(
                    "area_maxima deve ser None ou um inteiro "
                    "maior ou igual a area_minima."
                )

        if (
            not isinstance(limiar_binario, int)
            or isinstance(limiar_binario, bool)
            or not 0 <= limiar_binario <= 254
        ):
            raise ValueError(
                "limiar_binario deve ser um inteiro entre 0 e 254."
            )

        self.modelo = modelo
        self.preprocessador = (
            preprocessador or PreprocessadorCodeVision()
        )
        self.confianca_minima = float(confianca_minima)
        self.area_minima = area_minima
        self.area_maxima = area_maxima
        self.limiar_binario = limiar_binario

    @staticmethod
    def _carregar_imagem(caminho: CaminhoImagem) -> Image.Image:
        """Carrega uma imagem de arquivo sem perder a orientação EXIF."""
        if not isinstance(caminho, (str, Path)):
            raise TypeError("O caminho deve ser uma string ou Path.")

        arquivo = Path(caminho).expanduser()

        if not arquivo.exists():
            raise FileNotFoundError(
                f"Imagem não encontrada: {arquivo}"
            )

        if not arquivo.is_file():
            raise ErroReconhecimento(
                f"O caminho não é um arquivo: {arquivo}"
            )

        try:
            from PIL import ImageOps

            with Image.open(arquivo) as imagem:
                imagem = ImageOps.exif_transpose(imagem)
                imagem.load()
                return imagem.convert("RGB")

        except (OSError, ValueError) as erro:
            raise ErroReconhecimento(
                f"Não foi possível abrir a imagem: {erro}"
            ) from erro

    @staticmethod
    def _validar_imagem_array(imagem: np.ndarray) -> np.ndarray:
        """Valida e converte um array de imagem para uint8."""
        if not isinstance(imagem, np.ndarray):
            raise TypeError("A imagem deve ser um array NumPy.")

        if imagem.size == 0:
            raise ErroReconhecimento("A imagem está vazia.")

        if imagem.ndim not in (2, 3):
            raise ErroReconhecimento(
                "A imagem deve possuir duas ou três dimensões."
            )

        if imagem.ndim == 3 and imagem.shape[2] not in (3, 4):
            raise ErroReconhecimento(
                "A imagem colorida deve possuir 3 canais (RGB/BGR) "
                "ou 4 canais (RGBA/BGRA)."
            )

        if imagem.dtype.kind not in "uif":
            raise ErroReconhecimento(
                "A imagem deve conter valores numéricos reais."
            )

        if not np.isfinite(imagem).all():
            raise ErroReconhecimento(
                "A imagem contém valores NaN ou infinitos."
            )

        if imagem.dtype.kind == "f":
            if imagem.min() < 0 or imagem.max() > 1:
                raise ErroReconhecimento(
                    "Arrays float devem ter valores entre 0 e 1."
                )

            imagem = np.round(imagem * 255).astype(np.uint8)

        else:
            if imagem.min() < 0 or imagem.max() > 255:
                raise ErroReconhecimento(
                    "Os valores dos pixels devem estar entre 0 e 255."
                )

            imagem = imagem.astype(np.uint8)

        return imagem

    def _obter_imagem_cinza(
        self,
        imagem: Union[Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Converte a entrada para um array de tons de cinza."""
        if isinstance(imagem, Image.Image):
            imagem_pil = imagem.convert("L")
            array = np.asarray(imagem_pil)

        elif isinstance(imagem, np.ndarray):
            array = self._validar_imagem_array(imagem)

            if array.ndim == 2:
                pass
            elif array.shape[2] == 4:
                array = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
            else:
                # Arrays recebidos diretamente pelo módulo são
                # interpretados como RGB, não BGR.
                array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

        else:
            raise TypeError(
                "A imagem deve ser um objeto PIL.Image ou NumPy."
            )

        if array.ndim != 2 or array.size == 0:
            raise ErroReconhecimento(
                "Não foi possível converter a imagem para escala de cinza."
            )

        return np.ascontiguousarray(array, dtype=np.uint8)

    def _binarizar(self, imagem_cinza: np.ndarray) -> np.ndarray:
        """
        Produz uma imagem binária com símbolos brancos sobre fundo preto.

        Otsu é usado por padrão. A limiarização fixa pode ser selecionada
        explicitamente por limiar_binario.
        """
        if self.limiar_binario == 0:
            _, binaria = cv2.threshold(
                imagem_cinza,
                0,
                255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
            )
        else:
            _, binaria = cv2.threshold(
                imagem_cinza,
                self.limiar_binario,
                255,
                cv2.THRESH_BINARY_INV,
            )

        return binaria

    def _localizar_componentes(
        self,
        imagem_cinza: np.ndarray,
    ) -> list[dict[str, int]]:
        """
        Localiza componentes conectados potencialmente correspondentes
        a pontos e traços.

        Cada componente contém x, y, largura, altura e área.
        As coordenadas têm origem no canto superior esquerdo.
        """
        binaria = self._binarizar(imagem_cinza)

        numero, rotulos, estatisticas, _ = (
            cv2.connectedComponentsWithStats(
                binaria,
                connectivity=8,
            )
        )

        componentes = []

        for indice in range(1, numero):
            x, y, largura, altura, area = (
                int(valor)
                for valor in estatisticas[indice]
            )

            if area < self.area_minima:
                continue

            if (
                self.area_maxima is not None
                and area > self.area_maxima
            ):
                continue

            # Evita considerar um componente que ocupe praticamente
            # toda a imagem como um símbolo individual.
            if (
                largura >= imagem_cinza.shape[1] * 0.98
                and altura >= imagem_cinza.shape[0] * 0.98
            ):
                continue

            componentes.append(
                {
                    "x": x,
                    "y": y,
                    "largura": largura,
                    "altura": altura,
                    "area": area,
                }
            )

        # Ordenação inicial: de cima para baixo e, na mesma linha,
        # da esquerda para a direita.
        componentes.sort(key=lambda c: (c["y"], c["x"]))

        return componentes

    @staticmethod
    def _recortar_com_margem(
        imagem_cinza: np.ndarray,
        componente: dict[str, int],
        margem: int = 2,
    ) -> np.ndarray:
        """Extrai um componente com uma pequena margem ao redor."""
        altura_imagem, largura_imagem = imagem_cinza.shape

        x1 = max(0, componente["x"] - margem)
        y1 = max(0, componente["y"] - margem)

        x2 = min(
            largura_imagem,
            componente["x"] + componente["largura"] + margem,
        )
        y2 = min(
            altura_imagem,
            componente["y"] + componente["altura"] + margem,
        )

        recorte = imagem_cinza[y1:y2, x1:x2]

        if recorte.size == 0:
            raise ErroReconhecimento(
                "Foi encontrado um componente sem pixels válidos."
            )

        return recorte

    def _classificar_recorte(
        self,
        recorte: np.ndarray,
    ) -> dict[str, Any]:
        """Prepara e classifica um recorte individual."""
        tensor = self.preprocessador.preparar_simbolo(recorte)

        previsao = self.modelo.prever(tensor)

        if not isinstance(previsao, dict):
            raise ErroReconhecimento(
                "ModeloCodeVision.prever() deveria retornar um dicionário."
            )

        classe = previsao.get("classe")
        simbolo = previsao.get("simbolo")
        confianca = previsao.get("confianca")
        probabilidades = previsao.get("probabilidades", {})

        if classe not in CLASSES:
            raise ErroReconhecimento(
                f"A classe retornada pelo modelo é inválida: {classe!r}"
            )

        if not isinstance(confianca, (int, float)):
            raise ErroReconhecimento(
                "O modelo retornou uma confiança inválida."
            )

        confianca = float(confianca)

        if not 0 <= confianca <= 1:
            raise ErroReconhecimento(
                "A confiança retornada deve estar entre 0 e 1."
            )

        # O fundo não representa um símbolo da mensagem.
        if classe == 0:
            simbolo_final = None
            status = "fundo"
        elif confianca < self.confianca_minima:
            simbolo_final = None
            status = "baixa_confianca"
        else:
            simbolo_final = "·" if classe == 1 else "−"
            status = "reconhecido"

        return {
            "classe": int(classe),
            "nome_classe": CLASSES[int(classe)],
            "simbolo": simbolo_final,
            "confianca": confianca,
            "probabilidades": probabilidades,
            "status": status,
        }

    def reconhecer_recorte(
        self,
        imagem: Union[Image.Image, np.ndarray],
    ) -> dict[str, Any]:
        """
        Reconhece um único recorte que já contenha um ponto ou traço.

        Não realiza segmentação da imagem.
        """
        resultado = self._classificar_recorte(
            self._obter_imagem_cinza(imagem)
        )

        return resultado

    def reconhecer_imagem(
        self,
        imagem: Union[Image.Image, np.ndarray],
    ) -> dict[str, Any]:
        """
        Localiza e classifica possíveis símbolos em uma imagem completa.

        Args:
            imagem: imagem PIL ou array NumPy.

        Returns:
            Dicionário com dimensões, quantidade de componentes,
            resultados individuais e possíveis símbolos reconhecidos.

        Limitação:
            A segmentação por componentes conectados não resolve
            automaticamente símbolos encostados, ruídos complexos,
            perspectivas extremas ou organização de várias linhas.
        """
        imagem_cinza = self._obter_imagem_cinza(imagem)
        componentes = self._localizar_componentes(imagem_cinza)

        resultados = []

        for indice, componente in enumerate(componentes):
            recorte = self._recortar_com_margem(
                imagem_cinza,
                componente,
            )

            classificacao = self._classificar_recorte(recorte)

            resultados.append(
                {
                    "indice": indice,
                    "coordenadas": {
                        "x": componente["x"],
                        "y": componente["y"],
                        "largura": componente["largura"],
                        "altura": componente["altura"],
                    },
                    "area": componente["area"],
                    **classificacao,
                }
            )

        simbolos_reconhecidos = [
            resultado["simbolo"]
            for resultado in resultados
            if resultado["status"] == "reconhecido"
        ]

        return {
            "dimensoes_imagem": {
                "largura": int(imagem_cinza.shape[1]),
                "altura": int(imagem_cinza.shape[0]),
            },
            "quantidade_componentes": len(componentes),
            "quantidade_reconhecidos": len(simbolos_reconhecidos),
            "simbolos": simbolos_reconhecidos,
            "resultados": resultados,
            "observacao": (
                "Os símbolos estão ordenados inicialmente por posição "
                "vertical e horizontal. A separação de letras e palavras "
                "ainda deve ser feita pelo módulo de decodificação."
            ),
        }

    def reconhecer_arquivo(
        self,
        caminho: CaminhoImagem,
    ) -> dict[str, Any]:
        """Carrega um arquivo e reconhece os símbolos encontrados."""
        imagem = self._carregar_imagem(caminho)
        return self.reconhecer_imagem(imagem)