"""
Reconhecimento visual de símbolos Morse do CodeVision.

Responsabilidades:
- Receber imagens completas ou recortes individuais.
- Localizar componentes visuais que possam representar símbolos.
- Preparar os recortes com PreprocessadorCodeVision.
- Classificar os recortes usando ModeloCodeVision.
- Preservar coordenadas, ordem e confiança das classificações.
- Decodificar cada linha de texto em letras e palavras.

Dependências do projeto:
- ia/modelo.py
- ia/preprocessamento.py
- ia/segmentacao.py
- ia/decodificacao.py

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

from ia.decodificacao import SimboloPosicionado, decodificar_linha
from ia.modelo import ModeloCodeVision
from ia.preprocessamento import PreprocessadorCodeVision
from ia.segmentacao import (
    Componente,
    ResultadoSegmentacao,
    alongamento,
    recorte_isolado,
    segmentar,
)


CaminhoImagem = Union[str, Path]

CLASSES = {
    0: "fundo",
    1: "ponto",
    2: "traco",
}

# Alongamento (comprimento ÷ espessura) a partir do qual o formato é
# inequívoco. Entre os dois limites, vale a decisão da CNN.
ALONGAMENTO_PONTO = 1.5
ALONGAMENTO_TRACO = 2.5


class ErroReconhecimento(Exception):
    """Erro específico durante o reconhecimento visual."""


class ReconhecedorCodeVision:
    """
    Localiza e classifica possíveis símbolos Morse em imagens.

    A segmentação (ia/segmentacao.py) trata fotos reais: corrige
    iluminação e inclinação, ignora o que está fora do papel e remove
    as pautas do caderno. A CNN decide ponto, traço ou fundo, e a
    decodificação separa letras e palavras pelos espaços e barras "/".
    """

    def __init__(
        self,
        modelo: ModeloCodeVision,
        preprocessador: Optional[PreprocessadorCodeVision] = None,
        confianca_minima: float = 0.60,
    ):
        """
        Args:
            modelo: instância de ModeloCodeVision com pesos carregados.
            preprocessador: instância do preprocessador. Se omitida,
                uma instância padrão será criada.
            confianca_minima: confiança mínima para aceitar uma previsão.
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

        self.modelo = modelo
        self.preprocessador = (
            preprocessador or PreprocessadorCodeVision()
        )
        self.confianca_minima = float(confianca_minima)

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

    @staticmethod
    def _conferir_formato(
        classificacao: dict[str, Any],
        razao: float,
    ) -> dict[str, Any]:
        """
        Confere ponto/traço da CNN com o alongamento do símbolo.

        Em símbolos muito pequenos (fotos de longe ou comprimidas) a CNN
        pode confundir ponto e traço. Quando o formato é inequívoco, ele
        prevalece; a correção fica registrada em "corrigido_por_formato".
        O fundo continua sendo decidido só pela CNN.
        """
        resultado = {
            **classificacao,
            "alongamento": round(float(razao), 2),
            "corrigido_por_formato": False,
        }

        if classificacao["classe"] == 0:
            return resultado

        if razao >= ALONGAMENTO_TRACO:
            classe = 2
        elif razao <= ALONGAMENTO_PONTO:
            classe = 1
        else:
            return resultado

        if classe != classificacao["classe"]:
            resultado.update(
                classe=classe,
                nome_classe=CLASSES[classe],
                corrigido_por_formato=True,
            )
            if resultado["status"] == "reconhecido":
                resultado["simbolo"] = "·" if classe == 1 else "−"

        return resultado

    @staticmethod
    def _coordenadas_originais(
        componente: Componente,
        segmentacao: ResultadoSegmentacao,
    ) -> dict[str, int]:
        """
        Converte a caixa da imagem de trabalho (reduzida e endireitada)
        para a imagem original enviada.
        """
        altura, largura = segmentacao.imagem.shape
        matriz = cv2.getRotationMatrix2D(
            (largura / 2, altura / 2), segmentacao.angulo, 1.0
        )
        inversa = cv2.invertAffineTransform(matriz)

        cantos = np.array(
            [
                [componente.x, componente.y],
                [componente.direita, componente.y],
                [componente.x, componente.y + componente.altura],
                [componente.direita, componente.y + componente.altura],
            ],
            dtype=np.float64,
        )
        originais = cantos @ inversa[:, :2].T + inversa[:, 2]
        originais /= segmentacao.escala

        x_min, y_min = np.floor(originais.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(originais.max(axis=0)).astype(int)

        return {
            "x": int(max(0, x_min)),
            "y": int(max(0, y_min)),
            "largura": int(x_max - x_min),
            "altura": int(y_max - y_min),
        }

    def _classificar_recorte(
        self,
        recorte: np.ndarray,
    ) -> dict[str, Any]:
        """Prepara e classifica um recorte individual."""
        # PIL, como no treino: evita a inversão automática de arrays.
        tensor = self.preprocessador.preparar_simbolo(
            Image.fromarray(recorte)
        )

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
        Localiza, classifica e decodifica os símbolos de uma imagem.

        Args:
            imagem: imagem PIL ou array NumPy (RGB ou cinza).

        Returns:
            Dicionário com os resultados de cada símbolo, as linhas de
            texto encontradas (Morse e texto decodificado) e a mensagem
            completa, uma linha de texto por linha da imagem.

        Observação:
            Na decodificação, um símbolo abaixo da confiança mínima entra
            com a classe mais provável (ponto ou traço) e fica marcado
            como "baixa_confianca" nos resultados.
        """
        imagem_cinza = self._obter_imagem_cinza(imagem)
        segmentacao = segmentar(imagem_cinza)

        resultados = []
        linhas = []

        for numero_linha, componentes in enumerate(segmentacao.linhas):
            posicionados = []

            for componente in componentes:
                coordenadas = self._coordenadas_originais(
                    componente, segmentacao
                )

                if componente.barra:
                    classificacao = {
                        "classe": None,
                        "nome_classe": "separador",
                        "simbolo": "/",
                        "confianca": None,
                        "probabilidades": {},
                        "status": "separador",
                    }
                    posicionados.append(
                        SimboloPosicionado(
                            "/", componente.x, componente.largura
                        )
                    )
                else:
                    classificacao = self._classificar_recorte(
                        recorte_isolado(segmentacao, componente)
                    )
                    classificacao = self._conferir_formato(
                        classificacao,
                        alongamento(segmentacao, componente),
                    )
                    if classificacao["classe"] != 0:
                        posicionados.append(
                            SimboloPosicionado(
                                "." if classificacao["classe"] == 1 else "-",
                                componente.x,
                                componente.largura,
                            )
                        )

                resultados.append(
                    {
                        "indice": len(resultados),
                        "linha": numero_linha,
                        "coordenadas": coordenadas,
                        "area": int(componente.area / segmentacao.escala**2),
                        **classificacao,
                    }
                )

            if sum(1 for s in posicionados if s.simbolo != "/") == 0:
                continue

            decodificada = decodificar_linha(
                posicionados, segmentacao.espessura
            )
            linhas.append({"linha": numero_linha, **decodificada})

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
            "quantidade_componentes": len(resultados),
            "quantidade_reconhecidos": len(simbolos_reconhecidos),
            "simbolos": simbolos_reconhecidos,
            "resultados": resultados,
            "linhas": linhas,
            "morse": "\n".join(linha["morse"] for linha in linhas),
            "mensagem": "\n".join(linha["texto"] for linha in linhas),
            "inclinacao_graus": round(segmentacao.angulo, 2),
        }

    def reconhecer_arquivo(
        self,
        caminho: CaminhoImagem,
    ) -> dict[str, Any]:
        """Carrega um arquivo e reconhece os símbolos encontrados."""
        imagem = self._carregar_imagem(caminho)
        return self.reconhecer_imagem(imagem)