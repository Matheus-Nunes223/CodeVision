import logging
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ia.modelo import ModeloCodeVision
from ia.preprocessamento import PreprocessadorCodeVision
from ia.reconhecimento import ReconhecedorCodeVision

from reconhecimento.exceptions import (
    ErroProcessamentoImagem,
    ErroReconhecimento,
    ModeloIAIndisponivel,
)


logger = logging.getLogger(__name__)


class ServicoReconhecimento:

    NOME_MODELO = "modelo_morse.pth"

    def __init__(self, caminho_modelo=None):

        self.raiz_projeto = Path(__file__).resolve().parents[2]

        if caminho_modelo is None:
            caminho_modelo = (
                self.raiz_projeto
                / "modelos"
                / self.NOME_MODELO
            )

        self.caminho_modelo = Path(caminho_modelo)

        self.preprocessador = PreprocessadorCodeVision()
        self.modelo = None
        self.reconhecedor = None

        self._lock = threading.Lock()

    def carregar_modelo(self):

        if self.reconhecedor is not None:
            return self.reconhecedor

        with self._lock:
            if self.reconhecedor is not None:
                return self.reconhecedor

            if not self.caminho_modelo.is_file():
                logger.error(
                    "Modelo de IA não encontrado: %s",
                    self.caminho_modelo,
                )
                raise ModeloIAIndisponivel(
                    "O modelo de reconhecimento ainda não foi treinado "
                    "ou não está disponível."
                )

            try:
                modelo = ModeloCodeVision(caminho_modelo=self.caminho_modelo)
                modelo.carregar()

                reconhecedor = ReconhecedorCodeVision(
                    modelo=modelo,
                    preprocessador=self.preprocessador,
                )

                self.modelo = modelo
                self.reconhecedor = reconhecedor

                logger.info(
                    "Modelo CodeVision carregado com sucesso."
                )

                return self.reconhecedor

            except Exception as exc:
                logger.exception(
                    "Falha ao carregar o modelo de reconhecimento."
                )

                raise ModeloIAIndisponivel(
                    "Não foi possível inicializar o modelo de IA."
                ) from exc

    @staticmethod
    def _abrir_imagem(arquivo):

        try:
            if hasattr(arquivo, "seek"):
                arquivo.seek(0)

            with Image.open(arquivo) as imagem:
                imagem = ImageOps.exif_transpose(imagem)
                imagem.load()

                imagem_rgb = imagem.convert("RGB")

                return np.array(imagem_rgb)

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise ErroProcessamentoImagem(
                "Não foi possível abrir ou interpretar a imagem enviada."
            ) from exc

        finally:
            if hasattr(arquivo, "seek"):
                try:
                    arquivo.seek(0)
                except (OSError, ValueError):
                    pass

    @staticmethod
    def _normalizar_simbolos(resultado):

        simbolos_brutos = resultado.get("simbolos", [])

        if not isinstance(simbolos_brutos, (list, tuple)):
            raise ErroReconhecimento(
                "O reconhecedor retornou uma estrutura de símbolos inválida."
            )

        simbolos = []

        for item in simbolos_brutos:
            if isinstance(item, str):
                simbolos.append({
                    "simbolo": item,
                })

            elif isinstance(item, dict):
                simbolos.append(item.copy())

            else:
                raise ErroReconhecimento(
                    "O reconhecedor retornou um símbolo em formato inválido."
                )

        return simbolos

    @staticmethod
    def _extrair_resultados_detalhados(resultado):

        resultados = resultado.get("resultados", [])

        if not isinstance(resultados, (list, tuple)):
            return []

        return list(resultados)

    @staticmethod
    def _montar_texto_simbolos(simbolos):

        partes = []

        for item in simbolos:
            simbolo = item.get("simbolo")

            if isinstance(simbolo, str) and simbolo:
                partes.append(simbolo)

        return " ".join(partes)

    def reconhecer_imagem(self, arquivo):

        inicio = time.perf_counter()

        try:
            imagem = self._abrir_imagem(arquivo)

            reconhecedor = self.carregar_modelo()

            resultado_bruto = reconhecedor.reconhecer_imagem(imagem)

            if not isinstance(resultado_bruto, dict):
                raise ErroReconhecimento(
                    "O reconhecedor retornou um resultado inválido."
                )

            resultados_detalhados = self._extrair_resultados_detalhados(
                resultado_bruto
            )

            if resultados_detalhados:
                simbolos = [
                    {
                        "simbolo": item["simbolo"],
                        "classe": item["classe"],
                        "nome_classe": item["nome_classe"],
                        "confianca": item["confianca"],
                        "probabilidades": item["probabilidades"],
                        "coordenadas": item["coordenadas"],
                    }
                    for item in resultados_detalhados
                    if isinstance(item, dict)
                    and item.get("status") == "reconhecido"
                    and item.get("simbolo")
                ]
            else:
                simbolos = self._normalizar_simbolos(resultado_bruto)

            morse = resultado_bruto.get("morse")
            texto_reconhecido = (
                morse
                if isinstance(morse, str) and morse
                else self._montar_texto_simbolos(simbolos)
            )

            mensagem = resultado_bruto.get("mensagem")
            mensagem_decodificada = (
                mensagem if isinstance(mensagem, str) else ""
            )

            tempo_ms = (time.perf_counter() - inicio) * 1000

            return {
                "sucesso": True,
                "mensagem": (
                    "Reconhecimento visual concluído."
                    if simbolos
                    else "Nenhum símbolo foi reconhecido na imagem."
                ),
                "texto_reconhecido": texto_reconhecido,
                "mensagem_decodificada": mensagem_decodificada,
                "quantidade_simbolos": len(simbolos),
                "simbolos": simbolos,
                "tempo_processamento_ms": round(tempo_ms, 2),
                "dimensoes_imagem": {
                    "largura": int(imagem.shape[1]),
                    "altura": int(imagem.shape[0]),
                },
                "resultados_detalhados": resultados_detalhados,
            }

        except (
            ModeloIAIndisponivel,
            ErroProcessamentoImagem,
            ErroReconhecimento,
        ):
            raise

        except Exception as exc:
            logger.exception(
                "Erro inesperado durante o reconhecimento da imagem."
            )

            raise ErroReconhecimento(
                "Ocorreu um erro inesperado durante o reconhecimento."
            ) from exc


servico_reconhecimento = ServicoReconhecimento()