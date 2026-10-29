"""
Serviços de reconhecimento de imagens do CodeVision.

Responsabilidades:
- Carregar e preparar o modelo de inteligência artificial.
- Receber imagens validadas pela API.
- Executar o reconhecimento visual dos símbolos.
- Organizar os resultados para o backend.
- Tratar falhas de carregamento e processamento.
"""

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
    """
    Coordena o reconhecimento visual das imagens do CodeVision.

    O modelo é carregado sob demanda e reutilizado nas próximas
    requisições, evitando recarregá-lo a cada imagem.
    """

    NOME_MODELO = "modelo_morse.pth"

    def __init__(self, caminho_modelo=None):
        """
        Inicializa o serviço.

        Args:
            caminho_modelo: caminho opcional para os pesos treinados.
        """

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
        """
        Carrega o modelo de IA uma única vez por instância.

        Returns:
            ReconhecedorCodeVision: reconhecedor inicializado.

        Raises:
            ModeloIAIndisponivel: se o modelo não puder ser carregado.
        """

        if self.reconhecedor is not None:
            return self.reconhecedor

        with self._lock:
            # Outra requisição pode ter carregado o modelo
            # enquanto esta aguardava o bloqueio.
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
                modelo = ModeloCodeVision()
                modelo.carregar(str(self.caminho_modelo))

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
        """
        Abre o arquivo enviado e o converte para RGB.

        Args:
            arquivo: arquivo de imagem recebido pela API.

        Returns:
            numpy.ndarray: imagem RGB.

        Raises:
            ErroProcessamentoImagem: se a imagem não puder ser lida.
        """

        try:
            # Retorna o arquivo ao início caso já tenha sido lido
            # durante a validação do serializer.
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
            # Permite que o arquivo seja lido novamente, se necessário.
            if hasattr(arquivo, "seek"):
                try:
                    arquivo.seek(0)
                except (OSError, ValueError):
                    pass

    @staticmethod
    def _normalizar_simbolos(resultado):
        """
        Converte os símbolos reconhecidos em uma lista padronizada.

        O reconhecedor visual deve retornar um dicionário contendo
        a chave 'simbolos'. Cada item pode ser um símbolo textual
        ou um dicionário com informações de classificação.

        Returns:
            list: símbolos encontrados na imagem.
        """

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
        """
        Obtém os resultados detalhados produzidos pelo reconhecedor.

        Se não houver resultados detalhados, utiliza uma lista vazia.
        """

        resultados = resultado.get("resultados", [])

        if not isinstance(resultados, (list, tuple)):
            return []

        return list(resultados)

    @staticmethod
    def _montar_texto_simbolos(simbolos):
        """
        Monta uma representação textual dos símbolos identificados.

        Nesta etapa, os símbolos são apresentados na ordem fornecida
        pelo reconhecedor. A separação entre letras e palavras deverá
        ser resolvida pelo módulo de decodificação.
        """

        partes = []

        for item in simbolos:
            simbolo = item.get("simbolo")

            if isinstance(simbolo, str) and simbolo:
                partes.append(simbolo)

        return " ".join(partes)

    def reconhecer_imagem(self, arquivo):
        """
        Executa o reconhecimento visual de uma imagem.

        Args:
            arquivo: arquivo de imagem validado pela API.

        Returns:
            dict: resultado organizado para a API.

        Raises:
            ModeloIAIndisponivel: se o modelo não estiver disponível.
            ErroProcessamentoImagem: se a imagem não puder ser processada.
            ErroReconhecimento: se o reconhecimento falhar.
        """

        inicio = time.perf_counter()

        try:
            imagem = self._abrir_imagem(arquivo)

            reconhecedor = self.carregar_modelo()

            resultado_bruto = reconhecedor.reconhecer_imagem(imagem)

            if not isinstance(resultado_bruto, dict):
                raise ErroReconhecimento(
                    "O reconhecedor retornou um resultado inválido."
                )

            simbolos = self._normalizar_simbolos(resultado_bruto)
            resultados_detalhados = self._extrair_resultados_detalhados(
                resultado_bruto
            )

            texto_reconhecido = self._montar_texto_simbolos(simbolos)

            tempo_ms = (time.perf_counter() - inicio) * 1000

            return {
                "sucesso": True,
                "mensagem": (
                    "Reconhecimento visual concluído."
                    if simbolos
                    else "Nenhum símbolo foi reconhecido na imagem."
                ),
                "texto_reconhecido": texto_reconhecido,
                # A decodificação semântica será integrada ao módulo
                # de códigos quando sua interface estiver definida.
                "mensagem_decodificada": "",
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


# Instância compartilhada pelo processo do backend.
# O modelo só será carregado quando ocorrer a primeira requisição.
servico_reconhecimento = ServicoReconhecimento()