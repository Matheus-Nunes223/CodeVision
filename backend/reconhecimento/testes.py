"""
Testes automatizados da aplicação de reconhecimento do CodeVision.

Executar com:
    python manage.py test reconhecimento
"""

import io
import time
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from rest_framework.exceptions import ValidationError

from reconhecimento.exceptions import (
    ErroProcessamentoImagem,
    ErroReconhecimento,
    ModeloIAIndisponivel,
)
from reconhecimento.serializers import (
    ImagemReconhecimentoSerializer,
    ResultadoReconhecimentoSerializer,
)
from reconhecimento.services import ServicoReconhecimento


# ============================================================
# FUNÇÕES AUXILIARES DOS TESTES
# ============================================================

def criar_imagem_bytes(
    formato="PNG",
    tamanho=(100, 100),
    cor="white",
):
    """
    Cria uma imagem de teste em memória.

    Args:
        formato: formato da imagem.
        tamanho: largura e altura.
        cor: cor de preenchimento.

    Returns:
        bytes: conteúdo da imagem.
    """

    imagem = Image.new("RGB", tamanho, color=cor)
    buffer = io.BytesIO()
    imagem.save(buffer, format=formato)

    return buffer.getvalue()


def criar_arquivo_imagem(
    nome="teste.png",
    formato="PNG",
    tamanho=(100, 100),
    cor="white",
):
    """
    Cria um arquivo de imagem compatível com os testes do Django.
    """

    conteudo = criar_imagem_bytes(
        formato=formato,
        tamanho=tamanho,
        cor=cor,
    )

    extensao = nome.rsplit(".", 1)[-1].lower()

    tipos_mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }

    return SimpleUploadedFile(
        name=nome,
        content=conteudo,
        content_type=tipos_mime.get(extensao, "application/octet-stream"),
    )


# ============================================================
# TESTES DO SERIALIZER DE ENTRADA
# ============================================================

class ImagemReconhecimentoSerializerTests(SimpleTestCase):
    """Testa a validação das imagens recebidas pela API."""

    def test_aceita_imagem_png_valida(self):
        """Uma imagem PNG válida deve passar pela validação."""

        arquivo = criar_arquivo_imagem()

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertTrue(
            serializer.is_valid(),
            msg=str(serializer.errors),
        )

    def test_rejeita_requisicao_sem_imagem(self):
        """A imagem é um campo obrigatório."""

        serializer = ImagemReconhecimentoSerializer(data={})

        self.assertFalse(serializer.is_valid())
        self.assertIn("imagem", serializer.errors)

    def test_rejeita_arquivo_que_nao_e_imagem(self):
        """Um arquivo de texto não deve ser aceito como imagem."""

        arquivo = SimpleUploadedFile(
            name="arquivo.txt",
            content=b"Este arquivo nao e uma imagem.",
            content_type="text/plain",
        )

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("imagem", serializer.errors)

    def test_rejeita_imagem_acima_do_limite(self):
        """Imagens maiores que o limite configurado devem ser rejeitadas."""

        # A imagem é válida, mas o arquivo enviado excede 10 MB.
        arquivo = criar_arquivo_imagem()

        arquivo.size = 10 * 1024 * 1024 + 1

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("imagem", serializer.errors)

    def test_aceita_imagem_jpeg(self):
        """O formato JPEG deve ser aceito."""

        arquivo = criar_arquivo_imagem(
            nome="teste.jpg",
            formato="JPEG",
        )

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertTrue(
            serializer.is_valid(),
            msg=str(serializer.errors),
        )

    def test_aceita_imagem_webp(self):
        """O formato WEBP deve ser aceito pelo serializer."""

        arquivo = criar_arquivo_imagem(
            nome="teste.webp",
            formato="WEBP",
        )

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertTrue(
            serializer.is_valid(),
            msg=str(serializer.errors),
        )

    def test_aceita_imagem_bmp(self):
        """O formato BMP deve ser aceito."""

        arquivo = criar_arquivo_imagem(
            nome="teste.bmp",
            formato="BMP",
        )

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertTrue(
            serializer.is_valid(),
            msg=str(serializer.errors),
        )

    def test_rejeita_dimensoes_excessivas(self):
        """Uma imagem com dimensão superior ao limite deve ser rejeitada."""

        arquivo = criar_arquivo_imagem(
            tamanho=(10001, 1),
        )

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("imagem", serializer.errors)


# ============================================================
# TESTES DO SERIALIZER DE SAÍDA
# ============================================================

class ResultadoReconhecimentoSerializerTests(SimpleTestCase):
    """Testa a estrutura dos resultados devolvidos pela API."""

    def criar_resultado_valido(self):
        """Retorna um exemplo válido de resultado."""

        return {
            "sucesso": True,
            "mensagem": "Reconhecimento visual concluído.",
            "texto_reconhecido": ". -",
            "mensagem_decodificada": "",
            "quantidade_simbolos": 2,
            "simbolos": [
                {
                    "simbolo": ".",
                    "classe": 1,
                    "confianca": 0.98,
                    "probabilidades": {
                        "fundo": 0.01,
                        "ponto": 0.98,
                        "traco": 0.01,
                    },
                },
                {
                    "simbolo": "-",
                    "classe": 2,
                    "confianca": 0.96,
                    "probabilidades": {
                        "fundo": 0.02,
                        "ponto": 0.02,
                        "traco": 0.96,
                    },
                },
            ],
            "tempo_processamento_ms": 120.5,
        }

    def test_aceita_resultado_valido(self):
        """O resultado estruturado corretamente deve ser válido."""

        serializer = ResultadoReconhecimentoSerializer(
            data=self.criar_resultado_valido()
        )

        self.assertTrue(
            serializer.is_valid(),
            msg=str(serializer.errors),
        )

    def test_rejeita_confianca_acima_de_um(self):
        """A confiança deve estar entre 0 e 1."""

        dados = self.criar_resultado_valido()
        dados["simbolos"][0]["confianca"] = 1.5

        serializer = ResultadoReconhecimentoSerializer(data=dados)

        self.assertFalse(serializer.is_valid())

    def test_rejeita_classe_inexistente(self):
        """As classes válidas do modelo são 0, 1 e 2."""

        dados = self.criar_resultado_valido()
        dados["simbolos"][0]["classe"] = 5

        serializer = ResultadoReconhecimentoSerializer(data=dados)

        self.assertFalse(serializer.is_valid())

    def test_rejeita_quantidade_negativa(self):
        """A quantidade de símbolos não pode ser negativa."""

        dados = self.criar_resultado_valido()
        dados["quantidade_simbolos"] = -1

        serializer = ResultadoReconhecimentoSerializer(data=dados)

        self.assertFalse(serializer.is_valid())


# ============================================================
# TESTES DOS SERVIÇOS DE RECONHECIMENTO
# ============================================================

class ServicoReconhecimentoTests(SimpleTestCase):
    """Testa o serviço sem carregar o modelo real de IA."""

    def setUp(self):
        """Prepara uma instância isolada para cada teste."""

        self.servico = ServicoReconhecimento(
            caminho_modelo="/caminho/falso/modelo_morse.pth"
        )

    def test_normaliza_lista_de_simbolos_textuais(self):
        """Símbolos textuais devem ser convertidos para dicionários."""

        resultado = {
            "simbolos": [".", "-"],
        }

        simbolos = self.servico._normalizar_simbolos(resultado)

        self.assertEqual(
            simbolos,
            [
                {"simbolo": "."},
                {"simbolo": "-"},
            ],
        )

    def test_normaliza_lista_de_simbolos_detalhados(self):
        """Os detalhes dos símbolos devem ser preservados."""

        resultado = {
            "simbolos": [
                {
                    "simbolo": ".",
                    "classe": 1,
                    "confianca": 0.95,
                }
            ]
        }

        simbolos = self.servico._normalizar_simbolos(resultado)

        self.assertEqual(len(simbolos), 1)
        self.assertEqual(simbolos[0]["classe"], 1)
        self.assertEqual(simbolos[0]["confianca"], 0.95)

    def test_rejeita_estrutura_de_simbolos_invalida(self):
        """A lista de símbolos deve ter uma estrutura reconhecida."""

        with self.assertRaises(ErroReconhecimento):
            self.servico._normalizar_simbolos(
                {"simbolos": "estrutura inválida"}
            )

    def test_rejeita_item_de_simbolo_invalido(self):
        """Cada símbolo deve ser texto ou dicionário."""

        with self.assertRaises(ErroReconhecimento):
            self.servico._normalizar_simbolos(
                {"simbolos": [123]}
            )

    def test_monta_texto_dos_simbolos(self):
        """Os símbolos devem ser unidos em uma representação textual."""

        simbolos = [
            {"simbolo": "."},
            {"simbolo": "-"},
        ]

        texto = self.servico._montar_texto_simbolos(simbolos)

        self.assertEqual(texto, ". -")

    def test_monta_texto_ignora_simbolos_vazios(self):
        """Símbolos vazios não devem ser incluídos no texto."""

        simbolos = [
            {"simbolo": "."},
            {"simbolo": ""},
            {},
            {"simbolo": "-"},
        ]

        texto = self.servico._montar_texto_simbolos(simbolos)

        self.assertEqual(texto, ". -")

    def test_abre_imagem_valida(self):
        """O serviço deve conseguir abrir uma imagem válida."""

        conteudo = criar_imagem_bytes(
            formato="PNG",
            tamanho=(20, 30),
        )

        arquivo = SimpleUploadedFile(
            name="teste.png",
            content=conteudo,
            content_type="image/png",
        )

        imagem = self.servico._abrir_imagem(arquivo)

        self.assertIsInstance(imagem, np.ndarray)
        self.assertEqual(imagem.shape, (30, 20, 3))

    def test_rejeita_arquivo_corrompido(self):
        """O serviço deve tratar arquivos que não podem ser abertos."""

        arquivo = SimpleUploadedFile(
            name="corrompido.png",
            content=b"conteudo invalido",
            content_type="image/png",
        )

        with self.assertRaises(ErroProcessamentoImagem):
            self.servico._abrir_imagem(arquivo)

    def test_falha_quando_modelo_nao_existe(self):
        """O serviço deve informar quando o modelo não está disponível."""

        with self.assertRaises(ModeloIAIndisponivel):
            self.servico.carregar_modelo()

    @patch("reconhecimento.services.ReconhecedorCodeVision")
    @patch("reconhecimento.services.ModeloCodeVision")
    def test_carrega_modelo_com_sucesso(
        self,
        mock_modelo_classe,
        mock_reconhecedor_classe,
    ):
        """O modelo e o reconhecedor devem ser inicializados."""

        with patch.object(
            self.servico.caminho_modelo.__class__,
            "is_file",
            return_value=True,
        ):
            reconhecedor_esperado = Mock()
            mock_reconhecedor_classe.return_value = reconhecedor_esperado

            reconhecedor = self.servico.carregar_modelo()

        mock_modelo_classe.return_value.carregar.assert_called_once_with(
            str(self.servico.caminho_modelo)
        )

        mock_reconhecedor_classe.assert_called_once()

        self.assertIs(reconhecedor, reconhecedor_esperado)

    def test_reconhece_imagem_com_sucesso(self):
        """O serviço deve organizar os resultados do reconhecedor."""

        imagem_fake = np.zeros((30, 40, 3), dtype=np.uint8)

        reconhecedor_fake = Mock()
        reconhecedor_fake.reconhecer_imagem.return_value = {
            "simbolos": [
                {
                    "simbolo": ".",
                    "classe": 1,
                    "confianca": 0.98,
                },
                {
                    "simbolo": "-",
                    "classe": 2,
                    "confianca": 0.96,
                },
            ],
            "resultados": [],
        }

        with (
            patch.object(
                self.servico,
                "_abrir_imagem",
                return_value=imagem_fake,
            ),
            patch.object(
                self.servico,
                "carregar_modelo",
                return_value=reconhecedor_fake,
            ),
        ):
            arquivo = Mock()
            resultado = self.servico.reconhecer_imagem(arquivo)

        self.assertTrue(resultado["sucesso"])
        self.assertEqual(resultado["quantidade_simbolos"], 2)
        self.assertEqual(resultado["texto_reconhecido"], ". -")
        self.assertEqual(resultado["dimensoes_imagem"]["largura"], 40)
        self.assertEqual(resultado["dimensoes_imagem"]["altura"], 30)

        reconhecedor_fake.reconhecer_imagem.assert_called_once_with(
            imagem_fake
        )

    def test_propaga_erro_de_carregamento_do_modelo(self):
        """A indisponibilidade do modelo deve ser propagada."""

        imagem_fake = np.zeros((30, 40, 3), dtype=np.uint8)

        with (
            patch.object(
                self.servico,
                "_abrir_imagem",
                return_value=imagem_fake,
            ),
            patch.object(
                self.servico,
                "carregar_modelo",
                side_effect=ModeloIAIndisponivel(
                    "Modelo indisponível."
                ),
            ),
        ):
            with self.assertRaises(ModeloIAIndisponivel):
                self.servico.reconhecer_imagem(Mock())

    def test_trata_erro_inesperado_do_reconhecedor(self):
        """Erros inesperados devem ser convertidos em erro de reconhecimento."""

        imagem_fake = np.zeros((30, 40, 3), dtype=np.uint8)

        reconhecedor_fake = Mock()
        reconhecedor_fake.reconhecer_imagem.side_effect = RuntimeError(
            "Erro interno de teste."
        )

        with (
            patch.object(
                self.servico,
                "_abrir_imagem",
                return_value=imagem_fake,
            ),
            patch.object(
                self.servico,
                "carregar_modelo",
                return_value=reconhecedor_fake,
            ),
        ):
            with self.assertRaises(ErroReconhecimento):
                self.servico.reconhecer_imagem(Mock())


# ============================================================
# EXECUÇÃO
# ============================================================

# O Django descobre automaticamente as classes de teste deste arquivo.
# Não é necessário executar este arquivo diretamente.