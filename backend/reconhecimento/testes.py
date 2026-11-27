import io
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings

from rest_framework.exceptions import ValidationError
from rest_framework.test import APIClient

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
from ia.modelo import ModeloCodeVision


def criar_imagem_bytes(
    formato="PNG",
    tamanho=(100, 100),
    cor="white",
):
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


class ImagemReconhecimentoSerializerTests(SimpleTestCase):

    def test_aceita_imagem_png_valida(self):
        arquivo = criar_arquivo_imagem()

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertTrue(
            serializer.is_valid(),
            msg=str(serializer.errors),
        )

    def test_rejeita_requisicao_sem_imagem(self):
        serializer = ImagemReconhecimentoSerializer(data={})

        self.assertFalse(serializer.is_valid())
        self.assertIn("imagem", serializer.errors)

    def test_rejeita_arquivo_que_nao_e_imagem(self):
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
        arquivo = criar_arquivo_imagem()

        arquivo.size = 10 * 1024 * 1024 + 1

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("imagem", serializer.errors)

    def test_aceita_imagem_jpeg(self):
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
        arquivo = criar_arquivo_imagem(
            tamanho=(10001, 1),
        )

        serializer = ImagemReconhecimentoSerializer(
            data={"imagem": arquivo}
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("imagem", serializer.errors)


class ResultadoReconhecimentoSerializerTests(SimpleTestCase):

    def criar_resultado_valido(self):
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
        serializer = ResultadoReconhecimentoSerializer(
            data=self.criar_resultado_valido()
        )

        self.assertTrue(
            serializer.is_valid(),
            msg=str(serializer.errors),
        )

    def test_rejeita_confianca_acima_de_um(self):
        dados = self.criar_resultado_valido()
        dados["simbolos"][0]["confianca"] = 1.5

        serializer = ResultadoReconhecimentoSerializer(data=dados)

        self.assertFalse(serializer.is_valid())

    def test_rejeita_classe_inexistente(self):
        dados = self.criar_resultado_valido()
        dados["simbolos"][0]["classe"] = 5

        serializer = ResultadoReconhecimentoSerializer(data=dados)

        self.assertFalse(serializer.is_valid())

    def test_rejeita_quantidade_negativa(self):
        dados = self.criar_resultado_valido()
        dados["quantidade_simbolos"] = -1

        serializer = ResultadoReconhecimentoSerializer(data=dados)

        self.assertFalse(serializer.is_valid())


class ServicoReconhecimentoTests(SimpleTestCase):

    def setUp(self):
        self.servico = ServicoReconhecimento(
            caminho_modelo="/caminho/falso/modelo_morse.pth"
        )

    def test_normaliza_lista_de_simbolos_textuais(self):
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
        with self.assertRaises(ErroReconhecimento):
            self.servico._normalizar_simbolos(
                {"simbolos": "estrutura inválida"}
            )

    def test_rejeita_item_de_simbolo_invalido(self):
        with self.assertRaises(ErroReconhecimento):
            self.servico._normalizar_simbolos(
                {"simbolos": [123]}
            )

    def test_monta_texto_dos_simbolos(self):
        simbolos = [
            {"simbolo": "."},
            {"simbolo": "-"},
        ]

        texto = self.servico._montar_texto_simbolos(simbolos)

        self.assertEqual(texto, ". -")

    def test_monta_texto_ignora_simbolos_vazios(self):
        simbolos = [
            {"simbolo": "."},
            {"simbolo": ""},
            {},
            {"simbolo": "-"},
        ]

        texto = self.servico._montar_texto_simbolos(simbolos)

        self.assertEqual(texto, ". -")

    def test_abre_imagem_valida(self):
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
        arquivo = SimpleUploadedFile(
            name="corrompido.png",
            content=b"conteudo invalido",
            content_type="image/png",
        )

        with self.assertRaises(ErroProcessamentoImagem):
            self.servico._abrir_imagem(arquivo)

    def test_falha_quando_modelo_nao_existe(self):
        with self.assertRaises(ModeloIAIndisponivel):
            self.servico.carregar_modelo()

    @patch("reconhecimento.services.ReconhecedorCodeVision")
    @patch("reconhecimento.services.ModeloCodeVision")
    def test_carrega_modelo_com_sucesso(
        self,
        mock_modelo_classe,
        mock_reconhecedor_classe,
    ):
        with patch.object(
            self.servico.caminho_modelo.__class__,
            "is_file",
            return_value=True,
        ):
            reconhecedor_esperado = Mock()
            mock_reconhecedor_classe.return_value = reconhecedor_esperado

            reconhecedor = self.servico.carregar_modelo()

        mock_modelo_classe.return_value.carregar.assert_called_once_with(
        )

        mock_modelo_classe.assert_called_once_with(
            caminho_modelo=self.servico.caminho_modelo
        )

        mock_reconhecedor_classe.assert_called_once()

        self.assertIs(reconhecedor, reconhecedor_esperado)

    def test_reconhece_imagem_com_sucesso(self):
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

    def test_preserva_detalhes_e_expoe_apenas_simbolos_reconhecidos(self):
        imagem_fake = np.zeros((30, 40, 3), dtype=np.uint8)
        coordenadas = {"x": 4, "y": 8, "largura": 12, "altura": 6}
        reconhecedor_fake = Mock()
        reconhecedor_fake.reconhecer_imagem.return_value = {
            "simbolos": ["·"],
            "resultados": [
                {
                    "indice": 0,
                    "coordenadas": coordenadas,
                    "area": 72,
                    "classe": 1,
                    "nome_classe": "ponto",
                    "simbolo": "·",
                    "confianca": 0.97,
                    "probabilidades": {
                        "fundo": 0.01,
                        "ponto": 0.97,
                        "traco": 0.02,
                    },
                    "status": "reconhecido",
                },
                {
                    "indice": 1,
                    "coordenadas": {"x": 20, "y": 8, "largura": 2, "altura": 2},
                    "area": 4,
                    "classe": 0,
                    "nome_classe": "fundo",
                    "simbolo": None,
                    "confianca": 0.99,
                    "probabilidades": {
                        "fundo": 0.99,
                        "ponto": 0.005,
                        "traco": 0.005,
                    },
                    "status": "fundo",
                },
            ],
        }

        with (
            patch.object(self.servico, "_abrir_imagem", return_value=imagem_fake),
            patch.object(self.servico, "carregar_modelo", return_value=reconhecedor_fake),
        ):
            resultado = self.servico.reconhecer_imagem(Mock())

        self.assertEqual(resultado["quantidade_simbolos"], 1)
        self.assertEqual(resultado["simbolos"][0]["classe"], 1)
        self.assertEqual(resultado["simbolos"][0]["confianca"], 0.97)
        self.assertEqual(resultado["simbolos"][0]["coordenadas"], coordenadas)
        self.assertEqual(len(resultado["resultados_detalhados"]), 2)

        serializer = ResultadoReconhecimentoSerializer(data=resultado)
        self.assertTrue(serializer.is_valid(), msg=str(serializer.errors))


    def test_propaga_erro_de_carregamento_do_modelo(self):
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


class PersistenciaModeloTests(SimpleTestCase):

    def test_salva_e_carrega_pesos_com_metadados_em_caminho_explicito(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "modelo_morse.pth"
            modelo_original = ModeloCodeVision(dispositivo="cpu")
            modelo_original.treinado = True

            salvo = modelo_original.salvar(caminho)

            self.assertEqual(salvo, caminho)
            self.assertTrue(caminho.is_file())
            self.assertTrue(caminho.with_suffix(".json").is_file())

            modelo_carregado = ModeloCodeVision(dispositivo="cpu")
            modelo_carregado.carregar(caminho)

            self.assertTrue(modelo_carregado.pronto)


class IntegracaoApiReconhecimentoTests(SimpleTestCase):

    @patch("reconhecimento.views.servico_reconhecimento.reconhecer_imagem")
    def test_post_devolve_detalhes_esperados_pelo_frontend(self, mock_reconhecer):
        coordenadas = {"x": 2, "y": 3, "largura": 8, "altura": 8}
        probabilidades = {"fundo": 0.01, "ponto": 0.97, "traco": 0.02}
        mock_reconhecer.return_value = {
            "sucesso": True,
            "mensagem": "Reconhecimento visual concluído.",
            "texto_reconhecido": "·",
            "mensagem_decodificada": "",
            "quantidade_simbolos": 1,
            "simbolos": [{
                "simbolo": "·",
                "classe": 1,
                "nome_classe": "ponto",
                "confianca": 0.97,
                "probabilidades": probabilidades,
                "coordenadas": coordenadas,
            }],
            "tempo_processamento_ms": 12.3,
            "dimensoes_imagem": {"largura": 20, "altura": 20},
            "resultados_detalhados": [{
                "indice": 0,
                "coordenadas": coordenadas,
                "area": 64,
                "classe": 1,
                "nome_classe": "ponto",
                "simbolo": "·",
                "confianca": 0.97,
                "probabilidades": probabilidades,
                "status": "reconhecido",
            }],
        }
        resposta = APIClient().post(
            "/api/reconhecimento/",
            {"imagem": criar_arquivo_imagem()},
            format="multipart",
        )

        self.assertEqual(resposta.status_code, 200, resposta.content)
        dados = resposta.json()
        self.assertEqual(dados["simbolos"][0]["classe"], 1)
        self.assertEqual(dados["simbolos"][0]["coordenadas"]["x"], 2)
        self.assertEqual(dados["resultados_detalhados"][0]["status"], "reconhecido")

    @override_settings(
        CORS_ALLOWED_ORIGINS=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    def test_cors_permite_frontend_local_e_rejeita_origem_inesperada(self):
        client = APIClient()
        permitida = client.get(
            "/api/health/",
            HTTP_ORIGIN="http://127.0.0.1:5173",
        )
        rejeitada = client.get(
            "/api/health/",
            HTTP_ORIGIN="http://localhost:3000",
        )

        self.assertEqual(
            permitida["Access-Control-Allow-Origin"],
            "http://127.0.0.1:5173",
        )
        self.assertNotIn("Access-Control-Allow-Origin", rejeitada)