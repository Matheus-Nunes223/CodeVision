"""
Exceções personalizadas do backend do CodeVision.

Centraliza os erros relacionados ao recebimento de imagens,
ao reconhecimento visual e à integração com a inteligência
artificial.
"""


class CodeVisionError(Exception):
    """Exceção base para erros específicos do CodeVision."""

    def __init__(self, mensagem="Ocorreu um erro no CodeVision."):
        self.mensagem = mensagem
        super().__init__(mensagem)


class ErroValidacaoImagem(CodeVisionError):
    """Erro quando a imagem enviada não atende aos requisitos."""

    def __init__(
        self,
        mensagem="A imagem enviada é inválida ou não pode ser processada.",
    ):
        super().__init__(mensagem)


class FormatoImagemNaoSuportado(ErroValidacaoImagem):
    """Erro quando o formato do arquivo não é aceito."""

    def __init__(
        self,
        mensagem="O formato da imagem não é suportado.",
    ):
        super().__init__(mensagem)


class TamanhoImagemExcedido(ErroValidacaoImagem):
    """Erro quando a imagem ultrapassa o limite permitido."""

    def __init__(
        self,
        mensagem="A imagem excede o tamanho máximo permitido.",
    ):
        super().__init__(mensagem)


class ErroProcessamentoImagem(CodeVisionError):
    """Erro durante a leitura ou o processamento da imagem."""

    def __init__(
        self,
        mensagem="Não foi possível processar a imagem enviada.",
    ):
        super().__init__(mensagem)


class ModeloIAIndisponivel(CodeVisionError):
    """Erro quando o modelo de inteligência artificial não está disponível."""

    def __init__(
        self,
        mensagem="O modelo de inteligência artificial não está disponível.",
    ):
        super().__init__(mensagem)


class ErroReconhecimento(CodeVisionError):
    """Erro durante o reconhecimento dos símbolos da imagem."""

    def __init__(
        self,
        mensagem="Não foi possível reconhecer os símbolos da imagem.",
    ):
        super().__init__(mensagem)


class MensagemNaoDecodificavel(CodeVisionError):
    """Erro quando não é possível decodificar a mensagem reconhecida."""

    def __init__(
        self,
        mensagem="Não foi possível decodificar a mensagem reconhecida.",
    ):
        super().__init__(mensagem)


class ErroConfiguracaoIA(CodeVisionError):
    """Erro relacionado à configuração ou ao carregamento da IA."""

    def __init__(
        self,
        mensagem="Ocorreu um erro na configuração da inteligência artificial.",
    ):
        super().__init__(mensagem)