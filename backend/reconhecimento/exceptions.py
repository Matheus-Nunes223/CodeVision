class CodeVisionError(Exception):

    def __init__(self, mensagem="Ocorreu um erro no CodeVision."):
        self.mensagem = mensagem
        super().__init__(mensagem)


class ErroValidacaoImagem(CodeVisionError):

    def __init__(
        self,
        mensagem="A imagem enviada é inválida ou não pode ser processada.",
    ):
        super().__init__(mensagem)


class FormatoImagemNaoSuportado(ErroValidacaoImagem):

    def __init__(
        self,
        mensagem="O formato da imagem não é suportado.",
    ):
        super().__init__(mensagem)


class TamanhoImagemExcedido(ErroValidacaoImagem):

    def __init__(
        self,
        mensagem="A imagem excede o tamanho máximo permitido.",
    ):
        super().__init__(mensagem)


class ErroProcessamentoImagem(CodeVisionError):

    def __init__(
        self,
        mensagem="Não foi possível processar a imagem enviada.",
    ):
        super().__init__(mensagem)


class ModeloIAIndisponivel(CodeVisionError):

    def __init__(
        self,
        mensagem="O modelo de inteligência artificial não está disponível.",
    ):
        super().__init__(mensagem)


class ErroReconhecimento(CodeVisionError):

    def __init__(
        self,
        mensagem="Não foi possível reconhecer os símbolos da imagem.",
    ):
        super().__init__(mensagem)


class MensagemNaoDecodificavel(CodeVisionError):

    def __init__(
        self,
        mensagem="Não foi possível decodificar a mensagem reconhecida.",
    ):
        super().__init__(mensagem)


class ErroConfiguracaoIA(CodeVisionError):

    def __init__(
        self,
        mensagem="Ocorreu um erro na configuração da inteligência artificial.",
    ):
        super().__init__(mensagem)