"""
Treinamento da CNN do CodeVision.

Responsabilidades:
- Carregar imagens organizadas por classe.
- Associar rótulos numéricos às imagens.
- Pré-processar as imagens usando PreprocessadorCodeVision.
- Separar os dados em treinamento e validação.
- Verificar se as classes possuem exemplos suficientes.
- Treinar a CNN usando ModeloCodeVision.
- Salvar o modelo treinado.

Estrutura esperada:

datasets/
    fundo/
    ponto/
    traco/

Rótulos:
    0 = fundo
    1 = ponto
    2 = traco

As imagens devem representar exemplos individuais das classes,
não mensagens Morse completas.
"""

from pathlib import Path
from typing import Optional

import random

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ia.modelo import ModeloCodeVision
from ia.preprocessamento import (
    ErroPreprocessamento,
    PreprocessadorCodeVision,
)


CLASSES = {
    "fundo": 0,
    "ponto": 1,
    "traco": 2,
}

EXTENSOES_VALIDAS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class ErroTreinamento(Exception):
    """Erro específico durante a preparação ou execução do treinamento."""


class TreinadorCodeVision:
    """Gerencia o conjunto de dados e o treinamento da CNN."""

    def __init__(
        self,
        diretorio_dados: str | Path = "datasets",
        diretorio_modelos: str | Path = "modelos",
        preprocessador: Optional[PreprocessadorCodeVision] = None,
        proporcao_validacao: float = 0.20,
        semente: int = 42,
        diretorio_teste: str | Path | None = "datasets_teste",
        peso_real: int = 4,
    ):
        """
        Args:
            diretorio_dados: diretório que contém as pastas das classes.
            diretorio_modelos: diretório onde o modelo será salvo.
            preprocessador: instância responsável pelo pré-processamento.
            proporcao_validacao: fração reservada para validação.
            semente: semente para tornar a divisão reproduzível.
            diretorio_teste: teste independente (mesmas pastas de
                classes). Nunca entra no treinamento. Se não existir,
                a avaliação independente é ignorada.
            peso_real: quantas vezes os exemplos reais (arquivos que
                começam com 'real_') aparecem no treinamento, para não
                serem diluídos pelos sintéticos.
        """
        if (
            not isinstance(peso_real, int)
            or isinstance(peso_real, bool)
            or peso_real < 1
        ):
            raise ValueError("peso_real deve ser um inteiro positivo.")

        if not isinstance(proporcao_validacao, (int, float)):
            raise TypeError("A proporção de validação deve ser numérica.")

        if not 0 < proporcao_validacao < 1:
            raise ValueError(
                "A proporção de validação deve estar entre 0 e 1."
            )

        if (
            not isinstance(semente, int)
            or isinstance(semente, bool)
        ):
            raise TypeError("A semente deve ser um número inteiro.")

        self.diretorio_dados = Path(diretorio_dados).expanduser()
        self.diretorio_modelos = Path(diretorio_modelos).expanduser()

        self.preprocessador = (
            preprocessador or PreprocessadorCodeVision()
        )

        self.diretorio_teste = (
            Path(diretorio_teste).expanduser()
            if diretorio_teste is not None
            else None
        )
        self.peso_real = peso_real
        self.copias_reais = 0

        self.proporcao_validacao = float(proporcao_validacao)
        self.semente = semente

        self.historico = []
        self.quantidade_por_classe = {}

    def _validar_diretorio(self) -> None:
        """Verifica se o conjunto de dados existe e está organizado."""
        if not self.diretorio_dados.exists():
            raise FileNotFoundError(
                f"Diretório de dados não encontrado: "
                f"{self.diretorio_dados.resolve()}"
            )

        if not self.diretorio_dados.is_dir():
            raise ErroTreinamento(
                "O caminho dos dados não é um diretório."
            )

        for nome_classe in CLASSES:
            diretorio_classe = self.diretorio_dados / nome_classe

            if not diretorio_classe.is_dir():
                raise ErroTreinamento(
                    f"A pasta da classe '{nome_classe}' não existe: "
                    f"{diretorio_classe}"
                )

    def _listar_imagens(self) -> dict[str, list[Path]]:
        """Lista as imagens válidas de cada classe."""
        self._validar_diretorio()

        imagens_por_classe = {}

        for nome_classe in CLASSES:
            diretorio_classe = self.diretorio_dados / nome_classe

            arquivos = sorted(
                caminho
                for caminho in diretorio_classe.iterdir()
                if caminho.is_file()
                and caminho.suffix.lower() in EXTENSOES_VALIDAS
            )

            if len(arquivos) < 2:
                raise ErroTreinamento(
                    f"A classe '{nome_classe}' precisa ter pelo menos "
                    "duas imagens para permitir a divisão entre "
                    "treinamento e validação."
                )

            imagens_por_classe[nome_classe] = arquivos

        self.quantidade_por_classe = {
            classe: len(arquivos)
            for classe, arquivos in imagens_por_classe.items()
        }

        return imagens_por_classe

    def _carregar_imagem(self, caminho: Path) -> torch.Tensor:
        """
        Carrega e pré-processa uma imagem individual.

        Retorna um tensor com formato (1, 32, 32).
        """
        try:
            with Image.open(caminho) as imagem_original:
                imagem_original.load()
                imagem = imagem_original.convert("L")

            tensor = self.preprocessador.preparar_simbolo(imagem)

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
            ErroPreprocessamento,
        ) as erro:
            raise ErroTreinamento(
                f"Erro ao processar a imagem '{caminho}': {erro}"
            ) from erro

        if tensor.shape != (1, 32, 32):
            raise ErroTreinamento(
                f"A imagem '{caminho}' produziu um tensor com formato "
                f"inesperado: {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all():
            raise ErroTreinamento(
                f"A imagem '{caminho}' contém valores inválidos."
            )

        return tensor

    @staticmethod
    def _origem(caminho: Path) -> str:
        """
        Identifica a imagem-fonte de um recorte.

        Os recortes gerados por preparar_datasets.py se chamam
        '<origem>_<sub>_<letra>_<id>_<indice>.png'. Remover o sufixo
        '_<indice>' agrupa todos os recortes da mesma imagem.
        """
        nome = caminho.stem
        base, _, sufixo = nome.rpartition("_")
        return base if base and sufixo.isdigit() else nome

    def _preparar_dados(
        self,
        imagens_por_classe: dict[str, list[Path]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Converte as imagens em tensores, rótulos e origens."""
        imagens = []
        rotulos = []
        origens = []

        total = sum(
            len(arquivos)
            for arquivos in imagens_por_classe.values()
        )

        processadas = 0

        print("\nPreparando conjunto de dados...")

        for nome_classe, rotulo in CLASSES.items():
            arquivos = imagens_por_classe[nome_classe]

            for caminho in arquivos:
                tensor = self._carregar_imagem(caminho)

                imagens.append(tensor)
                rotulos.append(rotulo)
                origens.append(self._origem(caminho))

                processadas += 1

                if processadas % 50 == 0 or processadas == total:
                    print(
                        f"Imagens processadas: "
                        f"{processadas}/{total}"
                    )

        imagens_tensor = torch.stack(imagens).float()
        rotulos_tensor = torch.tensor(
            rotulos,
            dtype=torch.long,
        )

        return imagens_tensor, rotulos_tensor, origens

    def _dividir_dados(
        self,
        imagens: torch.Tensor,
        rotulos: torch.Tensor,
        origens: list[str],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Divide os dados por imagem-fonte, evitando vazamento.

        Todos os recortes de uma mesma imagem ficam no mesmo
        conjunto (treinamento ou validação). A validação recebe
        aproximadamente `proporcao_validacao` das origens.
        """
        gerador = random.Random(self.semente)

        todas_origens = sorted(set(origens))
        gerador.shuffle(todas_origens)

        quantidade_validacao = max(
            1,
            round(len(todas_origens) * self.proporcao_validacao),
        )
        quantidade_validacao = min(
            quantidade_validacao,
            len(todas_origens) - 1,
        )

        origens_validacao = set(todas_origens[:quantidade_validacao])

        indices_treino = []
        indices_validacao = []

        for indice, origem in enumerate(origens):
            if origem in origens_validacao:
                indices_validacao.append(indice)
            else:
                indices_treino.append(indice)

        # Repete os exemplos reais; cada repetição recebe um aumento
        # de dados diferente a cada época.
        self.copias_reais = 0

        if self.peso_real > 1:
            reais = [
                i for i in indices_treino
                if origens[i].startswith("real_")
            ]
            self.copias_reais = len(reais) * (self.peso_real - 1)
            indices_treino = indices_treino + reais * (self.peso_real - 1)

        for rotulo in CLASSES.values():
            for nome_conjunto, indices in (
                ("treinamento", indices_treino),
                ("validação", indices_validacao),
            ):
                if not any(int(rotulos[i]) == rotulo for i in indices):
                    raise ErroTreinamento(
                        f"A classe {rotulo} não tem exemplos no "
                        f"conjunto de {nome_conjunto} após a divisão "
                        "por origem."
                    )

        gerador.shuffle(indices_treino)
        gerador.shuffle(indices_validacao)

        indices_treino = torch.tensor(
            indices_treino,
            dtype=torch.long,
        )

        indices_validacao = torch.tensor(
            indices_validacao,
            dtype=torch.long,
        )

        return (
            imagens[indices_treino],
            rotulos[indices_treino],
            imagens[indices_validacao],
            rotulos[indices_validacao],
        )

    def _carregar_teste_independente(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]] | None:
        """
        Carrega o teste independente, se a pasta existir.

        Se a pasta existir, as três pastas de classes também precisam
        existir e conter imagens; caso contrário, o teste cobriria só
        parte das classes sem aviso.
        """
        if self.diretorio_teste is None or not self.diretorio_teste.is_dir():
            return None

        imagens = []
        rotulos = []
        origens = []

        for nome_classe, rotulo in CLASSES.items():
            pasta = self.diretorio_teste / nome_classe

            if not pasta.is_dir():
                raise ErroTreinamento(
                    f"O teste independente não tem a pasta da classe "
                    f"'{nome_classe}': {pasta}"
                )

            arquivos = sorted(
                caminho
                for caminho in pasta.iterdir()
                if caminho.is_file()
                and caminho.suffix.lower() in EXTENSOES_VALIDAS
            )

            if not arquivos:
                raise ErroTreinamento(
                    f"O teste independente não tem imagens da classe "
                    f"'{nome_classe}': {pasta}"
                )

            for caminho in arquivos:
                imagens.append(self._carregar_imagem(caminho))
                rotulos.append(rotulo)
                origens.append(self._origem(caminho))

        return (
            torch.stack(imagens).float(),
            torch.tensor(rotulos, dtype=torch.long),
            origens,
        )

    @staticmethod
    def _verificar_sobreposicao(
        origens_treino: list[str],
        origens_teste: list[str],
    ) -> None:
        """Impede que uma mesma imagem-fonte esteja no treino e no teste."""
        comuns = sorted(set(origens_treino) & set(origens_teste))

        if comuns:
            exemplos = ", ".join(comuns[:5])
            raise ErroTreinamento(
                f"{len(comuns)} imagem(ns)-fonte aparecem em "
                f"'datasets' e no teste independente (ex.: {exemplos}). "
                "O teste deixaria de ser independente."
            )

    def executar(
        self,
        epocas: int = 20,
        tamanho_lote: int = 32,
        taxa_aprendizado: float = 0.001,
        peso_decay: float = 0.0001,
        nome_modelo: str = "modelo_morse.pth",
    ) -> ModeloCodeVision:
        """
        Executa o fluxo completo de treinamento.

        Args:
            epocas: número de épocas de treinamento.
            tamanho_lote: quantidade de exemplos por lote.
            taxa_aprendizado: taxa de aprendizado do otimizador.
            peso_decay: regularização L2 do otimizador.
            nome_modelo: nome do arquivo de pesos a salvar.

        Returns:
            Instância de ModeloCodeVision treinada.
        """
        if (
            not isinstance(epocas, int)
            or isinstance(epocas, bool)
            or epocas < 1
        ):
            raise ValueError("epocas deve ser um inteiro positivo.")

        if (
            not isinstance(tamanho_lote, int)
            or isinstance(tamanho_lote, bool)
            or tamanho_lote < 1
        ):
            raise ValueError(
                "tamanho_lote deve ser um inteiro positivo."
            )

        if (
            not isinstance(taxa_aprendizado, (int, float))
            or isinstance(taxa_aprendizado, bool)
            or not np.isfinite(taxa_aprendizado)
            or taxa_aprendizado <= 0
        ):
            raise ValueError(
                "taxa_aprendizado deve ser um número positivo e finito."
            )

        if (
            not isinstance(peso_decay, (int, float))
            or isinstance(peso_decay, bool)
            or not np.isfinite(peso_decay)
            or peso_decay < 0
        ):
            raise ValueError(
                "peso_decay deve ser um número não negativo e finito."
            )

        if (
            not isinstance(nome_modelo, str)
            or not nome_modelo.strip()
            or Path(nome_modelo).name != nome_modelo
            or Path(nome_modelo).suffix.lower() != ".pth"
        ):
            raise ValueError(
                "nome_modelo deve ser um nome de arquivo .pth, "
                "sem diretórios."
            )

        random.seed(self.semente)
        np.random.seed(self.semente)
        torch.manual_seed(self.semente)

        imagens_por_classe = self._listar_imagens()

        print("\nQuantidade de imagens por classe:")

        for classe, quantidade in self.quantidade_por_classe.items():
            print(f"  {classe}: {quantidade}")

        imagens, rotulos, origens = self._preparar_dados(
            imagens_por_classe
        )

        # Carregado antes do treino: um erro aqui não desperdiça épocas.
        teste = self._carregar_teste_independente()

        if teste is not None:
            self._verificar_sobreposicao(origens, teste[2])

        (
            imagens_treino,
            rotulos_treino,
            imagens_validacao,
            rotulos_validacao,
        ) = self._dividir_dados(imagens, rotulos, origens)

        distintas = len(rotulos_treino) - self.copias_reais

        print("\nResumo da divisão:")
        print(
            f"  Treinamento: {distintas} imagens distintas"
            f" (+{self.copias_reais} repetições de exemplos reais"
            f" = {len(rotulos_treino)} por época)"
        )
        print(f"  Validação: {len(rotulos_validacao)} imagens")

        if teste is None:
            print(
                f"  Teste independente: não encontrado "
                f"({self.diretorio_teste}); etapa será ignorada."
            )
        else:
            print(f"  Teste independente: {len(teste[1])} imagens")

        modelo = ModeloCodeVision()

        print("\nIniciando treinamento da CNN...")

        self.historico = modelo.treinar(
            imagens_treino,
            rotulos_treino,
            dados_validacao=(imagens_validacao, rotulos_validacao),
            epocas=epocas,
            tamanho_lote=tamanho_lote,
            taxa_aprendizado=float(taxa_aprendizado),
            peso_decay=float(peso_decay),
            semente=self.semente,
        )

        print("\nAvaliando o modelo...")

        resultado_avaliacao = modelo.avaliar(
            imagens_validacao,
            rotulos_validacao,
        )

        print("\nResultados da validação:")

        for metrica, valor in resultado_avaliacao.items():
            print(f"  {metrica}: {valor}")

        if teste is not None:
            resultado_teste = modelo.avaliar(teste[0], teste[1])

            print("\nResultados no teste independente (fotos reais):")

            for metrica, valor in resultado_teste.items():
                print(f"  {metrica}: {valor}")

        self.diretorio_modelos.mkdir(
            parents=True,
            exist_ok=True,
        )

        caminho_modelo = self.diretorio_modelos / nome_modelo

        modelo.salvar(caminho_modelo)

        print(f"\nModelo salvo em: {caminho_modelo.resolve()}")

        return modelo


def main() -> None:
    """Ponto de entrada para executar o treinamento pelo terminal."""
    treinador = TreinadorCodeVision(
        diretorio_dados="datasets",
        diretorio_modelos="modelos",
        proporcao_validacao=0.20,
        semente=42,
    )

    treinador.executar(
        epocas=40,
        tamanho_lote=32,
        taxa_aprendizado=0.001,
        peso_decay=0.0001,
        nome_modelo="modelo_morse.pth",
    )


if __name__ == "__main__":
    main()