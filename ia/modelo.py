"""
Modelo de inteligência artificial do CodeVision.

Classifica recortes de símbolos Morse em três classes:
    0: fundo
    1: ponto
    2: traco

Responsabilidades:
    - Definir a arquitetura CNN.
    - Treinar e avaliar o classificador.
    - Salvar e carregar pesos e metadados.
    - Classificar imagens individuais e retornar probabilidades.

Formato esperado:
    - Imagens: (N, 1, 32, 32), com valores entre 0 e 1.
    - Rótulos: (N,), com valores inteiros entre 0 e 2.
    - Previsão: imagem com formato (1, 32, 32).

O pré-processamento de imagens brutas e a localização de símbolos
em imagens completas são responsabilidades de outros módulos.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RedeNeuralMorse(nn.Module):
    """Rede neural convolucional para classificar símbolos Morse."""

    TAMANHO_IMAGEM = 32
    CANAIS_ENTRADA = 1

    def __init__(self, numero_classes: int = 3) -> None:
        super().__init__()

        if (
            isinstance(numero_classes, bool)
            or not isinstance(numero_classes, int)
            or numero_classes < 2
        ):
            raise ValueError(
                "numero_classes deve ser um inteiro maior ou igual a 2."
            )

        self.extrator_caracteristicas = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, numero_classes),
        )

    def forward(self, imagem: torch.Tensor) -> torch.Tensor:
        caracteristicas = self.extrator_caracteristicas(imagem)
        return self.classificador(caracteristicas)


class ModeloCodeVision:
    """Gerencia treinamento, avaliação, persistência e previsão."""

    CLASSES = {
        0: "fundo",
        1: "ponto",
        2: "traco",
    }

    TAMANHO_IMAGEM = 32
    VERSAO_FORMATO = 1

    def __init__(
        self,
        caminho_modelo: str | Path = "modelos/modelo_morse.pth",
        dispositivo: str | torch.device | None = None,
    ) -> None:
        self.caminho_modelo = Path(caminho_modelo)
        self.caminho_metadados = self.caminho_modelo.with_suffix(".json")

        if dispositivo is None:
            dispositivo = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        try:
            self.dispositivo = torch.device(dispositivo)
        except (TypeError, RuntimeError) as erro:
            raise ValueError(
                f"Dispositivo inválido: {dispositivo!r}"
            ) from erro

        if (
            self.dispositivo.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise ValueError(
                "CUDA foi selecionado, mas não está disponível."
            )

        self.modelo = RedeNeuralMorse(
            numero_classes=len(self.CLASSES)
        ).to(self.dispositivo)

        self.treinado = False
        self.epocas_treinadas = 0
        self.historico_treinamento: list[dict[str, float]] = []

    @property
    def pronto(self) -> bool:
        """Indica se o modelo possui pesos treinados ou carregados."""
        return self.treinado

    @staticmethod
    def _validar_parametros_treinamento(
        epocas: int,
        tamanho_lote: int,
        taxa_aprendizado: float,
        peso_decay: float,
    ) -> None:
        if (
            isinstance(epocas, bool)
            or not isinstance(epocas, int)
            or epocas < 1
        ):
            raise ValueError("epocas deve ser um inteiro maior que zero.")

        if (
            isinstance(tamanho_lote, bool)
            or not isinstance(tamanho_lote, int)
            or tamanho_lote < 1
        ):
            raise ValueError(
                "tamanho_lote deve ser um inteiro maior que zero."
            )

        if (
            not isinstance(taxa_aprendizado, (int, float))
            or not math.isfinite(taxa_aprendizado)
            or taxa_aprendizado <= 0
        ):
            raise ValueError(
                "taxa_aprendizado deve ser finita e maior que zero."
            )

        if (
            not isinstance(peso_decay, (int, float))
            or not math.isfinite(peso_decay)
            or peso_decay < 0
        ):
            raise ValueError(
                "peso_decay deve ser finito e não negativo."
            )

    @classmethod
    def _validar_dados(
        cls,
        imagens: Any,
        rotulos: Any,
        nome_conjunto: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Valida e converte imagens e rótulos para tensores."""

        try:
            imagens_tensor = torch.as_tensor(
                imagens,
                dtype=torch.float32,
            )
            rotulos_tensor = torch.as_tensor(rotulos)
        except (TypeError, ValueError, RuntimeError) as erro:
            raise ValueError(
                f"{nome_conjunto}: não foi possível converter "
                "imagens e rótulos em tensores."
            ) from erro

        formato_esperado = (
            imagens_tensor.ndim == 4
            and tuple(imagens_tensor.shape[1:])
            == (1, cls.TAMANHO_IMAGEM, cls.TAMANHO_IMAGEM)
        )

        if not formato_esperado:
            raise ValueError(
                f"{nome_conjunto}: as imagens devem possuir formato "
                f"(N, 1, {cls.TAMANHO_IMAGEM}, {cls.TAMANHO_IMAGEM})."
            )

        if rotulos_tensor.ndim != 1:
            raise ValueError(
                f"{nome_conjunto}: os rótulos devem possuir formato (N,)."
            )

        if imagens_tensor.shape[0] == 0:
            raise ValueError(
                f"{nome_conjunto}: o conjunto de imagens está vazio."
            )

        if imagens_tensor.shape[0] != rotulos_tensor.shape[0]:
            raise ValueError(
                f"{nome_conjunto}: a quantidade de imagens "
                "não corresponde à quantidade de rótulos."
            )

        if not torch.isfinite(imagens_tensor).all().item():
            raise ValueError(
                f"{nome_conjunto}: existem valores inválidos nas imagens."
            )

        if (
            (imagens_tensor < 0).any().item()
            or (imagens_tensor > 1).any().item()
        ):
            raise ValueError(
                f"{nome_conjunto}: os pixels devem estar entre 0 e 1."
            )

        if (
            rotulos_tensor.dtype == torch.bool
            or torch.is_floating_point(rotulos_tensor)
            or torch.is_complex(rotulos_tensor)
        ):
            if (
                torch.is_floating_point(rotulos_tensor)
                and not torch.isfinite(rotulos_tensor).all().item()
            ):
                raise ValueError(
                    f"{nome_conjunto}: existem rótulos inválidos."
                )

            if (
                torch.is_complex(rotulos_tensor)
                or not torch.equal(
                    rotulos_tensor,
                    rotulos_tensor.round(),
                )
            ):
                raise ValueError(
                    f"{nome_conjunto}: os rótulos devem ser inteiros."
                )

        rotulos_tensor = rotulos_tensor.to(dtype=torch.long)

        rotulos_validos = (
            (rotulos_tensor >= 0)
            & (rotulos_tensor < len(cls.CLASSES))
        )

        if not rotulos_validos.all().item():
            raise ValueError(
                f"{nome_conjunto}: os rótulos válidos são "
                f"{list(cls.CLASSES.keys())}."
            )

        return (
            imagens_tensor.contiguous(),
            rotulos_tensor.contiguous(),
        )

    def _criar_carregador(
        self,
        imagens: torch.Tensor,
        rotulos: torch.Tensor,
        tamanho_lote: int,
        embaralhar: bool,
    ) -> DataLoader:
        conjunto = TensorDataset(imagens, rotulos)

        return DataLoader(
            conjunto,
            batch_size=tamanho_lote,
            shuffle=embaralhar,
            num_workers=0,
            pin_memory=self.dispositivo.type == "cuda",
        )

    def _executar_epoca(
        self,
        carregador: DataLoader,
        criterio: nn.Module,
        otimizador: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Executa uma época de treinamento ou avaliação."""

        treinando = otimizador is not None
        self.modelo.train(mode=treinando)

        perda_acumulada = 0.0
        total = 0
        acertos = 0

        contexto = (
            torch.enable_grad()
            if treinando
            else torch.inference_mode()
        )

        with contexto:
            for imagens_lote, rotulos_lote in carregador:
                imagens_lote = imagens_lote.to(
                    self.dispositivo,
                    non_blocking=True,
                )
                rotulos_lote = rotulos_lote.to(
                    self.dispositivo,
                    non_blocking=True,
                )

                if treinando:
                    otimizador.zero_grad(set_to_none=True)

                logits = self.modelo(imagens_lote)
                perda = criterio(logits, rotulos_lote)

                if treinando:
                    perda.backward()
                    otimizador.step()

                quantidade = rotulos_lote.size(0)

                perda_acumulada += perda.item() * quantidade
                total += quantidade
                acertos += (
                    logits.argmax(dim=1) == rotulos_lote
                ).sum().item()

        if total == 0:
            raise ValueError(
                "Não há amostras suficientes para calcular as métricas."
            )

        return {
            "perda": perda_acumulada / total,
            "acuracia": acertos / total,
            "amostras": float(total),
        }

    def treinar(
        self,
        imagens: Any,
        rotulos: Any,
        *,
        epocas: int = 15,
        tamanho_lote: int = 32,
        taxa_aprendizado: float = 0.001,
        peso_decay: float = 0.0001,
        dados_validacao: tuple[Any, Any] | None = None,
        semente: int | None = 42,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """
        Treina a CNN e retorna o histórico das métricas.

        imagens:
            Tensor (N, 1, 32, 32), com pixels entre 0 e 1.

        rotulos:
            Tensor (N,), com classes 0, 1 ou 2.

        dados_validacao:
            Opcionalmente, tupla (imagens_validacao, rotulos_validacao).
        """

        self._validar_parametros_treinamento(
            epocas,
            tamanho_lote,
            taxa_aprendizado,
            peso_decay,
        )

        imagens_treino, rotulos_treino = self._validar_dados(
            imagens,
            rotulos,
            "Treinamento",
        )

        carregador_treino = self._criar_carregador(
            imagens_treino,
            rotulos_treino,
            tamanho_lote,
            embaralhar=True,
        )

        carregador_validacao = None

        if dados_validacao is not None:
            if (
                not isinstance(dados_validacao, (tuple, list))
                or len(dados_validacao) != 2
            ):
                raise ValueError(
                    "dados_validacao deve conter "
                    "(imagens_validacao, rotulos_validacao)."
                )

            imagens_val, rotulos_val = self._validar_dados(
                dados_validacao[0],
                dados_validacao[1],
                "Validação",
            )

            carregador_validacao = self._criar_carregador(
                imagens_val,
                rotulos_val,
                tamanho_lote,
                embaralhar=False,
            )

        if semente is not None:
            if isinstance(semente, bool) or not isinstance(semente, int):
                raise ValueError(
                    "semente deve ser um inteiro ou None."
                )

            torch.manual_seed(semente)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(semente)

        criterio = nn.CrossEntropyLoss()

        otimizador = torch.optim.Adam(
            self.modelo.parameters(),
            lr=float(taxa_aprendizado),
            weight_decay=float(peso_decay),
        )

        historico: list[dict[str, float]] = []

        for epoca in range(1, epocas + 1):
            metricas_treino = self._executar_epoca(
                carregador_treino,
                criterio,
                otimizador,
            )

            registro = {
                "epoca": float(epoca),
                "perda_treino": metricas_treino["perda"],
                "acuracia_treino": metricas_treino["acuracia"],
            }

            if carregador_validacao is not None:
                metricas_val = self._executar_epoca(
                    carregador_validacao,
                    criterio,
                )

                registro["perda_validacao"] = metricas_val["perda"]
                registro["acuracia_validacao"] = metricas_val["acuracia"]

            historico.append(registro)

            if verbose:
                mensagem = (
                    f"Época {epoca}/{epocas}"
                    f" | perda treino: {registro['perda_treino']:.4f}"
                    f" | acurácia treino: "
                    f"{registro['acuracia_treino']:.4f}"
                )

                if "perda_validacao" in registro:
                    mensagem += (
                        f" | perda validação: "
                        f"{registro['perda_validacao']:.4f}"
                        f" | acurácia validação: "
                        f"{registro['acuracia_validacao']:.4f}"
                    )

                print(mensagem)

        self.modelo.eval()
        self.treinado = True
        self.epocas_treinadas += epocas
        self.historico_treinamento = historico

        return historico

    @torch.inference_mode()
    def avaliar(
        self,
        imagens: Any,
        rotulos: Any,
        *,
        tamanho_lote: int = 32,
    ) -> dict[str, Any]:
        """
        Avalia o modelo em dados separados do treinamento.

        Retorna perda, acurácia, matriz de confusão e métricas por classe.
        """

        if not self.treinado:
            raise RuntimeError(
                "Treine ou carregue o modelo antes de avaliá-lo."
            )

        if (
            isinstance(tamanho_lote, bool)
            or not isinstance(tamanho_lote, int)
            or tamanho_lote < 1
        ):
            raise ValueError(
                "tamanho_lote deve ser um inteiro maior que zero."
            )

        imagens_tensor, rotulos_tensor = self._validar_dados(
            imagens,
            rotulos,
            "Avaliação",
        )

        carregador = self._criar_carregador(
            imagens_tensor,
            rotulos_tensor,
            tamanho_lote,
            embaralhar=False,
        )

        self.modelo.eval()

        matriz = torch.zeros(
            (len(self.CLASSES), len(self.CLASSES)),
            dtype=torch.int64,
        )

        perda_total = 0.0
        total = 0
        criterio = nn.CrossEntropyLoss(reduction="sum")

        for imagens_lote, rotulos_lote in carregador:
            imagens_lote = imagens_lote.to(self.dispositivo)
            rotulos_lote = rotulos_lote.to(self.dispositivo)

            logits = self.modelo(imagens_lote)
            perda_total += criterio(logits, rotulos_lote).item()

            previsoes = logits.argmax(dim=1).cpu()
            rotulos_cpu = rotulos_lote.cpu()
            total += rotulos_lote.size(0)

            for real, previsto in zip(
                rotulos_cpu.tolist(),
                previsoes.tolist(),
            ):
                matriz[real, previsto] += 1

        acertos = int(matriz.diag().sum().item())
        metricas_por_classe: dict[str, dict[str, float]] = {}

        for indice, nome in self.CLASSES.items():
            verdadeiros_positivos = int(matriz[indice, indice].item())
            suporte = int(matriz[indice, :].sum().item())
            previstos = int(matriz[:, indice].sum().item())

            precisao = (
                verdadeiros_positivos / previstos
                if previstos
                else 0.0
            )

            recall = (
                verdadeiros_positivos / suporte
                if suporte
                else 0.0
            )

            f1 = (
                2 * precisao * recall / (precisao + recall)
                if precisao + recall
                else 0.0
            )

            metricas_por_classe[nome] = {
                "precisao": precisao,
                "recall": recall,
                "f1": f1,
                "suporte": float(suporte),
            }

        return {
            "perda": perda_total / total,
            "acuracia": acertos / total,
            "amostras": total,
            "matriz_confusao": matriz.tolist(),
            "metricas_por_classe": metricas_por_classe,
        }

    def salvar(self) -> Path:
        """Salva os pesos e os metadados do modelo."""

        if not self.treinado:
            raise RuntimeError(
                "Treine ou carregue o modelo antes de salvá-lo."
            )

        self.caminho_modelo.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        estado = {
            chave: tensor.detach().cpu()
            for chave, tensor in self.modelo.state_dict().items()
        }

        torch.save(estado, self.caminho_modelo)

        metadados = {
            "versao_formato": self.VERSAO_FORMATO,
            "arquitetura": "RedeNeuralMorse",
            "tamanho_imagem": self.TAMANHO_IMAGEM,
            "canais_entrada": 1,
            "classes": {
                str(chave): valor
                for chave, valor in self.CLASSES.items()
            },
            "epocas_treinadas": self.epocas_treinadas,
        }

        self.caminho_metadados.write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return self.caminho_modelo

    def carregar(self) -> None:
        """Carrega os pesos e verifica os metadados, quando disponíveis."""

        if not self.caminho_modelo.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado: {self.caminho_modelo}"
            )

        metadados = None

        if self.caminho_metadados.is_file():
            try:
                metadados = json.loads(
                    self.caminho_metadados.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as erro:
                raise ValueError(
                    "O arquivo de metadados do modelo é inválido."
                ) from erro

            if metadados.get("versao_formato") != self.VERSAO_FORMATO:
                raise ValueError(
                    "A versão dos metadados é incompatível."
                )

            classes_salvas = {
                int(chave): valor
                for chave, valor in metadados.get("classes", {}).items()
            }

            if classes_salvas != self.CLASSES:
                raise ValueError(
                    "As classes salvas não correspondem às classes atuais."
                )

            if metadados.get("tamanho_imagem") != self.TAMANHO_IMAGEM:
                raise ValueError(
                    "O tamanho de imagem salvo não é compatível."
                )

        try:
            estado = torch.load(
                self.caminho_modelo,
                map_location=self.dispositivo,
                weights_only=True,
            )

            self.modelo.load_state_dict(estado, strict=True)

        except (OSError, RuntimeError, ValueError) as erro:
            raise ValueError(
                "Não foi possível carregar os pesos do modelo."
            ) from erro

        self.modelo.to(self.dispositivo)
        self.modelo.eval()
        self.treinado = True

        if metadados is not None:
            self.epocas_treinadas = int(
                metadados.get("epocas_treinadas", 0)
            )

    @torch.inference_mode()
    def prever(self, imagem: Any) -> dict[str, Any]:
        """
        Classifica um recorte de símbolo.

        imagem:
            Tensor (1, 32, 32), com pixels entre 0 e 1.

        Retorna:
            Classe, símbolo, confiança e probabilidades por classe.
        """

        if not self.treinado:
            raise RuntimeError(
                "O modelo ainda não foi treinado ou carregado."
            )

        try:
            tensor = torch.as_tensor(
                imagem,
                dtype=torch.float32,
            )
        except (TypeError, ValueError, RuntimeError) as erro:
            raise ValueError(
                "A imagem não pôde ser convertida em tensor."
            ) from erro

        formato_esperado = (
            1,
            self.TAMANHO_IMAGEM,
            self.TAMANHO_IMAGEM,
        )

        if tuple(tensor.shape) != formato_esperado:
            raise ValueError(
                f"A imagem deve ter formato {formato_esperado}; "
                f"recebido {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all().item():
            raise ValueError(
                "A imagem contém valores NaN ou infinitos."
            )

        if (
            (tensor < 0).any().item()
            or (tensor > 1).any().item()
        ):
            raise ValueError(
                "Os pixels devem estar normalizados entre 0 e 1."
            )

        self.modelo.eval()

        entrada = tensor.unsqueeze(0).to(self.dispositivo)
        logits = self.modelo(entrada)
        probabilidades = torch.softmax(logits, dim=1)[0].cpu()

        indice = int(probabilidades.argmax().item())

        return {
            "classe": indice,
            "simbolo": self.CLASSES[indice],
            "confianca": float(probabilidades[indice].item()),
            "probabilidades": {
                self.CLASSES[i]: float(probabilidades[i].item())
                for i in self.CLASSES
            },
        }