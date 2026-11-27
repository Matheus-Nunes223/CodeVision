import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union


Caminho = Union[str, Path]


class ErroUtilitario(Exception):
    pass


def obter_raiz_projeto() -> Path:
    return Path(__file__).resolve().parent.parent


def resolver_caminho(
    caminho: Caminho,
    relativo_a_raiz: bool = True,
) -> Path:
    if not isinstance(caminho, (str, Path)):
        raise TypeError("O caminho deve ser uma string ou Path.")

    caminho_path = Path(caminho).expanduser()

    if caminho_path.is_absolute():
        return caminho_path.resolve()

    if relativo_a_raiz:
        return (obter_raiz_projeto() / caminho_path).resolve()

    return caminho_path.resolve()


def garantir_diretorio(
    caminho: Caminho,
    relativo_a_raiz: bool = True,
) -> Path:
    diretorio = resolver_caminho(caminho, relativo_a_raiz)

    if diretorio.exists() and not diretorio.is_dir():
        raise ErroUtilitario(
            f"O caminho existe, mas não é um diretório: {diretorio}"
        )

    try:
        diretorio.mkdir(parents=True, exist_ok=True)
    except OSError as erro:
        raise ErroUtilitario(
            f"Não foi possível criar o diretório '{diretorio}': {erro}"
        ) from erro

    return diretorio


def validar_arquivo(
    caminho: Caminho,
    extensoes_permitidas: Optional[set[str]] = None,
    relativo_a_raiz: bool = True,
) -> Path:
    arquivo = resolver_caminho(caminho, relativo_a_raiz)

    if not arquivo.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {arquivo}"
        )

    if not arquivo.is_file():
        raise ErroUtilitario(
            f"O caminho informado não é um arquivo: {arquivo}"
        )

    if extensoes_permitidas is not None:
        if not isinstance(extensoes_permitidas, set):
            raise TypeError(
                "extensoes_permitidas deve ser um conjunto (set)."
            )

        extensoes = {
            extensao.lower()
            if extensao.startswith(".")
            else f".{extensao.lower()}"
            for extensao in extensoes_permitidas
        }

        if arquivo.suffix.lower() not in extensoes:
            raise ErroUtilitario(
                f"Extensão não permitida: {arquivo.suffix}. "
                f"Extensões aceitas: {sorted(extensoes)}"
            )

    return arquivo


def validar_numero(
    valor: Union[int, float],
    nome: str,
    minimo: Optional[float] = None,
    maximo: Optional[float] = None,
    permitir_igual_minimo: bool = True,
    permitir_igual_maximo: bool = True,
) -> float:
    if (
        not isinstance(valor, (int, float))
        or isinstance(valor, bool)
    ):
        raise TypeError(f"{nome} deve ser numérico.")

    numero = float(valor)

    if not math.isfinite(numero):
        raise ValueError(f"{nome} deve ser finito.")

    if minimo is not None:
        if permitir_igual_minimo:
            invalido = numero < minimo
        else:
            invalido = numero <= minimo

        if invalido:
            operador = ">=" if permitir_igual_minimo else ">"
            raise ValueError(
                f"{nome} deve ser {operador} {minimo}."
            )

    if maximo is not None:
        if permitir_igual_maximo:
            invalido = numero > maximo
        else:
            invalido = numero >= maximo

        if invalido:
            operador = "<=" if permitir_igual_maximo else "<"
            raise ValueError(
                f"{nome} deve ser {operador} {maximo}."
            )

    return numero


def salvar_json(
    dados: Any,
    caminho: Caminho,
    relativo_a_raiz: bool = True,
) -> Path:
    arquivo = resolver_caminho(caminho, relativo_a_raiz)

    if arquivo.suffix.lower() != ".json":
        raise ErroUtilitario(
            "O arquivo de saída precisa ter a extensão .json."
        )

    try:
        arquivo.parent.mkdir(parents=True, exist_ok=True)

        with arquivo.open("w", encoding="utf-8") as destino:
            json.dump(
                dados,
                destino,
                ensure_ascii=False,
                indent=4,
                allow_nan=False,
            )

            destino.write("\n")

    except (OSError, TypeError, ValueError) as erro:
        raise ErroUtilitario(
            f"Não foi possível salvar o JSON '{arquivo}': {erro}"
        ) from erro

    return arquivo


def carregar_json(
    caminho: Caminho,
    relativo_a_raiz: bool = True,
) -> Any:
    arquivo = validar_arquivo(
        caminho,
        extensoes_permitidas={".json"},
        relativo_a_raiz=relativo_a_raiz,
    )

    try:
        with arquivo.open("r", encoding="utf-8") as origem:
            return json.load(origem)

    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as erro:
        raise ErroUtilitario(
            f"Não foi possível carregar o JSON '{arquivo}': {erro}"
        ) from erro


def configurar_logger(
    nome: str = "CodeVision",
    nivel: int = logging.INFO,
    arquivo_log: Optional[Caminho] = None,
) -> logging.Logger:
    if not isinstance(nome, str) or not nome.strip():
        raise ValueError("O nome do logger não pode estar vazio.")

    if not isinstance(nivel, int):
        raise TypeError("O nível do logger deve ser um inteiro.")

    logger = logging.getLogger(nome)
    logger.setLevel(nivel)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler_terminal = logging.StreamHandler()
    handler_terminal.setLevel(nivel)
    handler_terminal.setFormatter(formato)
    logger.addHandler(handler_terminal)

    if arquivo_log is not None:
        caminho_log = resolver_caminho(arquivo_log)
        garantir_diretorio(caminho_log.parent)

        handler_arquivo = logging.FileHandler(
            caminho_log,
            encoding="utf-8",
        )
        handler_arquivo.setLevel(nivel)
        handler_arquivo.setFormatter(formato)
        logger.addHandler(handler_arquivo)

    return logger


def obter_timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resumir_resultado_reconhecimento(
    resultado: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(resultado, dict):
        raise TypeError("O resultado deve ser um dicionário.")

    simbolos = resultado.get("simbolos", [])
    resultados = resultado.get("resultados", [])

    if not isinstance(simbolos, list):
        raise ErroUtilitario(
            "O campo 'simbolos' precisa ser uma lista."
        )

    if not isinstance(resultados, list):
        raise ErroUtilitario(
            "O campo 'resultados' precisa ser uma lista."
        )

    confiancas = []

    for item in resultados:
        if not isinstance(item, dict):
            continue

        confianca = item.get("confianca")

        if isinstance(confianca, (int, float)):
            confianca = float(confianca)

            if math.isfinite(confianca) and 0 <= confianca <= 1:
                confiancas.append(confianca)

    return {
        "data_utc": obter_timestamp_utc(),
        "quantidade_componentes": resultado.get(
            "quantidade_componentes",
            len(resultados),
        ),
        "quantidade_reconhecidos": resultado.get(
            "quantidade_reconhecidos",
            len(simbolos),
        ),
        "simbolos": simbolos.copy(),
        "confianca_media": (
            sum(confiancas) / len(confiancas)
            if confiancas
            else None
        ),
    }