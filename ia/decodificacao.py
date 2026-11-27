from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MORSE_PARA_TEXTO = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", "-.-.--": "!",
    "-..-.": "/", "-.--.": "(", "-.--.-": ")", "---...": ":",
    "-...-": "=", ".-.-.": "+", "-....-": "-", ".--.-.": "@",
}

DESCONHECIDO = "?"


@dataclass
class SimboloPosicionado:
    simbolo: str
    x: float
    largura: float


def _limiar_entre_letras(espacos: list[float], referencia: float) -> float:
    padrao = 2.5 * referencia
    positivos = [e for e in espacos if e > 0]

    if len(positivos) < 2:
        return padrao

    valores = np.log1p(np.array(positivos, dtype=float))
    baixo, alto = valores.min(), valores.max()

    if alto - baixo < np.log(2):
        return padrao

    for _ in range(20):
        corte = (baixo + alto) / 2
        grupo_baixo = valores[valores <= corte]
        grupo_alto = valores[valores > corte]
        if len(grupo_baixo) == 0 or len(grupo_alto) == 0:
            break
        novo_baixo, novo_alto = grupo_baixo.mean(), grupo_alto.mean()
        if np.isclose(novo_baixo, baixo) and np.isclose(novo_alto, alto):
            break
        baixo, alto = novo_baixo, novo_alto

    return float(np.expm1((baixo + alto) / 2))


def _espacos_da_linha(linha: list[SimboloPosicionado]) -> list[float]:
    ordenada = sorted(linha, key=lambda s: s.x)
    pontos_tracos = [s for s in ordenada if s.simbolo in ".-"]
    barras = [s.x for s in ordenada if s.simbolo == "/"]
    return [
        b.x - (a.x + a.largura)
        for a, b in zip(pontos_tracos, pontos_tracos[1:])
        if not any(a.x < x < b.x for x in barras)
    ]


def estimar_limiar_letra(
    linhas: list[list[SimboloPosicionado]],
    diametro_ponto: float,
) -> float:
    espacos = [
        e for linha in linhas for e in _espacos_da_linha(linha)
        if e >= 0.5 * diametro_ponto
    ]
    limiar = _limiar_entre_letras(espacos, diametro_ponto)
    return max(limiar, 1.2 * diametro_ponto)


def separar_letras(
    linha: list[SimboloPosicionado],
    referencia: float,
    limiar_letra: float | None = None,
) -> list[list[str]]:
    ordenada = sorted(linha, key=lambda s: s.x)

    espacos_linha = _espacos_da_linha(linha)
    if limiar_letra is None:
        limiar_letra = _limiar_entre_letras(espacos_linha, referencia)

    entre_letras = [e for e in espacos_linha if e > limiar_letra]
    limiar_palavra = max(
        3 * limiar_letra,
        6 * referencia,
        2.2 * float(np.median(entre_letras)) if len(entre_letras) >= 2 else 0.0,
    )

    palavras: list[list[str]] = []
    letras: list[str] = []
    codigo = ""
    anterior = None

    def fechar_letra():
        nonlocal codigo
        if codigo:
            letras.append(codigo)
            codigo = ""

    def fechar_palavra():
        nonlocal letras
        fechar_letra()
        if letras:
            palavras.append(letras)
            letras = []

    for simbolo in ordenada:
        if simbolo.simbolo == "/":
            fechar_palavra()
            anterior = None
            continue

        if anterior is not None:
            espaco = simbolo.x - (anterior.x + anterior.largura)
            if espaco > limiar_palavra:
                fechar_palavra()
            elif espaco > limiar_letra:
                fechar_letra()

        codigo += simbolo.simbolo
        anterior = simbolo

    fechar_palavra()
    return palavras


def codigos_para_texto(palavras: list[list[str]]) -> str:
    return " ".join(
        "".join(MORSE_PARA_TEXTO.get(codigo, DESCONHECIDO) for codigo in palavra)
        for palavra in palavras
    )


def codigos_para_morse(palavras: list[list[str]]) -> str:
    return " / ".join(" ".join(palavra) for palavra in palavras)


def decodificar_linha(
    linha: list[SimboloPosicionado],
    referencia: float,
    limiar_letra: float | None = None,
) -> dict:
    palavras = separar_letras(linha, referencia, limiar_letra)
    return {
        "morse": codigos_para_morse(palavras),
        "texto": codigos_para_texto(palavras),
    }