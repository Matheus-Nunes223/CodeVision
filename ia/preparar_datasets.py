"""
Prepara o dataset MorseCode (apple1986/MorseCodeImageClassify) para o
treinamento do CodeVision.

Cada imagem do ZIP original é uma letra Morse (512x512) com pontos
(elipses) e traços (retângulos). O script segmenta os símbolos, recorta
cada um e salva em datasets/{fundo,ponto,traco}.

Uma imagem só é aceita se a geometria dos símbolos, lida da esquerda
para a direita, coincidir com o código Morse da letra da pasta.

Uso (na raiz do repositório):
    python -m ia.preparar_datasets caminho/MorseCode.zip
"""

import argparse
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
    "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
    "Z": "--..",
}

NOMES_CLASSE = {".": "ponto", "-": "traco"}

MARGEM = 2
AREA_MINIMA = 150
RAZAO_PREENCHIMENTO_TRACO = 0.88
LIMIAR_FUNDO = 245


def decodificar_imagem(dados: bytes) -> np.ndarray:
    """Decodifica bytes de PNG em RGB/cinza sem alterar a profundidade."""
    imagem = cv2.imdecode(
        np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if imagem is None:
        raise ValueError("Imagem inválida.")
    return imagem


def mascara_simbolos(imagem_bgr: np.ndarray) -> np.ndarray:
    """Máscara dos símbolos, robusta aos ruídos de pulso e gaussiano."""
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    cinza = cv2.medianBlur(cinza, 5)
    mascara = (cinza < LIMIAR_FUNDO).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    preenchida = np.zeros_like(mascara)
    cv2.drawContours(preenchida, contornos, -1, 255, thickness=cv2.FILLED)
    return preenchida


Caixa = tuple[int, int, int, int, int]


def ordenar_leitura(caixas: list[Caixa]) -> list[Caixa]:
    """Ordena por linha (centro vertical) e, dentro da linha, por x."""
    if not caixas:
        return []
    altura_media = sum(c[3] for c in caixas) / len(caixas)
    por_centro = sorted(caixas, key=lambda c: c[1] + c[3] / 2)
    linhas: list[list[Caixa]] = [[por_centro[0]]]
    for caixa in por_centro[1:]:
        anterior = linhas[-1][-1]
        if (caixa[1] + caixa[3] / 2) - (anterior[1] + anterior[3] / 2) > altura_media / 2:
            linhas.append([])
        linhas[-1].append(caixa)
    return [c for linha in linhas for c in sorted(linha, key=lambda c: c[0])]


def localizar_simbolos(mascara: np.ndarray) -> list[Caixa]:
    """Retorna caixas (x, y, largura, altura, area) em ordem de leitura."""
    numero, _, estatisticas, _ = cv2.connectedComponentsWithStats(
        mascara, connectivity=8
    )
    caixas = [
        tuple(int(v) for v in estatisticas[i][:5])
        for i in range(1, numero)
        if estatisticas[i][4] >= AREA_MINIMA
    ]
    return ordenar_leitura(caixas)  # type: ignore[arg-type]


def tipo_da_caixa(caixa: Caixa) -> str:
    """Elipse (ponto) preenche ~0,79 da caixa; retângulo (traço) ~1."""
    _, _, largura, altura, area = caixa
    return "-" if area / (largura * altura) >= RAZAO_PREENCHIMENTO_TRACO else "."


def toca_borda(caixa: Caixa, forma: tuple[int, int]) -> bool:
    """Indica se o símbolo está cortado pela borda da imagem."""
    x, y, largura, altura, _ = caixa
    return x <= 0 or y <= 0 or x + largura >= forma[1] or y + altura >= forma[0]


def recortar(cinza: np.ndarray, caixa: Caixa) -> np.ndarray:
    """Recorta a caixa com a mesma margem usada pelo reconhecedor."""
    x, y, largura, altura, _ = caixa
    altura_img, largura_img = cinza.shape
    return cinza[
        max(0, y - MARGEM): min(altura_img, y + altura + MARGEM),
        max(0, x - MARGEM): min(largura_img, x + largura + MARGEM),
    ]


def recortes_de_fundo(
    cinza: np.ndarray,
    mascara: np.ndarray,
    caixas: list[Caixa],
    quantidade: int,
    gerador: random.Random,
) -> list[np.ndarray]:
    """Sorteia recortes sem símbolo, com tamanhos parecidos aos reais."""
    altura_img, largura_img = cinza.shape
    livre = cv2.dilate(mascara, np.ones((15, 15), np.uint8)) == 0
    saidas = []

    for _ in range(quantidade * 20):
        if len(saidas) == quantidade:
            break
        base = gerador.choice(caixas)
        largura = int(base[2] * gerador.uniform(0.8, 1.2))
        altura = int(base[3] * gerador.uniform(0.8, 1.2))
        largura = min(max(8, largura), largura_img // 2)
        altura = min(max(8, altura), altura_img // 2)
        x = gerador.randint(0, largura_img - largura)
        y = gerador.randint(0, altura_img - altura)
        if livre[y: y + altura, x: x + largura].all():
            saidas.append(cinza[y: y + altura, x: x + largura])

    return saidas


def processar(
    caminho_zip: Path,
    destino: Path,
    fundo_por_imagem: int,
    semente: int,
) -> None:
    gerador = random.Random(semente)
    for nome in NOMES_CLASSE.values():
        (destino / nome).mkdir(parents=True, exist_ok=True)
    (destino / "fundo").mkdir(parents=True, exist_ok=True)

    padrao = re.compile(r"^(?:.*/)?(?P<sub>[^/]+)/(?P<letra>[A-Z])/(?P<id>[^/]+)\.png$")
    contagem = Counter()
    rejeitadas = Counter()

    with zipfile.ZipFile(caminho_zip) as zf:
        for info in sorted(zf.namelist()):
            m = padrao.match(info)
            if not m:
                continue

            letra = m["letra"]
            partes = info.split("/")
            origem = next(
                (p.split("_")[1].lower() for p in partes if p.startswith("MorseCode_") and "Train_Test" in p),
                "x",
            )
            etiqueta = f"{origem}_{m['sub']}_{letra}_{m['id']}"

            imagem = decodificar_imagem(zf.read(info))
            mascara = mascara_simbolos(imagem)
            caixas = localizar_simbolos(mascara)
            lido = "".join(tipo_da_caixa(c) for c in caixas)

            if lido != MORSE[letra]:
                rejeitadas[origem] += 1
                continue

            cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            for indice, (caixa, simbolo) in enumerate(zip(caixas, lido)):
                if toca_borda(caixa, cinza.shape):
                    continue
                classe = NOMES_CLASSE[simbolo]
                cv2.imwrite(
                    str(destino / classe / f"{etiqueta}_{indice}.png"),
                    recortar(cinza, caixa),
                )
                contagem[classe] += 1

            fundos = recortes_de_fundo(
                cinza, mascara, caixas, fundo_por_imagem, gerador
            )
            for indice, recorte in enumerate(fundos):
                cv2.imwrite(
                    str(destino / "fundo" / f"{etiqueta}_{indice}.png"),
                    recorte,
                )
                contagem["fundo"] += 1

            contagem[f"aceitas_{origem}"] += 1

    print("Recortes salvos:", {k: v for k, v in contagem.items() if not k.startswith("aceitas_")})
    print("Imagens aceitas:", {k[8:]: v for k, v in contagem.items() if k.startswith("aceitas_")})
    print("Imagens rejeitadas (não bateram com o código Morse):", dict(rejeitadas))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, help="MorseCode.zip do apple1986")
    parser.add_argument("--destino", type=Path, default=Path("datasets"))
    parser.add_argument("--fundo-por-imagem", type=int, default=2)
    parser.add_argument("--semente", type=int, default=42)
    argumentos = parser.parse_args()

    processar(
        argumentos.zip,
        argumentos.destino,
        argumentos.fundo_por_imagem,
        argumentos.semente,
    )


if __name__ == "__main__":
    main()
