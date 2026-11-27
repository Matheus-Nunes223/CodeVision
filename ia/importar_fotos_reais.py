from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import cv2

from ia.avaliar_fotos import carregar_gabarito
from ia.decodificacao import MORSE_PARA_TEXTO
from ia.segmentacao import recorte_isolado, segmentar


TEXTO_PARA_MORSE = {letra: codigo for codigo, letra in MORSE_PARA_TEXTO.items()}
CLASSE = {".": "ponto", "-": "traco"}


def morse_da_linha(texto: str) -> str:
    return "".join(TEXTO_PARA_MORSE.get(letra, "") for letra in texto if letra != " ")


def _parear_linhas(contagens: list[int], esperadas: list[str]) -> dict[int, str]:
    if len(contagens) == len(esperadas):
        return dict(enumerate(esperadas))

    pares = {}
    for indice, quantidade in enumerate(contagens):
        candidatas = [e for e in esperadas if len(e) == quantidade]
        if len(candidatas) == 1:
            pares[indice] = candidatas[0]
    return pares


def limpar(destino: Path, prefixo: str) -> int:
    removidos = 0
    for classe in CLASSE.values():
        pasta = destino / classe
        if pasta.is_dir():
            for arquivo in pasta.glob(f"{prefixo}*.png"):
                arquivo.unlink()
                removidos += 1
    return removidos


def importar(
    pasta: Path,
    fotos: list[str] | None,
    destino: Path,
    fonte: str,
) -> Counter:
    gabarito = carregar_gabarito(pasta)
    nomes = fotos if fotos else sorted(gabarito)
    prefixo = f"real_{fonte}_"
    contagem: Counter = Counter()

    removidos = limpar(destino, prefixo)
    if removidos:
        print(f"Removidos {removidos} recortes da importação anterior.")

    for nome in nomes:
        if nome not in gabarito:
            raise SystemExit(f"{nome} não está no gabarito de {pasta}.")

        imagem = cv2.imread(str(pasta / nome), cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            raise SystemExit(f"Não foi possível abrir {pasta / nome}.")

        segmentacao = segmentar(imagem)
        linhas = [
            [c for c in linha if not c.barra] for linha in segmentacao.linhas
        ]
        esperadas = [morse_da_linha(texto) for texto in gabarito[nome]]
        pares = _parear_linhas([len(l) for l in linhas], esperadas)

        foto_id = re.sub(r"[^A-Za-z0-9-]", "-", Path(nome).stem)
        indice = 0
        for numero, componentes in enumerate(linhas):
            esperado = pares.get(numero)
            if esperado is None or len(esperado) != len(componentes):
                contagem["linhas_descartadas"] += 1
                continue

            contagem["linhas_rotuladas"] += 1
            for componente, simbolo in zip(componentes, esperado):
                classe = CLASSE[simbolo]
                saida = destino / classe
                saida.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(saida / f"{prefixo}{foto_id}_{indice}.png"),
                    recorte_isolado(segmentacao, componente),
                )
                indice += 1
                contagem[classe] += 1

    return contagem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pasta", type=Path, help="pasta com as fotos e o gabarito")
    parser.add_argument("--fotos", nargs="*", help="só estas fotos (padrão: todas)")
    parser.add_argument("--destino", type=Path, default=Path("datasets"))
    parser.add_argument("--fonte", default="fotos")
    argumentos = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9-]+", argumentos.fonte):
        parser.error("--fonte deve conter apenas letras, números e '-'.")

    contagem = importar(
        argumentos.pasta, argumentos.fotos, argumentos.destino, argumentos.fonte
    )
    print(
        f"Linhas rotuladas: {contagem['linhas_rotuladas']} | "
        f"descartadas (quantidade não bateu): {contagem['linhas_descartadas']}"
    )
    print(f"Recortes: ponto {contagem['ponto']}, traco {contagem['traco']}")


if __name__ == "__main__":
    main()