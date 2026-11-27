from __future__ import annotations

import argparse
import json
from pathlib import Path

from ia.modelo import ModeloCodeVision
from ia.reconhecimento import ReconhecedorCodeVision


SEPARADOR_LINHAS = " | "


def distancia_edicao(a: str, b: str) -> int:
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        atual = [i]
        for j, cb in enumerate(b, start=1):
            atual.append(min(
                anterior[j] + 1,
                atual[j - 1] + 1,
                anterior[j - 1] + (ca != cb),
            ))
        anterior = atual
    return anterior[-1]


def carregar_gabarito(pasta: Path) -> dict[str, list[str]]:
    json_path = pasta / "gabarito.json"
    if json_path.is_file():
        dados = json.loads(json_path.read_text(encoding="utf-8"))
        return {nome: item["linhas"] for nome, item in dados.items()}

    txt_path = pasta / "gabarito.txt"
    if txt_path.is_file():
        gabarito = {}
        for linha in txt_path.read_text(encoding="utf-8").splitlines():
            if not linha.strip() or linha.lstrip().startswith("#"):
                continue
            nome, _, texto = linha.partition(":")
            gabarito[nome.strip()] = [
                parte.strip().upper() for parte in texto.split("|") if parte.strip()
            ]
        return gabarito

    raise SystemExit(f"Nenhum gabarito.json ou gabarito.txt em {pasta}")


def avaliar(pasta: Path, caminho_modelo: Path, detalhes: int) -> float:
    gabarito = carregar_gabarito(pasta)
    modelo = ModeloCodeVision(caminho_modelo)
    modelo.carregar()
    reconhecedor = ReconhecedorCodeVision(modelo)

    erros = tamanho = exatas = 0
    resultados = []

    for nome, linhas in sorted(gabarito.items()):
        arquivo = pasta / nome
        if not arquivo.is_file():
            print(f"  (ausente) {nome}")
            continue

        lido = reconhecedor.reconhecer_arquivo(arquivo)
        esperado = SEPARADOR_LINHAS.join(linhas)
        obtido = SEPARADOR_LINHAS.join(l["texto"] for l in lido["linhas"])

        distancia = distancia_edicao(esperado, obtido)
        erros += distancia
        tamanho += len(esperado)
        exatas += distancia == 0
        resultados.append((distancia / max(1, len(esperado)), nome, esperado, obtido))

    if not resultados:
        raise SystemExit("Nenhuma foto avaliada.")

    acuracia = 1 - erros / max(1, tamanho)

    resultados.sort(reverse=True)
    if detalhes:
        print("\nPiores leituras:")
        for taxa, nome, esperado, obtido in resultados[:detalhes]:
            print(f"  {nome}: erro {taxa:.0%}\n    esperado: {esperado}\n    lido:     {obtido}")

    print(
        f"\nFotos: {len(resultados)} | acurácia por caractere: {acuracia:.1%} | "
        f"fotos 100% certas: {exatas} ({exatas / len(resultados):.0%})"
    )
    return acuracia


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pasta", type=Path)
    parser.add_argument("--modelo", type=Path, default=Path("modelos/modelo_morse.pth"))
    parser.add_argument("--detalhes", type=int, default=5, help="quantas piores leituras mostrar")
    argumentos = parser.parse_args()
    avaliar(argumentos.pasta, argumentos.modelo, argumentos.detalhes)


if __name__ == "__main__":
    main()