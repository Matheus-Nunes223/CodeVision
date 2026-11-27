from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ia.gerador_manuscrito import PaginaSintetica, gerar_pagina
from ia.segmentacao import (
    Componente,
    ResultadoSegmentacao,
    recorte_isolado,
    segmentar,
)


PREFIXO = "manuscrito_"
CLASSE_DO_TIPO = {".": "ponto", "-": "traco"}
FUNDO_POR_PAGINA = 4


def _para_trabalho(
    segmentacao: ResultadoSegmentacao,
    caixa: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    altura, largura = segmentacao.imagem.shape
    matriz = cv2.getRotationMatrix2D(
        (largura / 2, altura / 2), segmentacao.angulo, 1.0
    )
    x, y, w, h = caixa
    cantos = np.array(
        [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float64
    ) * segmentacao.escala
    girados = cantos @ matriz[:, :2].T + matriz[:, 2]
    x0, y0 = girados.min(axis=0)
    x1, y1 = girados.max(axis=0)
    return x0, y0, x1, y1


def _rotular(
    segmentacao: ResultadoSegmentacao,
    pagina: PaginaSintetica,
) -> tuple[list[tuple[Componente, str]], Counter]:
    caixas = []
    for simbolo in pagina.simbolos:
        x0, y0, x1, y1 = _para_trabalho(segmentacao, simbolo.caixa)
        folga_x = 0.25 * (x1 - x0) + 2
        folga_y = 0.25 * (y1 - y0) + 2
        caixas.append((simbolo.tipo, x0 - folga_x, y0 - folga_y, x1 + folga_x, y1 + folga_y))

    componentes = [c for linha in segmentacao.linhas for c in linha]
    candidatos: dict[int, list[int]] = {}
    for indice, c in enumerate(componentes):
        cx = c.x + c.largura / 2
        cy = c.y + c.altura / 2
        candidatos[indice] = [
            j for j, (_, x0, y0, x1, y1) in enumerate(caixas)
            if x0 <= cx <= x1 and y0 <= cy <= y1
        ]

    usos = Counter(j for lista in candidatos.values() for j in lista)
    rotulados = []
    estatisticas = Counter()

    for indice, c in enumerate(componentes):
        lista = candidatos[indice]
        if c.barra:
            continue
        if not lista:
            rotulados.append((c, "fundo"))
            continue
        if len(lista) == 1 and usos[lista[0]] == 1:
            tipo = caixas[lista[0]][0]
            if tipo in CLASSE_DO_TIPO:
                rotulados.append((c, CLASSE_DO_TIPO[tipo]))
                estatisticas["encontrados"] += 1
            continue
        estatisticas["ambiguos"] += 1

    estatisticas["no_gabarito"] = sum(1 for s in pagina.simbolos if s.tipo in ".-")
    return rotulados, estatisticas


def _limpar(raiz: Path) -> int:
    removidos = 0
    for classe in ("ponto", "traco", "fundo"):
        pasta = raiz / classe
        if pasta.is_dir():
            for arquivo in pasta.glob(f"{PREFIXO}*.png"):
                arquivo.unlink()
                removidos += 1
    return removidos


def gerar(
    paginas: int,
    paginas_teste: int,
    destino_fotos: Path,
    destino_recortes: Path,
    semente: int,
) -> None:
    total = Counter()

    removidos = _limpar(destino_recortes)
    if removidos:
        print(f"Removidos {removidos} recortes de uma geração anterior.")

    for divisao, quantidade in (("treino", paginas), ("teste", paginas_teste)):
        pasta = destino_fotos / divisao
        pasta.mkdir(parents=True, exist_ok=True)
        for antigo in pasta.glob("pagina_*.*"):
            antigo.unlink()
        gabarito = {}

        for numero in range(quantidade):
            semente_pagina = zlib.crc32(f"{semente}-{divisao}-{numero}".encode())
            gerador = random.Random(semente_pagina)
            pagina = gerar_pagina(gerador)
            nome = f"pagina_{numero:04d}.jpg"
            cv2.imwrite(str(pasta / nome), pagina.imagem, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gabarito[nome] = {
                "linhas": pagina.linhas,
                "simbolos": [
                    {"tipo": s.tipo, "caixa": list(s.caixa), "linha": s.linha}
                    for s in pagina.simbolos
                ],
            }

            if divisao == "treino":
                segmentacao = segmentar(pagina.imagem)
                rotulados, estatisticas = _rotular(segmentacao, pagina)
                total.update(estatisticas)

                fundos = [r for r in rotulados if r[1] == "fundo"]
                simbolos = [r for r in rotulados if r[1] != "fundo"]
                gerador.shuffle(fundos)

                for indice, (componente, classe) in enumerate(
                    simbolos + fundos[:FUNDO_POR_PAGINA]
                ):
                    saida = destino_recortes / classe
                    saida.mkdir(parents=True, exist_ok=True)
                    recorte = recorte_isolado(segmentacao, componente)
                    cv2.imwrite(
                        str(saida / f"{PREFIXO}{divisao}{numero:04d}_{indice}.png"),
                        recorte,
                    )
                    total[f"recortes_{classe}"] += 1

            if (numero + 1) % 50 == 0:
                print(f"  {divisao}: {numero + 1}/{quantidade} páginas")

        (pasta / "gabarito.json").write_text(
            json.dumps(gabarito, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    if total["no_gabarito"]:
        print(
            f"\nSegmentação nas páginas de treino: {total['encontrados']} de "
            f"{total['no_gabarito']} pontos/traços encontrados isolados "
            f"({total['encontrados'] / total['no_gabarito']:.1%}); "
            f"{total['ambiguos']} componentes ambíguos descartados."
        )
    print("Recortes salvos:", {k[9:]: v for k, v in total.items() if k.startswith("recortes_")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paginas", type=int, default=400)
    parser.add_argument("--paginas-teste", type=int, default=100)
    parser.add_argument("--destino-fotos", type=Path, default=Path("fotos_sinteticas"))
    parser.add_argument("--destino-recortes", type=Path, default=Path("datasets"))
    parser.add_argument("--semente", type=int, default=42)
    argumentos = parser.parse_args()

    gerar(
        argumentos.paginas,
        argumentos.paginas_teste,
        argumentos.destino_fotos,
        argumentos.destino_recortes,
        argumentos.semente,
    )


if __name__ == "__main__":
    main()