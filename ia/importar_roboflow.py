"""
Importa um dataset de fotos reais exportado do Roboflow (formato COCO).

Cada foto contém um ou mais símbolos anotados (classes 'dash' e 'dots').
O script recorta cada símbolo e também recortes de fundo (papel vazio)
da mesma foto, longe de todos os símbolos anotados.

Destino:
    - Divisão 'train' do Roboflow -> datasets/{ponto,traco,fundo}
      (usada no treinamento).
    - Divisões 'valid' e 'test' -> datasets_teste/{ponto,traco,fundo}
      (teste independente: nunca entra no treinamento).

Os arquivos se chamam 'real_<fonte>_<foto>_<indice>.png':
    - 'real_' marca exemplos reais (o treinamento os repete mais vezes);
    - '<fonte>' identifica a importação (--fonte), permitindo importar
      vários datasets sem que um apague o outro;
    - '<foto>' mantém todos os recortes de uma foto no mesmo conjunto.

Reexecutar o script com a mesma --fonte substitui a importação anterior
dessa fonte (apaga apenas os arquivos 'real_<fonte>_*'). Antes de gravar,
o script verifica que nenhuma foto está no treino e no teste ao mesmo
tempo.

Uso (a partir de CodeVision/):
    python -m ia.importar_roboflow caminho/dataset.coco.zip
    python -m ia.importar_roboflow outro.coco.zip --fonte colegas
"""

import argparse
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = {"dash": "traco", "dots": "ponto"}
PASTAS = ("fundo", "ponto", "traco")
DIVISAO_PARA_DESTINO = {"train": "treino", "valid": "teste", "test": "teste"}

MARGEM_RELATIVA = 0.05
LADO_MAXIMO = 256
CONTRASTE_MAXIMO_FUNDO = 60

Caixa = list[float]


class ErroImportacao(Exception):
    """Erro que impede uma importação segura."""


def nome_da_foto(arquivo: str) -> str:
    """Remove o sufixo '_jpg.rf.<hash>' que o Roboflow acrescenta."""
    base = Path(arquivo).name
    return re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", base, flags=re.I)


def reduzir(imagem: Image.Image) -> Image.Image:
    """Limita o lado maior; o treinamento redimensiona para 32x32."""
    imagem = imagem.copy()
    imagem.thumbnail((LADO_MAXIMO, LADO_MAXIMO), Image.Resampling.LANCZOS)
    return imagem


def recortar_simbolo(foto: Image.Image, caixa: Caixa) -> Image.Image:
    x, y, largura, altura = caixa
    margem_x = MARGEM_RELATIVA * largura
    margem_y = MARGEM_RELATIVA * altura
    return foto.crop(
        (
            max(0, int(x - margem_x)),
            max(0, int(y - margem_y)),
            min(foto.width, int(x + largura + margem_x)),
            min(foto.height, int(y + altura + margem_y)),
        )
    )


def _area_proibida(caixa: Caixa) -> tuple[float, float, float, float]:
    """A caixa ampliada em 50% da sua dimensão para cada lado."""
    x, y, largura, altura = caixa
    return (
        x - 0.5 * largura,
        y - 0.5 * altura,
        x + 1.5 * largura,
        y + 1.5 * altura,
    )


def _sobrepoe(
    x0: int,
    y0: int,
    largura: int,
    altura: int,
    area: tuple[float, float, float, float],
) -> bool:
    return not (
        x0 + largura <= area[0]
        or x0 >= area[2]
        or y0 + altura <= area[1]
        or y0 >= area[3]
    )


def recortes_de_fundo(
    foto: Image.Image,
    caixas: list[Caixa],
    quantidade: int,
    gerador: random.Random,
) -> list[Image.Image]:
    """
    Sorteia recortes sem símbolos, com tamanho parecido com o deles.

    Os recortes evitam todas as caixas anotadas na foto.
    """
    proibidas = [_area_proibida(caixa) for caixa in caixas]
    saidas = []

    for _ in range(quantidade * 40):
        if len(saidas) == quantidade:
            break

        _, _, largura, altura = gerador.choice(caixas)
        c_larg = int(largura * gerador.uniform(0.8, 1.2))
        c_alt = int(altura * gerador.uniform(0.8, 1.2))
        c_larg = min(max(16, c_larg), foto.width // 2)
        c_alt = min(max(16, c_alt), foto.height // 2)

        x0 = gerador.randint(0, foto.width - c_larg)
        y0 = gerador.randint(0, foto.height - c_alt)

        if any(_sobrepoe(x0, y0, c_larg, c_alt, area) for area in proibidas):
            continue

        recorte = foto.crop((x0, y0, x0 + c_larg, y0 + c_alt))
        pixels = np.asarray(recorte, dtype=np.float32)

        # Descarta recortes com tinta não anotada, bordas ou sombras.
        contraste = np.percentile(pixels, 50) - np.percentile(pixels, 1)
        if contraste > CONTRASTE_MAXIMO_FUNDO:
            continue

        saidas.append(recorte)

    return saidas


def ler_divisoes(zf: zipfile.ZipFile) -> list[dict]:
    """
    Lê as anotações e agrupa por imagem.

    Retorna uma lista de fotos com destino, arquivo, foto_id e símbolos.
    """
    fotos = []

    for divisao, destino in DIVISAO_PARA_DESTINO.items():
        caminho = f"{divisao}/_annotations.coco.json"

        if caminho not in zf.namelist():
            continue

        dados = json.loads(zf.read(caminho))
        categorias = {c["id"]: c["name"] for c in dados["categories"]}
        simbolos_por_imagem = defaultdict(list)

        for anotacao in dados["annotations"]:
            simbolos_por_imagem[anotacao["image_id"]].append(
                (categorias[anotacao["category_id"]], anotacao["bbox"])
            )

        for imagem in sorted(dados["images"], key=lambda i: i["id"]):
            fotos.append(
                {
                    "destino": destino,
                    "arquivo": f"{divisao}/{imagem['file_name']}",
                    "foto_id": nome_da_foto(imagem["file_name"]),
                    "simbolos": simbolos_por_imagem.get(imagem["id"], []),
                }
            )

    if not fotos:
        raise ErroImportacao(
            "Nenhum arquivo _annotations.coco.json encontrado no ZIP."
        )

    return fotos


def verificar_sobreposicao(fotos: list[dict]) -> None:
    """Impede que a mesma foto (ou cópia dela) vá para treino e teste."""
    por_destino = defaultdict(set)

    for foto in fotos:
        por_destino[foto["destino"]].add(foto["foto_id"])

    comuns = sorted(por_destino["treino"] & por_destino["teste"])

    if comuns:
        raise ErroImportacao(
            f"{len(comuns)} foto(s) aparecem no treino e no teste "
            f"(ex.: {', '.join(comuns[:5])}). Gere a versão no Roboflow "
            "sem augmentation, ou refaça a divisão."
        )


def limpar_importacao_anterior(raizes: list[Path], prefixo: str) -> int:
    """Apaga apenas os arquivos gerados por esta fonte."""
    removidos = 0

    for raiz in raizes:
        for pasta in PASTAS:
            diretorio = raiz / pasta

            if not diretorio.is_dir():
                continue

            for arquivo in diretorio.glob(f"{prefixo}*.png"):
                arquivo.unlink()
                removidos += 1

    return removidos


def importar(
    caminho_zip: Path,
    destino_treino: Path,
    destino_teste: Path,
    fonte: str,
    fundo_por_foto: int,
    semente: int,
) -> Counter:
    gerador = random.Random(semente)
    raizes = {"treino": destino_treino, "teste": destino_teste}
    prefixo = f"real_{fonte}_"
    contagem: Counter = Counter()

    with zipfile.ZipFile(caminho_zip) as zf:
        fotos = ler_divisoes(zf)

        # Valida tudo antes de apagar ou gravar qualquer arquivo.
        verificar_sobreposicao(fotos)

        removidos = limpar_importacao_anterior(
            list(raizes.values()), prefixo
        )
        if removidos:
            print(f"Removidos {removidos} recortes da importação anterior.")

        for pasta in PASTAS:
            for raiz in raizes.values():
                (raiz / pasta).mkdir(parents=True, exist_ok=True)

        # Cópias do Roboflow de uma mesma foto compartilham o foto_id;
        # o índice continua entre elas para que nenhum arquivo seja
        # sobrescrito.
        proximo_indice: Counter = Counter()

        def salvar(imagem: Image.Image, raiz: Path, classe: str, foto_id: str):
            indice = proximo_indice[foto_id]
            proximo_indice[foto_id] += 1
            reduzir(imagem).save(
                raiz / classe / f"{prefixo}{foto_id}_{indice}.png"
            )

        for foto in fotos:
            raiz = raizes[foto["destino"]]
            validos = [
                (CLASSES[nome], caixa)
                for nome, caixa in foto["simbolos"]
                if nome in CLASSES
            ]
            contagem["ignorados"] += len(foto["simbolos"]) - len(validos)

            if not validos:
                continue

            with Image.open(io.BytesIO(zf.read(foto["arquivo"]))) as aberta:
                imagem = aberta.convert("L")

            for classe, caixa in validos:
                salvar(recortar_simbolo(imagem, caixa), raiz, classe, foto["foto_id"])
                contagem[f"{foto['destino']}/{classe}"] += 1

            fundos = recortes_de_fundo(
                imagem,
                [caixa for _, caixa in validos],
                fundo_por_foto,
                gerador,
            )
            for recorte in fundos:
                salvar(recorte, raiz, "fundo", foto["foto_id"])
                contagem[f"{foto['destino']}/fundo"] += 1

    return contagem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, help="ZIP do Roboflow (COCO)")
    parser.add_argument("--destino", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--destino-teste", type=Path, default=Path("datasets_teste")
    )
    parser.add_argument(
        "--fonte",
        default="roboflow",
        help="identificador desta importação (letras, números e '-')",
    )
    parser.add_argument("--fundo-por-foto", type=int, default=2)
    parser.add_argument("--semente", type=int, default=42)
    argumentos = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9-]+", argumentos.fonte):
        parser.error("--fonte deve conter apenas letras, números e '-'.")

    try:
        contagem = importar(
            argumentos.zip,
            argumentos.destino,
            argumentos.destino_teste,
            argumentos.fonte,
            argumentos.fundo_por_foto,
            argumentos.semente,
        )
    except ErroImportacao as erro:
        raise SystemExit(f"Importação cancelada: {erro}") from erro

    print("Recortes salvos:")
    for chave in sorted(contagem):
        print(f"  {chave}: {contagem[chave]}")


if __name__ == "__main__":
    main()
