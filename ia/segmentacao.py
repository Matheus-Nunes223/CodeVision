"""
Segmentação de símbolos Morse em fotos reais.

Localiza pontos, traços e barras de separação ("/") em fotos de papel,
inclusive cadernos pautados, com sombras, inclinação e objetos sobre a
folha (canetas, lapiseiras, espiral, mesa ao fundo).

Etapas:
    1. Reduz a foto para um tamanho de trabalho fixo.
    2. Normaliza a iluminação (divide pelo fundo estimado): sombras e
       regiões escuras uniformes viram "papel".
    3. Estima a região do papel e descarta o que estiver fora dela.
    4. Mede a inclinação pelas pautas do caderno e endireita a imagem.
    5. Binariza a tinta e remove as pautas (linhas horizontais longas).
    6. Filtra componentes pelo tamanho típico do traço da caneta.
    7. Agrupa os símbolos em linhas de texto (pela pauta, quando há).

A classificação ponto/traço é feita depois, pela CNN; aqui só se decide
o que é candidato a símbolo e como os candidatos se organizam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


LADO_TRABALHO = 1600

# A tinta é bem mais escura que a pauta impressa no papel normalizado.
LIMIAR_TINTA = 140
LIMIAR_PAUTA = 215

# O papel na sombra fica em ~45% do brilho do papel claro; objetos
# escuros (lapiseira, mesa, tecido) ficam abaixo de ~35%.
FRACAO_PAPEL = 0.38


@dataclass
class Componente:
    """Candidato a símbolo, em coordenadas da imagem de trabalho."""

    x: int
    y: int
    largura: int
    altura: int
    area: int
    rotulo: int = 0  # índice do componente em ResultadoSegmentacao.rotulos
    barra: bool = False

    @property
    def centro_y(self) -> float:
        return self.y + self.altura / 2

    @property
    def direita(self) -> int:
        return self.x + self.largura


@dataclass
class ResultadoSegmentacao:
    """Saída da segmentação."""

    imagem: np.ndarray  # cinza, reduzida e endireitada
    linhas: list[list[Componente]] = field(default_factory=list)
    # Mapa de componentes da tinta (sem pautas): permite recortar só os
    # pixels de um símbolo, sem a pauta ou vizinhos que cruzam o recorte.
    rotulos: np.ndarray | None = None
    espessura: float = 0.0  # espessura típica do traço da caneta (px)
    angulo: float = 0.0  # rotação aplicada (graus)
    escala: float = 1.0  # imagem de trabalho / original


def _normalizar_iluminacao(cinza: np.ndarray) -> np.ndarray:
    k = max(15, int(round(max(cinza.shape) / 50)) | 1)
    fundo = cv2.morphologyEx(
        cinza,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    norm = cinza.astype(np.float32) / np.maximum(fundo, 1) * 255
    return np.clip(norm, 0, 255).astype(np.uint8)


def _mascara_papel(cinza: np.ndarray) -> np.ndarray:
    """Regiões claras da foto (o papel), afastadas das bordas."""
    borrada = cv2.medianBlur(
        cv2.morphologyEx(
            cinza,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ),
        21,
    )
    referencia = float(np.percentile(borrada, 95))
    clara = (borrada > FRACAO_PAPEL * referencia).astype(np.uint8) * 255

    numero, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(
        clara, connectivity=4
    )
    if numero <= 1:
        return np.full_like(cinza, 255)

    # Um objeto sobre a folha pode dividir o papel em partes.
    maior = estatisticas[1:, cv2.CC_STAT_AREA].max()
    partes = [
        i for i in range(1, numero)
        if estatisticas[i, cv2.CC_STAT_AREA] >= 0.05 * maior
    ]
    papel = np.isin(rotulos, partes).astype(np.uint8) * 255
    papel = cv2.morphologyEx(
        papel,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    papel = _preencher_buracos(papel)
    k = max(11, int(max(cinza.shape) / 75) | 1)
    return cv2.erode(
        papel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )


def _preencher_buracos(papel: np.ndarray) -> np.ndarray:
    """
    Marca como papel toda região escura cercada de papel.

    Pontos grossos de caneta parecem "regiões escuras" para a máscara;
    mesa e objetos sobre a folha sempre chegam à borda da foto, então
    só o que não toca a borda é preenchido.
    """
    fora = (papel == 0).astype(np.uint8)
    numero, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(
        fora, connectivity=4
    )
    altura, largura = papel.shape
    preenchido = papel.copy()

    for i in range(1, numero):
        x, y, w, h, _ = estatisticas[i]
        toca_borda = x == 0 or y == 0 or x + w >= largura or y + h >= altura
        if not toca_borda:
            preenchido[rotulos == i] = 255

    return preenchido


def _angulo_pautas(norm: np.ndarray) -> float:
    """Ângulo dominante das linhas longas quase horizontais (graus)."""
    bordas = cv2.Canny(norm, 50, 150)
    linhas = cv2.HoughLinesP(
        bordas,
        1,
        np.pi / 720,
        80,
        minLineLength=norm.shape[1] // 4,
        maxLineGap=15,
    )
    if linhas is None:
        return 0.0

    angulos = [
        math.degrees(math.atan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in linhas[:, 0]
    ]
    angulos = [a for a in angulos if abs(a) < 25]
    return float(np.median(angulos)) if len(angulos) >= 3 else 0.0


def _girar(imagem: np.ndarray, angulo: float, borda: int) -> np.ndarray:
    if angulo == 0:
        return imagem
    altura, largura = imagem.shape
    matriz = cv2.getRotationMatrix2D((largura / 2, altura / 2), angulo, 1.0)
    return cv2.warpAffine(
        imagem,
        matriz,
        (largura, altura),
        flags=cv2.INTER_LINEAR,
        borderValue=(borda,),
    )


def _posicoes_pautas(pautas: np.ndarray) -> list[float]:
    """Coordenada y de cada pauta (linha impressa) detectada."""
    projecao = (pautas > 0).sum(axis=1)
    ativas = projecao > 0.15 * pautas.shape[1]
    posicoes = []
    inicio = None

    for y, ativa in enumerate(ativas):
        if ativa and inicio is None:
            inicio = y
        elif not ativa and inicio is not None:
            posicoes.append((inicio + y - 1) / 2)
            inicio = None

    if inicio is not None:
        posicoes.append((inicio + len(ativas) - 1) / 2)

    return posicoes


def _espessura_pautas(pautas: np.ndarray, quantidade: int) -> int:
    """Espessura média das pautas, em pixels (ímpar, de 3 a 15)."""
    if quantidade == 0:
        return 3
    pixels = int((pautas > 0).sum())
    largura = max(1, int((pautas > 0).any(axis=0).sum()))
    espessura = int(round(pixels / (quantidade * largura)))
    return int(np.clip(espessura | 1, 3, 15))


def _componentes(mascara: np.ndarray) -> tuple[list[Componente], np.ndarray]:
    _, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(
        mascara, connectivity=8
    )
    componentes = [
        Componente(int(x), int(y), int(w), int(h), int(a), rotulo=indice)
        for indice, (x, y, w, h, a) in enumerate(estatisticas[1:], start=1)
        if a >= 6
    ]
    return componentes, rotulos


def _sem_colunas(componentes: list[Componente]) -> list[Componente]:
    """
    Remove manchas empilhadas numa coluna vertical (furos da espiral,
    marcas da margem). Texto Morse corre na horizontal e nunca forma
    colunas de 6 ou mais manchas alinhadas.
    """
    relevantes = [c for c in componentes if c.area >= 15]
    if len(relevantes) < 6:
        return componentes

    cx = np.array([c.x + c.largura / 2 for c in relevantes])
    cy = np.array([c.centro_y for c in relevantes])
    largura = np.array([c.largura for c in relevantes], dtype=float)
    altura = np.array([c.altura for c in relevantes], dtype=float)

    mesma_coluna = np.abs(cx[:, None] - cx[None, :]) < 0.6 * np.maximum(
        largura[:, None], largura[None, :]
    )
    outra_linha = np.abs(cy[:, None] - cy[None, :]) > 1.5 * np.maximum(
        altura[:, None], altura[None, :]
    )
    vizinhos = (mesma_coluna & outra_linha).sum(axis=1)

    em_coluna = {id(c) for c, n in zip(relevantes, vizinhos) if n >= 5}
    return [c for c in componentes if id(c) not in em_coluna]


def _espessura_tipica(componentes: list[Componente], forma) -> float:
    """Espessura típica do traço da caneta, ignorando poeira e contornos."""
    area_minima = max(15, forma[0] * forma[1] * 1e-5)
    candidatos = [
        min(c.largura, c.altura)
        for c in componentes
        if c.area >= area_minima
        and c.area / (c.largura * c.altura) >= 0.4
        and max(c.largura, c.altura) <= forma[1] / 15
    ]
    return float(np.median(candidatos)) if candidatos else 4.0


def _filtrar(componentes: list[Componente], d: float) -> list[Componente]:
    aceitos = []

    for c in componentes:
        maior = max(c.largura, c.altura)
        menor = min(c.largura, c.altura)
        preenchimento = c.area / (c.largura * c.altura)

        # A barra "/" é testada antes do filtro de tamanho: inclinada, sua
        # caixa é larga, mas o traço é fino e ocupa pouco da caixa.
        if (
            2.5 * d <= c.altura <= 14 * d
            and c.altura >= 1.2 * c.largura
            and preenchimento <= 0.5
            and c.area >= c.altura * 0.5 * d
        ):
            c.barra = True
            aceitos.append(c)
            continue

        if menor > 3.0 * d or maior > 14 * d:
            continue  # lapiseira, espiral, manchas grandes
        if c.area < 0.4 * d * d:
            continue  # poeira do papel
        if preenchimento < 0.25:
            continue  # contornos finos (logotipos, bordas)
        if c.altura >= 2.5 * d and c.altura > 1.6 * c.largura:
            c.barra = True  # barra quase vertical

        aceitos.append(c)

    return aceitos


def _agrupar_linhas(
    componentes: list[Componente],
    pautas: list[float],
    d: float,
) -> list[list[Componente]]:
    """
    Agrupa os símbolos em linhas de texto.

    Com pautas: cada símbolo pertence à pauta logo abaixo dele (o ponto
    costuma ficar sobre a pauta e o traço um pouco acima). Sem pautas:
    agrupa por distância vertical.
    """
    grupos: dict[int, list[Componente]] = {}

    espacamento = (
        float(np.median(np.diff(pautas))) if len(pautas) >= 2 else 0.0
    )

    if espacamento > 3 * d:
        for c in componentes:
            limite = c.centro_y - 0.35 * espacamento
            indice = next(
                (i for i, y in enumerate(pautas) if y >= limite),
                None,
            )
            if indice is None:
                # Abaixo da última pauta detectada: continua a contagem
                # com o mesmo espaçamento, em vez de juntar tudo num grupo.
                indice = len(pautas) + int(
                    (limite - pautas[-1]) // espacamento
                )
            grupos.setdefault(indice, []).append(c)
        linhas = [grupos[i] for i in sorted(grupos)]
    else:
        linhas, atual, ultimo = [], [], None
        for c in sorted(componentes, key=lambda c: c.centro_y):
            if ultimo is not None and c.centro_y - ultimo > 6 * d:
                linhas.append(atual)
                atual = []
            atual.append(c)
            ultimo = c.centro_y
        if atual:
            linhas.append(atual)

    # Se a pauta entre duas linhas de texto não foi detectada, as duas
    # caem no mesmo grupo: separa pelo vão vertical entre os símbolos.
    vao = max(4.5 * d, 0.5 * espacamento)
    linhas = [parte for linha in linhas for parte in _separar_por_altura(linha, vao)]

    linhas = [sorted(linha, key=lambda c: c.x) for linha in linhas]
    limpas = (_limpar_linha(linha, d) for linha in linhas)
    return [linha for linha in limpas if linha]


def _separar_por_altura(
    linha: list[Componente],
    vao: float,
) -> list[list[Componente]]:
    """Divide um grupo onde houver um vão vertical maior que `vao`."""
    ordenada = sorted(linha, key=lambda c: c.centro_y)
    partes = [[ordenada[0]]] if ordenada else []
    for anterior, atual in zip(ordenada, ordenada[1:]):
        if atual.centro_y - anterior.centro_y > vao:
            partes.append([])
        partes[-1].append(atual)
    return partes


def _limpar_linha(
    linha: list[Componente],
    d: float,
) -> list[Componente] | None:
    """
    Mantém só o que se alinha horizontalmente com o texto da linha.

    Remove intrusos (componentes de outra altura que caíram no mesmo
    grupo) e descarta "linhas" que não são texto: furos da espiral
    empilhados, letras de logotipos, poeira espalhada.
    """
    simbolos = [c for c in linha if not c.barra]
    if len(simbolos) < 2:
        return None

    # Texto é quase horizontal: os centros ficam numa faixa estreita em
    # torno de uma reta pouco inclinada (o traço fica um pouco acima do
    # ponto). A mediana torna a reta resistente a intrusos.
    xs = np.array([c.x + c.largura / 2 for c in simbolos])
    ys = np.array([c.centro_y for c in simbolos])
    inclinacao = 0.0
    if len(simbolos) >= 3 and np.ptp(xs) > 0:
        pares = [
            (ys[j] - ys[i]) / (xs[j] - xs[i])
            for i in range(len(xs)) for j in range(i + 1, len(xs))
            if xs[j] != xs[i]
        ]
        if pares:
            inclinacao = float(np.clip(np.median(pares), -0.18, 0.18))
    centro = float(np.median(ys - inclinacao * xs))

    def alinhado(c: Componente) -> bool:
        residuo = c.centro_y - inclinacao * (c.x + c.largura / 2) - centro
        return abs(residuo) <= (8 if c.barra else 7) * d

    limpa = [c for c in linha if alinhado(c)]
    simbolos = [c for c in limpa if not c.barra]

    if len(simbolos) < 2:
        return None

    # Dois componentes soltos e distantes são sujeira, não uma letra.
    if len(simbolos) == 2:
        a, b = simbolos
        if b.x - a.direita > 10 * d:
            return None

    return limpa


def recorte_isolado(
    resultado: ResultadoSegmentacao,
    componente: Componente,
) -> np.ndarray:
    """
    Recorta um símbolo com margem, mantendo só os seus próprios pixels.

    Tudo o que não pertence ao componente (pauta que o atravessa,
    símbolos vizinhos, manchas) é pintado com a cor do papel ao redor.
    """
    imagem = resultado.imagem
    altura_img, largura_img = imagem.shape
    margem = max(2, round(0.15 * max(componente.largura, componente.altura)))

    x1 = max(0, componente.x - margem)
    y1 = max(0, componente.y - margem)
    x2 = min(largura_img, componente.direita + margem)
    y2 = min(altura_img, componente.y + componente.altura + margem)

    recorte = imagem[y1:y2, x1:x2].copy()

    if resultado.rotulos is None or componente.rotulo == 0:
        return recorte

    proprio = (resultado.rotulos[y1:y2, x1:x2] == componente.rotulo)
    # Inclui a borda suave (antisserrilhado) do traço da caneta.
    proprio = cv2.dilate(proprio.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    fora = recorte[~proprio]
    if fora.size:
        recorte[~proprio] = np.uint8(np.percentile(fora, 90))

    return recorte


def alongamento(
    resultado: ResultadoSegmentacao,
    componente: Componente,
) -> float:
    """
    Comprimento ÷ espessura do símbolo, medido no retângulo mínimo que
    o envolve (funciona também com traços inclinados).

    Ponto fica perto de 1; traço, bem acima de 2.
    """
    if resultado.rotulos is None or componente.rotulo == 0:
        lados = (componente.largura, componente.altura)
    else:
        janela = resultado.rotulos[
            componente.y:componente.y + componente.altura,
            componente.x:componente.direita,
        ]
        ys, xs = np.nonzero(janela == componente.rotulo)
        pontos = np.column_stack([xs, ys]).astype(np.float32)
        # +1: um pixel tem tamanho 1, não 0.
        lados = tuple(v + 1 for v in cv2.minAreaRect(pontos)[1])

    return max(lados) / max(1.0, min(lados))


def segmentar(cinza: np.ndarray) -> ResultadoSegmentacao:
    """
    Localiza e agrupa candidatos a símbolos em uma foto.

    Args:
        cinza: imagem em tons de cinza (uint8), em qualquer resolução.
    """
    # Toda foto vai para a mesma escala de trabalho: reduzida se for
    # grande, ampliada se for um recorte pequeno. Os filtros e operações
    # morfológicas abaixo foram calibrados nessa escala.
    escala = LADO_TRABALHO / max(cinza.shape)
    if escala != 1:
        cinza = cv2.resize(
            cinza,
            None,
            fx=escala,
            fy=escala,
            interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC,
        )

    norm = _normalizar_iluminacao(cinza)
    papel = _mascara_papel(cinza)

    angulo = _angulo_pautas(cv2.bitwise_or(norm, cv2.bitwise_not(papel)))
    norm = _girar(norm, angulo, 255)
    papel = _girar(papel, angulo, 0)
    imagem = _girar(cinza, angulo, 255)

    tinta = (norm < LIMIAR_TINTA).astype(np.uint8) * 255

    # Com um limiar tolerante a pauta impressa fica contínua, e uma
    # abertura horizontal longa isola só as linhas do caderno.
    comprimento = max(40, norm.shape[1] // 12)
    pautas = cv2.morphologyEx(
        (norm < LIMIAR_PAUTA).astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (comprimento, 1)),
    )
    pautas = cv2.bitwise_and(pautas, papel)
    posicoes = _posicoes_pautas(pautas)

    # A faixa removida acompanha a espessura real da pauta nesta foto.
    faixa = _espessura_pautas(pautas, len(posicoes)) + 2
    tinta = cv2.subtract(
        tinta, cv2.dilate(pautas, np.ones((faixa, 1), np.uint8))
    )
    # Religa pontos, traços e barras "/" cortados pela faixa removida da
    # pauta (a barra cruza a pauta na diagonal, por isso 3 px de largura).
    tinta = cv2.morphologyEx(
        tinta, cv2.MORPH_CLOSE, np.ones((faixa + 4, 3), np.uint8)
    )
    tinta = cv2.bitwise_and(tinta, papel)
    tinta = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    componentes, rotulos = _componentes(tinta)
    if not componentes:
        return ResultadoSegmentacao(imagem, [], None, 0.0, angulo, escala)

    componentes = _sem_colunas(componentes)
    d = _espessura_tipica(componentes, tinta.shape)

    # Componentes cortados pela borda da foto não são símbolos inteiros.
    altura, largura = tinta.shape
    componentes = [
        c for c in componentes
        if c.x > 1 and c.y > 1
        and c.direita < largura - 1 and c.y + c.altura < altura - 1
    ]
    candidatos = _filtrar(componentes, d)
    linhas = _agrupar_linhas(candidatos, posicoes, d)

    return ResultadoSegmentacao(imagem, linhas, rotulos, d, angulo, escala)
