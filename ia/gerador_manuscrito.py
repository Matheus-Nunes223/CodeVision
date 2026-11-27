from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from ia.decodificacao import MORSE_PARA_TEXTO


TEXTO_PARA_MORSE = {
    letra: codigo
    for codigo, letra in MORSE_PARA_TEXTO.items()
    if letra.isalnum()
}

PALAVRAS = (
    "OLA MUNDO BOM DIA BOA NOITE TARDE PAZ AMOR SOS CASA REDES IA CODIGO "
    "MORSE VISAO AULA ESTUDAR VAMOS ESTOU CHEGANDO OBRIGADO SIM NAO QUE "
    "HORAS SAO EU GOSTO DE VOCE AJUDA AGORA HOJE AMANHA ESCOLA PROJETO "
    "TESTE IMAGEM SINAL LUZ FIM INICIO CERTO ERRO PERIGO SAIDA NORTE SUL "
    "LESTE OESTE AGUA FOGO TERRA AR CEU MAR SOL LUA RIO PONTE TREM BARCO"
).split()


@dataclass
class SimboloGabarito:
    tipo: str
    caixa: tuple[int, int, int, int]
    linha: int


@dataclass
class PaginaSintetica:
    imagem: np.ndarray
    linhas: list[str] = field(default_factory=list)
    simbolos: list[SimboloGabarito] = field(default_factory=list)


@dataclass
class _Estilo:
    espessura: float
    raio_ponto: float
    comprimento_traco: float
    espaco_simbolo: float
    espaco_letra: float
    espaco_palavra: float
    elevacao_traco: float
    tinta: float
    usa_barra: bool
    chance_anel: float


def _mensagem(gerador: random.Random) -> str:
    if gerador.random() < 0.15:
        simbolos = list(TEXTO_PARA_MORSE)
        return "".join(gerador.choice(simbolos) for _ in range(gerador.randint(2, 6)))
    return " ".join(gerador.choice(PALAVRAS) for _ in range(gerador.randint(1, 3)))


def _estilo(gerador: random.Random, espacamento: float) -> _Estilo:
    espessura = gerador.uniform(0.045, 0.11) * espacamento
    raio = espessura * gerador.uniform(0.7, 1.5)
    espaco_simbolo = espessura * gerador.uniform(1.2, 3.5)
    return _Estilo(
        espessura=espessura,
        raio_ponto=raio,
        comprimento_traco=espessura * gerador.uniform(3.5, 8),
        espaco_simbolo=espaco_simbolo,
        espaco_letra=espaco_simbolo * gerador.uniform(2.2, 4),
        espaco_palavra=espaco_simbolo * gerador.uniform(5, 9),
        elevacao_traco=gerador.uniform(0.0, 0.3),
        tinta=gerador.uniform(0.0, 0.35),
        usa_barra=gerador.random() < 0.6,
        chance_anel=gerador.choice((0.0, 0.0, 0.15)),
    )


def _desenhar_ponto(mascara, cx, cy, estilo, gerador):
    raio = estilo.raio_ponto * gerador.uniform(0.8, 1.2)
    eixos = (raio * gerador.uniform(0.8, 1.25), raio * gerador.uniform(0.8, 1.25))
    angulo = gerador.uniform(0, 180)
    centro = (int(cx * 4), int(cy * 4))
    eixos_px = (max(1, int(eixos[0] * 4)), max(1, int(eixos[1] * 4)))

    if gerador.random() < estilo.chance_anel:
        cv2.ellipse(mascara, centro, eixos_px, angulo, 0, 360, 1.0,
                    max(1, int(raio * 1.6)), cv2.LINE_AA)
    else:
        cv2.ellipse(mascara, centro, eixos_px, angulo, 0, 360, 1.0,
                    -1, cv2.LINE_AA)

    return (cx - eixos[0], cy - eixos[1], 2 * eixos[0], 2 * eixos[1])


def _desenhar_traco(mascara, x0, cy, estilo, gerador):
    comprimento = estilo.comprimento_traco * gerador.uniform(0.8, 1.25)
    inclinacao = math.radians(gerador.uniform(-8, 8))
    onda = gerador.uniform(0, 0.25) * estilo.espessura
    passos = 12
    pontos = []
    for i in range(passos + 1):
        t = i / passos
        x = x0 + t * comprimento * math.cos(inclinacao)
        y = cy + t * comprimento * math.sin(inclinacao) + onda * math.sin(t * math.pi * 2)
        pontos.append((x, y))

    for (xa, ya), (xb, yb) in zip(pontos, pontos[1:]):
        espessura = estilo.espessura * gerador.uniform(0.85, 1.15)
        cv2.line(mascara, (int(xa * 4), int(ya * 4)), (int(xb * 4), int(yb * 4)),
                 1.0, max(1, int(espessura * 4)), cv2.LINE_AA)

    xs = [p[0] for p in pontos]
    ys = [p[1] for p in pontos]
    meia = estilo.espessura / 2
    return (min(xs) - meia, min(ys) - meia,
            max(xs) - min(xs) + 2 * meia, max(ys) - min(ys) + 2 * meia)


def _desenhar_barra(mascara, x0, base, espacamento, estilo, gerador):
    altura = espacamento * gerador.uniform(0.55, 0.95)
    largura = altura * gerador.uniform(0.2, 0.5)
    topo = base - altura * gerador.uniform(0.75, 1.0)
    p1 = (x0, topo + altura)
    p2 = (x0 + largura, topo)
    cv2.line(mascara, (int(p1[0] * 4), int(p1[1] * 4)), (int(p2[0] * 4), int(p2[1] * 4)),
             1.0, max(1, int(estilo.espessura * 0.8 * 4)), cv2.LINE_AA)
    return (x0, topo, largura, altura)


def _largura_palavra(palavra: str, estilo: _Estilo) -> float:
    total = 0.0
    for letra in palavra:
        codigo = TEXTO_PARA_MORSE[letra]
        total += sum(
            estilo.comprimento_traco * 1.3 if s == "-" else estilo.raio_ponto * 2.6
            for s in codigo
        )
        total += estilo.espaco_simbolo * 1.3 * (len(codigo) - 1)
        total += estilo.espaco_letra * 1.3
    return total


def _escrever_linha(mascara, texto, x_inicio, x_limite, base, espacamento,
                    estilo, gerador):
    simbolos = []
    escritas = []
    x = x_inicio

    for palavra in texto.split():
        separador = estilo.espaco_letra * 1.5 + espacamento * 0.5 if escritas else 0
        if x + separador + _largura_palavra(palavra, estilo) > x_limite:
            break

        if escritas:
            if estilo.usa_barra:
                x += estilo.espaco_letra * 0.6
                caixa = _desenhar_barra(mascara, x, base, espacamento, estilo, gerador)
                simbolos.append(("/", caixa))
                x += caixa[2] + estilo.espaco_letra * 0.6
            else:
                x += estilo.espaco_palavra

        for letra in palavra:
            for posicao, simbolo in enumerate(TEXTO_PARA_MORSE[letra]):
                if posicao > 0:
                    x += estilo.espaco_simbolo * gerador.uniform(0.7, 1.3)
                if simbolo == ".":
                    raio = estilo.raio_ponto
                    cy = base - raio * gerador.uniform(0.3, 1.3)
                    caixa = _desenhar_ponto(mascara, x + raio, cy, estilo, gerador)
                else:
                    cy = base - espacamento * estilo.elevacao_traco * gerador.uniform(0.7, 1.3)
                    cy -= estilo.espessura
                    caixa = _desenhar_traco(mascara, x, cy, estilo, gerador)
                simbolos.append((simbolo, caixa))
                x = caixa[0] + caixa[2]

            x += estilo.espaco_letra * gerador.uniform(0.75, 1.3)

        escritas.append(palavra)

    return simbolos, " ".join(escritas)


def _texto_fantasma(pagina, gerador, area):
    x0, y0, x1, y1 = area
    camada = np.zeros_like(pagina)
    for _ in range(gerador.randint(10, 40)):
        x = gerador.uniform(x0, x1)
        y = gerador.uniform(y0, y1)
        pontos = []
        for _ in range(gerador.randint(3, 8)):
            x += gerador.uniform(-15, 25)
            y += gerador.uniform(-10, 10)
            pontos.append([int(x), int(y)])
        cv2.polylines(camada, [np.array(pontos)], False, 1.0, gerador.randint(2, 5), cv2.LINE_AA)
    camada = cv2.GaussianBlur(camada, (0, 0), 2.5)
    return pagina * (1 - camada * gerador.uniform(0.06, 0.15))


def gerar_pagina(gerador: random.Random) -> PaginaSintetica:
    largura, altura = gerador.choice(((1500, 2000), (2000, 1500), (1700, 1700), (1800, 1200)))

    mesa = gerador.uniform(0.05, 0.4)
    pagina = np.full((altura, largura), mesa, np.float32)
    margem_x = int(largura * gerador.choice((0, 0, 0.04, 0.1)))
    margem_y = int(altura * gerador.choice((0, 0, 0.05, 0.12)))
    papel = gerador.uniform(0.8, 1.0)
    pagina[margem_y:altura - margem_y or altura, margem_x:largura - margem_x or largura] = papel
    area = (margem_x, margem_y, largura - margem_x, altura - margem_y)

    espacamento = gerador.uniform(55, 110)
    pautado = gerador.random() < 0.8
    primeira = area[1] + espacamento * gerador.uniform(0.8, 2.5)
    bases = list(np.arange(primeira, area[3] - espacamento * 0.5, espacamento))
    if pautado:
        cor_pauta = papel - gerador.uniform(0.15, 0.4)
        espessura_pauta = gerador.randint(1, 3)
        x_pauta = area[0] + int(gerador.uniform(0, 0.08) * largura)
        for y in bases:
            cv2.line(pagina, (x_pauta, int(y)), (area[2], int(y)), cor_pauta,
                     espessura_pauta, cv2.LINE_AA)

    if gerador.random() < 0.35:
        pagina = _texto_fantasma(pagina, gerador, area)

    if pautado and gerador.random() < 0.3:
        for y in np.arange(area[1] + 20, area[3], gerador.uniform(35, 60)):
            cv2.ellipse(pagina, (area[0] + 18, int(y)), (14, 9), 0, 0, 360,
                        gerador.uniform(0.05, 0.3), -1, cv2.LINE_AA)

    mascara = np.zeros((altura * 4, largura * 4), np.float32)
    estilo = _estilo(gerador, espacamento)
    x_inicio = area[0] + largura * gerador.uniform(0.05, 0.2)
    x_limite = area[2] - largura * gerador.uniform(0.03, 0.15)

    quantidade = gerador.randint(1, min(4, max(1, len(bases) // 2)))
    escolhidas = sorted(gerador.sample(range(len(bases)), quantidade)) if bases else []
    linhas_texto, desenhados = [], []
    for numero, indice in enumerate(escolhidas):
        simbolos, texto = _escrever_linha(
            mascara, _mensagem(gerador), x_inicio, x_limite, bases[indice],
            espacamento, estilo, gerador,
        )
        if texto:
            linhas_texto.append(texto)
            desenhados.extend((tipo, caixa, len(linhas_texto) - 1) for tipo, caixa in simbolos)

    tinta = cv2.resize(mascara, (largura, altura), interpolation=cv2.INTER_AREA)
    pagina = pagina * (1 - tinta) + estilo.tinta * tinta

    if gerador.random() < 0.2:
        livres = [y for i, y in enumerate(bases) if i not in escolhidas and i - 1 not in escolhidas]
        if livres:
            y = int(gerador.choice(livres)) - int(espacamento * 0.6)
            cv2.rectangle(pagina, (int(largura * 0.25), y), (largura, y + int(espacamento * 0.7)),
                          gerador.uniform(0.02, 0.12), -1)

    yy, xx = np.mgrid[0:altura, 0:largura].astype(np.float32)
    angulo = gerador.uniform(0, 2 * math.pi)
    gradiente = (xx * math.cos(angulo) + yy * math.sin(angulo)) / max(largura, altura)
    gradiente = gradiente - gradiente.min()
    luz = 1 - gerador.uniform(0, 0.4) * gradiente / max(gradiente.max(), 1e-6)
    if gerador.random() < 0.35:
        sombra = np.zeros_like(pagina)
        cv2.ellipse(sombra, (gerador.randint(0, largura), gerador.randint(0, altura)),
                    (gerador.randint(200, 900), gerador.randint(150, 600)),
                    gerador.uniform(0, 180), 0, 360, 1.0, -1)
        sombra = cv2.GaussianBlur(sombra, (0, 0), 60)
        luz *= 1 - gerador.uniform(0.2, 0.5) * sombra
    pagina *= luz

    lado_final = gerador.choice((700, 900, 1000, 1280, 1600, 2000))
    escala = lado_final / max(largura, altura)
    origem = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])
    giro = math.radians(gerador.uniform(-7, 7))
    cx, cy = largura / 2, altura / 2
    destino = []
    for x, y in origem:
        dx, dy = x - cx, y - cy
        rx = cx + dx * math.cos(giro) - dy * math.sin(giro)
        ry = cy + dx * math.sin(giro) + dy * math.cos(giro)
        rx += gerador.uniform(-0.04, 0.04) * largura
        ry += gerador.uniform(-0.04, 0.04) * altura
        destino.append([rx * escala, ry * escala])
    destino = np.float32(destino)
    homografia = cv2.getPerspectiveTransform(origem, destino)
    tamanho = (int(largura * escala), int(altura * escala))
    foto = cv2.warpPerspective(pagina, homografia, tamanho,
                               flags=cv2.INTER_AREA, borderValue=float(mesa))

    if gerador.random() < 0.7:
        foto = cv2.GaussianBlur(foto, (0, 0), gerador.uniform(0.3, 1.3))
    foto = foto * 255 + np.random.default_rng(gerador.randint(0, 2**31)).normal(
        0, gerador.uniform(1, 6), foto.shape)
    foto = np.clip(foto, 0, 255).astype(np.uint8)
    qualidade = gerador.randint(35, 90)
    _, codificada = cv2.imencode(".jpg", foto, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    foto = cv2.imdecode(codificada, cv2.IMREAD_GRAYSCALE)

    simbolos = []
    for tipo, (x, y, w, h), linha in desenhados:
        cantos = np.float32([[[x, y]], [[x + w, y]], [[x, y + h]], [[x + w, y + h]]])
        transformados = cv2.perspectiveTransform(cantos, homografia)[:, 0]
        x0, y0 = transformados.min(axis=0)
        x1, y1 = transformados.max(axis=0)
        simbolos.append(SimboloGabarito(
            tipo, (int(x0), int(y0), int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))), linha))

    return PaginaSintetica(foto, linhas_texto, simbolos)