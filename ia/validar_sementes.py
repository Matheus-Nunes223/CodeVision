import argparse
import statistics
import time
from pathlib import Path

from ia.modelo import ModeloCodeVision
from ia.treinamento import TreinadorCodeVision


def executar_semente(
    treinador: TreinadorCodeVision,
    imagens,
    rotulos,
    origens,
    semente: int,
    epocas: int,
    tamanho_lote: int,
    taxa_aprendizado: float,
    peso_decay: float,
    paciencia: int,
) -> dict:
    treinador.semente = semente

    (
        imagens_treino,
        rotulos_treino,
        imagens_validacao,
        rotulos_validacao,
    ) = treinador._dividir_dados(imagens, rotulos, origens)

    modelo = ModeloCodeVision()

    historico = modelo.treinar(
        imagens_treino,
        rotulos_treino,
        dados_validacao=(imagens_validacao, rotulos_validacao),
        epocas=epocas,
        tamanho_lote=tamanho_lote,
        taxa_aprendizado=taxa_aprendizado,
        peso_decay=peso_decay,
        semente=semente,
        paciencia=paciencia,
        verbose=False,
    )

    resultado = modelo.avaliar(imagens_validacao, rotulos_validacao)

    amostras = resultado["amostras"]
    erros = round((1 - resultado["acuracia"]) * amostras)

    return {
        "semente": semente,
        "acuracia": resultado["acuracia"],
        "perda": resultado["perda"],
        "erros": erros,
        "amostras": amostras,
        "epocas": len(historico),
        "recall_minimo": min(
            m["recall"]
            for m in resultado["metricas_por_classe"].values()
        ),
    }


def resumir(resultados: list[dict]) -> None:
    print("\nResultados por semente:")
    print(
        f"  {'semente':>7} | {'acurácia':>8} | {'erros':>5} | "
        f"{'perda':>10} | {'recall mín.':>11} | {'épocas':>6}"
    )

    for r in resultados:
        print(
            f"  {r['semente']:>7} | {r['acuracia']:>8.4f} | "
            f"{r['erros']:>5} | {r['perda']:>10.6f} | "
            f"{r['recall_minimo']:>11.4f} | {r['epocas']:>6}"
        )

    acuracias = [r["acuracia"] for r in resultados]
    perdas = [r["perda"] for r in resultados]
    pior = min(resultados, key=lambda r: r["acuracia"])

    desvio = (
        statistics.stdev(acuracias) if len(acuracias) > 1 else 0.0
    )

    print("\nResumo:")
    print(f"  Acurácia média : {statistics.mean(acuracias):.4f}")
    print(f"  Desvio padrão  : {desvio:.4f}")
    print(
        f"  Pior caso      : {pior['acuracia']:.4f} "
        f"(semente {pior['semente']}, {pior['erros']} erros)"
    )
    print(f"  Perda média    : {statistics.mean(perdas):.6f}")
    print(
        f"  Erros totais   : {sum(r['erros'] for r in resultados)} "
        f"em {sum(r['amostras'] for r in resultados)} amostras"
    )

    if pior["acuracia"] >= 0.995:
        print("\nConclusão: estável. A nota não depende da divisão.")
    else:
        print(
            "\nConclusão: instável. Há sementes com queda relevante; "
            "vale investigar os erros desse caso."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--sementes", type=int, nargs="+", default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--taxa", type=float, default=0.001)
    parser.add_argument("--peso-decay", type=float, default=0.0001)
    parser.add_argument("--paciencia", type=int, default=6)
    argumentos = parser.parse_args()

    treinador = TreinadorCodeVision(
        diretorio_dados=argumentos.dados,
        proporcao_validacao=0.20,
    )

    imagens_por_classe = treinador._listar_imagens()
    imagens, rotulos, origens = treinador._preparar_dados(
        imagens_por_classe
    )

    resultados = []
    inicio = time.time()

    for posicao, semente in enumerate(argumentos.sementes, start=1):
        print(
            f"\n[{posicao}/{len(argumentos.sementes)}] "
            f"Treinando com semente {semente}..."
        )

        resultado = executar_semente(
            treinador,
            imagens,
            rotulos,
            origens,
            semente,
            argumentos.epocas,
            argumentos.lote,
            argumentos.taxa,
            argumentos.peso_decay,
            argumentos.paciencia,
        )

        print(
            f"  acurácia {resultado['acuracia']:.4f} | "
            f"erros {resultado['erros']} | "
            f"épocas {resultado['epocas']}"
        )

        resultados.append(resultado)

    resumir(resultados)
    print(f"\nTempo total: {time.time() - inicio:.0f}s")


if __name__ == "__main__":
    main()