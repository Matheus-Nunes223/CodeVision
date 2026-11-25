import json
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from ia.modelo import RedeNeuralMorse
from ia.preprocessamento import PreprocessadorCodeVision

CAMINHO_PESOS = os.path.join(RAIZ, "modelos", "modelo_morse.pth")
CAMINHO_CONFIG = os.path.join(RAIZ, "modelos", "modelo_morse.json")
PASTA_TESTE = os.path.join(RAIZ, "datasets_teste")
SAIDA = os.path.join(RAIZ, "docs", "resultados")

os.makedirs(SAIDA, exist_ok=True)

with open(CAMINHO_CONFIG) as f:
    config = json.load(f)

itens = sorted(config["classes"].items(), key=lambda par: int(par[0]))
classes = [str(v) for _, v in itens]

dataset = datasets.ImageFolder(PASTA_TESTE)
mapa = {i: classes.index(nome) for i, nome in enumerate(dataset.classes)}

preprocessador = PreprocessadorCodeVision()

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
pesos = torch.load(CAMINHO_PESOS, map_location=dispositivo, weights_only=False)
modelo = RedeNeuralMorse(numero_classes=len(classes))
modelo.load_state_dict(pesos)
modelo.to(dispositivo)
modelo.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for caminho, rotulo in dataset.samples:
        imagem = Image.open(caminho).convert("L")
        tensor = preprocessador.preparar_simbolo(imagem).unsqueeze(0).to(dispositivo)
        saida = modelo(tensor)
        y_pred.append(int(saida.argmax(dim=1).item()))
        y_true.append(mapa[rotulo])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

relatorio = classification_report(y_true, y_pred, target_names=classes, digits=3)
print(relatorio)
with open(os.path.join(SAIDA, "metricas_pipeline.txt"), "w") as f:
    f.write(relatorio)

for normalizar, nome in [(None, "contagem"), ("true", "normalizada")]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes,
        cmap="Blues", normalize=normalizar, ax=ax)
    ax.set_title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA, f"matriz_pipeline_{nome}.png"), dpi=300)
    plt.close()
