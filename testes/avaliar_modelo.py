import json
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

CAMINHO_PESOS = os.path.join(RAIZ, "modelos", "modelo_morse.pth")
CAMINHO_CONFIG = os.path.join(RAIZ, "modelos", "modelo_morse.json")
PASTA_TESTE = os.path.join(RAIZ, "datasets_teste")
SAIDA = os.path.join(RAIZ, "docs", "resultados")

os.makedirs(SAIDA, exist_ok=True)

with open(CAMINHO_CONFIG) as f:
    config = json.load(f)

classes_cfg = config.get("classes", ["fundo", "ponto", "traco"])
if isinstance(classes_cfg, dict):
    itens = list(classes_cfg.items())
    if all(str(k).isdigit() for k, _ in itens):
        itens.sort(key=lambda par: int(par[0]))
        classes = [str(v) for _, v in itens]
    else:
        itens.sort(key=lambda par: int(par[1]))
        classes = [str(k) for k, _ in itens]
else:
    classes = list(classes_cfg)

tamanho = config.get("img_size", 32)
if isinstance(tamanho, int):
    tamanho = (tamanho, tamanho)
tons_de_cinza = True

etapas = []
if tons_de_cinza:
    etapas.append(transforms.Grayscale(num_output_channels=1))
etapas += [transforms.Resize(tamanho), transforms.ToTensor()]
if "mean" in config and "std" in config:
    etapas.append(transforms.Normalize(config["mean"], config["std"]))

dataset = datasets.ImageFolder(PASTA_TESTE, transform=transforms.Compose(etapas))
mapa = {i: classes.index(nome) for i, nome in enumerate(dataset.classes)}
loader = DataLoader(dataset, batch_size=32, shuffle=False)

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
objeto = torch.load(CAMINHO_PESOS, map_location=dispositivo, weights_only=False)

if isinstance(objeto, dict):
    from ia.modelo import RedeNeuralMorse
    modelo = RedeNeuralMorse(numero_classes=len(classes))
    modelo.load_state_dict(objeto)
else:
    modelo = objeto

modelo.to(dispositivo)
modelo.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imagens, rotulos in loader:
        saida = modelo(imagens.to(dispositivo))
        y_pred.extend(saida.argmax(dim=1).cpu().tolist())
        y_true.extend(mapa[r] for r in rotulos.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

relatorio = classification_report(y_true, y_pred, target_names=classes, digits=3)
print(relatorio)
with open(os.path.join(SAIDA, "metricas.txt"), "w") as f:
    f.write(relatorio)

for normalizar, nome in [(None, "contagem"), ("true", "normalizada")]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes,
        cmap="Blues", normalize=normalizar, ax=ax)
    ax.set_title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA, f"matriz_{nome}.png"), dpi=300)
    plt.close()
