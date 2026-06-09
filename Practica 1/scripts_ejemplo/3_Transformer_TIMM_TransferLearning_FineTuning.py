import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import cv2
import torch
import timm

import matplotlib.pyplot as plt

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from timm.data import resolve_model_data_config, create_transform
from ModeloTransformerTimmFactory import ModeloTransformerTimmFactory


# ============================================================
# DATASET
# ============================================================
class DatasetImagenes(Dataset):
    def __init__(self, rutas, etiquetas, transform=None):
        self.rutas = rutas
        self.etiquetas = etiquetas
        self.transform = transform

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        ruta = self.rutas[idx]
        etiqueta = self.etiquetas[idx]

        imagen = cv2.imread(ruta)

        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta}")

        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(imagen)

        if self.transform is not None:
            imagen = self.transform(imagen)

        return imagen, etiqueta


# ============================================================
# CARGAR RUTAS
# ============================================================
def cargarRutas(rutaOrigen, numeroCategorias, limite):
    rutas = []
    etiquetas = []

    for categoria in range(numeroCategorias):
        for idImagen in range(1, limite[categoria]):
            ruta = f"{rutaOrigen}{categoria}/{categoria}_{idImagen}.jpg"

            if os.path.exists(ruta):
                rutas.append(ruta)
                etiquetas.append(categoria)
            else:
                print(f"No existe: {ruta}")

    return rutas, np.array(etiquetas, dtype=np.int64)


# ============================================================
# ENTRENAR UNA EPOCA
# ============================================================
def entrenar_una_epoca(model, dataloader, criterio, optimizador, device):
    model.train()

    perdida_total = 0.0
    correctos = 0
    total = 0

    for imagenes, etiquetas in dataloader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        optimizador.zero_grad()

        salidas = model(imagenes)
        perdida = criterio(salidas, etiquetas)

        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)

        predicciones = torch.argmax(salidas, dim=1)
        correctos += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

    loss = perdida_total / total
    accuracy = correctos / total

    return loss, accuracy


# ============================================================
# VALIDAR / EVALUAR
# ============================================================
def evaluar(model, dataloader, criterio, device):
    model.eval()

    perdida_total = 0.0
    correctos = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            salidas = model(imagenes)
            perdida = criterio(salidas, etiquetas)

            perdida_total += perdida.item() * imagenes.size(0)

            predicciones = torch.argmax(salidas, dim=1)

            correctos += (predicciones == etiquetas).sum().item()
            total += etiquetas.size(0)

            y_true.extend(etiquetas.cpu().numpy())
            y_pred.extend(predicciones.cpu().numpy())

    loss = perdida_total / total
    accuracy = correctos / total

    return loss, accuracy, np.array(y_true), np.array(y_pred)


# ============================================================
# EARLY STOPPING MANUAL
# ============================================================
class EarlyStoppingTorch:
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.mejor_loss = float("inf")
        self.contador = 0
        self.mejor_estado = None

    def step(self, val_loss, model):
        if val_loss < self.mejor_loss - self.min_delta:
            self.mejor_loss = val_loss
            self.contador = 0
            self.mejor_estado = copy.deepcopy(model.state_dict())
            return False

        self.contador += 1
        return self.contador >= self.patience

    def restaurar_mejores_pesos(self, model):
        if self.mejor_estado is not None:
            model.load_state_dict(self.mejor_estado)


# ============================================================
# CONFIGURACIÓN
# ============================================================
numeroCategorias = 10

porcentajeValidacion = 0.2

cantidaDatosEntrenamiento = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
cantidaDatosPruebas = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

# Opciones:
# "vit_tiny", "vit_small", "vit_base",
# "deit_tiny", "deit_small",
# "swin_tiny", "swin_small",
# "coat_tiny", "coat_lite_tiny",
# "convnext_tiny"
nombreModelo = "deit_tiny"

# "imagenet" = usar pesos preentrenados
# "none" = entrenar desde cero
pesos = "imagenet"

# True = transfer learning
# False = fine tuning
congelar_base = True

epochs = 2
batch_size = 4
learning_rate = 0.0001
patience = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# CREAR MODELO CON FACTORY
# ============================================================
model, nombre_timm = ModeloTransformerTimmFactory.crear(
    nombreModelo=nombreModelo,
    numeroCategorias=numeroCategorias,
    pesos=pesos,
    congelar_base=congelar_base
)

model = model.to(device)

parametros = ModeloTransformerTimmFactory.contar_parametros(model)

print("\nModelo creado:")
print("Alias:", nombreModelo)
print("TIMM:", nombre_timm)
print("Pesos:", pesos)
print("Base congelada:", congelar_base)
print("Parámetros totales:", parametros["total"])
print("Parámetros entrenables:", parametros["entrenables"])
print("Parámetros congelados:", parametros["congelados"])


# ============================================================
# TRANSFORMS OFICIALES DE TIMM
# ============================================================
data_config = resolve_model_data_config(model)

transform_train = create_transform(
    **data_config,
    is_training=True
)

transform_eval = create_transform(
    **data_config,
    is_training=False
)

print("\nConfig de datos TIMM:")
print(data_config)


# ============================================================
# CARGAR RUTAS TRAIN
# ============================================================
rutas, etiquetas = cargarRutas(
    "dataset/train/",
    numeroCategorias,
    cantidaDatosEntrenamiento
)

rutas_train, rutas_val, y_train, y_val = train_test_split(
    rutas,
    etiquetas,
    test_size=porcentajeValidacion,
    random_state=42,
    shuffle=True,
    stratify=etiquetas
)

print("Datos entrenamiento:", len(rutas_train))
print("Datos validación:", len(rutas_val))


# ============================================================
# DATALOADERS
# ============================================================
dataset_train = DatasetImagenes(
    rutas=rutas_train,
    etiquetas=y_train,
    transform=transform_train
)

dataset_val = DatasetImagenes(
    rutas=rutas_val,
    etiquetas=y_val,
    transform=transform_eval
)

loader_train = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

loader_val = DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# ============================================================
# COMPILAR EN PYTORCH: LOSS + OPTIMIZADOR
# ============================================================
criterio = nn.CrossEntropyLoss()

parametros_entrenables = [p for p in model.parameters() if p.requires_grad]

optimizador = AdamW(
    parametros_entrenables,
    lr=learning_rate
)

early_stop = EarlyStoppingTorch(
    patience=patience,
    min_delta=0.0001
)


# ============================================================
# ENTRENAR
# ============================================================
historial = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoca in range(1, epochs + 1):
    train_loss, train_acc = entrenar_una_epoca(
        model,
        loader_train,
        criterio,
        optimizador,
        device
    )

    val_loss, val_acc, _, _ = evaluar(
        model,
        loader_val,
        criterio,
        device
    )

    historial["loss"].append(train_loss)
    historial["accuracy"].append(train_acc)
    historial["val_loss"].append(val_loss)
    historial["val_accuracy"].append(val_acc)

    print(
        f"Época {epoca}/{epochs} "
        f"- loss: {train_loss:.4f} "
        f"- acc: {train_acc:.4f} "
        f"- val_loss: {val_loss:.4f} "
        f"- val_acc: {val_acc:.4f}"
    )

    if early_stop.step(val_loss, model):
        print("Early stopping activado.")
        break

early_stop.restaurar_mejores_pesos(model)


# ============================================================
# CURVA DE PÉRDIDA
# ============================================================
plt.figure()
plt.plot(historial["loss"], label="Pérdida entrenamiento")
plt.plot(historial["val_loss"], label="Pérdida validación")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title(f"Curva de pérdida - {nombreModelo}")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# CURVA DE ACCURACY
# ============================================================
plt.figure()
plt.plot(historial["accuracy"], label="Accuracy entrenamiento")
plt.plot(historial["val_accuracy"], label="Accuracy validación")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title(f"Curva de accuracy - {nombreModelo}")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# CARGAR DATOS DE PRUEBA
# ============================================================
rutas_test, y_test = cargarRutas(
    "dataset/test/",
    numeroCategorias,
    cantidaDatosPruebas
)

dataset_test = DatasetImagenes(
    rutas=rutas_test,
    etiquetas=y_test,
    transform=transform_eval
)

loader_test = DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# ============================================================
# EVALUAR MODELO
# ============================================================
test_loss, test_acc, y_true, y_pred = evaluar(
    model,
    loader_test,
    criterio,
    device
)

print("Loss test =", test_loss)
print("Accuracy test =", test_acc)


# ============================================================
# MÉTRICAS SKLEARN Y MATRIZ DE CONFUSIÓN
# ============================================================
print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, digits=4))

matriz = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=list(range(numeroCategorias))
)

disp.plot(cmap="Blues")
plt.title(f"Matriz de confusión - {nombreModelo}")
plt.show()


# ============================================================
# GUARDAR MODELO
# ============================================================
modo = "scratch" if pesos == "none" else "imagenet"
estado_base = "frozen" if congelar_base else "finetuning"

os.makedirs("models", exist_ok=True)

ruta_pesos = f"models/modelo_{nombreModelo}_{modo}_{estado_base}.pth"
ruta_checkpoint = f"models/checkpoint_{nombreModelo}_{modo}_{estado_base}.pth"

torch.save(model.state_dict(), ruta_pesos)

torch.save(
    {
        "alias_modelo": nombreModelo,
        "nombre_timm": nombre_timm,
        "numeroCategorias": numeroCategorias,
        "pesos": pesos,
        "congelar_base": congelar_base,
        "state_dict": model.state_dict(),
        "data_config": data_config,
        "historial": historial
    },
    ruta_checkpoint
)

print(f"Pesos guardados en: {ruta_pesos}")
print(f"Checkpoint guardado en: {ruta_checkpoint}")
