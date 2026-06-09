import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from ModeloCNNPreentrenadoFactory import ModeloPreentrenadoFactory

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# CARGAR DATOS
# ============================================================
CATEGORIAS = ["dosmil", "diesmil"]

def cargarDatos(directorio, numeroCategorias, limites, ancho, alto, numeroCanales=3):
    imagenesCargadas = []
    valorEsperado    = []
    for idx in range(numeroCategorias):
        carpeta  = os.path.join(directorio, CATEGORIAS[idx])
        archivos = sorted([f for f in os.listdir(carpeta)
                           if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if limites[idx] is not None:
            archivos = archivos[:limites[idx]]
        for archivo in archivos:
            imagen = cv2.imread(os.path.join(carpeta, archivo))
            if imagen is None:
                continue
            if numeroCanales == 3:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = np.expand_dims(imagen, axis=-1)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.astype("float32")
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[idx] = 1
            valorEsperado.append(probabilidades)
    return np.array(imagenesCargadas, dtype="float32"), np.array(valorEsperado, dtype="float32")


# ============================================================
# CONFIGURACIÓN
# ============================================================
# Las arquitecturas de Keras con ImageNet esperan 3 canales y mínimo 32x32.
ancho         = 32
alto          = 32
numeroCanales = 3

formaImagen      = (alto, ancho, numeroCanales)
numeroCategorias = len(CATEGORIAS)

# Opciones: "vgg16", "vgg19", "resnet50", "mobilenetv2", "densenet121"
nombreModelo = os.environ.get("EXP_MODELO",   "vgg16")
pesos        = os.environ.get("EXP_PESOS",    "imagenet")

# Transfer learning: congelar_base = True
# Fine tuning:       congelar_base = False
congelar_base = os.environ.get("EXP_CONGELAR", "True") == "True"

learning_rate = 0.0001

DIR_BASE    = os.path.dirname(os.path.abspath(__file__))
DIR_TRAIN   = os.path.join(DIR_BASE, "dataset", "train")
DIR_VAL     = os.path.join(DIR_BASE, "dataset", "val")
DIR_TEST    = os.path.join(DIR_BASE, "dataset", "test")
DIR_MODELOS = os.path.join(DIR_BASE, "models")

os.makedirs(DIR_MODELOS, exist_ok=True)

def _contar(carpeta):
    return len([f for f in os.listdir(carpeta) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

cantidaDatosEntrenamiento = [_contar(os.path.join(DIR_TRAIN, c)) for c in CATEGORIAS]
cantidaDatosPruebas       = [_contar(os.path.join(DIR_TEST,  c)) for c in CATEGORIAS]


# ============================================================
# CARGAR TRAIN COMPLETO
# ============================================================
print("Cargando datos...")
imagenes, probabilidades = cargarDatos(
    DIR_TRAIN, numeroCategorias, cantidaDatosEntrenamiento, ancho, alto, numeroCanales
)

# ============================================================
# CARGAR VALIDACIÓN
# ============================================================
x_train, y_train = imagenes, probabilidades
x_val,   y_val   = cargarDatos(
    DIR_VAL, numeroCategorias, cantidaDatosEntrenamiento, ancho, alto, numeroCanales
)

print("Datos entrenamiento:", x_train.shape)
print("Datos validación:", x_val.shape)


# ============================================================
# CREAR MODELO CON FACTORY
# ============================================================
model = ModeloPreentrenadoFactory.crear(
    nombreModelo=nombreModelo,
    formaImagen=formaImagen,
    numeroCategorias=numeroCategorias,
    pesos=pesos,
    congelar_base=congelar_base,
    learning_rate=learning_rate
)

print("\nModelo creado:")
print("Arquitectura:", nombreModelo)
print("Pesos:", pesos)
print("Base congelada:", congelar_base)
print()
model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    min_delta=0.0001,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# ENTRENAR
# ============================================================
historial = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    shuffle=True
)


# ============================================================
# CURVA DE PÉRDIDA
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(historial.history["loss"],     label="Entrenamiento")
ax1.plot(historial.history["val_loss"], label="Validacion")
ax1.set_xlabel("Epoca"); ax1.set_ylabel("Perdida")
ax1.set_title(f"Curva de perdida - {nombreModelo}")
ax1.legend(); ax1.grid(True)

ax2.plot(historial.history["accuracy"],     label="Entrenamiento")
ax2.plot(historial.history["val_accuracy"], label="Validacion")
ax2.set_xlabel("Epoca"); ax2.set_ylabel("Accuracy")
ax2.set_title(f"Curva de accuracy - {nombreModelo}")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(DIR_MODELOS, f"curvas_{nombreModelo}_{'TL' if congelar_base else 'FT'}.png"), dpi=150)
plt.show(block=False)
plt.pause(5)
plt.close('all')


# ============================================================
# CARGAR DATOS DE PRUEBA
# ============================================================
imagenesPrueba, probabilidadesPrueba = cargarDatos(
    DIR_TEST, numeroCategorias, cantidaDatosPruebas, ancho, alto, numeroCanales
)


# ============================================================
# EVALUAR MODELO
# ============================================================
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Loss test =", resultados[0])
print("Accuracy test =", resultados[1])


# ============================================================
# MÉTRICAS SKLEARN Y MATRIZ DE CONFUSIÓN
# ============================================================
predicciones = model.predict(imagenesPrueba)
y_pred = np.argmax(predicciones, axis=1)
y_true = np.argmax(probabilidadesPrueba, axis=1)

print("\nReporte de clasificacion:")
print(classification_report(y_true, y_pred, target_names=CATEGORIAS, digits=4))

report   = classification_report(y_true, y_pred, target_names=CATEGORIAS, digits=4, output_dict=True)
f1_macro = report["macro avg"]["f1-score"]
tipo     = "Transfer learning" if congelar_base else "Fine tunning y pesos"
with open(os.path.join(DIR_MODELOS, "resultados_f1.txt"), "a") as f:
    f.write(f"{nombreModelo} | {tipo} | {f1_macro:.4f}\n")
print(f"\nF1 Score Macro: {f1_macro:.4f} (guardado en resultados_f1.txt)")

matriz = confusion_matrix(y_true, y_pred)
disp   = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=CATEGORIAS)
disp.plot(cmap="Blues")
plt.title(f"Matriz de confusion - {nombreModelo}")
plt.savefig(os.path.join(DIR_MODELOS, f"confusion_{nombreModelo}_{'TL' if congelar_base else 'FT'}.png"), dpi=150)
plt.show(block=False)
plt.pause(5)
plt.close('all')


# ============================================================
# GUARDAR MODELO
# ============================================================
modo        = "scratch" if pesos == "none" else "imagenet"
estado_base = "frozen" if congelar_base else "finetuning"
ruta        = os.path.join(DIR_MODELOS, f"modelo_{nombreModelo}_{modo}_{estado_base}.keras")
model.save(ruta)
print(f"\nModelo guardado en: {ruta}")

