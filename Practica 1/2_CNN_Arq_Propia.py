import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =========================
# CONFIGURACIÓN
# =========================
ancho    = 28
alto     = 28
pixeles  = ancho * alto

numeroCanales    = 1
formaImagen      = (ancho, alto, numeroCanales)
numeroCategorias = 2
CATEGORIAS       = ["dosmil", "diesmil"]

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


# =========================
# CARGAR DATOS
# =========================
def cargarDatos(directorio, limites):
    imagenesCargadas = []
    valorEsperado    = []
    for idx, clase in enumerate(CATEGORIAS):
        carpeta  = os.path.join(directorio, clase)
        archivos = sorted([f for f in os.listdir(carpeta)
                           if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if limites[idx] is not None:
            archivos = archivos[:limites[idx]]
        for archivo in archivos:
            imagen = cv2.imread(os.path.join(carpeta, archivo))
            if imagen is None:
                continue
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.astype("float32") / 255.0
            imagen = imagen.flatten()
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[idx] = 1
            valorEsperado.append(probabilidades)
    return np.array(imagenesCargadas, dtype="float32"), np.array(valorEsperado, dtype="float32")


# =========================
# CARGAR TRAIN COMPLETO
# =========================
print("Cargando datos...")
imagenes, probabilidades = cargarDatos(DIR_TRAIN, cantidaDatosEntrenamiento)

# =========================
# CARGAR VALIDACIÓN
# =========================
x_train, y_train = imagenes, probabilidades
x_val,   y_val   = cargarDatos(DIR_VAL, cantidaDatosEntrenamiento)

print("Datos entrenamiento:", x_train.shape)
print("Datos validación:", x_val.shape)

# =========================
# CARGAR DATOS DE PRUEBA
# =========================
imagenesPrueba, probabilidadesPrueba = cargarDatos(DIR_TEST, cantidaDatosPruebas)
print("Datos prueba:", imagenesPrueba.shape)


# =========================
# CREAR MODELO
# =========================
model = Sequential([
    Input(shape=(pixeles,)),
    Reshape(formaImagen),

    Conv2D(
        filters=16,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
        name="capa_1"
    ),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(
        filters=36,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu",
        name="capa_2"
    ),
    MaxPooling2D(pool_size=2, strides=2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(numeroCategorias, activation="softmax")
])


# =========================
# COMPILAR
# =========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    min_delta=0.0001,
    restore_best_weights=True,
    verbose=1
)


# =========================
# ENTRENAR
# =========================
historial = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop],
    shuffle=True
)


# =========================
# CURVA DE PÉRDIDA
# =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(historial.history["loss"],     label="Entrenamiento")
ax1.plot(historial.history["val_loss"], label="Validacion")
ax1.set_xlabel("Epoca"); ax1.set_ylabel("Perdida")
ax1.set_title("Curva de perdida"); ax1.legend(); ax1.grid(True)

ax2.plot(historial.history["accuracy"],     label="Entrenamiento")
ax2.plot(historial.history["val_accuracy"], label="Validacion")
ax2.set_xlabel("Epoca"); ax2.set_ylabel("Accuracy")
ax2.set_title("Curva de accuracy"); ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(DIR_MODELOS, "curvas_CNN_Arq_Propia.png"), dpi=150)
plt.show(block=False)
plt.pause(5)
plt.close('all')


# =========================
# EVALUAR MODELO
# =========================
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Loss test =", resultados[0])
print("Accuracy test =", resultados[1])


# =========================
# MÉTRICAS SKLEARN Y MATRIZ DE CONFUSIÓN
# =========================
predicciones = model.predict(imagenesPrueba)
y_pred = np.argmax(predicciones, axis=1)
y_true = np.argmax(probabilidadesPrueba, axis=1)

print("\nReporte de clasificacion:")
print(classification_report(y_true, y_pred, target_names=CATEGORIAS, digits=4))

report   = classification_report(y_true, y_pred, target_names=CATEGORIAS, digits=4, output_dict=True)
f1_macro = report["macro avg"]["f1-score"]
with open(os.path.join(DIR_MODELOS, "resultados_f1.txt"), "a") as f:
    f.write(f"Propia | Sin pesos | {f1_macro:.4f}\n")
print(f"\nF1 Score Macro: {f1_macro:.4f} (guardado en resultados_f1.txt)")

matriz = confusion_matrix(y_true, y_pred)
disp   = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=CATEGORIAS)
disp.plot(cmap="Blues")
plt.title("Matriz de confusion")
plt.savefig(os.path.join(DIR_MODELOS, "confusion_CNN_Arq_Propia.png"), dpi=150)
plt.show(block=False)
plt.pause(5)
plt.close('all')


# =========================
# GUARDAR MODELO
# =========================
ruta = os.path.join(DIR_MODELOS, "CNN_Arq_Propia.keras")
model.save(ruta)
print(f"\nModelo guardado en: {ruta}")

