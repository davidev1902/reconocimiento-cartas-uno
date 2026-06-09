import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# =========================
# CARGAR DATOS
# =========================
def cargarDatos(rutaOrigen, numeroCategorias, limite, ancho, alto):
    imagenesCargadas = []
    valorEsperado = []

    for categoria in range(numeroCategorias):
        for idImagen in range(1, limite[categoria]):
            ruta = f"{rutaOrigen}{categoria}/{categoria}_{idImagen}.jpg"
            print(ruta)

            imagen = cv2.imread(ruta)

            if imagen is None:
                print(f"No se pudo cargar la imagen: {ruta}")
                continue

            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.astype("float32") / 255.0
            imagen = imagen.flatten()

            imagenesCargadas.append(imagen)

            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)

    imagenes = np.array(imagenesCargadas, dtype="float32")
    valoresEsperados = np.array(valorEsperado, dtype="float32")

    return imagenes, valoresEsperados


# =========================
# CONFIGURACIÓN
# =========================
ancho = 28
alto = 28
pixeles = ancho * alto

numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
numeroCategorias = 2

porcentajeValidacion = 0.2

cantidaDatosEntrenamiento = [60,60]
cantidaDatosPruebas = [20,20]


# =========================
# CARGAR TRAIN COMPLETO
# =========================
imagenes, probabilidades = cargarDatos(
    "dataset/train/",
    numeroCategorias,
    cantidaDatosEntrenamiento,
    ancho,
    alto
)


# =========================
# SEPARAR TRAIN Y VALIDACIÓN
# =========================
x_train, x_val, y_train, y_val = train_test_split(
    imagenes,
    probabilidades,
    test_size=porcentajeValidacion,
    random_state=42,
    shuffle=True,
    stratify=np.argmax(probabilidades, axis=1)
)

print("Datos entrenamiento:", x_train.shape)
print("Datos validación:", x_val.shape)


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
    batch_size=60,
    callbacks=[early_stop],
    shuffle=True
)


# =========================
# CURVA DE PÉRDIDA
# =========================
plt.figure()
plt.plot(historial.history["loss"], label="Pérdida entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida validación")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Curva de pérdida")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# CARGAR DATOS DE PRUEBA
# =========================
imagenesPrueba, probabilidadesPrueba = cargarDatos(
    "dataset/test/",
    numeroCategorias,
    cantidaDatosPruebas,
    ancho,
    alto
)


# =========================
# EVALUAR MODELO
# =========================
resultados = model.evaluate(
    x=imagenesPrueba,
    y=probabilidadesPrueba
)

print("Loss test =", resultados[0])
print("Accuracy test =", resultados[1])


# =========================
# MÉTRICAS SKLEARN Y MATRIZ DE CONFUSIÓN
# =========================
predicciones = model.predict(imagenesPrueba)

y_pred = np.argmax(predicciones, axis=1)
y_true = np.argmax(probabilidadesPrueba, axis=1)

print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, digits=4))

matriz = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=list(range(numeroCategorias))
)

disp.plot(cmap="Blues")
plt.title("Matriz de confusión")
plt.show()


# =========================
# GUARDAR MODELO
# =========================
ruta = "models/modeloA.keras"
model.save(ruta)

model.summary()