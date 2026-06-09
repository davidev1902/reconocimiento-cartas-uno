
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from ModeloCNNPreentrenadoFactory import ModeloPreentrenadoFactory

# ============================================================
# CARGAR DATOS
# ============================================================
def cargarDatos(rutaOrigen, numeroCategorias, limite, ancho, alto, numeroCanales=3):
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

            if numeroCanales == 3:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = np.expand_dims(imagen, axis=-1)

            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.astype("float32")

            imagenesCargadas.append(imagen)

            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)

    imagenes = np.array(imagenesCargadas, dtype="float32")
    valoresEsperados = np.array(valorEsperado, dtype="float32")

    return imagenes, valoresEsperados





# ============================================================
# CONFIGURACIÓN
# ============================================================

# OJO:
# Las arquitecturas oficiales de Keras con ImageNet esperan 3 canales.
# Además, VGG/ResNet/MobileNet/DenseNet necesitan al menos 32x32.
ancho = 32
alto = 32
numeroCanales = 3

formaImagen = (alto, ancho, numeroCanales)
numeroCategorias = 10

porcentajeValidacion = 0.2

cantidaDatosEntrenamiento = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
cantidaDatosPruebas = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

# Opciones:
# "vgg16", "vgg19", "resnet50", "mobilenetv2", "densenet121"
nombreModelo = "vgg16"

# Para entrenar desde cero usa:
# pesos = "none"
#
# Para usar pesos preentrenados de ImageNet usa:
# pesos = "imagenet"
# si se quiere desde 0 entonces colocar "none"
pesos = "imagenet"

# Transfer learning:
# congelar_base = True
#
# Fine tuning:
# congelar_base = False
congelar_base = True

learning_rate = 0.0001


# ============================================================
# CARGAR TRAIN COMPLETO
# ============================================================
imagenes, probabilidades = cargarDatos(
    "dataset/train/",
    numeroCategorias,
    cantidaDatosEntrenamiento,
    ancho,
    alto,
    numeroCanales
)


# ============================================================
# SEPARAR TRAIN Y VALIDACIÓN
# ============================================================
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
    batch_size=32,
    callbacks=[early_stop],
    shuffle=True
)


# ============================================================
# CURVA DE PÉRDIDA
# ============================================================
plt.figure()
plt.plot(historial.history["loss"], label="Pérdida entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida validación")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title(f"Curva de pérdida - {nombreModelo}")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# CARGAR DATOS DE PRUEBA
# ============================================================
imagenesPrueba, probabilidadesPrueba = cargarDatos(
    "dataset/test/",
    numeroCategorias,
    cantidaDatosPruebas,
    ancho,
    alto,
    numeroCanales
)


# ============================================================
# EVALUAR MODELO
# ============================================================
resultados = model.evaluate(
    x=imagenesPrueba,
    y=probabilidadesPrueba
)

print("Loss test =", resultados[0])
print("Accuracy test =", resultados[1])


# ============================================================
# MÉTRICAS SKLEARN Y MATRIZ DE CONFUSIÓN
# ============================================================
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
plt.title(f"Matriz de confusión - {nombreModelo}")
plt.show()


# ============================================================
# GUARDAR MODELO
# ============================================================
modo = "scratch" if pesos == "none" else "imagenet"
estado_base = "frozen" if congelar_base else "finetuning"

ruta = f"models/modelo_{nombreModelo}_{modo}_{estado_base}.h5"
model.save(ruta)

print(f"Modelo guardado en: {ruta}")
