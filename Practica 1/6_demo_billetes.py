"""
Demo detección de billetes en tiempo real
- Detecta contorno rectangular del billete (polígono)
- Predicción con 3 modelos SOLO al presionar F
- Sin billete detectado o sin haber presionado F: no muestra números
- Tecla Q para salir
"""
import os
import cv2
import numpy as np
import torch
import timm
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELOS    = os.path.join(BASE, "models")
CATEGORIAS = ["dosmil", "diesmil"]

# ============================================================
# CARGAR MODELOS
# ============================================================
print("Cargando modelos...")
modelo_propio = tf.keras.models.load_model(os.path.join(MODELOS, "CNN_Arq_Propia.keras"))
modelo_vgg16  = tf.keras.models.load_model(os.path.join(MODELOS, "modelo_vgg16_imagenet_finetuning.keras"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_deit = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=2)
modelo_deit.load_state_dict(torch.load(
    os.path.join(MODELOS, "modelo_deit_small_imagenet_frozen.pth"), map_location=DEVICE
))
modelo_deit.to(DEVICE).eval()
print(f"Modelos listos. Device: {DEVICE}")


# ============================================================
# PREDICCIONES
# ============================================================
def predecir_propio(roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28)).astype("float32") / 255.0
    probs = modelo_propio.predict(img.flatten().reshape(1, -1), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CATEGORIAS[idx], float(probs[idx])

def predecir_vgg16(roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32)).astype("float32")
    probs = modelo_vgg16.predict(img.reshape(1, 32, 32, 3), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CATEGORIAS[idx], float(probs[idx])

def predecir_deit(roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32") / 255.0
    t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(modelo_deit(t), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return CATEGORIAS[idx], float(probs[idx])


# ============================================================
# DETECCIÓN DE POLÍGONO (billete)
# ============================================================
def detectar_billete(frame, canny_bajo=30, canny_alto=80, blur=5, area_minima=8000):
    gris   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k      = blur if blur % 2 == 1 else blur + 1
    gauss  = cv2.GaussianBlur(gris, (k, k), 0)
    bordes = cv2.Canny(gauss, canny_bajo, canny_alto)
    bordes = cv2.dilate(bordes, np.ones((5, 5), np.uint8), iterations=2)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return None, 0, 0, 0, 0, None

    # Tomar el contorno de mayor área
    c = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(c) < area_minima:
        return None, 0, 0, 0, 0, None

    x, y, w, h = cv2.boundingRect(c)
    # Filtrar por proporción de billete (debe ser rectangular horizontal o vertical)
    razon = w / float(h) if h > 0 else 0
    if not (0.4 < razon < 3.5):
        return None, 0, 0, 0, 0, None

    return c, x, y, w, h, frame[y:y+h, x:x+w]


# ============================================================
# MAIN
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

resultados = None   # None = no mostrar hasta que se presione F
COLORES    = [(255, 200, 0), (0, 200, 255), (200, 0, 255)]

print("Iniciando... (F=predecir, Q=salir)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    aprox, x, y, w, h, roi = detectar_billete(frame)
    tecla = cv2.waitKey(1) & 0xFF

    # Solo predice al presionar F y si hay billete detectado
    if tecla == ord('f'):
        if roi is not None and roi.size > 0:
            try:
                resultados = [
                    ("CNN Propia", *predecir_propio(roi)),
                    ("VGG16 FT",  *predecir_vgg16(roi)),
                    ("DeiT-S TL", *predecir_deit(roi)),
                ]
            except Exception as e:
                print(f"Error: {e}")
        else:
            resultados = None   # Limpiar si no hay billete

    # Dibujar contorno del billete
    if aprox is not None:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, "Billete detectado", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Sin billete", (10, frame.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    # Mostrar predicciones solo si ya se presionó F
    if resultados is not None:
        for i, (nombre, clase, confianza) in enumerate(resultados):
            texto = f"{nombre}: {clase} ({confianza*100:.1f}%)"
            yp    = 30 + i * 35
            cv2.rectangle(frame, (8, yp - 22), (390, yp + 8), (0, 0, 0), -1)
            cv2.putText(frame, texto, (12, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES[i], 2)

    cv2.putText(frame, "F=predecir  Q=salir", (8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Demo Billetes - 3 Modelos", frame)

    if tecla == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
