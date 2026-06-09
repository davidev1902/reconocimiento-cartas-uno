import cv2
import numpy as np
import os
import time

# =============================================
# CONFIGURACIÓN: cambia aquí qué billete capturar
# Opciones: "dosmil" o "diesmil"
CLASE_CAPTURA = "diesmil"
# =============================================


# ============================================
# FUNCIONES GENERALES
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def crear_carpetas():
    """
    Crea las carpetas para almacenar las imágenes de los billetes.
    """
    for clase in ["dosmil", "diesmil"]:
        carpeta = os.path.join(BASE_DIR, "data_origen", clase)
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            print(f"Carpeta '{carpeta}' creada.")



def obtener_siguiente_numero(carpeta):
    """
    Obtiene el siguiente número consecutivo para nombrar la imagen.
    """
    if not os.path.exists(carpeta):
        return 1
    
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.png')]
    if not archivos:
        return 1
    
    numeros = []
    for archivo in archivos:
        # Extrae el número del nombre (ej: dosmil_5.png -> 5)
        try:
            numero = int(archivo.split('_')[1].split('.')[0])
            numeros.append(numero)
        except:
            pass
    
    return max(numeros) + 1 if numeros else 1

def nada(x):
    """Función vacía requerida por los trackbars de OpenCV."""
    pass


def obtener_nombre_poligono(lados):
    """
    Retorna el nombre del polígono según la cantidad de lados.
    Si tiene más lados, se deja abierto como polígono genérico.
    """
    nombres = {
        3: "Triangulo",
        4: "Cuadrilatero",
        5: "Pentagono",
        6: "Hexagono",
        7: "Heptagono",
        8: "Octagono"
    }

    return nombres.get(lados, f"Poligono {lados} lados")


def ajustar_kernel_impar(valor):
    """
    OpenCV necesita que el kernel del Gaussian Blur sea impar.
    """
    if valor < 1:
        valor = 1

    if valor % 2 == 0:
        valor += 1

    return valor


# ============================================
# INTERFAZ
# ============================================

def crear_interfaz(nombre_ventana):
    """
    Crea la ventana principal y los controles interactivos.
    """

    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    cv2.createTrackbar(
        "Umbral Canny Bajo",
        nombre_ventana,
        48,
        255,
        nada
    )

    cv2.createTrackbar(
        "Umbral Canny Alto",
        nombre_ventana,
        48,
        255,
        nada
    )

    cv2.createTrackbar(
        "Gaussian Blur",
        nombre_ventana,
        0,
        31,
        nada
    )

    cv2.createTrackbar(
        "Area Minima",
        nombre_ventana,
        2500,
        3000,
        nada
    )

    cv2.createTrackbar(
        "Precision Poligono",
        nombre_ventana,
        0,
        20,
        nada
    )


def leer_controles(nombre_ventana):
    """
    Lee los valores actuales de los trackbars.
    """

    canny_bajo = cv2.getTrackbarPos(
        "Umbral Canny Bajo",
        nombre_ventana
    )

    canny_alto = cv2.getTrackbarPos(
        "Umbral Canny Alto",
        nombre_ventana
    )

    # Asegurar que el umbral alto sea siempre mayor que el bajo
    if canny_alto <= canny_bajo:
        canny_alto = canny_bajo + 1

    blur = cv2.getTrackbarPos(
        "Gaussian Blur",
        nombre_ventana
    )

    area_minima = cv2.getTrackbarPos(
        "Area Minima",
        nombre_ventana
    )

    precision = cv2.getTrackbarPos(
        "Precision Poligono",
        nombre_ventana
    )

    blur = ajustar_kernel_impar(blur)

    if precision < 1:
        precision = 1

    return canny_bajo, canny_alto, blur, area_minima, precision


# ============================================
# PROCESAMIENTO DE IMAGEN
# ============================================

def preprocesar_imagen(frame, blur):
    """
    Convierte la imagen a escala de grises y aplica Gaussian Blur.
    """

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gauss = cv2.GaussianBlur(
        gris,
        (blur, blur),
        0
    )

    return gris, gauss


def detectar_bordes(gauss, canny_bajo, canny_alto):
    """
    Aplica Canny para detectar bordes.
    """

    bordes = cv2.Canny(
        gauss,
        canny_bajo,
        canny_alto
    )

    kernel = np.ones((3, 3), np.uint8)

    bordes = cv2.dilate(
        bordes,
        kernel,
        iterations=1
    )

    return bordes


def encontrar_contornos(bordes):
    """
    Encuentra contornos externos en la imagen de bordes.
    """

    contornos, _ = cv2.findContours(
        bordes,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contornos


# ============================================
# DETECCIÓN DE POLÍGONOS
# ============================================

def clasificar_cuadrilatero(w, h):
    """
    Diferencia entre cuadrado y rectángulo usando relación ancho/alto.
    """

    relacion = w / float(h)

    if 0.90 <= relacion <= 1.10:
        return "Cuadrado"

    return "Rectangulo"


def clasificar_billete(w, h, area):
    """
    Clasifica el billete según sus dimensiones y área.
    Filtra por proporción de aspecto para evitar manos/caras.
    Retorna CLASE_CAPTURA o None.
    """
    # Los billetes colombianos tienen proporción ~2.35:1 (horizontal)
    # Aceptamos un rango flexible: 1.5 a 3.5 horizontal, o su inverso vertical
    razon = w / float(h) if h > 0 else 0
    es_horizontal = 1.5 <= razon <= 3.5
    es_vertical   = 1.5 <= (1.0 / razon) <= 3.5 if razon > 0 else False

    if not (es_horizontal or es_vertical):
        return None   # No tiene forma de billete — rechazar

    if area < 15000:  # Muy pequeño para ser billete
        return None

    return CLASE_CAPTURA


def dibujar_poligonos(frame, contornos, area_minima, precision):
    """
    Aproxima contornos a polígonos y los dibuja sobre la imagen.
    Retorna la imagen procesada y una lista de billetes detectados.
    """

    salida = frame.copy()
    billetes_detectados = []

    # Ordenar contornos desde el más grande para detectar primero el billete
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for contorno in contornos:

        area = cv2.contourArea(contorno)

        if area < area_minima:
            continue

        perimetro = cv2.arcLength(contorno, True)

        epsilon = (precision / 100) * perimetro

        aproximacion = cv2.approxPolyDP(
            contorno,
            epsilon,
            True
        )

        lados = len(aproximacion)
        x, y, w, h = cv2.boundingRect(aproximacion)

        # Aceptar también contornos grandes con bounding rect razonable
        razon_aspecto = w / float(h) if h > 0 else 0
        relleno = area / float(w * h) if w * h > 0 else 0

        # Aceptar el contorno del billete aunque no tenga exactamente 4 lados.
        if lados == 4 or (0.3 < razon_aspecto < 4.0 and 0.2 < relleno <= 1.0):
            forma = clasificar_cuadrilatero(w, h)
        else:
            continue

        cv2.drawContours(
            salida,
            [aproximacion],
            0,
            (0, 255, 0),
            3
        )

        for punto in aproximacion:
            px, py = punto[0]

            cv2.circle(
                salida,
                (px, py),
                5,
                (0, 0, 255),
                -1
            )

        cv2.rectangle(
            salida,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            salida,
            f"{forma} ({lados} lados)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Clasificar el billete y guardar información solo del rectángulo principal
        tipo_billete = clasificar_billete(w, h, area)
        if tipo_billete is None:
            continue

        billetes_detectados.append({
            'tipo': tipo_billete,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'area': area,
            'contorno': aproximacion
        })

    return salida, billetes_detectados


# ============================================
# VISUALIZACIÓN
# ============================================

def convertir_gris_a_bgr(imagen):
    """
    Convierte una imagen en escala de grises a BGR para poder unirla en mosaico.
    """

    return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)


def crear_mosaico(frame, gris, gauss, bordes, salida):
    """
    Crea un mosaico con:
    Original, grises, Gaussian, Canny y resultado final.
    """

    ancho = 420
    alto = 300

    frame_r = cv2.resize(frame, (ancho, alto))
    gris_r = cv2.resize(convertir_gris_a_bgr(gris), (ancho, alto))
    gauss_r = cv2.resize(convertir_gris_a_bgr(gauss), (ancho, alto))
    bordes_r = cv2.resize(convertir_gris_a_bgr(bordes), (ancho, alto))
    salida_r = cv2.resize(salida, (ancho, alto))

    negro = np.zeros_like(frame_r)

    fila1 = np.hstack((frame_r, gris_r, gauss_r))
    fila2 = np.hstack((bordes_r, salida_r, negro))

    panel = np.vstack((fila1, fila2))

    return panel


def agregar_titulos(panel):
    """
    Agrega títulos sobre cada sección del mosaico.
    """

    cv2.putText(panel, "Original", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(panel, "Grises", (440, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(panel, "Gaussian Blur", (860, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(panel, "Canny", (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(panel, "Poligonos Detectados", (440, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(panel, "Presiona Q para salir", (860, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return panel


# ============================================
# PROGRAMA PRINCIPAL
# ============================================

def main():
    """
    Programa principal:
    - Captura video en tiempo real cada 1 segundo
    - Aplica filtros
    - Detecta bordes
    - Detecta y muestra polígonos de billetes
    - Guarda imágenes en carpetas según tipo de billete
    """

    nombre_ventana = "Detector de Billetes - Captura cada 1 segundo"
    
    # =============================================
    # CONFIGURACIÓN: cambia aquí qué billete capturar
    # Opciones: "dosmil" o "diesmil"
    # (también definido a nivel de módulo arriba)
    # =============================================

    # Crear carpetas
    crear_carpetas()
    
    # Variables para control de tiempo
    ultimo_tiempo_captura = time.time()
    intervalo_captura = 0.25  # segundos
    contador_capturas = {'dosmil': 0, 'diesmil': 0}

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    crear_interfaz(nombre_ventana)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer el frame.")
            break

        canny_bajo, canny_alto, blur, area_minima, precision = leer_controles(
            nombre_ventana
        )

        gris, gauss = preprocesar_imagen(
            frame,
            blur
        )

        bordes = detectar_bordes(
            gauss,
            canny_bajo,
            canny_alto
        )

        contornos = encontrar_contornos(
            bordes
        )

        salida, billetes_detectados = dibujar_poligonos(
            frame,
            contornos,
            area_minima,
            precision
        )
        
        # Verificar si ha pasado el intervalo de captura
        tiempo_actual = time.time()
        if tiempo_actual - ultimo_tiempo_captura >= intervalo_captura:
            # Capturar imagen si se detectó billete
            if billetes_detectados:
                for billete in billetes_detectados:
                    # Extraer la región de interés (ROI) - solo el billete
                    x = billete['x']
                    y = billete['y']
                    w = billete['w']
                    h = billete['h']
                    contorno = billete['contorno']
                    
                    # Extraer directamente la región rectangular del billete
                    roi = frame[y:y+h, x:x+w].copy()
                    
                    # Guardar siempre en la clase configurada (no por área)
                    contador_capturas[CLASE_CAPTURA] += 1
                    nombre_archivo = os.path.join(BASE_DIR, "data_origen", CLASE_CAPTURA,
                                                  f"{CLASE_CAPTURA}_{contador_capturas[CLASE_CAPTURA]}.png")
                    cv2.imwrite(nombre_archivo, roi)
                    print(f"✓ {nombre_archivo} (Area: {int(billete['area'])})")
                    
                    # Verificar si se alcanzó el total de 100 imágenes
                    total_capturas = contador_capturas['dosmil'] + contador_capturas['diesmil']
                    if total_capturas >= 200:
                        print(f"\n✓ ¡SE ALCANZARON 100 IMÁGENES!")
                        break
            
            ultimo_tiempo_captura = tiempo_actual
        
        # Verificar si se completó la captura de 100 imágenes
        total_capturas = contador_capturas['dosmil'] + contador_capturas['diesmil']
        if total_capturas >= 200:
            break

        panel = crear_mosaico(
            frame,
            gris,
            gauss,
            bordes,
            salida
        )

        panel = agregar_titulos(panel)
        
        # Mostrar información detallada de billetes detectados y progreso
        total_capturas = contador_capturas['dosmil'] + contador_capturas['diesmil']
        info_texto = f"Detectados: {len(billetes_detectados)} | Dosmil: {contador_capturas['dosmil']} | Diesmil: {contador_capturas['diesmil']} | Total: {total_capturas}/200"
        cv2.putText(panel, info_texto, (20, panel.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Mostrar detalles de cada billete detectado
        for i, billete in enumerate(billetes_detectados):
            detalle = f"Billete {i+1}: {billete['tipo'].upper()} (Area: {int(billete['area'])})"
            cv2.putText(panel, detalle, (20, panel.shape[0] - 20 - (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(nombre_ventana, panel)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    print(f"\n{'='*50}")
    print(f"CAPTURA COMPLETADA")
    print(f"{'='*50}")
    print(f"Billetes de 2k capturados: {contador_capturas['dosmil']}")
    print(f"Billetes de 10k capturados: {contador_capturas['diesmil']}")
    print(f"Total: {contador_capturas['dosmil'] + contador_capturas['diesmil']}")
    print(f"{'='*50}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
