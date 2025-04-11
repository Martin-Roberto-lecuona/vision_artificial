import cv2 as cv
import numpy as np
import os

def nothing(x):
    pass

# Cargar imágenes de referencia
def cargar_formas_referencia(path="referencias"):
    referencias = {}
    for archivo in os.listdir(path):
        nombre = os.path.splitext(archivo)[0]
        img = cv.imread(os.path.join(path, archivo), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(os.path.join(path, archivo))
        _, bin_ref = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        bin_ref = cv.morphologyEx(bin_ref, cv.MORPH_OPEN, kernel)
        bin_ref = cv.morphologyEx(bin_ref, cv.MORPH_CLOSE, kernel)

        contornos, _ = cv.findContours(bin_ref, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contornos:
            referencias[nombre] = contornos[0]
        color = (0, 255, 255) 
        img2 = cv.drawContours(img2, contornos, -1, color, 5)
        cv.imshow(nombre,img2)
       
    return referencias

# Inicializar webcam
cap = cv.VideoCapture(0)
cv.namedWindow("Parametros")

# Crear sliders
cv.createTrackbar("Umbral binario", "Parametros", 127, 255, nothing)
cv.createTrackbar("Tam Kernel", "Parametros", 1, 20, nothing)
cv.createTrackbar("Match max dist", "Parametros", 20, 100, nothing)

# Cargar referencias
formas_referencia = cargar_formas_referencia()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_out = frame.copy()

    # Convertir a escala de grises
    gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("blancoYNegro", gris)

    # Threshold binario
    umbral = cv.getTrackbarPos("Umbral binario", "Parametros")
    _, binaria = cv.threshold(gris, umbral, 255, cv.THRESH_BINARY_INV)

    cv.imshow("Parametros", binaria)
    # Operación morfológica
    tam = cv.getTrackbarPos("Tam Kernel", "Parametros") * 2 + 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (tam, tam))
    binaria = cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel)
    binaria = cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel)

    # Encontrar contornos
    contornos, _ = cv.findContours(binaria, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    match_umbral = cv.getTrackbarPos("Match max dist", "Parametros") / 100.0

    for cont in contornos:
        area = cv.contourArea(cont)
        if area < 1000:  # descartar contornos pequeños
            continue

        # Comparar con formas de referencia
        mejor_match = None
        mejor_dist = float("inf")
        for nombre, ref_cont in formas_referencia.items():
            dist = cv.matchShapes(cont, ref_cont, cv.CONTOURS_MATCH_I1, 0.0)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_match = nombre

        # Clasificar
        if mejor_dist < match_umbral:
            color = (0, 255, 0)  # verde
            etiqueta = mejor_match
        else:
            color = (0, 0, 255)  # rojo
            etiqueta = "Desconocido"

        # Dibujar contorno y etiqueta
        cv.drawContours(frame_out, [cont], -1, color, 2)
        x, y, w, h = cv.boundingRect(cont)
        cv.putText(frame_out, etiqueta, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostrar resultados
    cv.imshow("Original", frame)
    cv.imshow("Anotado", frame_out)

    if cv.waitKey(30) == 27:  # ESC para salir
        break

cap.release()
cv.destroyAllWindows()
