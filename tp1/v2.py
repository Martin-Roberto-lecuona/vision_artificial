import cv2 as cv
import numpy as np
import os

#from matplotlib.pyplot import imshow


def nothing(x):
    pass

# Inicializar webcam
cap = cv.VideoCapture(0)
cv.namedWindow("Setup")



referencias = {}
contornosValidados = {
    "triangulo": [],
    "cuadrado": [],
    "circulo": []
}


def createWindow():
    cv.namedWindow("Parametros")
    # Crear sliders
    cv.createTrackbar("Umbral binario", "Parametros", 127, 255, nothing)
    cv.createTrackbar("Tam Kernel", "Parametros", 1, 20, nothing)
    # cv.createTrackbar("Match max dist", "Parametros", 20, 100, nothing)


# Cargar imágenes de referencia
def setup():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_out = frame.copy()
        #----------
        # Convertir a escala de grises
        contornos = getContour(frame)
        #----
        res = cv.drawContours(frame_out, contornos, -1, (0,255,0), 2)
        cv.imshow("ok", res)
        key = cv.waitKey(30) & 0xFF
        if key == ord('c'):
            print("El contorno se guardará como un círculo")
            contornosValidados["circulo"].append(contornos[0])
        if key == ord('t'):
            print("El contorno se guardará como un triangulo")
            contornosValidados["triángulo"].append(contornos[0])
        if key == ord('x'):
            print("El contorno se guardará como un cuadrado")
            contornosValidados["cuadrado"].append(contornos[0])
        if cv.waitKey(30) == 27:
            break

    return referencias


def getContour(frame):


    gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("blancoYNegro", gris)
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
    contornos, _ = cv.findContours(binaria, cv.RETR_EXTERNAL, cv.CONTOURS_MATCH_I1)
    return contornos

createWindow()
setup()
#cap.release()
cv.destroyAllWindows()
print("Setup finalizado. Comenzando reconocimiento")
print(contornosValidados)

def detectar_forma(contorno_actual, contornosValidados):
    menor_distancia = float('inf')
    forma_detectada = None

    for tipo, lista_contornos in contornosValidados.items():
        for ref in lista_contornos:
            # Comparar el contorno actual con cada contorno guardado
            distancia = cv.matchShapes(contorno_actual, ref, cv.CONTOURS_MATCH_I1, 0.0)
            if distancia < menor_distancia:
                menor_distancia = distancia
                forma_detectada = tipo

    return forma_detectada, menor_distancia

def executeModel():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_out = frame.copy()
        cv.imshow("escaneando", frame_out)
        key = cv.waitKey(30) & 0xFF
        if key == ord('c'):
            contorno = getContour(frame)
            resultadoContorno, distancia = detectar_forma(contorno[0], contornosValidados)
            print("El contorno obtenido es un " + resultadoContorno + " con una distancia " + str(distancia))
        if cv.waitKey(30) == 27:
            break

createWindow()
executeModel()