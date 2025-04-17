import cv2 as cv
import numpy as np
import os

#from matplotlib.pyplot import imshow
green = (0, 255, 0)

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
    cv.createTrackbar("Umbral binario", "Parametros", 115, 255, nothing)
    cv.createTrackbar("Tam Kernel", "Parametros", 1, 20, nothing)

# Cargar imágenes de referencia
def setup():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_out = frame.copy()
        contornos = getContour(frame)
        res = cv.drawContours(frame_out, contornos, -1, (0,255,0), 2)
        cv.imshow("ok", res)
        key = cv.waitKey(30) & 0xFF
        if key == ord('c'):
            if len(contornos) > 1:
                print("Hay mas de un elemento detectado")
            else:
                print("El contorno se guardará como un círculo")
                contornosValidados["circulo"].append(contornos[0])
        if key == ord('t'):
            if len(contornos) > 1:
                print("Hay mas de un elemento detectado")
            else:
                print("El contorno se guardará como un triangulo")
                contornosValidados["triangulo"].append(contornos[0])
        if key == ord('x'):
            if len(contornos) > 1:
                print("Hay mas de un elemento detectado")
            else:
                print("El contorno se guardará como un cuadrado")
                contornosValidados["cuadrado"].append(contornos[0])
        if cv.waitKey(30) == 27:
            break

    return referencias


def getContour(frame):
    gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Threshold binario
    umbral = cv.getTrackbarPos("Umbral binario", "Parametros")
    _, binaria = cv.threshold(gris, umbral, 255, cv.THRESH_BINARY_INV)
    cv.imshow("Calibración", binaria)
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
cv.destroyAllWindows()
print("Calibración finalizada. Comenzando reconocimiento.")

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
        contornos = getContour(frame)
        cv.drawContours(frame_out, contornos, -1, green, 2)
        for contorno in contornos:
            x, y, w, h = cv.boundingRect(contorno)
            resultadoContorno, distancia = detectar_forma(contorno, contornosValidados)
            if distancia < 0.2 :
                text = resultadoContorno + " distancia: " + str(round(distancia, 4))
                cv.putText(frame_out, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        cv.imshow("Detección", frame_out)
        if cv.waitKey(30) == 27:
            break

createWindow()
executeModel()

#MatchShapes
#0.0	Formas idénticas
#0.01 - 0.1	Formas muy parecidas
#0.1 - 0.3	Parecidas (puede variar mucho)
#0.3	Poco parecidas o distintas
#1.0	Formas claramente diferentes