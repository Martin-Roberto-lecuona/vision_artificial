import numpy as np
import cv2 as cv

print("Pulse ESC para terminar.")	# Instrucciones en consola
webcam = cv.VideoCapture(0) # webcam 0, podría ser 1, 2... para abrir otra si hay más de una

# Aquí se escribe el código de setup
# ...

while True:
    ret, imWebcam = webcam.read()
    cv.imshow('webcam', imWebcam)

    # Aquí se escribe el código para procesar la imagen imWebcam
    imGris = cv.cvtColor(imWebcam, cv.COLOR_BGR2GRAY)	# Código ejemplo


    # Aquí se escribe el código de visualización
    cv.imshow('blancoYNegro', imGris)	# Código ejemplo

    # Lee el teclado y decide qué hacer con cada tecla
    tecla = cv.waitKey(30)  # espera 30 ms. El mínimo es 1 ms.  tecla == 0 si no se pulsó ninguna.
    if tecla == 27:	# tecla ESC para salir
        break
    # aquí se pueden agregar else if y procesar otras teclas


cv.destroyAllWindows()
