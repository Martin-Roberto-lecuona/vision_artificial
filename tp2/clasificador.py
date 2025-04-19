from joblib import load

import cv2 as cv

DETECTION_WINDOW = "Detección en vivo"

CLASIFICATION_WINDOW = "Clasificación"
GREEN = (0, 255, 0)
PARAMS_WINDOW = "Parametros"
ESC_KEY = 27
clasificador = load('filename.joblib')
def nothing(x):
    pass

X = []  # Descriptores de Hu
Y = []  # Etiquetas

dictionary = {
    1: "triangulo",
    2: "cuadrado",
    3: "circulo"
}
result = []
def create_trackbars():
    """Crea sliders para ajustar umbral y tamaño de kernel."""
    cv.createTrackbar("Umbral binario", PARAMS_WINDOW, 115, 255, nothing)
    cv.createTrackbar("Tam Kernel", PARAMS_WINDOW, 1, 20, nothing)
    cv.createTrackbar("Tamaño mínimo contornos", PARAMS_WINDOW, 5000, 50000, nothing)

def obtener_parametros():
    """Obtiene los valores actuales de los sliders."""
    umbral = cv.getTrackbarPos("Umbral binario", PARAMS_WINDOW)
    tam_kernel = cv.getTrackbarPos("Tam Kernel", PARAMS_WINDOW) * 2 + 1
    tam_min_contornos = cv.getTrackbarPos("Tamaño mínimo contornos", PARAMS_WINDOW)
    return umbral, tam_kernel, tam_min_contornos

def obtener_contornos(frame):
    """Procesa la imagen para detectar contornos."""
    gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    umbral, tam_kernel, tam_min_contornos = obtener_parametros()
    _, binaria = cv.threshold(gris, umbral, 255, cv.THRESH_BINARY_INV)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (tam_kernel, tam_kernel))
    binaria = cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel)
    binaria = cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel)

    cv.imshow(PARAMS_WINDOW, binaria)
    contornos, _ = cv.findContours(binaria, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filteredContours = [contorno for contorno in contornos if cv.contourArea(contorno) > tam_min_contornos]
    return filteredContours

def modo_deteccion(cap):
    """Reconocimiento de formas en tiempo real."""
    print("[MODO] Detección activa. Presione ESC para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo acceder a la cámara.")
            break

        frame_out = frame.copy()
        contornos = obtener_contornos(frame)
        cv.drawContours(frame_out, contornos, -1, GREEN, 2)

        for contorno in contornos:
            momentos = cv.moments(contornos[0])
            hu = cv.HuMoments(momentos)
            etiquetaPredicha = clasificador.predict(hu.flatten().reshape(1, -1))
            x, y, w, h = cv.boundingRect(contorno)
            prediction = dictionary[int(etiquetaPredicha[0])]
            cv.putText(frame_out, prediction, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

            cv.imshow(DETECTION_WINDOW, frame_out)

        if cv.waitKey(30) == ESC_KEY:
            print("[MODO] Finalizando detección...")
            break


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cv.namedWindow(PARAMS_WINDOW)
    create_trackbars()
    cv.namedWindow(DETECTION_WINDOW)
    modo_deteccion(cap)
    # Liberar recursos
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
