import cv2 as cv
import numpy as np

# Constantes
GREEN = (0, 255, 0)
ESC_KEY = 27
CALIBRATION_WINDOW = "Calibración - Visualización"
DETECTION_WINDOW = "Detección en vivo"
# Variables globales
cap = cv.VideoCapture(0)
contornos_validados = {
    "triangulo": [],
    "cuadrado": [],
    "circulo": []
}

def nothing(x):
    pass

def create_trackbars():
    """Crea sliders para ajustar parámetros."""
    cv.createTrackbar("Umbral binario", "Parametros", 115, 255, nothing)
    cv.createTrackbar("Tam Kernel", "Parametros", 1, 20, nothing)

def obtener_contornos(frame):
    """Procesa la imagen para detectar contornos."""
    gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    umbral = cv.getTrackbarPos("Umbral binario", "Parametros")
    _, binaria = cv.threshold(gris, umbral, 255, cv.THRESH_BINARY_INV)

    tam = cv.getTrackbarPos("Tam Kernel", "Parametros") * 2 + 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (tam, tam))
    binaria = cv.morphologyEx(binaria, cv.MORPH_OPEN, kernel)
    binaria = cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel)

    cv.imshow("Parametros", binaria)
    contornos, _ = cv.findContours(binaria, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contornos


def guardar_contorno(contorno, forma):
    """Guarda un contorno como referencia para una forma específica."""
    if contorno is not None:
        contornos_validados[forma].append(contorno)
        print(f"Contorno guardado como {forma}.")
    else:
        print("No se detectó un contorno válido.")


def modo_calibracion():
    """Permite capturar formas de referencia manualmente."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_out = frame.copy()
        contornos = obtener_contornos(frame)
        cv.drawContours(frame_out, contornos, -1, GREEN, 2)
        cv.imshow(CALIBRATION_WINDOW, frame_out)

        key = cv.waitKey(30) & 0xFF

        if key in [ord('c'), ord('t'), ord('x')]:
            forma = {'c': 'circulo', 't': 'triangulo', 'x': 'cuadrado'}[chr(key)]
            if len(contornos) != 1:
                print("Debe haber exactamente un solo contorno para guardar.")
            else:
                guardar_contorno(contornos[0], forma)

        if key == ESC_KEY:
            break


def detectar_forma(contorno_actual):
    """Compara un contorno con las formas calibradas y devuelve la más parecida."""
    menor_distancia = float('inf')
    forma_detectada = None

    for forma, contornos in contornos_validados.items():
        for ref in contornos:
            distancia = cv.matchShapes(contorno_actual, ref, cv.CONTOURS_MATCH_I1, 0.0)
            if distancia < menor_distancia:
                menor_distancia = distancia
                forma_detectada = forma

    return forma_detectada, menor_distancia


def modo_deteccion():
    """Ejecuta el reconocimiento de formas en tiempo real."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_out = frame.copy()
        cv.imshow("Parametros", frame_out)
        contornos = obtener_contornos(frame)
        cv.drawContours(frame_out, contornos, -1, GREEN, 2)

        for contorno in contornos:
            x, y, w, h = cv.boundingRect(contorno)
            forma, distancia = detectar_forma(contorno)

            if distancia < 0.2:
                texto = f"{forma} (dist: {round(distancia, 4)})"
                cv.putText(frame_out, texto, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        cv.imshow(DETECTION_WINDOW, frame_out)

        if cv.waitKey(30) == ESC_KEY:
            break


def main():
    cv.namedWindow("Parametros")  # Crear una vez la ventana de sliders
    create_trackbars()            # Crear sliders dentro de esa ventana

    print("Modo calibración. Presione ESC para finalizar.")
    cv.namedWindow(CALIBRATION_WINDOW)
    modo_calibracion()
    print("Calibración finalizada. Comenzando detección...")
    cv.destroyAllWindows()
    cv.namedWindow("Parametros")  # Crear una vez la ventana de sliders
    create_trackbars()
    cv.namedWindow(DETECTION_WINDOW)
    modo_deteccion()

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
