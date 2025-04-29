import cv2 as cv

# Constantes
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ESC_KEY = 27
CALIBRATION_WINDOW = "Calibración - Visualización"
DETECTION_WINDOW = "Detección en vivo"
PARAMS_WINDOW = "Parametros"

# Diccionario de contornos calibrados por forma
contornos_validados = {
    "triangulo": [],
    "cuadrado": [],
    "circulo": []
}

def nothing(x):
    pass

def create_trackbars():
    """Crea sliders para ajustar umbral y tamaño de kernel."""
    cv.createTrackbar("Umbral binario", PARAMS_WINDOW, 115, 255, nothing)
    cv.createTrackbar("Tam Kernel", PARAMS_WINDOW, 1, 20, nothing)
    cv.createTrackbar("Tamaño mínimo contornos", PARAMS_WINDOW, 5000, 50000, nothing)
    cv.createTrackbar("Distancia Máxima", PARAMS_WINDOW, 20, 100, nothing)

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

def detectar_forma(contorno_actual):
    """Compara un contorno con los calibrados y retorna la forma más parecida."""
    menor_distancia = float('inf')
    forma_detectada = None

    for forma, referencias in contornos_validados.items():
        for ref in referencias:
            distancia = cv.matchShapes(contorno_actual, ref, cv.CONTOURS_MATCH_I1, 0.0)
            if distancia < menor_distancia:
                menor_distancia = distancia
                forma_detectada = forma

    return forma_detectada, menor_distancia

def modo_calibracion(cap):
    """Permite capturar manualmente formas de referencia."""
    print("[MODO] Calibración activa. Presione 'c', 't' o 'x' para guardar forma.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo acceder a la cámara.")
            break

        frame_out = frame.copy()
        contornos = obtener_contornos(frame)
        cv.drawContours(frame_out, contornos, -1, GREEN, 2)
        cv.imshow(CALIBRATION_WINDOW, frame_out)

        key = cv.waitKey(30) & 0xFF

        if key in [ord('c'), ord('t'), ord('x')]:
            forma = {'c': 'circulo', 't': 'triangulo', 'x': 'cuadrado'}[chr(key)]
            if len(contornos) != 1:
                print("[ADVERTENCIA] Debe haber exactamente un solo contorno para guardar.")
            else:
                contornos_validados[forma].append(contornos[0])
                print(f"[INFO] Contorno guardado como {forma}.")

        if key == ESC_KEY:
            print("[MODO] Finalizando calibración...")
            break

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
            forma, distancia = detectar_forma(contorno)
            x, y, w, h = cv.boundingRect(contorno)
            match_umbral = cv.getTrackbarPos("Distancia Máxima", "Parametros") / 100.0
            if distancia < match_umbral:
                texto = f"{forma} (dist: {round(distancia, 4)})"
                cv.putText(frame_out, texto, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            else:
                cv.putText(frame_out, "Desconocido", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        cv.imshow(DETECTION_WINDOW, frame_out)

        if cv.waitKey(30) == ESC_KEY:
            print("[MODO] Finalizando detección...")
            break

def main():
    cap = cv.VideoCapture(2)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    # Configurar UI
    cv.namedWindow(PARAMS_WINDOW)
    create_trackbars()

    cv.namedWindow(CALIBRATION_WINDOW)
    modo_calibracion(cap)
    cv.destroyWindow(CALIBRATION_WINDOW)

    cv.namedWindow(DETECTION_WINDOW)
    modo_deteccion(cap)

    # Liberar recursos
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
