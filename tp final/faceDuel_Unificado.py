# TAREAS:
# TESTING:
    # verificar si los relojes estan sincronizados en red
    # permitir reanudar el juego cuando termina.
    # poner pistola en mano
    # poner sonido de 3 2 1 disparo
# DONE
    # al finalizar el juego mostrar quien ha ganado y quien ha perdido
    # No mostrar diana en tu frame para no saber donde apunta el enemigo
    # pausar 2 segundos el juego al disparar
    # liberar recursos. importante puerto
    # Mejorara apagado de juego o salidad de juego
    # usar menu de seleccion de crear o unirse


import socket
import struct
import json
import sys

import cv2
import mediapipe as mp
import pickle
import math
import numpy as np
import time
import ntplib

from playsound import playsound
#import beepy as beep
# Configuración del servidor
SERVER_HOST = '192.168.3.9'
CLIENT_HOST = '192.168.3.9'
PORT = 65432
start_time = None
face_radius = 110
hand_radius = 40
lifes_left = 10
beat_frames_jugador = 0  # Frames restantes para el efecto de latido
beat_frames_oponente = 0
last_face_image = None
last_face_time = None
default_face = False
oponent_default_face = False
rol = 'Servidor'
# Inicialización de MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
first_frame = 1
# Configuración visual
target_face_ratio = 0.4  # Porcentaje deseado del alto que debe ocupar el rostro
countdown_seconds = 3
# pistola_img = cv2.imread("pistola.png", cv2.IMREAD_UNCHANGED)

def superponer_pistola(frame, x, y, pistola_img):
    # h, w = pistola_img.shape[:2]
    # y -= int(h * 0.5)   # Ajuste vertical hacia arriba
    # x -= int(w * 0.3)   # Ajuste horizontal hacia la izquierda (depende de orientación del PNG)
    
    # y = max(0, min(y, frame.shape[0] - h))
    # x = max(0, min(x, frame.shape[1] - w))

    # alpha = pistola_img[:, :, 3] / 255.0
    # for c in range(3):
    #     frame[y:y+h, x:x+w, c] = frame[y:y+h, x:x+w, c] * (1 - alpha) + pistola_img[:, :, c] * alpha
    return frame

def obtener_coordenada_y(frame, posicion):
    # Obtener altura del frame
    alto = frame.shape[0]
    # Determinar la coordenada Y basada en la posición
    if posicion == 'arriba':
        return int(alto * 0.1)  # 10% desde la parte superior
    elif posicion == 'centro':
        return int(alto * 0.5)  # Centro del eje Y
    elif posicion == 'abajo':
        return int(alto * 0.9)  # 10% desde la parte inferior
    else:
        raise ValueError("Posición inválida para el eje Y: usa 'arriba', 'centro' o 'abajo'.")

def obtener_coordenada_x(frame, posicion):
    # Obtener ancho del frame
    ancho = frame.shape[1]

    # Determinar la coordenada X basada en la posición
    if posicion == 'izquierda':
        return int(ancho * 0.1)  # 10% desde la parte izquierda
    elif posicion == 'centro':
        return int(ancho * 0.5)  # Centro del eje X
    elif posicion == 'derecha':
        return int(ancho * 0.9)  # 10% desde la parte derecha
    else:
        raise ValueError("Posición inválida para el eje X: usa 'izquierda', 'centro' o 'derecha'.")

def recibir_datos_adversario(conn):
    global oponent_default_face
    # Leer exactamente 4 bytes para determinar el tamaño del mensaje
    raw_msglen = conn.recv(4)
    if not raw_msglen:
        return None

    # Desempaquetar el tamaño del mensaje (4 bytes big-endian)
    msglen = struct.unpack('>I', raw_msglen)[0]

    # Leer los datos completos basados en el tamaño recibido
    serialized_data = b''
    while len(serialized_data) < msglen:
        # Leer en partes hasta completar el tamaño esperado
        packet = conn.recv(msglen - len(serialized_data))
        if not packet:
            raise ConnectionError("La conexión se cerró antes de recibir los datos completos")
        serialized_data += packet

    # Deserializar el contenido usando pickle
    data_received = pickle.loads(serialized_data)
    np_frame = np.frombuffer(data_received['frame'], dtype=np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # Extraer frame y mis_datos del diccionario recibido
    del_oponente = data_received['datos']
    oponent_default_face = data_received['default_face']
    derrota_oponente = data_received['derrota']
    return frame, del_oponente, derrota_oponente

def accept_clients(server_socket):
    print("Esperando jugadores...")
    conn, addr = server_socket.accept()
    print(f"Jugador conectado desde {addr}")
    conn.sendall(json.dumps({"msg": "connected", "player_id": 0}).encode())
    return conn

def get_ntp_time():
    try:
        c = ntplib.NTPClient()
        response = c.request('pool.ntp.org', version=3)
        return response.tx_time
    except Exception as e:
        print(f"No se pudo obtener la hora NTP: {e}")
        return time.time()

def obtain_time_left():
    global start_time
    now = get_ntp_time()
    if start_time is None:
        start_time = now
    time_left = countdown_seconds - (now - start_time)
    return max(0, int(time_left))

def reproducir_cuenta_regresiva(i):
    # anda mal en windows. 
    if i < 0:
        playsound("beep.mp3")
    elif i == 0:
        playsound("disparo.mp3")

def captura_datos_jugador(cap, last_hand_x, last_hand_y, last_face_x, last_face_y):
    global default_face, first_frame
    global last_face_image, last_face_time
    datos = {}

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)
    results_hand = hands.process(frame_rgb)
    frame_height, frame_width = frame.shape[:2]

    time_left = obtain_time_left()
    cv2.putText(frame, f"Disparo en {time_left}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    reproducir_cuenta_regresiva(time_left)

    current_time = time.time()
    face_detected = False
    if results_face.detections:
        bbox = results_face.detections[0].location_data.relative_bounding_box
        cx = int((bbox.xmin + bbox.width / 2) * frame_width)
        cy = int((bbox.ymin + bbox.height / 2) * frame_height)
        datos['face_x'] = cx
        datos['face_y'] = cy
        default_face = False
        face_detected = True
        # Guardar la imagen de la cara cada 5 segundos
        if (last_face_time is None) or (current_time - last_face_time > 5):
            # Recortar la cara del frame
            x1 = max(int(bbox.xmin * frame_width), 0)
            y1 = max(int(bbox.ymin * frame_height), 0)
            x2 = min(int((bbox.xmin + bbox.width) * frame_width), frame_width)
            y2 = min(int((bbox.ymin + bbox.height) * frame_height), frame_height)
            last_face_image = frame[y1:y2, x1:x2].copy()
            last_face_time = current_time
    else:
        datos['face_x'] = last_face_x
        datos['face_y'] = last_face_y
        default_face = True

    if results_hand.multi_hand_landmarks:
        hand = results_hand.multi_hand_landmarks[0]
        index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        hand_x = int(index.x * frame_width)
        hand_y = int(index.y * frame_height)
        datos['hand_x'] = hand_x
        datos['hand_y'] = hand_y
    else:
        datos['hand_x'] = last_hand_x
        datos['hand_y'] = last_hand_y

    return datos, frame

def verificar_superposicion(mis_datos, del_oponente):
    # Obtener coordenadas de los centros
    x1, y1 = mis_datos['face_x'], mis_datos['face_y']
    x2, y2 = del_oponente['hand_x'], del_oponente['hand_y']

    # Calcular la distancia euclidiana entre los centros
    distancia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Verificar superposición
    if distancia <= (face_radius + hand_radius):
        return True  # Los círculos se superponen
    else:
        return False  # No se superponen

def aplicar_latido(frame, beat_frames_var, alpha=0.4, color=(0, 0, 255)):
    """Aplica un overlay de latido rojo (o color elegido) si beat_frames_var > 0."""
    if beat_frames_var > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        beat_frames_var -= 1
    return beat_frames_var

def dibujar_diana_sobre_frame(frame, x, y, radio_max):
    colores = [(0, 0, 255), (255, 255, 255)]  # rojo y blanco
    num_circulos = 5
    radios = [int(radio_max * (1 - i / num_circulos)) for i in range(num_circulos)]

    for i, radio in enumerate(radios):
        color = colores[i % 2]  # alternar rojo/blanco
        cv2.circle(frame, (x, y), radio, color, thickness=-1)

    # (Opcional) Cruz central
    cruz_size = radio_max // 5
    cv2.line(frame, (x - cruz_size, y), (x + cruz_size, y), (0, 0, 0), 2)
    cv2.line(frame, (x, y - cruz_size), (x, y + cruz_size), (0, 0, 0), 2)

    return frame

def renderiza_frames(frame_oponente, frame_jugador, mis_datos, del_oponente):
    global start_time, lifes_left, beat_frames_jugador, beat_frames_oponente, default_face, oponent_default_face, rol
    global first_frame
    # dibujar_diana_sobre_frame(frame_oponente, mis_datos['hand_x'], mis_datos['hand_y'], hand_radius)

    msj = "Te quedan " + str(lifes_left) + " vidas"
    cv2.putText(frame_jugador, msj, (10, frame_jugador.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    dibujar_diana_sobre_frame(frame_jugador, del_oponente['hand_x'], del_oponente['hand_y'], hand_radius)

    beat_frames_jugador = aplicar_latido(frame_jugador, beat_frames_jugador)
    beat_frames_oponente = aplicar_latido(frame_oponente, beat_frames_oponente)
    if(obtain_time_left() == 0):
        start_time = None
        impacto_jugador = verificar_superposicion(mis_datos, del_oponente)
        impacto_oponente = verificar_superposicion(del_oponente, mis_datos)
        if(impacto_jugador):
            lifes_left -= 1
            beat_frames_jugador = 5  # Activa el efecto de latido por 5 frames
        if(impacto_oponente):
            beat_frames_oponente = 5  # Activa el efecto de latido por 5 frames
        if impacto_jugador or impacto_oponente:
            time.sleep(0.5)  # Pausa post disparo
        if(lifes_left == 0):
            return 1
    # combinada = np.hstack((frame_oponente, frame_jugador))
    # frame_jugador = superponer_pistola(frame_jugador, mis_datos['hand_x'], mis_datos['hand_y'], pistola_img)
    cv2.imshow(rol + "_oponente", frame_oponente)
    cv2.waitKey(1)
    cv2.imshow(rol + "_jugador", frame_jugador)
    if(first_frame):
        first_frame = 0
        if(rol == 'Servidor'):
            cv2.moveWindow(rol + "_jugador", 1000, 100)
        else:
            cv2.moveWindow(rol + "_jugador", 1000, 700)
    cv2.waitKey(1)


def mostrar_derrota():
    cv2.destroyAllWindows()
    altura, ancho = 400, 800
    derrota_img = np.zeros((altura, ancho, 3), dtype=np.uint8)
    texto = "DERROTA"
    color_base = (0, 0, 255)
    for i in range(30):  # 30 frames ~1 segundo
        img = derrota_img.copy()
        # Efecto de destello: alterna el color del texto
        if i % 6 < 3:
            color = (0, 0, 255)
        else:
            color = (0, 0, 100 + 50 * (i % 3))
        # Efecto de "zoom": el texto crece
        escala = 3 + 0.1 * i
        grosor = 8 + i // 5
        # Fondo con líneas rojas animadas
        for j in range(0, ancho, 40):
            cv2.line(img, (j + (i*10)%40, 0), (j + (i*10)%40, altura), (0,0,80), 2)
        # Texto central
        (text_w, text_h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        x = (ancho - text_w) // 2
        y = (altura + text_h) // 2
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)
        cv2.imshow("Fin del juego", img)
        cv2.waitKey(30)
    # Mantener cartel final
    for i in range(100):
        img = derrota_img.copy()
        for j in range(0, ancho, 40):
            cv2.line(img, (j, 0), (j, altura), (0,0,80), 2)
        (text_w, text_h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 6, 14)
        x = (ancho - text_w) // 2
        y = (altura + text_h) // 2
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 6, color_base, 14, cv2.LINE_AA)
        cv2.imshow("Fin del juego", img)
        cv2.waitKey(50)
    cv2.destroyAllWindows()
    sys.exit()

def mostrar_victoria():
    cv2.destroyAllWindows()
    altura, ancho = 400, 800
    victoria_img = np.zeros((altura, ancho, 3), dtype=np.uint8)
    texto = "VICTORIA"
    color_base = (0, 255, 0)
    for i in range(30):  # 30 frames ~1 segundo
        img = victoria_img.copy()
        # Efecto de destello: alterna el color del texto
        if i % 6 < 3:
            color = (0, 255, 0)
        else:
            color = (0, 100 + 50 * (i % 3), 0)
        # Efecto de "zoom": el texto crece
        escala = 3 + 0.1 * i
        grosor = 8 + i // 5
        # Fondo con líneas verdes animadas
        for j in range(0, ancho, 40):
            cv2.line(img, (j + (i*10)%40, 0), (j + (i*10)%40, altura), (0,80,0), 2)
        # Calcular posición y tamaño del texto
        (text_w, text_h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        x = (ancho - text_w) // 2
        y = (altura + text_h) // 2
        # Añadir destellos animados acompañando el texto
        if i % 5 == 0:
            for k in range(8):
                ang = np.deg2rad(k * 45 + i*10)
                # Líneas desde el borde del texto hacia afuera
                x1 = int(x + text_w/2 + (text_w/2) * np.cos(ang))
                y1 = int(y - text_h/2 + (text_h/2) * np.sin(ang))
                x2 = int(x + text_w/2 + (text_w/2 + 80) * np.cos(ang))
                y2 = int(y - text_h/2 + (text_h/2 + 80) * np.sin(ang))
                cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 4)
        # Texto central
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)
        cv2.imshow("Fin del juego", img)
        cv2.waitKey(30)
    # Mantener cartel final
    for i in range(100):
        img = victoria_img.copy()
        for j in range(0, ancho, 40):
            cv2.line(img, (j, 0), (j, altura), (0,80,0), 2)
        (text_w, text_h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 6, 14)
        x = (ancho - text_w) // 2
        y = (altura + text_h) // 2
        # Destellos fijos acompañando el texto
        for k in range(8):
            ang = np.deg2rad(k * 45)
            x1 = int(x + text_w/2 + (text_w/2) * np.cos(ang))
            y1 = int(y - text_h/2 + (text_h/2) * np.sin(ang))
            x2 = int(x + text_w/2 + (text_w/2 + 80) * np.cos(ang))
            y2 = int(y - text_h/2 + (text_h/2 + 80) * np.sin(ang))
            cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 4)
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 6, color_base, 14, cv2.LINE_AA)
        cv2.imshow("Fin del juego", img)
        cv2.waitKey(50)
    cv2.destroyAllWindows()
    sys.exit()

def enviar_datos_adversario(conn, frame, mis_datos, derrota = 0):
    global default_face
    try:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if conn:
            data_to_send = {
                'frame': buffer.tobytes(),
                'datos': mis_datos,
                'default_face': default_face,
                'derrota': derrota
            }

            # Serializar el diccionario
            serialized_data = pickle.dumps(data_to_send)

            # Empaquetar el tamaño de los datos y enviar los datos serializados
            conn.sendall(struct.pack('>I', len(serialized_data)) + serialized_data)
        else:
            print("Socket no válido o desconectado.")
    except (BrokenPipeError, OSError) as e:
        print(f"Error al enviar datos, el cliente se desconectó: {e}")


def use_default_face(frame_jugador, mis_datos):
    global last_face_image
    if default_face and last_face_image is not None:
        # Si no se detecta la cara, mostrar la última imagen guardada en la ubicación estimada
        size = 2 * face_radius
        resized_face = cv2.resize(last_face_image, (size, size))
        x = int(mis_datos['face_x'] - size // 2)
        y = int(mis_datos['face_y'] - size // 2)
        # Asegurarse de que la imagen no se salga del frame
        x = max(0, min(x, frame_jugador.shape[1] - size))
        y = max(0, min(y, frame_jugador.shape[0] - size))
        frame_jugador[y:y + size, x:x + size] = resized_face
    return frame_jugador

def mostrar_menu():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Presiona 'c' para Crear partida", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Presiona 'u' para Unirse a partida", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Menu", img)

    while True:
        key = cv2.waitKey(0)
        if key == ord('c'):
            cv2.destroyAllWindows()
            return 1
        elif key == ord('u'):
            cv2.destroyAllWindows()
            return 2
def mostrar_new_game():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Presiona 'r' para volver a jugar", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Presiona 'esc' para salir", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Menu", img)

    while True:
        key = cv2.waitKey(0)
        if key == ord('r'):
            cv2.destroyAllWindows()
            return True
        else:
            cv2.destroyAllWindows()
            return False
def sincronizar_relojes(conn):
    t0 = time.time()
    conn.sendall(json.dumps({"sync": t0}).encode())
    response = conn.recv(1024)
    t1 = time.time()
    t_server = json.loads(response.decode())['server_time']
    print(f"Diferencia estimada: {(t_server - (t0 + t1)/2)*1000:.2f} ms")

def main():
    global rol, start_time
    newGame = False
    servidor = mostrar_menu()
    if(servidor == 1):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((SERVER_HOST, PORT))
        server_socket.listen(2)
        conn = accept_clients(server_socket)
        cap = cv2.VideoCapture(0)
        # Sincronización: el servidor obtiene la hora NTP y la envía al cliente
        ntp_start_time = get_ntp_time()
        start_time = ntp_start_time
        conn.sendall(json.dumps({"ntp_start_time": ntp_start_time}).encode())
    else:
        rol = 'Cliente'
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect((CLIENT_HOST, PORT))
        print("Conectado al servidor. Esperando asignación...")
        init_data = server_socket.recv(1024)
        init_info = json.loads(init_data.decode())
        print(f"Asignado como Jugador {init_info['player_id']}")
        conn = server_socket
        cap = cv2.VideoCapture(2)
        # Recibir hora NTP del servidor
        ntp_data = conn.recv(1024)
        ntp_info = json.loads(ntp_data.decode())
        start_time = ntp_info["ntp_start_time"]
    mis_datos = {}
    mis_datos['hand_x'] = 100
    mis_datos['hand_y'] = 100
    mis_datos['face_x'] = 100
    mis_datos['face_y'] = 100

    if servidor == 1:
        data = conn.recv(1024)
        if b"sync" in data:
            t_cliente = json.loads(data.decode())["sync"]
            t_servidor = time.time()
            conn.sendall(json.dumps({"server_time": t_servidor}).encode())
    else:
        sincronizar_relojes(conn)
    
    try:
        #Limitamos la cantidad de frames para mejorar el rendimiento
        frame_count = 0  
        while True:
            frame_count += 1
            if frame_count == 2: # pasa 1 y no pasa 2
                frame_count = 0
                continue  
            mis_datos, frame_jugador = captura_datos_jugador(cap, mis_datos['hand_x'], mis_datos['hand_y'], mis_datos['face_x'], mis_datos['face_y'])
            frame_jugador = use_default_face(frame_jugador, mis_datos)
            enviar_datos_adversario(conn, frame_jugador, mis_datos)
            frame_oponente, del_oponente, derrota_oponente = recibir_datos_adversario(conn)
            if(derrota_oponente == 1):
                mostrar_victoria()
            derrota = renderiza_frames(frame_oponente, frame_jugador, mis_datos, del_oponente)
            if(derrota == 1):
                enviar_datos_adversario(conn, frame_jugador, mis_datos, 1)
                mostrar_derrota()
            if cv2.waitKey(30) == 27:  # ESC para salir
                newGame = mostrar_new_game()
                break
    except KeyboardInterrupt:
        print("\nServidor detenido manualmente.")
    finally:
        if not newGame:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            if conn:
                conn.close()
            if servidor == 1 and server_socket:
                server_socket.close()
        return newGame

if __name__ == "__main__":
    newGame = True
    while newGame:
        newGame = False
        try:
            newGame = main()
        except Exception as e:
            newGame = False
            print("FINALIZANDO JUEGO")
