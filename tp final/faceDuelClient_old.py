# faceDuelClient.py
import socket
import json
import struct
import mediapipe as mp
import pickle
import math
import cv2
import numpy as np
import time

# Dirección IP del servidor (Jugador 1)
SERVER_HOST = '127.0.0.1'  # Reemplazar con la IP del host si están en distintas PCs
SERVER_PORT = 65432

# Inicialización de MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

face_radius = 120
hand_radius = 40
lifes_left = 10
beat_frames_jugador = 0  # Frames restantes para el efecto de latido
beat_frames_oponente = 0
default_face = False
oponent_default_face = False
start_time = None
countdown_seconds = 3

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

def obtain_time_left():
    global start_time
    if start_time is None:
        start_time = cv2.getTickCount()
    elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    return max(0, countdown_seconds - int(elapsed))

def captura_datos_jugador(cap, last_hand_x, last_hand_y, last_face_x, last_face_y):
    global default_face
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

# Variables globales para la última imagen de la cara y el tiempo
last_face_image = None
last_face_time = None

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
    global start_time, lifes_left, beat_frames_jugador, beat_frames_oponente, default_face
    global last_face_image
    #if(oponent_default_face == True):
    #    cv2.circle(frame_oponente, (del_oponente['face_x'], del_oponente['face_y']), face_radius, (255, 255, 255), 2)
    #cv2.circle(frame_oponente, (mis_datos['hand_x'], mis_datos['hand_y']), hand_radius, (0, 255, 0), -1)
    dibujar_diana_sobre_frame(frame_oponente, mis_datos['hand_x'], mis_datos['hand_y'], hand_radius)
    msj = "Te quedan " + str(lifes_left) + " vidas"
    cv2.putText(frame_jugador, msj, (10, frame_jugador.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #if(default_face == True):
    #    cv2.circle(frame_jugador, (mis_datos['face_x'], mis_datos['face_y']), face_radius, (255, 255, 255), 2)

    cv2.circle(frame_jugador, (del_oponente['hand_x'], del_oponente['hand_y']), hand_radius, (0, 255, 0), -1)
    dibujar_diana_sobre_frame(frame_jugador, del_oponente['hand_x'], del_oponente['hand_y'], hand_radius)

    # Efecto de latido rojo si corresponde
    beat_frames_jugador = aplicar_latido(frame_jugador, beat_frames_jugador)
    beat_frames_oponente = aplicar_latido(frame_oponente, beat_frames_oponente)
    if(obtain_time_left() == 0):
        start_time = None
        if(verificar_superposicion(mis_datos, del_oponente) == True):
            print("disparo acertado")
            lifes_left = lifes_left - 1
            beat_frames_jugador = 5  # Activa el efecto de latido por 5 frames
        if(verificar_superposicion(del_oponente, mis_datos) == True):
            beat_frames_oponente = 5  # Activa el efecto de latido por 5 frames

    combinada = np.hstack((frame_oponente, frame_jugador))

    cv2.imshow("Juego cliente", combinada)
    cv2.waitKey(1)

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

    # Extraer frame y mis_datos del diccionario recibido
    frame = data_received['frame']
    del_oponente = data_received['datos']
    oponent_default_face = data_received['default_face']

    return frame, del_oponente


def enviar_datos_adversario(conn, frame, mis_datos):
    global default_face
    try:
        if conn:
            data_to_send = {
                'frame': frame,
                'datos': mis_datos,
                'default_face': default_face,
            }

            # Serializar el diccionario
            serialized_data = pickle.dumps(data_to_send)

            # Empaquetar el tamaño de los datos y enviar los datos serializados
            conn.sendall(struct.pack('>I', len(serialized_data)) + serialized_data)
        else:
            print("Socket no válido o desconectado.")
    except (BrokenPipeError, OSError) as e:
        print(f"Error al enviar datos, el cliente se desconectó: {e}")

def use_default_face(frame_jugador):
    if default_face and last_face_image is not None:
        # Si no se detecta la cara, mostrar la última imagen guardada en la ubicación estimada
        h, w, _ = last_face_image.shape
        x = int(mis_datos['face_x'] - w // 2)
        y = int(mis_datos['face_y'] - h // 2)
        # Asegurarse de que la imagen no se salga del frame
        x = max(0, min(x, frame_jugador.shape[1] - w))
        y = max(0, min(y, frame_jugador.shape[0] - h))
        frame_jugador[y:y + h, x:x + w] = last_face_image
    return frame_jugador


if __name__ == "__main__":
    # Conexión al servidor
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cap = cv2.VideoCapture(2)
    s.connect((SERVER_HOST, SERVER_PORT))
    print("Conectado al servidor. Esperando asignación...")

    # Recibir confirmación e ID de jugador
    init_data = s.recv(1024)
    init_info = json.loads(init_data.decode())
    print(f"Asignado como Jugador {init_info['player_id']}")
    #Limitamos la cantidad de frames para mejorar el rendimiento
    frame_count = 0  # Contador de frames
    mis_datos = {}
    mis_datos['hand_x'] = 100
    mis_datos['hand_y'] = 100
    mis_datos['face_x'] = 100
    mis_datos['face_y'] = 100
    while True:
        frame_count += 1
        if frame_count % 3 != 0:
            continue  # Saltar este frame para reducir la carga
        mis_datos, frame_jugador = captura_datos_jugador(cap, mis_datos['hand_x'], mis_datos['hand_y'], mis_datos['face_x'], mis_datos['face_y'])
        frame_jugador = use_default_face(frame_jugador)
        # Enviar datos locales
        enviar_datos_adversario(s, frame_jugador, mis_datos)
        #print("Datos enviados. Esperando datos del oponente...")
        frame_oponente, del_oponente = recibir_datos_adversario(s)
        renderiza_frames(frame_oponente, frame_jugador, mis_datos, del_oponente)