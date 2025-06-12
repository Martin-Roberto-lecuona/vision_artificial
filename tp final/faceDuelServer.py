# faceduel_server_client.py - Juego completo con captura y sincronización

import socket
import struct
import json
import cv2
import mediapipe as mp
import pickle
import math

# Configuración del servidor
HOST = '0.0.0.0'
PORT = 65432
start_time = None
face_radius = 120
hand_radius = 40
# Inicialización de MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Configuración visual
target_face_ratio = 0.4  # Porcentaje deseado del alto que debe ocupar el rostro
countdown_seconds = 3

def recibir_datos_adversario(conn):
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

    return frame, del_oponente

def accept_clients(server_socket):
    print("Esperando jugadores...")
    conn, addr = server_socket.accept()
    print(f"Jugador conectado desde {addr}")
    conn.sendall(json.dumps({"msg": "connected", "player_id": 0}).encode())
    return conn

def obtain_time_left():
    global start_time
    if start_time is None:
        start_time = cv2.getTickCount()
    elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    return max(0, countdown_seconds - int(elapsed))

def captura_datos_jugador(cap):
    datos = {}

    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)
    results_hand = hands.process(frame_rgb)
    frame_height, frame_width = frame.shape[:2]

    time_left = obtain_time_left()
    cv2.putText(frame, f"Disparo en {time_left}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    if results_face.detections:
        bbox = results_face.detections[0].location_data.relative_bounding_box
        cx = int((bbox.xmin + bbox.width / 2) * frame_width)
        cy = int((bbox.ymin + bbox.height / 2) * frame_height)
        datos['face_x'] = cx
        datos['face_y'] = cy
    else:
        datos['face_x'] = 100
        datos['face_y'] = 100

    if results_hand.multi_hand_landmarks:
        hand = results_hand.multi_hand_landmarks[0]
        index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        hand_x = int(index.x * frame_width)
        hand_y = int(index.y * frame_height)
        datos['hand_x'] = hand_x
        datos['hand_y'] = hand_y
    else:
        datos['hand_x'] = 200
        datos['hand_y'] = 200

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

def renderiza_disparo(frame, mis_datos, del_oponente):
    global start_time
    #canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (mis_datos['face_x'], mis_datos['face_y']), face_radius, (255, 255, 255), 2)
    cv2.circle(frame, (del_oponente['hand_x'], del_oponente['hand_y']), hand_radius, (0, 255, 0), -1)
    #cv2.line(canvas, (del_oponente['hand_x'], del_oponente['hand_y']), (mis_datos['face_x'], mis_datos['face_y']), (0, 0, 255), 4)
    #cv2.putText(canvas, "Disparo recibido!", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    if(obtain_time_left() == 0):
        start_time = None
        if(verificar_superposicion(mis_datos, del_oponente) == True):
            print("disparo acertado")
    cv2.imshow("Servidor-Juego", frame)
    cv2.waitKey(1)

def renderiza_oponente(frame, mis_datos, del_oponente):
    cv2.circle(frame, (del_oponente['face_x'], del_oponente['face_y']), face_radius, (255, 255, 255), 2)
    cv2.circle(frame, (mis_datos['hand_x'], mis_datos['hand_y']), hand_radius, (0, 255, 0), -1)
    cv2.imshow("Servidor-Oponente", frame)


def enviar_datos_adversario(conn, frame, mis_datos):
    try:
        if conn:
            data_to_send = {
                'frame': frame,
                'datos': mis_datos
            }

            # Serializar el diccionario
            serialized_data = pickle.dumps(data_to_send)

            # Empaquetar el tamaño de los datos y enviar los datos serializados
            conn.sendall(struct.pack('>I', len(serialized_data)) + serialized_data)
        else:
            print("Socket no válido o desconectado.")
    except (BrokenPipeError, OSError) as e:
        print(f"Error al enviar datos, el cliente se desconectó: {e}")


if __name__ == "__main__":
    # Iniciar servidor y aceptar jugador 2
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(2)
    cap = cv2.VideoCapture(0)

    try:
        conn = accept_clients(server_socket)
        while True:
            mis_datos, frameJug = captura_datos_jugador(cap)
            enviar_datos_adversario(conn, frameJug, mis_datos)
            frame, del_oponente = recibir_datos_adversario(conn)
            renderiza_oponente(frame, mis_datos, del_oponente)
            cv2.waitKey(1)
            renderiza_disparo(frameJug, mis_datos, del_oponente)
    except KeyboardInterrupt:
        print("\nServidor detenido manualmente.")
    finally:
        server_socket.close()
