import socket
import json
import struct
import mediapipe as mp
import pickle

import cv2
import numpy as np

# Dirección IP del servidor (Jugador 1)
SERVER_HOST = '127.0.0.1'  # Reemplazar con la IP del host si están en distintas PCs
SERVER_PORT = 65432

# Inicialización de MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Datos de ejemplo para enviar (simulan coordenadas del disparo y de la cara)

my_data = {
    "hand_x": 120,
    "hand_y": 200,
    "face_x": 300,
    "face_y": 250
}
start_time = None
countdown_seconds = 3

def captura_datos_jugador(cap):
    datos = {}
    global start_time

    ret, frame = cap.read()
    #if not ret:
    #    continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)
    results_hand = hands.process(frame_rgb)
    frame_height, frame_width = frame.shape[:2]

    if start_time is None:
        start_time = cv2.getTickCount()

    elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    time_left = max(0, countdown_seconds - int(elapsed))
    cv2.putText(frame, f"Disparo en {time_left}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    #if elapsed >= countdown_seconds:
    if results_face.detections:
        bbox = results_face.detections[0].location_data.relative_bounding_box
        cx = int((bbox.xmin + bbox.width / 2) * frame_width)
        cy = int((bbox.ymin + bbox.height / 2) * frame_height)
        datos['face_x'] = cx
        datos['face_y'] = cy
    else:
        datos['face_x'] = 500
        datos['face_y'] = 500

    if results_hand.multi_hand_landmarks:
        hand = results_hand.multi_hand_landmarks[0]
        index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        hand_x = int(index.x * frame_width)
        hand_y = int(index.y * frame_height)
        datos['hand_x'] = hand_x
        datos['hand_y'] = hand_y
    else:
        datos['hand_x'] = 400
        datos['hand_y'] = 400
    if time_left == 0:
        start_time = None
    return datos, frame

def recv_all(sock, count):
    """Recibe exactamente `count` bytes del socket"""
    buf = b''
    while len(buf) < count:
        newbuf = sock.recv(count - len(buf))
        if not newbuf:
            return None
        buf += newbuf
    return buf

def renderiza_disparo(frame, mis_datos, del_oponente):
    #canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (mis_datos['face_x'], mis_datos['face_y']), 70, (255, 255, 255), 2)
    cv2.circle(frame, (del_oponente['hand_x'], del_oponente['hand_y']), 40, (0, 255, 0), -1)
    #cv2.line(canvas, (del_oponente['hand_x'], del_oponente['hand_y']), (mis_datos['face_x'], mis_datos['face_y']), (0, 0, 255), 4)
    #cv2.putText(canvas, "Disparo recibido!", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Cliente-Juego", frame)
    cv2.waitKey(1)

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

# Conexión al servidor
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    cap = cv2.VideoCapture(2)
    s.connect((SERVER_HOST, SERVER_PORT))
    print("Conectado al servidor. Esperando asignación...")

    # Recibir confirmación e ID de jugador
    init_data = s.recv(1024)
    init_info = json.loads(init_data.decode())
    print(f"Asignado como Jugador {init_info['player_id']}")
    while True:
        mis_datos, frameJug = captura_datos_jugador(cap)
        # Enviar datos locales
        s.sendall(json.dumps(mis_datos).encode())
        print("Datos enviados. Esperando datos del oponente...")

        # Recibir datos del oponente
        #data = s.recv(6000000)
        #opponent_data = json.loads(data.decode())
        #frame = np.array(opponent_data["frame"])

        #------------------------------------------------
        frame, del_oponente = recibir_datos_adversario(s)

        if frame is not None:
            cv2.imshow("Client-Oponente", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #------------------------------------------------

        print("Datos recibidos del oponente:")
        renderiza_disparo(frameJug, mis_datos, del_oponente)
        #print(opponent_data)


