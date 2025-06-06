# faceduel_server_client.py - Juego completo con captura y sincronización

import socket
import threading
import json
import cv2
import mediapipe as mp
import numpy as np

# Configuración del servidor
HOST = '0.0.0.0'
PORT = 65432

clients = []
player_data = [None, None]
lock = threading.Lock()

# Inicialización de MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Configuración visual
target_face_ratio = 0.4  # Porcentaje deseado del alto que debe ocupar el rostro
countdown_seconds = 1

def handle_client(conn, player_id):
    global player_data
    conn.sendall(json.dumps({"msg": "connected", "player_id": player_id}).encode())

    while True:
        try:
            data = conn.recv(4096)
            if not data:
                break

            decoded = json.loads(data.decode())
            with lock:
                player_data[player_id] = decoded

            with lock:
                if all(player_data):
                    opponent_id = 1 - player_id
                    opponent_data = player_data[opponent_id]
                    conn.sendall(json.dumps({"opponent_data": opponent_data}).encode())
                    player_data = [None, None]

        except Exception as e:
            print(f"Error en cliente {player_id}: {e}")
            break

    conn.close()
    with lock:
        clients[player_id] = None

def accept_clients(server_socket):
    print("Esperando jugadores...")
    while len(clients) < 1:
        conn, addr = server_socket.accept()
        print(f"Jugador conectado desde {addr}")
        clients.append(conn)
        threading.Thread(target=handle_client, args=(conn, len(clients)-1)).start()


def captura_datos_jugador():
    cap = cv2.VideoCapture(0)
    datos = {}

    while True:
        start_time = None
        captured = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_detection.process(frame_rgb)
            results_hand = hands.process(frame_rgb)
            frame_height, frame_width = frame.shape[:2]

            if start_time is None:
                start_time = cv2.getTickCount()

            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            time_left = max(0, countdown_seconds - int(elapsed))
            cv2.putText(frame, f"Disparo en {time_left}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            if elapsed >= countdown_seconds and not captured:
                if results_face.detections:
                    bbox = results_face.detections[0].location_data.relative_bounding_box
                    cx = int((bbox.xmin + bbox.width / 2) * frame_width)
                    cy = int((bbox.ymin + bbox.height / 2) * frame_height)
                    datos['face_x'] = cx
                    datos['face_y'] = cy

                if results_hand.multi_hand_landmarks:
                    hand = results_hand.multi_hand_landmarks[0]
                    index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    hand_x = int(index.x * frame_width)
                    hand_y = int(index.y * frame_height)
                    datos['hand_x'] = hand_x
                    datos['hand_y'] = hand_y

                captured = True

            cv2.imshow("Captura Jugador", frame)
            if cv2.waitKey(1) & 0xFF == 27 or captured:
                break

    cap.release()
    cv2.destroyAllWindows()
    return datos


def renderiza_disparo(mis_datos, del_oponente):
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(canvas, (mis_datos['face_x'], mis_datos['face_y']), 30, (255, 255, 255), 2)
    cv2.circle(canvas, (del_oponente['hand_x'], del_oponente['hand_y']), 10, (0, 255, 0), -1)
    cv2.line(canvas, (del_oponente['hand_x'], del_oponente['hand_y']), (mis_datos['face_x'], mis_datos['face_y']), (0, 0, 255), 4)
    cv2.putText(canvas, "Disparo recibido!", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Impacto", canvas)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Iniciar servidor y aceptar jugador 2
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(2)

    try:
        accept_clients(server_socket)

        # Captura datos locales del jugador 1 (host)
        mis_datos = captura_datos_jugador()
        renderiza_disparo(mis_datos, player_data[1])

    except KeyboardInterrupt:
        print("\nServidor detenido manualmente.")
    finally:
        for conn in clients:
            if conn:
                conn.close()
        server_socket.close()

    # Espera datos del oponente desde la estructura compartida
    import time
    while player_data[1] is None:
        time.sleep(0.1)

    renderiza_disparo(mis_datos, player_data[1])
