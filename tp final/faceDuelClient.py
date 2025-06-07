import socket
import json
import struct

import cv2
import numpy as np

# Dirección IP del servidor (Jugador 1)
SERVER_HOST = '127.0.0.1'  # Reemplazar con la IP del host si están en distintas PCs
SERVER_PORT = 65432

# Datos de ejemplo para enviar (simulan coordenadas del disparo y de la cara)
my_data = {
    "hand_x": 120,
    "hand_y": 200,
    "face_x": 300,
    "face_y": 250
}

def recv_all(sock, count):
    """Recibe exactamente `count` bytes del socket"""
    buf = b''
    while len(buf) < count:
        newbuf = sock.recv(count - len(buf))
        if not newbuf:
            return None
        buf += newbuf
    return buf

# Conexión al servidor
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER_HOST, SERVER_PORT))
    print("Conectado al servidor. Esperando asignación...")

    # Recibir confirmación e ID de jugador
    init_data = s.recv(1024)
    init_info = json.loads(init_data.decode())
    print(f"Asignado como Jugador {init_info['player_id']}")
    while True:
        # Enviar datos locales
        s.sendall(json.dumps(my_data).encode())
        print("Datos enviados. Esperando datos del oponente...")

        # Recibir datos del oponente
        #data = s.recv(6000000)
        #opponent_data = json.loads(data.decode())
        #frame = np.array(opponent_data["frame"])

        #------------------------------------------------
        raw_msglen = recv_all(s, 4)
        if not raw_msglen:
            break
        msglen = struct.unpack('>I', raw_msglen)[0]

        # Ahora recibimos los bytes del frame
        frame_data = recv_all(s, msglen)
        if not frame_data:
            break

        # Convertimos los bytes en una imagen
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Oponente", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #------------------------------------------------

        print("Datos recibidos del oponente:")
        cv2.imshow("Oponente", frame)
        #print(opponent_data)
