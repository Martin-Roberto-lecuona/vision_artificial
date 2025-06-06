import socket
import json

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

# Conexión al servidor
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER_HOST, SERVER_PORT))
    print("Conectado al servidor. Esperando asignación...")

    # Recibir confirmación e ID de jugador
    init_data = s.recv(1024)
    init_info = json.loads(init_data.decode())
    print(f"Asignado como Jugador {init_info['player_id']}")

    # Enviar datos locales
    s.sendall(json.dumps(my_data).encode())
    print("Datos enviados. Esperando datos del oponente...")

    # Recibir datos del oponente
    data = s.recv(4096)
    opponent_data = json.loads(data.decode())
    print("Datos recibidos del oponente:")
    print(opponent_data)
