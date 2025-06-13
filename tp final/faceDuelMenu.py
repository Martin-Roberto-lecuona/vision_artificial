import os
import platform
import threading

current_os = platform.system()

def iniciar_servidor():
    if current_os == "Windows":
        os.system('start cmd /k "python faceDuelServer.py"')
    elif current_os == "Linux":
        os.system(f'gnome-terminal -- bash -c python3 faceDuelServer.py; exec bash"')

def iniciar_cliente():
    if current_os == "Windows":
        os.system('start cmd /k "python faceDuelClient.py"')
    elif current_os == "Linux":
        os.system(f'gnome-terminal -- bash -c python3 faceDuelClient.py; exec bash"')

CREAR_UNIR = input("¿Crear o unirse? (C/U): ").strip().upper()

if CREAR_UNIR == "C":
    hilo = threading.Thread(target=iniciar_servidor)
elif CREAR_UNIR == "U":
    hilo = threading.Thread(target=iniciar_cliente)
else:
    print("Opción inválida.")
    exit()

hilo.start()
hilo.join()
