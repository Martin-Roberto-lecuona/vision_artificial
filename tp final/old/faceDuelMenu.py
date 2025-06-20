import os
import platform
import threading

current_os = platform.system()

def iniciar_servidor():
    if current_os == "Windows":
        os.system('start cmd /k "python faceDuel_Unificado.py"')
    elif current_os == "Linux":
        os.system(f'gnome-terminal -- bash -c "python3 faceDuel_Unificado.py; exec bash"')

def iniciar_cliente():
    if current_os == "Windows":
        os.system('start cmd /k "python faceDuelClient_old.py"')
    elif current_os == "Linux":
        os.system(f'gnome-terminal -- bash -c "python3 faceDuelClient_old.py; exec bash"')

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
