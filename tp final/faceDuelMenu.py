import os
import threading

def iniciar_servidor():
    os.system('start cmd /k "python faceDuelServer.py"')

def iniciar_cliente():
    os.system('start cmd /k "python faceDuelClient.py"')

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
