import socket
import requests

# URL base de la API
base_url = "http://localhost:8000"

# Ejemplo de cómo agregar un texto (POST)
def add_text_api(text):
    url = f"{base_url}/add/"
    payload = {
        "text": text
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("id del agregado:", response.json())
    else:
        print("Error al agregar el texto:", response.status_code, response.json())

# Ejemplo de cómo obtener un texto (GET)
def get_text_api(item_id):
    url = f"{base_url}/get/{item_id}"
    response = requests.get(url)
    if response.status_code == 200:
        print("Texto obtenido:", response.json())
    else:
        print("Error al obtener el texto:", response.status_code, response.json())

# Ejemplo de cómo eliminar un texto (DELETE)
def delete_text_api(item_id):
    url = f"{base_url}/delete/{item_id}"
    response = requests.delete(url)
    if response.status_code == 200:
        print("Texto eliminado:", response.json())
    else:
        print("Error al eliminar el texto:", response.status_code, response.json())

# Ejecución de ejemplos
if __name__ == "__main__":
    # Agregar texto
    # add_text_api("Hola, mundo")
    add_text_api(f"127.0.0.1:65432")

    codigo = input("ingrese condigo: ")
    # Obtener texto
    get_text_api(codigo)

    # Eliminar texto
    delete_text_api(codigo)

    # Intentar obtener el texto nuevamente
    get_text_api(codigo)