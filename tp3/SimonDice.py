import cv2
import mediapipe as mp
import random
import time
import random

import numpy as np

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Lista de gestos válidos
GESTURES = ['Abierto', 'Punio', 'Uno', 'Paz', 'Tres', 'Cuernos', 'Pulgar', 'Menique']

THRES_HOLD = 0.8
def pantalla_inicio():
    pantalla = 255 * np.ones((500, 600, 3), dtype=np.uint8)
    reglas_base = [
        "Bienvenido a Simon Dice - Gestos",
        "",
        "Reglas del juego:",
        "- Si dice 'Simon dice', haz el gesto.",
        "- Si NO dice 'Simon dice', no lo hagas",
        "- Si fallas, vuelves al nivel 1.",
        "- Si aciertas, subes de nivel.",
        "- El juego se juega con la palma de la mano ",
        "apuntando hacia la camara.",
        "",
        "Selecciona la dificultad y el juego comenzara:",
        "1 - Facil (8s)    2 - Normal (3s)    3 - Dificil (1s)",
        "'ESC' para salir..."
    ]

    dificultad = "Normal"
    tiempo = 3  # por defecto

    while True:
        pantalla.fill(255)  # limpiar fondo
        y = 50
        for regla in reglas_base:
            cv2.putText(pantalla, regla, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)
            y += 35

        cv2.putText(pantalla, f"Dificultad seleccionada: {dificultad}", (30, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Simon Dice - Gestos", pantalla)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            dificultad = "Facil"
            tiempo = 8
            cv2.destroyAllWindows()
            return tiempo
        elif key == ord('2'):
            dificultad = "Normal"
            tiempo = 3
            cv2.destroyAllWindows()
            return tiempo
        elif key == ord('3'):
            dificultad = "Dificil"
            tiempo = 1
            cv2.destroyAllWindows()
            return tiempo
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            exit()

TIME_TO_RESPOND = pantalla_inicio()
cap = cv2.VideoCapture(1)
level = 1
current_gesture = None
start_time = 0
output = cv2.namedWindow("Output de puntaje")
cv2.moveWindow("Output de puntaje", 800, 100)
output_frame = 255 * np.ones((400, 500, 3), dtype=np.uint8)  # fondo blanco
output_lines = []


# Función para detectar el gesto
def detect_gesture(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)  # thumb
    else:
        fingers.append(0)

    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    gestures_map = {
        (1, 1, 1, 1, 1): 'Abierto',
        (0, 0, 0, 0, 0): 'Punio',
        (0, 1, 0, 0, 0): 'Uno',
        (0, 1, 1, 0, 0): 'Paz',
        (1, 1, 1, 0, 0): 'Tres',
        (0, 1, 0, 0, 1): 'Cuernos',
        (1, 0, 0, 0, 0): 'Pulgar',
        (0, 0, 0, 0, 1): 'Menique'
    }

    return gestures_map.get(tuple(fingers), 'unknown')

def new_gesture(GESTURES, THRES_HOLD):
    current_gesture = random.choice(GESTURES)
    prob_simon_dice = random.random()
    simon_dice = False
    if(THRES_HOLD > prob_simon_dice):
        # print(f"🎯 Simon dice: '{current_gesture.upper()}'")
        simon_dice = True
    # else:
        # print(f"🎯'{current_gesture.upper()}'")
    return current_gesture,simon_dice

def show_msj():
    output_frame = 255 * np.ones((400, 500, 3), dtype=np.uint8)  # fondo blanco
    y = 20
    for line in output_lines:
        cv2.putText(output_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1)
        y+=25
    cv2.imshow("Output de puntaje", output_frame)



def add_output(message):
    print(message)
    output_lines.append(message)
    if len(output_lines) > 15:
        output_lines.pop(0)

current_gesture, simon_dice = new_gesture(GESTURES, THRES_HOLD)
start_time = time.time()


while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    detected = 'none'
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected = detect_gesture(hand_landmarks)

    # Mostrar temporizador
    time_left = int(TIME_TO_RESPOND - (time.time() - start_time))
    if time_left >= 0:
        cv2.putText(frame, f"Tiempo: {time_left}s", (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar gesto actual e interpretado
    if (simon_dice):
        cv2.putText(frame, f"Simon dice: {current_gesture.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        cv2.putText(frame, f"{current_gesture.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Tú: {detected}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Nivel: {level}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    change_gesture = time.time() - start_time > TIME_TO_RESPOND or detected == current_gesture
    if (simon_dice):
        if time.time() - start_time > TIME_TO_RESPOND:
            # print(" Tiempo agotado. Reiniciando juego.")
            add_output(" Tiempo agotado. Reiniciando juego.")
            level = 1
        elif detected == current_gesture:
            # print("Correcto!")
            add_output("Correcto!")
            level += 1
    else:
        if time.time() - start_time > TIME_TO_RESPOND:
            # print("Correcto!")
            add_output("Correcto!")
            level += 1
        elif detected == current_gesture:
            # print(" Simon No LO DIJO. Reiniciando juego.")
            add_output(" Simon No LO DIJO. Reiniciando juego.")
            level = 1
    if (change_gesture):
        current_gesture, simon_dice = new_gesture(GESTURES, THRES_HOLD)

        start_time = time.time() 

    cv2.imshow("Simon Dice - Gestos", frame)

    if level > 10:
        # print("🎉Ganaste!")
        cv2.putText(frame, f"Ganaste! Puntaje: {level - 1}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Simon Dice - Gestos", frame)
        cv2.waitKey(3000)
        break
    
    show_msj()
    
    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
