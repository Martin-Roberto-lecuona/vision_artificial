import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Constantes
GRID_SIZE = 3
TARGET_SQUARE = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)

# Variables globales
manual_points = []
homography = None
mode = 'view'

# OpenCV setup
cap = cv2.VideoCapture(0)
qr_detector = cv2.QRCodeDetector()

# Funciones auxiliares
def draw_grid(image, H, grid_size=GRID_SIZE):
    if H is None:
        return image
    result = image.copy()
    step = 300 // grid_size
    for i in range(0, 300 + 1, step):
        for j in range(0, 300 + 1, step):
            pt1 = np.array([[i, 0]], dtype=np.float32)
            pt2 = np.array([[i, 300]], dtype=np.float32)
            pt3 = np.array([[0, j]], dtype=np.float32)
            pt4 = np.array([[300, j]], dtype=np.float32)
            p1 = cv2.perspectiveTransform(np.array([pt1, pt2]), np.linalg.inv(H))
            p2 = cv2.perspectiveTransform(np.array([pt3, pt4]), np.linalg.inv(H))
            cv2.line(result, tuple(p1[0][0].astype(int)), tuple(p1[1][0].astype(int)), (0, 255, 0), 1)
            cv2.line(result, tuple(p2[0][0].astype(int)), tuple(p2[1][0].astype(int)), (0, 255, 0), 1)
    return result

def draw_frontal_view(frame, H):
    if H is None:
        return np.zeros((300, 300, 3), dtype=np.uint8)
    return cv2.warpPerspective(frame, H, (300, 300))

def on_click(event):
    global manual_points, homography, mode
    if mode == 'manual':
        manual_points.append([event.x, event.y])
        if len(manual_points) == 4:
            homography = cv2.getPerspectiveTransform(np.array(manual_points, dtype=np.float32), TARGET_SQUARE)
            manual_points.clear()
            mode = 'view'

def update_frame():
    global mode, homography

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    if mode == 'qr':
        retval, points = qr_detector.detect(frame)
        if retval and points is not None and points.shape[1] == 4:
            homography = cv2.getPerspectiveTransform(points[0].astype(np.float32), TARGET_SQUARE)
            mode = 'view'

    elif mode == 'manual':
        for pt in manual_points:
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)

    view = draw_grid(display, homography)
    topdown = draw_frontal_view(frame, homography)

    view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    topdown_rgb = cv2.cvtColor(topdown, cv2.COLOR_BGR2RGB)

    img1 = ImageTk.PhotoImage(Image.fromarray(view_rgb))
    img2 = ImageTk.PhotoImage(Image.fromarray(topdown_rgb))

    panel1.config(image=img1)
    panel1.image = img1

    panel2.config(image=img2)
    panel2.image = img2

    root.after(10, update_frame)

def set_mode_qr():
    global mode
    mode = 'qr'

def set_mode_manual():
    global mode, manual_points
    manual_points.clear()
    mode = 'manual'

def close_app():
    cap.release()
    root.destroy()

# GUI
root = tk.Tk()
root.title("Homografía - Proyecto")

btn_qr = tk.Button(root, text="Detección QR", command=set_mode_qr)
btn_qr.pack()

btn_manual = tk.Button(root, text="Homografía Manual", command=set_mode_manual)
btn_manual.pack()

panel1 = tk.Label(root)
panel1.pack()
panel1.bind("<Button-1>", on_click)

panel2 = tk.Label(root)
panel2.pack()

root.protocol("WM_DELETE_WINDOW", close_app)
update_frame()
root.mainloop()
