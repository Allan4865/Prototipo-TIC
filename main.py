import cv2
import time
import pyttsx3
from ultralytics import YOLO
from collections import deque

# ---------- parámetros ----------
PROX_NEAR = 0.4  # umbral de proximidad para "near"
PROX_MID = 0.7  # umbral de proximidad para "mid"
CONF_THRESHOLD = 0.60  # confianza mínima de detección
INTEREST = {
    "laptop", "keyboard", "mouse",
    "tv", "tvmonitor", "tv-monitor",
    "cell phone", "cellphone",
    "book", "books"
}
COOLDOWN = 1.5  # seg entre locuciones
FRAMES_TO_CONFIRM = 3  # frames consecutivos para confirmar detección
# ---------------------------------

last_tts_time = 0

detector = YOLO("yolov8n.pt")
depth_net = cv2.dnn.readNet("model-small.onnx")

tts = pyttsx3.init()
tts.setProperty("rate", 165)
voice_on = True

cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)  # Cambia al número de índice correcto
if not cap.isOpened():
    raise RuntimeError("No se encontró la OBS Virtual Camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Estado mejorado para tracking de objetos
object_states = {cls: {
    'is_near': False,
    'consecutive_near': 0,
    'consecutive_not_near': 0,
    'announced': False
} for cls in INTEREST}

# FPS
fps_times = deque(maxlen=10)

def depth_map(img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (256, 256), swapRB=True, crop=False)
    depth_net.setInput(blob)
    d = depth_net.forward()[0, 0]
    d = cv2.resize(d, (img.shape[1], img.shape[0]))
    return cv2.normalize(d, None, 0, 1, cv2.NORM_MINMAX)

def speak(txt):
    global last_tts_time
    if not voice_on:
        return
    now = time.time()
    if now - last_tts_time < COOLDOWN:
        return
    tts.stop()  # limpia cola
    tts.say(txt)
    tts.runAndWait()
    last_tts_time = now

def calculate_fps():
    now = time.time()
    fps_times.append(now)
    if len(fps_times) > 1:
        return len(fps_times) / (fps_times[-1] - fps_times[0])
    return 0

print("[INFO] 'q' = quit | 'm' = mute/unmute voice")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Depth map optimizado
        dmap = depth_map(frame)

        # Detección de objetos
        results = detector(frame, verbose=False)[0]
        names = detector.names

        # Resetear contadores para objetos no detectados en este frame
        detected_objects = set()
        
        for box in results.boxes:
            if float(box.conf) < CONF_THRESHOLD:
                continue
            cls = names[int(box.cls)]
            if cls not in INTEREST:
                continue

            detected_objects.add(cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth_val = dmap[cy, cx]

            # Categorizar distancias
            if depth_val < PROX_NEAR:
                tag = "near"
                is_near = True
            elif depth_val < PROX_MID:
                tag = "mid"
                is_near = False
            else:
                tag = "far"
                is_near = False

            # Dibujar cajas y texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {tag}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Actualizar estado del objeto
            state = object_states[cls]
            
            if is_near:
                state['consecutive_near'] += 1
                state['consecutive_not_near'] = 0
                
                # Confirmar que está cerca por varios frames consecutivos
                if state['consecutive_near'] >= FRAMES_TO_CONFIRM and not state['announced']:
                    print(f"[VOZ]: {cls} near")
                    speak(f"{cls} near")
                    state['announced'] = True
                    state['is_near'] = True
            else:
                state['consecutive_not_near'] += 1
                state['consecutive_near'] = 0
                
                # Si ya no está cerca por varios frames, resetear
                if state['consecutive_not_near'] >= FRAMES_TO_CONFIRM:
                    if state['is_near']:
                        print(f"[INFO]: {cls} ya no está cerca")
                    state['is_near'] = False
                    state['announced'] = False

        # Resetear objetos que ya no se detectan
        for cls in INTEREST:
            if cls not in detected_objects:
                state = object_states[cls]
                state['consecutive_near'] = 0
                state['consecutive_not_near'] += 1
                
                if state['consecutive_not_near'] >= FRAMES_TO_CONFIRM:
                    if state['is_near']:
                        print(f"[INFO]: {cls} desapareció")
                    state['is_near'] = False
                    state['announced'] = False

        # Mostrar FPS en pantalla
        fps = calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostrar estado de objetos detectados
        y_offset = 60
        for cls, state in object_states.items():
            if state['is_near']:
                cv2.putText(frame, f"{cls}: NEAR", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20

        cv2.imshow("Desk detection + depth", frame)

        # Control de teclado
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            voice_on = not voice_on
            print("[INFO] Voice", "enabled" if voice_on else "muted")

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()