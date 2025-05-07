import time
from collections import defaultdict, deque
import cv2
import numpy as np
import winsound;
from ultralytics import YOLO

# setup
AR_UP        = 1.3     # box maior que isso consideramos como em pé
AR_DOWN      = 0.8     # box menor que isso consideramos como uma queda
HISTORY      = 5       # quantos frames temos de guardar pós ID
ALERT_COOLDN = 3.0     # cooldown entre ID

model = YOLO("yolov8n-pose.pt") 
cap   = cv2.VideoCapture(0)
assert cap.isOpened(), "Webcam não encontrada"

aspect_hist   = defaultdict(lambda: deque(maxlen=HISTORY))
last_alert_ts = defaultdict(float)

# main loop -------------------------------------------------------------------
print("Loop em progresso, Pressione ESC para terminar...")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Loop finalizado.")
        break

    # track=True gives boxes.id (persisting) and keypoints
    res = model.track(frame, persist=True, imgsz=640, conf=0.25)[0]

    for box, pid in zip(res.boxes.xywh, res.boxes.id):
        if pid is None:              # shouldn’t happen, but be safe
            continue
        x, y, w, h = box.cpu().numpy()
        ar = h / (w + 1e-6)          # aspect ratio

        hist = aspect_hist[pid]
        hist.append(ar)

        # need enough history: was previously mostly upright?
        if len(hist) == HISTORY and max(hist) > AR_UP and ar < AR_DOWN:
            now = time.time()
            if now - last_alert_ts[pid] > ALERT_COOLDN:
                print(f"[{time.strftime('%H:%M:%S')}] FALL DETECTED  (ID {int(pid)})")
                winsound.Beep(1500, 300)
                last_alert_ts[pid] = now

        # draw bounding box & label
        color = (0, 0, 255) if ar < AR_DOWN else (0, 255, 0)
        p1 = int(x - w / 2), int(y - h / 2)
        p2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, p1, p2, color, 2)
        if ar < AR_DOWN:
            cv2.putText(frame, "FALL", (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Simple Fall Detector  –  ESC to quit", frame)
    if cv2.waitKey(1) == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()