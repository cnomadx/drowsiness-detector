"""
Drowsiness detection pipeline
  a. Image Operations  : grayscale, CLAHE, blur, crop, resize, overlay drawing
  c. Haar Cascade      : finds the face box that is fed to the landmark step
  f. Facial Landmarks  : MediaPipe eye points -> EAR -> awake / closed / drowsy

Keys: q = quit, s = save a screenshot (screenshots/ folder)
Events are logged to drowsiness_log.csv
"""
import csv
import math
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---- Settings you can tune ----
EAR_THRESHOLD = 0.20      # below this, the eyes count as closed
CLOSED_SECONDS = 1.5      # eyes closed this long = drowsy alert
FACE_MARGIN = 0.25        # extra space around the Haar box (fraction of size)
CROP_SIZE = 256           # face crop is resized to this before landmarking
LOST_GRACE_FRAMES = 5     # keep last face box this many frames if Haar misses

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(p):
    p1, p2, p3, p4, p5, p6 = p
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4))


# ---- (c) Haar cascade ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise SystemExit("Could not load Haar cascade")

# ---- (f) Landmark model ----
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ---- (a) CLAHE object ----
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def square_box(x, y, w, h, margin, W, H):
    """Expand a Haar box into a padded square, clamped to the frame."""
    side = int(max(w, h) * (1 + margin))
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, min(cx - side // 2, W - side))
    y0 = max(0, min(cy - side // 2, H - side))
    side = min(side, W, H)
    return x0, y0, side


os.makedirs("screenshots", exist_ok=True)
log_new = not os.path.exists("drowsiness_log.csv")
log_file = open("drowsiness_log.csv", "a", newline="")
log = csv.writer(log_file)
if log_new:
    log.writerow(["timestamp", "event", "ear", "closed_seconds"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

closed_start = None
drowsy_logged = False
last_box, lost_count = None, 0
last_ts = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    # ---------- (a) Image operations ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)   # light denoise
    gray = clahe.apply(gray)                   # even out lighting

    # ---------- (c) Haar cascade: find the face ----------
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])  # largest face
        last_box, lost_count = (fx, fy, fw, fh), 0
    else:
        lost_count += 1
        if lost_count > LOST_GRACE_FRAMES:
            last_box = None

    status, color = "NO FACE", (200, 200, 200)
    ear = None

    if last_box is not None:
        fx, fy, fw, fh = last_box
        x0, y0, side = square_box(fx, fy, fw, fh, FACE_MARGIN, W, H)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 128, 0), 2)

        # ---------- (a) crop + resize the Haar region ----------
        crop = frame[y0:y0 + side, x0:x0 + side]
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # ---------- (f) Landmarks on the Haar face region only ----------
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.time() * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            # map normalized crop coords back to full-frame pixels
            def to_px(i):
                return (x0 + lms[i].x * side, y0 + lms[i].y * side)
            left = [to_px(i) for i in LEFT_EYE]
            right = [to_px(i) for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

            for (x, y) in left + right:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ---------- Drowsiness logic ----------
    now = time.time()
    if ear is not None:
        if ear < EAR_THRESHOLD:
            if closed_start is None:
                closed_start = now
            closed_for = now - closed_start
            if closed_for >= CLOSED_SECONDS:
                status, color = "DROWSY! WAKE UP!", (0, 0, 255)
                if not drowsy_logged:
                    log.writerow([datetime.now().isoformat(timespec="seconds"),
                                  "DROWSY", f"{ear:.3f}", f"{closed_for:.1f}"])
                    log_file.flush()
                    drowsy_logged = True
            else:
                status, color = "Eyes closed", (0, 165, 255)
        else:
            closed_start, drowsy_logged = None, False
            status, color = "Awake", (0, 200, 0)
    else:
        closed_start, drowsy_logged = None, False

    cv2.putText(frame, status, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    if status.startswith("DROWSY"):
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 12)

    # small grayscale/CLAHE preview to show the image-ops stage in the demo
    preview = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (160, 120))
    frame[H - 120:H, W - 160:W] = preview

    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        name = f"screenshots/{datetime.now():%H%M%S}_{status.split()[0]}.png"
        cv2.imwrite(name, frame)
        print("saved", name)

cap.release()
log_file.close()
cv2.destroyAllWindows()
