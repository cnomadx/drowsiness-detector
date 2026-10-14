import time
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---- Settings you can tune ----
EAR_THRESHOLD = 0.20      # below this, the eyes count as closed
CLOSED_SECONDS = 1.5      # eyes closed this long = drowsy alert

# Landmark indices for each eye: [corner, upper1, upper2, corner, lower2, lower1]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def dist(a, b):
    """Straight-line distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(points):
    """EAR: vertical eye opening divided by horizontal eye width."""
    p1, p2, p3, p4, p5, p6 = points
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4))


options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
closed_start = None   # time when the eyes first closed (None = eyes open)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    status, color = "NO FACE", (200, 200, 200)

    if result.face_landmarks:
        h, w = frame.shape[:2]
        lms = result.face_landmarks[0]

        left = [(lms[i].x * w, lms[i].y * h) for i in LEFT_EYE]
        right = [(lms[i].x * w, lms[i].y * h) for i in RIGHT_EYE]
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

        # Draw the eye points and the EAR value
        for (x, y) in left + right:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Drowsiness logic: how long have the eyes been closed?
        if ear < EAR_THRESHOLD:
            if closed_start is None:
                closed_start = time.time()
            closed_for = time.time() - closed_start
            if closed_for >= CLOSED_SECONDS:
                status, color = "DROWSY! WAKE UP!", (0, 0, 255)
            else:
                status, color = "Eyes closed", (0, 165, 255)
        else:
            closed_start = None
            status, color = "Awake", (0, 200, 0)
    else:
        closed_start = None  # reset if the face is lost

    cv2.putText(frame, status, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Red border while the drowsy alert is active
    if status.startswith("DROWSY"):
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                      (0, 0, 255), 12)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()